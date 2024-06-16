from ._base import register_transform, register_mode_transform
import torch 
# from .atom_num_config import CONFIG
import numpy as np
from repo.utils.molecule.constants import *
from repo.utils.molecule.fg_constants import *
from repo.utils.protein.constants import *
from repo.models.utils.so3 import random_uniform_so3
from ._base import get_index
from easydict import EasyDict
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
config_atom_num = np.load(os.path.join(current_dir, '_atom_num_dist.npy'),
                          allow_pickle=True).item()
config_fg_num = np.load(os.path.join(current_dir, '_fg_num_dist.npy'),
                        allow_pickle=True).item()
config_linker_num = np.load(os.path.join(current_dir, '_linker_num_dist.npy'),
                            allow_pickle=True).item()


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index
 
def sample_atom_num(space_size):
    config = config_atom_num
    bin_idx = _get_bin_idx(space_size, config)
    num_atom_list, prob_list = config['bins'][bin_idx]
    return np.random.choice(num_atom_list, p=prob_list)

def sample_fg_num(space_size):
    config = config_fg_num
    bin_idx = _get_bin_idx(space_size, config)
    num_atom_list, prob_list = config['bins'][bin_idx]
    return np.random.choice(num_atom_list, p=prob_list)

def sample_linker_num(space_size):
    config = config_linker_num
    bin_idx = _get_bin_idx(space_size, config)
    num_atom_list, prob_list = config['bins'][bin_idx]
    return np.random.choice(num_atom_list, p=prob_list)


def _get_bin_idx(space_size, config):
    bounds = config['bounds']
    for i in range(len(bounds)):
        if bounds[i] > space_size:
            return i
    return len(bounds)

@register_mode_transform('load_ctx')
@register_transform('load_ctx')
class LoadCtx(object):
    def __init__(self, ctx_path, mode) -> None:
        self.ctx_path = ctx_path
        self.mode = mode

        if self.mode == 'add_aromatic':
            self.num_types = len(map_atom_type_aromatic_to_index)
        elif self.mode == 'basic':
            self.num_types = len(map_atom_type_only_to_index)
        elif self.mode == 'add_aromatic_hybrid':
            self.num_types = len(map_index_to_atom_type_full)
    
    def choose_latest_ctx_and_update_pool(self, data):
        ctx_correspond = os.path.join(self.ctx_path, data.entry[0][:-4])
        ctx_names = [name for name in os.listdir(ctx_correspond) if 'raw' not in name]
        ctx_num = [int(name.split('_')[-1].split('.')[0]) for name in ctx_names]
        ctx_latest = np.argmin(ctx_num)
        ctx_psvd = np.argmax(ctx_num)
        ctx_path_used = os.path.join(ctx_correspond, ctx_names[ctx_latest])
        ctx_used = torch.load(ctx_path_used)
        ctx_selected = ctx_used[0:1][0]
        del ctx_used[0]
        
        torch.save(ctx_used, os.path.join('/'.join(ctx_path_used.split('/')[:-1]), 
                                                   '_'.join(ctx_names[ctx_latest].split('_')[:-1] + ['%04d.pt' % len(ctx_used)])))
        
        ctx_names = [name for name in os.listdir(ctx_correspond) if 'raw' not in name]
        ctx_num = [int(name.split('_')[-1].split('.')[0]) for name in ctx_names]
        ctx_psvd = np.argmax(ctx_num)
        ctx_latest = np.argmin(ctx_num)
        ctx_removed_names = []
        for i in range(len(ctx_num)):
            if ctx_names[i] not in [ctx_names[ctx_latest], ctx_names[ctx_psvd]]:
                ctx_removed_names.append(ctx_names[i])
            if ctx_names[i][-7:-3] == '0000':
                ctx_removed_names.append(ctx_names[i])

        for name in ctx_removed_names:
            os.remove(os.path.join(ctx_correspond, name))
        return ctx_selected
        
    def __call__(self, data):
        ctx_selected = self.choose_latest_ctx_and_update_pool(data)
        if self.mode == 'basic':
            data.ligand_ctx = {'element': ctx_selected[0],
                               'pos': ctx_selected[1],
                               'lig_flag': torch.ones_like(ctx_selected[0], dtype=torch.bool)}
        elif self.mode == 'add_aromatic':
            data.ligand_ctx = {'element': ctx_selected[0],
                                'pos': ctx_selected[1],
                                'aromatic': ctx_selected[2],
                                'lig_flag': torch.ones_like(ctx_selected[0], dtype=torch.bool)}
        elif self.mode == 'add_aromatic_hybrid':
            data.ligand_ctx = {'element': ctx_selected[0],
                               'pos': ctx_selected[1],
                               'aromatic': ctx_selected[2],
                               'hybrid': ctx_selected[3],
                               'lig_flag': torch.ones_like(ctx_selected[0], dtype=torch.bool)}
            
        data.ligand_ctx.atom_type = torch.tensor([get_index(e.item(), h.item(), a.item(), self.mode) 
                                                  for e, h, a in zip(data.ligand_ctx.get('element', torch.zeros_like(ctx_selected[0])),
                                                                     data.ligand_ctx.get('hybrid', torch.zeros_like(ctx_selected[0])),
                                                                     data.ligand_ctx.get('aromatic', torch.zeros_like(ctx_selected[0])))])

        return data
        
@register_mode_transform('ar_init_mol_geo')
@register_transform('ar_init_mol_geo')
class ARInitMolGeo(object):
    def __init__(self, mode, num_gen=1) -> None:
        self.mode = mode

        if self.mode == 'add_aromatic':
            self.num_types = len(map_atom_type_aromatic_to_index)
        elif self.mode == 'basic':
            self.num_types = len(map_atom_type_only_to_index)
        elif self.mode == 'add_aromatic_hybrid':
            self.num_types = len(map_index_to_atom_type_full)
        
        self.num_gen = num_gen

    def __call__(self, data) -> torch.Any:
        data.ligand.atom_type = torch.empty([self.num_gen, 0], dtype=int)
        data.ligand.pos = torch.empty([self.num_gen, 0, 3])
        data.ligand.focuses = torch.empty([self.num_gen, 0], dtype=int)
        return EasyDict(data)


@register_mode_transform('ar_init_gen_geo')
@register_transform('ar_init_gen_geo')
class ARInitGenGeo(object):
    def __init__(self, mode, num_gen=1) -> None:
        self.mode = mode

        if self.mode == 'add_aromatic':
            self.num_types = len(map_atom_type_aromatic_to_index)
        elif self.mode == 'basic':
            self.num_types = len(map_atom_type_only_to_index)
        elif self.mode == 'add_aromatic_hybrid':
            self.num_types = len(map_index_to_atom_type_full)
        
        self.num_gen = num_gen

    def __call__(self, data) -> torch.Any:
    
        data_lig = EasyDict()

        data_lig.atom_type = data.ligand.atom_type
        data_lig.pos = data.ligand.pos
        data.ligand_ctx = data_lig

        data.ligand.atom_type = torch.empty([self.num_gen, 0], dtype=int)
        data.ligand.pos = torch.empty([self.num_gen, 0, 3])
        data.ligand.focuses = torch.empty([self.num_gen, 0], dtype=int)
        return EasyDict(data)


@register_transform('assign_linkernum')
class AssignLinkerNum(object):

    def __init__(self, distribution='prior_distcond'):
        super().__init__()
        self.distribution = distribution

    def __call__(self, data):
        if self.distribution == 'prior_distcond':
            pocket_size = self.get_space_size(data.protein.pos)
            num_atoms = sample_atom_num(pocket_size.item()).astype(int)
            num_atoms_linker = num_atoms - data.ligand_ctx['element'].shape[0] 
            if num_atoms_linker <= 0:
                num_atoms_linker = sample_linker_num(pocket_size.item()).astype(int)

        elif self.distribution == 'posterior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        data.ligand.atom_type = torch.zeros(num_atoms_linker, dtype=torch.long)
        data.ligand.lig_flag = torch.ones(num_atoms_linker, dtype=torch.bool)
        data.ligand.pos = torch.zeros(num_atoms_linker, 3, dtype=torch.float)
        data.ligand.element = torch.zeros(num_atoms_linker, dtype=torch.long)
        return data

    def get_space_size(self, pos):
        aa_dist = torch.pdist(pos)
        aa_dist = torch.sort(aa_dist, descending=True)[0]
        return torch.median(aa_dist[:10])
    

@register_transform('assign_fgnum')
class AssignFGNum(object):

    def __init__(self, distribution='prior_distcond'):
        super().__init__()
        self.distribution = distribution

    def __call__(self, data):
        if self.distribution == 'prior_distcond':
            pocket_size = self.get_space_size(data.protein.pos_heavyatom[:,BBHeavyAtom.CA])
            num_fgs_lig = sample_fg_num(pocket_size.item()).astype(int)

        elif self.distribution == 'posterior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        data.ligand.type_fg = torch.zeros(num_fgs_lig, dtype=torch.long)
        data.ligand.lig_flag = torch.ones(num_fgs_lig, dtype=torch.bool)
        data.ligand.pos_heavyatom = torch.zeros(num_fgs_lig, max_num_heavyatoms, 3, dtype=torch.float)
        return data

    def get_space_size(self, pos):
        aa_dist = torch.pdist(pos)
        aa_dist = torch.sort(aa_dist, descending=True)[0]
        return torch.median(aa_dist[:10])

@register_transform('assign_molsize')
class AssignMolSize(object):

    def __init__(self, distribution='prior_distcond'):
        super().__init__()
        self.distribution = distribution

    def __call__(self, data):
        if self.distribution == 'prior_distcond':
            pocket_size = self.get_space_size(data.protein.pos)
            num_atoms_lig = sample_atom_num(pocket_size.item()).astype(int)

        elif self.distribution == 'posterior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        data.ligand.atom_type = torch.zeros(num_atoms_lig, dtype=torch.long)
        data.ligand.lig_flag = torch.ones(num_atoms_lig, dtype=torch.bool)
        data.ligand.pos = torch.zeros(num_atoms_lig, 3, dtype=torch.float)
        data.ligand.element = torch.zeros(num_atoms_lig, dtype=torch.long)
        return data

    def get_space_size(self, pos):
        aa_dist = torch.pdist(pos)
        aa_dist = torch.sort(aa_dist, descending=True)[0]
        return torch.median(aa_dist[:10])

@register_transform('assign_gensize')
class AssignGenSize(object):

    def __init__(self, distribution='prior_distcond'):
        super().__init__()
        self.distribution = distribution

    def __call__(self, data):
        if self.distribution == 'prior_distcond':
            pocket_size = self.get_space_size(data.protein.pos)
            num_atoms_lig = sample_atom_num(pocket_size.item()).astype(int)

        elif self.distribution == 'posterior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        ctx_len = data.ligand.element.shape[0]
        if num_atoms_lig <= ctx_len:
            num_atoms_lig = ctx_len + torch.randint(1, 8, size=(1,))
        
        lig_flag = torch.ones(num_atoms_lig, dtype=torch.bool)
        pos = torch.zeros(num_atoms_lig, 3, dtype=torch.float)
        element = torch.zeros(num_atoms_lig, dtype=torch.long)
        ctx_flag = torch.zeros(num_atoms_lig, dtype=torch.bool)
        atom_type = torch.zeros(num_atoms_lig, dtype=torch.long)

        pos[:ctx_len] = data.ligand.pos
        element[:ctx_len] = data.ligand.element
        atom_type[:ctx_len] = data.ligand.atom_type
        ctx_flag[:ctx_len] = True
        data.ligand.pos = pos
        data.ligand.element = element
        data.ligand.lig_flag = lig_flag
        data.ligand.ctx_flag = ctx_flag
        data.ligand.gen_flag = torch.logical_not(ctx_flag)
        data.ligand.atom_type = atom_type
        return data

    def get_space_size(self, pos):
        aa_dist = torch.pdist(pos)
        aa_dist = torch.sort(aa_dist, descending=True)[0]
        return torch.median(aa_dist[:10])


@register_mode_transform('assign_genatomtype')
@register_transform('assign_genatomtype')
class AssignGenType(object):

    def __init__(self, distribution='uniform', mode='add_aromatic') -> None:
        super().__init__()
        self.distribution = distribution
        self.mode = mode
        if self.mode == 'add_aromatic':
            self.num_types = len(map_atom_type_aromatic_to_index)
        elif self.mode == 'basic':
            self.num_types = len(map_atom_type_only_to_index)

    def __call__(self, data) -> torch.Any:

        atom_type = data.ligand.atom_type.clone()

        if self.distribution == 'uniform':
            uniform_logits = torch.zeros(len(atom_type), self.num_types)
            atom_type = log_sample_categorical(uniform_logits)
        elif self.distribution == 'absorbing':
            atom_type = torch.ones(len(atom_type)) * absorbing_state
        elif self.distribution == 'gaussian':
            atom_type = torch.randn(len(atom_type), self.num_types)
        elif self.distribution == 'prior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        elif self.distribution == 'posterior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        if self.distribution not in ['gaussian']:
            data.ligand.atom_type = torch.where(data.ligand.gen_flag, atom_type, data.ligand.atom_type).long()
        else:
            data.ligand.atom_type = torch.where(data.ligand.gen_flag.unsqueeze(-1), atom_type, 
                                                F.one_hot(data.ligand.atom_type, num_classes=self.num_types))
        return data
    

@register_mode_transform('assign_fgtype')
@register_transform('assign_fgtype')
class AssignFGType(object):

    def __init__(self, distribution='uniform', mode='fg_only') -> None:
        super().__init__()
        self.distribution = distribution
        self.mode = mode

    def __call__(self, data) -> torch.Any:
        type_fg = data.ligand.type_fg

        if self.distribution == 'uniform':
            uniform_logits = torch.zeros(len(type_fg), num_fg_types)
            type_fg = log_sample_categorical(uniform_logits)
        elif self.distribution == 'absorbing':
            type_fg = torch.ones(len(type_fg)) * absorbing_state
        elif self.distribution == 'gaussian':
            type_fg =torch.randn(len(type_fg), num_fg_types)
        elif self.distribution == 'prior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        elif self.distribution == 'posterior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        data.ligand.type_fg = type_fg

        if self.distribution not in ['gaussian']:
            data.ligand.type_fg = type_fg.long()
        return data

@register_mode_transform('assign_atomtype')
@register_transform('assign_atomtype')
class AssignMolType(object):

    def __init__(self, distribution='uniform', mode='add_aromatic') -> None:
        super().__init__()
        self.distribution = distribution
        self.mode = mode
        if self.mode == 'add_aromatic':
            self.num_types = len(map_atom_type_aromatic_to_index)
        elif self.mode == 'basic':
            self.num_types = len(map_atom_type_only_to_index)

    def __call__(self, data) -> torch.Any:
        atom_type = data.ligand.atom_type

        if self.distribution == 'uniform':
            uniform_logits = torch.zeros(len(atom_type), self.num_types)
            atom_type = log_sample_categorical(uniform_logits)
        elif self.distribution == 'absorbing':
            atom_type = torch.ones(len(atom_type)) * absorbing_state
        elif self.distribution == 'gaussian':
            atom_type =torch.randn(len(atom_type), self.num_types)
        elif self.distribution == 'prior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        elif self.distribution == 'posterior':
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        data.ligand.atom_type = atom_type

        if self.distribution not in ['gaussian']:
            data.ligand.atom_type = atom_type.long()
        return data
    

@register_transform('assign_molpos')
class AssignMolPos(object):

    def __init__(self, distribution='gaussian') -> None:
        super().__init__()
        self.distribution = distribution
    
    def __call__(self, data) -> torch.Any:

        if self.distribution == 'gaussian':
            data.ligand.pos = torch.randn_like(data.ligand.pos)
        elif self.distribution == 'zero_mean_gaussian':
            data.ligand.pos = torch.randn_like(data.ligand.pos)
            data.ligand.pos -= torch.mean(data.ligand.pos, dim=0, keepdim=True)
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        return data


@register_transform('assign_fgpos')
class AssignFGPos(object):

    def __init__(self, distribution='gaussian') -> None:
        super().__init__()
        self.distribution = distribution
    
    def __call__(self, data) -> torch.Any:

        if self.distribution == 'gaussian':
            data.ligand.pos_heavyatom[:,BBHeavyAtom.CA] = torch.randn_like(
                data.ligand.pos_heavyatom[:,BBHeavyAtom.CA]
                )
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        return data
    

@register_transform('assign_genpos')
class AssignGenPos(object):

    def __init__(self, distribution='gaussian') -> None:
        super().__init__()
        self.distribution = distribution
    
    def __call__(self, data) -> torch.Any:

        if self.distribution == 'gaussian':
            data.ligand.pos = torch.where(data.ligand.gen_flag.unsqueeze(-1), torch.randn_like(data.ligand.pos), data.ligand.pos)
        elif self.distribution == 'zero_mean_gaussian':
            data.ligand.pos = torch.where(data.ligand.gen_flag.unsqueeze(-1), torch.randn_like(data.ligand.pos), data.ligand.pos)
            data.ligand.pos -= torch.mean(data.ligand.pos, dim=0, keepdim=True)
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        return data
    

@register_transform('assign_fgori')
class AssignFGPos(object):

    def __init__(self, distribution='uniform') -> None:
        super().__init__()
        self.distribution = distribution
    
    def __call__(self, data) -> torch.Any:

        if self.distribution == 'uniform':
            o_rand = random_uniform_so3((data.ligand.pos_heavyatom.shape[0],))
            data.ligand.o_fg = o_rand
        else:
            raise ValueError(f'Unknown distribution type: {self.distribution}')
        
        return data


@register_mode_transform('init_empty_mol')
@register_transform('init_empty_mol')
class InitEmptyMol(object):

    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode
    def __call__(self, data) -> torch.Any:
        data.ligand.atom_type = torch.empty([0,], dtype=torch.long)
        data.ligand.lig_flag = torch.empty([0,], dtype=torch.bool)
        data.ligand.pos = torch.empty([0, 3], dtype=torch.float)
        data.ligand.element = torch.empty([0,], dtype=torch.long)
        data.ligand.bond_index = torch.empty([2, 0], dtype=torch.long)
        data.ligand.bond_type = torch.empty([0,], dtype=torch.long)
        return data
    
