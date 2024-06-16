import math
import torch
from torch.utils.data._utils.collate import default_collate
from repo.utils.molecule.constants import map_atom_type_aromatic_to_index
from repo.utils.protein.constants import num_aa_types
from easydict import EasyDict
from torch_geometric.data import Data, Batch

DEFAULT_PAD_VALUES = {
    'aa': num_aa_types, 
    'atom_feature': len(map_atom_type_aromatic_to_index),
}

BOOL_KEY = ['flag', 'mask']

CNT_KEY = ['pos', 'feature']

LONG_KEY = ['element', 'type']


def get_collate_fn(config):
    fn_type = config.type
    if fn_type == 'padding':
        return PaddingCollate()
    elif fn_type == 'graphbp':
        return GraphBPCollate()
    elif fn_type == 'flag':
        return FlagCollate()

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return EasyDict({k: recursive_to(v, device=device) for k, v in obj.items()})

    else:
        return obj


class FlagCollate(object):
    def __init__(self) -> None:
        super().__init__()
        self.keys = ['protein_pos', 'protein_atom_feature', 'ligand_context_pos', 'ligand_context_atom_type',
                    'ligand_frontier', 'num_atoms', 'next_wid', 'current_wid', 'current_atoms', 'cand_labels',
                    'ligand_pos_torsion', 'ligand_atom_type_torsion', 'true_sin', 'true_cos', 'true_three_hop',
                    'dihedral_mask', 'protein_contact', 'true_dm', 'protein_alpha_carbon_indicator', 
                    'protein_aa_type', 'ligand_context_lig_flag', 'protein_lig_flag', 'ligand_lig_flag_torsion']
    
    def __call__(self, mol_dicts) -> torch.Any:
        data_batch = {}
        batch_size = len(mol_dicts)
        for key in self.keys:
            data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
        # unsqueeze dim0
        for key in ['xn_pos', 'yn_pos', 'ligand_torsion_xy_index', 'y_pos']:
            cat_list = [mol_dict[key].unsqueeze(0) for mol_dict in mol_dicts if len(mol_dict[key]) > 0]
            if len(cat_list) > 0:
                data_batch[key] = torch.cat(cat_list, dim=0)
            else:
                data_batch[key] = torch.tensor([])
        # follow batch
        for key in ['protein_element', 'ligand_context_element', 'current_atoms']:
            repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
            data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)
        for key in ['ligand_element_torsion']:
            repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts if len(mol_dict[key]) > 0])
            if len(repeats) > 0:
                data_batch[key + '_batch'] = torch.repeat_interleave(torch.arange(len(repeats)), repeats)
            else:
                data_batch[key + '_batch'] = torch.tensor([])
        # distance matrix prediction
        p_idx, q_idx = torch.cartesian_prod(torch.arange(4), torch.arange(2)).chunk(2, dim=-1)
        p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
        protein_offsets = torch.cumsum(data_batch['protein_element_batch'].bincount(), dim=0)
        ligand_offsets = torch.cumsum(data_batch['ligand_context_element_batch'].bincount(), dim=0)
        protein_offsets, ligand_offsets = torch.cat([torch.tensor([0]), protein_offsets]), torch.cat([torch.tensor([0]), ligand_offsets])
        ligand_idx, protein_idx = [], []
        for i, mol_dict in enumerate(mol_dicts):
            if len(mol_dict['true_dm']) > 0:
                protein_idx.append(mol_dict['dm_protein_idx'][p_idx] + protein_offsets[i])
                ligand_idx.append(mol_dict['dm_ligand_idx'][q_idx] + ligand_offsets[i])
        if len(ligand_idx) > 0:
            data_batch['dm_ligand_idx'], data_batch['dm_protein_idx'] = torch.cat(ligand_idx), torch.cat(protein_idx)

        # structure refinement (alpha carbon - ligand atom)
        sr_ligand_idx, sr_protein_idx = [], []
        for i, mol_dict in enumerate(mol_dicts):
            if len(mol_dict['true_dm']) > 0:
                ligand_atom_index = torch.arange(len(mol_dict['ligand_context_pos']))
                p_idx, q_idx = torch.cartesian_prod(torch.arange(len(mol_dict['ligand_context_pos'])), torch.arange(len(mol_dict['protein_alpha_carbon_index']))).chunk(2, dim=-1)
                p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
                sr_ligand_idx.append(ligand_atom_index[p_idx] + ligand_offsets[i])
                sr_protein_idx.append(mol_dict['protein_alpha_carbon_index'][q_idx] + protein_offsets[i])
        if len(sr_ligand_idx) > 0:
            data_batch['sr_ligand_idx'], data_batch['sr_protein_idx'] = torch.cat(sr_ligand_idx).long(), torch.cat(sr_protein_idx).long()

        # structure refinement (ligand atom - ligand atom)
        sr_ligand_idx0, sr_ligand_idx1 = [], []
        for i, mol_dict in enumerate(mol_dicts):
            if len(mol_dict['true_dm']) > 0:
                ligand_atom_index = torch.arange(len(mol_dict['ligand_context_pos']))
                p_idx, q_idx = torch.cartesian_prod(torch.arange(len(mol_dict['ligand_context_pos'])), torch.arange(len(mol_dict['ligand_context_pos']))).chunk(2, dim=-1)
                p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
                sr_ligand_idx0.append(ligand_atom_index[p_idx] + ligand_offsets[i])
                sr_ligand_idx1.append(ligand_atom_index[q_idx] + ligand_offsets[i])
        if len(ligand_idx) > 0:
            data_batch['sr_ligand_idx0'], data_batch['sr_ligand_idx1'] = torch.cat(sr_ligand_idx0).long(), torch.cat(sr_ligand_idx1).long()
        # index
        if len(data_batch['y_pos']) > 0:
            repeats = torch.tensor([len(mol_dict['ligand_element_torsion']) for mol_dict in mol_dicts if len(mol_dict['ligand_element_torsion']) > 0])
            offsets = torch.cat([torch.tensor([0]), torch.cumsum(repeats, dim=0)])[:-1]
            data_batch['ligand_torsion_xy_index'] += offsets.unsqueeze(1)

        offsets1 = torch.cat([torch.tensor([0]), torch.cumsum(data_batch['num_atoms'], dim=0)])[:-1]
        data_batch['current_atoms'] += torch.repeat_interleave(offsets1, data_batch['current_atoms_batch'].bincount())
        # cand mols: torch geometric Data
        cand_mol_list = []
        for data in mol_dicts:
            if len(data['cand_labels']) > 0:
                cand_mol_list.extend(data['cand_mols'])
        if len(cand_mol_list) > 0:
            data_batch['cand_mols'] = Batch.from_data_list(cand_mol_list)
        return data_batch



class GraphBPCollate(object):
    def __init__(self) -> None:
        super().__init__()
        self.keys = ['atom_type', 'pos', 'ctx_flag', 'gen_flag', 
                     'cannot_contact', 'new_atom_type', 'new_dist', 
                     'new_angle', 'new_torsion', 'cannot_focus']
        
    def __call__(self, mol_dicts) -> torch.Any:
        data_batch = {}

        for key in self.keys:
            data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
        
        num_steps_list = torch.tensor([0]+[len(mol_dicts[i]['new_atom_type']) for i in range(len(mol_dicts)-1)])
        batch_idx_offsets = torch.cumsum(num_steps_list, dim=0)
        repeats = torch.tensor([len(mol_dict['batch']) for mol_dict in mol_dicts])
        batch_idx_repeated_offsets = torch.repeat_interleave(batch_idx_offsets, repeats)
        batch_offseted = torch.cat([mol_dict['batch'] for mol_dict in mol_dicts], dim=0) + batch_idx_repeated_offsets
        data_batch['batch'] = batch_offseted

        num_atoms_list = torch.tensor([0]+[len(mol_dicts[i]['atom_type']) for i in range(len(mol_dicts)-1)])
        atom_idx_offsets = torch.cumsum(num_atoms_list, dim=0)
        for key in ['focus', 'c1_focus', 'c2_c1_focus', 'contact_y_or_n']:
            repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
            atom_idx_repeated_offsets = torch.repeat_interleave(atom_idx_offsets, repeats)
            if key in [ 'contact_y_or_n' , 'focus' ]:
                atom_offseted = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0) + atom_idx_repeated_offsets
            else:
                atom_offseted = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0) + atom_idx_repeated_offsets[:,None]
                if (atom_offseted[:,0].max() > data_batch['atom_type'].shape[0]): 
                    assert (atom_offseted[:,0].max() <= data_batch['atom_type'].shape[0])
            data_batch[key] = atom_offseted

        return data_batch


class PaddingCollate(object):

    def __init__(self, length_ref_key='element', pad_values=DEFAULT_PAD_VALUES, eight=False):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]
    
    def _convert_tensor_to_bool(self, data, key):
        for bool_key in BOOL_KEY:
            if bool_key in key:
                data = data.bool()
        return data
    
    def _convert_tensor_to_long(self, data, key):
        for long_key in LONG_KEY:
            if long_key in key:
                data = data.long()
        return data

    def _convert_tensor_to_float(self, data, key):
        for cnt_key in CNT_KEY:
            if cnt_key in key:
                data = data.float()
        return data

    def __call__(self, bound_data_list):
        bound_batch = {}
        hetero_graph_keys = bound_data_list[0].keys()
        for key in hetero_graph_keys:
            data_list = [bound_data[key] for bound_data in bound_data_list]
            max_length = max([data[self.length_ref_key].size(0) for data in data_list])

            pad_keys = self._get_common_keys(data_list)
            
            if self.eight:
                max_length = math.ceil(max_length / 8) * 8
            data_list_padded = []
            for data in data_list:
                data_padded = {
                    k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                    for k, v in data.items()
                    if k in pad_keys
                }

                data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
                data_padded = {
                    k: self._convert_tensor_to_bool(v, k) 
                    for k, v in data_padded.items()
                    }
                data_padded = {
                    k: self._convert_tensor_to_long(v, k) 
                    for k, v in data_padded.items()
                    }
                data_padded = {
                    k: self._convert_tensor_to_float(v, k) 
                    for k, v in data_padded.items()
                    }
                data_list_padded.append(data_padded)
            batch = default_collate(data_list_padded)
            batch['size'] = len(data_list_padded)
            bound_batch[key] = batch
        return bound_batch

