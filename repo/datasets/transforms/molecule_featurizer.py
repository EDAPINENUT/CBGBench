from repo.utils.molecule.constants import aromatic_feat_map_idx
from ._base import register_transform, register_mode_transform, get_fg_type, get_index
import torch


@register_mode_transform('featurize_ligand_fa')
@register_transform('featurize_ligand_fa')
class FeaturizeLigandFullAtom(object):

    def __init__(self, mode='add_aromatic'):
        super().__init__()
        self.mode = mode
        
    def __call__(self, data):
        data_lig = {}
        element_list = data.ligand.element
        hybridization_list = data.ligand.hybridization
        aromatic_list = [v[aromatic_feat_map_idx] for v in data.ligand.atom_feature]
        # add_aro: class_num=14 / no_aro: class_num=10
        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data_lig['atom_type'] = x
        data_lig['lig_flag'] = torch.ones_like(x, dtype=torch.bool)
        data_lig['pos'] = data.ligand.pos
        data_lig['element'] = data.ligand.element
        
        if hasattr(data.ligand, 'gen_flag'):
            data_lig['gen_flag'] = data.ligand.gen_flag
        else:
            data_lig['gen_flag'] = torch.ones_like(x, dtype=torch.bool)
        if hasattr(data.ligand, 'ctx_flag'):
            data_lig['ctx_flag'] = data.ligand.ctx_flag
        else:
            data_lig['ctx_flag'] = torch.zeros_like(x, dtype=torch.bool)


        data.ligand = data_lig
        return data

@register_mode_transform('featurize_ligand_ar')
@register_transform('featurize_ligand_ar')
class FeaturizeLigandAutoRegres(object):

    def __init__(self, mode='add_aromatic'):
        super().__init__()
        self.mode = mode

    def __call__(self, data):
        data_lig = {}
        element_list = data.ligand.element
        hybridization_list = data.ligand.hybridization
        aromatic_list = [v[aromatic_feat_map_idx] for v in data.ligand.atom_feature]
        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data_lig['atom_type'] = x
        data_lig['lig_flag'] = torch.ones_like(x, dtype=torch.bool)
        data_lig['pos'] = data.ligand.pos
        data_lig['element'] = data.ligand.element
        
        if hasattr(data.ligand, 'bond_index'):
            data_lig['bond_index'] = data.ligand.bond_index
            data_lig['bond_type'] = data.ligand.bond_type
        
        if hasattr(data.ligand, 'gen_bond_index'):
            data_lig['gen_bond_index'] = data.ligand.gen_bond_index
            data_lig['gen_bond_type'] = data.ligand.gen_bond_type
            data_lig['ctx_bond_index'] = data.ligand.ctx_bond_index
            data_lig['ctx_bond_type'] = data.ligand.ctx_bond_type
            data_lig['cross_bond_index'] = data.ligand.cross_bond_index
            data_lig['cross_bond_type'] = data.ligand.cross_bond_type

        if hasattr(data.ligand, 'gen_flag'):
            data_lig['gen_flag'] = data.ligand.gen_flag
        else:
            data_lig['gen_flag'] = torch.ones_like(x, dtype=torch.bool)
        if hasattr(data.ligand, 'ctx_flag'):
            data_lig['ctx_flag'] = data.ligand.ctx_flag
        else:
            data_lig['ctx_flag'] = torch.zeros_like(x, dtype=torch.bool)

        if hasattr(data.ligand, 'moltree'):
            data_lig['moltree'] = data.ligand.moltree
        if hasattr(data.ligand, 'neighbor_list'):
            data_lig['neighbors'] = data.ligand.neighbor_list

        data.ligand = data_lig
        return data

def map_mol_index_to_masked_index(mol_index, masked_flag):
    masked_index = torch.arange(0, masked_flag.sum())
    masked_index_map = torch.zeros_like(masked_flag).long()
    masked_index_map[masked_flag] = masked_index
    return masked_index_map[mol_index]

def map_masked_index_to_mol_index(masked_index, masked_flag):
    mol_index = torch.arange(0, len(masked_flag))
    mol_index_map = mol_index[masked_flag]
    return mol_index_map[masked_index]

@register_mode_transform('featurize_ligand_gen_ctx_ar')
@register_transform('featurize_ligand_gen_ctx_ar')
class FeaturizeLigandAutoRegres(object):

    def __init__(self, mode='add_aromatic'):
        super().__init__()
        self.mode = mode

    def __call__(self, data):
        data_lig = {}
        element_list = data.ligand.element
        hybridization_list = data.ligand.hybridization
        aromatic_list = [v[aromatic_feat_map_idx] for v in data.ligand.atom_feature]
        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data_lig['atom_type'] = x
        data_lig['lig_flag'] = torch.ones_like(x, dtype=torch.bool)
        data_lig['pos'] = data.ligand.pos
        data_lig['element'] = data.ligand.element
        
        if hasattr(data.ligand, 'bond_index'):
            data_lig['bond_index'] = data.ligand.bond_index
            data_lig['bond_type'] = data.ligand.bond_type
        
        if hasattr(data.ligand, 'gen_bond_index'):
            data_lig['gen_bond_index'] = data.ligand.gen_bond_index
            data_lig['gen_bond_type'] = data.ligand.gen_bond_type
            data_lig['ctx_bond_index'] = data.ligand.ctx_bond_index
            data_lig['ctx_bond_type'] = data.ligand.ctx_bond_type
            data_lig['cross_bond_index'] = data.ligand.cross_bond_index
            data_lig['cross_bond_type'] = data.ligand.cross_bond_type

        if hasattr(data.ligand, 'gen_flag'):
            data_lig['gen_flag'] = data.ligand.gen_flag
        else:
            data_lig['gen_flag'] = torch.ones_like(x, dtype=torch.bool)
        if hasattr(data.ligand, 'ctx_flag'):
            data_lig['ctx_flag'] = data.ligand.ctx_flag
        else:
            data_lig['ctx_flag'] = torch.zeros_like(x, dtype=torch.bool)
        
        if hasattr(data.ligand, 'neighbor_list'):
            data_lig['neighbors'] = data.ligand.neighbor_list
        data.ligand = data_lig
        return data
    

@register_mode_transform('featurize_ligand_fg')
@register_transform('featurize_ligand_fg')
class FeaturizeLigandFuncGroup(object):

    def __init__(self, mode='protein_fg_merge'):
        super().__init__()
        self.mode = mode

    def __call__(self, data) -> torch.Any:
        type_fg = data.ligand['type_fg']
        data.ligand['type_fg'] = torch.tensor([get_fg_type(fg, self.mode, is_aa=False) for fg in type_fg])
        data.ligand['lig_flag'] = torch.ones_like(type_fg, dtype=torch.bool)

        return data


@register_transform('remove_ligand')
class RemoveLigand(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand = {}
        return data

@register_transform('remove_ligand_gen')
@register_mode_transform('remove_ligand_gen')
class RemoveLigandGen(object):

    def __init__(self, mode='add_aromatic') -> None:
        self.mode = mode

    def __call__(self, data):
        ctx_flag = data.ligand['ctx_flag']

        element_list = data.ligand.element
        hybridization_list = data.ligand.hybridization
        aromatic_list = [v[aromatic_feat_map_idx] for v in data.ligand.atom_feature]
        # add_aro: class_num=14 / no_aro: class_num=10
        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data.ligand['atom_type'] = x[ctx_flag]
        data.ligand['element'] = data.ligand['element'][ctx_flag]
        data.ligand['pos'] = data.ligand['pos'][ctx_flag]
        data.ligand['ctx_flag'] = torch.ones_like(data.ligand['element'], dtype=bool)
        data.ligand['gen_flag'] = torch.zeros_like(data.ligand['element'], dtype=bool)
        data.ligand['lig_flag'] = torch.ones_like(data.ligand['element'], dtype=bool)


        if hasattr(data.ligand, 'ctx_bond_index'):
            data.ligand['bond_index'] = torch.stack([map_mol_index_to_masked_index(data.ligand['ctx_bond_index'][0], 
                                                                      ctx_flag),
                                                     map_mol_index_to_masked_index(data.ligand['ctx_bond_index'][1], 
                                                                      ctx_flag)], dim=0)
            data.ligand['bond_type'] = data.ligand['ctx_bond_type']
        
        
        return data