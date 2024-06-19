from repo.utils.protein.constants import *
import torch
import torch.nn.functional as F
from ._base import register_transform, register_mode_transform, get_fg_type, get_index
import numpy as np

@register_transform('featurize_protein_fa')
class FeaturizeProteinFullAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor(atomic_numbers)  # H, C, N, O, S, Se
        self.max_num_aa = len(aa_name_number)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data):
        data_prot = {}
        element = (data.protein.element.view(-1, 1) == self.atomic_numbers.view(1, -1)).float()
        amino_acid = data.protein.atom_to_aa_type
        is_backbone = data.protein.is_backbone.view(-1, 1).long()
        x = torch.cat([element, is_backbone], dim=-1)

        data_prot['atom_feature'] = x
        data_prot['aa_type'] = amino_acid
        data_prot['pos'] = data.protein.pos
        data_prot['element'] = data.protein.element
        data_prot['lig_flag'] = torch.zeros_like(data.protein.element, dtype=torch.bool)
        data_prot['atom_type'] = torch.tensor([get_index(e, h, a, 'basic') for e, h, a in zip(data.protein.element, 
                                                                                 torch.zeros_like(data.protein.element), 
                                                                                 torch.zeros_like(data.protein.element))])
        data_prot['alpha_carbon_indicator'] = torch.tensor([True if name =="CA" else False for name in data.protein['atom_name']])
        
        if hasattr(data.protein, 'contact'):
            data_prot['contact'] = data.protein['contact']
            data_prot['contact_idx'] = data.protein['contact_idx']

        data.protein = data_prot    

        return data

@register_transform('featurize_protein_fg')
class FeaturizeProteinFuncGroup(object):

    def __init__(self, mode='protein_fg_merge') -> None:
        self.mode = mode

    def __call__(self, data) -> torch.Any:
        aa_type = data.protein['aa']
        data.protein['type_fg'] = torch.tensor([get_fg_type(aa, self.mode, is_aa=True) for aa in aa_type])
        data.protein['lig_flag'] = torch.zeros_like(aa_type, dtype=torch.bool)
        chain_id_to_nb = {k:v for v,k in enumerate(np.unique(data.protein['chain_id']))}
        data.protein['chain_nb'] = torch.tensor([chain_id_to_nb[k] for k in data.protein['chain_id']])
        data.protein['num_chains'] = len(np.unique(data.protein['chain_id']))
        return data