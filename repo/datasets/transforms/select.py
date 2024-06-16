from typing import Any
from ._base import *
import random
import torch

@register_transform('select_fg')
class SelectFG(object):
    def __call__(self, data) -> Any:
        data.protein = data.protein.fg
        data.ligand = data.ligand.fg
        return data


@register_transform('select_linker')
class SelectLinker(object):
    def __call__(self, data) -> Any:
        data.protein = data.protein.linker
        data.ligand = data.ligand.linker
        return data
    
@register_transform('choose_ctx_gen')
class ChooseCtxGen(object):
    def __init__(self, sampling='uniform', ref_key='element') -> None:
        self.sampling = sampling
        self.ref_key = ref_key

    def __call__(self, data) -> Any:
        if self.sampling == 'uniform':
            chosen_idx = random.choice(range(len(data['ligand']['gen_index'])))
        elif self.sampling == 'fix_zero':
            chosen_idx = 0
        else:
            raise NotImplementedError()
        
        data_lig = data['ligand']

        gen_index = data_lig['gen_index'][chosen_idx]
        ctx_index = data_lig['ctx_index'][chosen_idx]

        gen_bond_type = data_lig['gen_bond_type'][chosen_idx]
        gen_bond_index = data_lig['gen_bond_index'][chosen_idx]

        ctx_bond_type = data_lig['ctx_bond_type'][chosen_idx]
        ctx_bond_index = data_lig['ctx_bond_index'][chosen_idx]

        cross_bond_type = data_lig['cross_bond_type'][chosen_idx]
        cross_bond_index = data_lig['cross_bond_index'][chosen_idx]

        gen_flag = torch.zeros(len(data_lig[self.ref_key]), dtype=torch.bool)
        gen_flag[gen_index] = True
        # ctx_flag = torch.ones(len(data_lig[self.ref_key]), dtype=torch.bool)
        ctx_flag = torch.logical_not(gen_flag)

        assert(torch.logical_xor(gen_flag, ctx_flag).all())
        ctx_index = torch.arange(0, len(data_lig[self.ref_key]))[ctx_flag]

        data['ligand']['gen_flag'] = gen_flag
        data['ligand']['ctx_flag'] = ctx_flag
        ctx_bond_index = []
        ctx_bond_type = []
        gen_bond_index = []
        gen_bond_type = []
        cross_bond_index = []
        cross_bond_type = []
        # update gen bond type and ctx bond type
        for bond_idx, bond_type in zip(data['ligand']['bond_index'].transpose(0,1),
                                       data['ligand']['bond_type']):
            if bond_idx[0] in ctx_index and bond_idx[1] in ctx_index:
                ctx_bond_index.append(bond_idx)
                ctx_bond_type.append(bond_type)
            elif bond_idx[0] in gen_index and bond_idx[1] in gen_index:
                gen_bond_index.append(bond_idx)
                gen_bond_type.append(bond_type)
            elif bond_idx[0] in gen_index and bond_idx[1] in ctx_index:
                cross_bond_index.append(bond_idx)
                cross_bond_type.append(bond_type)
            elif bond_idx[0] in ctx_index and bond_idx[1] in gen_index:
                cross_bond_index.append(bond_idx)
                cross_bond_type.append(bond_type)

        data['ligand']['gen_bond_type'] = torch.tensor(gen_bond_type) 
        data['ligand']['gen_bond_index'] = torch.stack(gen_bond_index, dim=0).transpose(0,1) if len(gen_bond_index)>0 else torch.empty((2,0)).long()
        data['ligand']['ctx_bond_type'] = torch.tensor(ctx_bond_type)
        data['ligand']['ctx_bond_index'] = torch.stack(ctx_bond_index, dim=0).transpose(0,1) if len(ctx_bond_index)>0 else torch.empty((2,0)).long()
        data['ligand']['cross_bond_type'] = torch.tensor(cross_bond_type)
        data['ligand']['cross_bond_index'] = torch.stack(cross_bond_index, dim=0).transpose(0,1) if len(cross_bond_index)>0 else torch.empty((2,0)).long()

        return data


        

        