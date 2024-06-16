from torch import nn 
import torch 
from repo.utils.molecule.constants import *

def get_atom_embedding_func(type, 
                            emb_dim, 
                            input_dim = len(map_atom_type_aromatic_to_index)):
    if type == 'linear':
        return nn.Linear(input_dim, emb_dim)

    else:
        raise ValueError(f'Unknown time embedding type: {type}')