from torch import nn 
import torch 
from repo.utils.molecule.fg_constants import *

def get_fg_embedding_func(type, 
                            emb_dim, 
                            input_dim = num_fg_types):
    if type == 'linear':
        return nn.Linear(input_dim, emb_dim)

    else:
        raise ValueError(f'Unknown time embedding type: {type}')