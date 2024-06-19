from torch import nn 
import torch 
from repo.utils.protein.constants import *

def get_vec_embedding_func(type, 
                            emb_dim):
    if type == 'linear':
        return VecLinear(emb_dim, 20.0)

    else:
        raise ValueError(f'Unknown time embedding type: {type}')



class VecLinear(nn.Module):
    def __init__(self, emb_dim, normalizer):
        super().__init__()
        self.linear = nn.Linear(1, emb_dim)
        self.normalizer = normalizer
    
    def forward(self, x):
        x = x.unsqueeze(-1) / self.normalizer
        return self.linear(x).transpose(-1, -2)
        