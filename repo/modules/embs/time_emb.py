from torch import nn 
from repo.modules.common import SinusoidalPosEmb

def get_time_embedding_func(type, emb_dim):

    if type == 'sin':
        return nn.Sequential(
                    SinusoidalPosEmb(emb_dim),
                    nn.Linear(emb_dim, emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(emb_dim * 4, emb_dim)
                )
    elif type == 'linear':
        return nn.Linear(emb_dim, emb_dim)

    else:
        raise ValueError(f'Unknown time embedding type: {type}')