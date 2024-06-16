import torch
from torch import nn

def register_from_numpy(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x