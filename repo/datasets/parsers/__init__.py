from .protein_parser import *
from .molecule_parser import *
import torch


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        elif isinstance(v, list):
            if isinstance(v[0], np.ndarray):
                output[k] = [torch.from_numpy(v[ii]) for ii in range(len(v))]
            else:
                output[k] = v
        else:
            output[k] = v
    return output