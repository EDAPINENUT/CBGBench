import torch 
from torch import nn
import numpy as np 
from torch.nn import functional as F



def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)


def get_dict_mean(loss_dicts):
    loss_dict_record = {k:[] for k in loss_dicts[0].keys()}
    for loss_dict in loss_dicts:
        for k in loss_dict_record.keys():
            loss_dict_record[k].append(loss_dict[k])
    
    loss_dict_mean = {}
    for k, v in loss_dict_record.items():
        loss_dict_mean[k] = torch.mean(torch.tensor(v))
    return loss_dict_mean


def _atom_index_select(v, index, n):
    if isinstance(v, torch.Tensor) and v.size(0) == n:
        return v[index]
    elif isinstance(v, list) and len(v) == n:
        return [v[i] for i in index]
    else:
        return v


def _atom_index_select_data(data, index):
    return {
        k: _atom_index_select(v, index, data['element'].size(0))
        for k, v in data.items()
    }


def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(-2)
            out = out.view(*out.shape[:-2], -1).unsqueeze(-1)
    return out.squeeze()

def merge_multiple_adjacency(adj_list, attr_adj_list):
    """
    Merge multiple adjacency matrices into a single adjacency matrix.
    Args:
        adj_list (list): List of adjacency matrices.
        attr_adj_list (list): List of edge types.
    Returns:
        torch.Tensor: Merged adjacency matrix.
    """

    num_nodes = adj_list[0].size(0)
    if len(attr_adj_list[0].shape)  == len(adj_list[0].shape):
        num_class = torch.stack(attr_adj_list).max()
        attr_adj_list = [F.one_hot(attr_adj, num_class) for attr_adj in attr_adj_list]

    num_edge_types = attr_adj_list.shape[-1]

    adj = torch.zeros(num_nodes, num_nodes, device=adj_list[0].device)
    attr_adj = torch.zeros(num_nodes, num_nodes, num_edge_types, device=adj_list[0].device)

    for i, (adj_i, attr_i) in enumerate(adj_list, attr_adj_list):
        adj = torch.logical_or(adj, adj_i)
        attr_adj = attr_adj + attr_i * adj_i.unsqueeze(-1)

    return adj, attr_adj

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    
NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            # customized offset
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist - self.offset.view(*[1 for i in range(len(dist.shape) - 1)], self.offset.shape[0])
        return torch.exp(self.coeff * torch.pow(dist, 2))
    

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


def masked_softmax(x, mask, dim=1):
    x_masked = x.clone()
    x_masked[~mask] = -1e6

    return torch.softmax(x_masked, dim=dim) * mask.float().unsqueeze(-1)

def compose_context(context_dict_lig, context_dict_rec, batch_idx_lig, batch_idx_rec):
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)

    batch_ctx = torch.cat([batch_idx_rec, batch_idx_lig], dim=0)
    mask_rec = torch.cat([torch.ones_like(batch_idx_rec, dtype=torch.bool), 
                          torch.zeros_like(batch_idx_lig, dtype=torch.bool)], dim=0)
    mask_lig = ~mask_rec
    
    idx_old_rec_lig = torch.arange(len(batch_idx_rec) + len(batch_idx_lig)).to(batch_idx_rec.device)
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices
    batch_ctx = batch_ctx[sort_idx]
    mask_lig_new = mask_lig[sort_idx]
    mask_rec_new = mask_rec[sort_idx]
    
    merged_context = {}
    keys = set(context_dict_lig.keys()).intersection(context_dict_rec.keys())

    for key in keys:
        merged_context[key] = torch.cat([context_dict_rec[key], context_dict_lig[key]], dim=0)[sort_idx]
    
    idx_new_lig = idx_old_rec_lig[mask_lig_new]
    idx_new_rec = idx_old_rec_lig[mask_rec_new]

    return merged_context, batch_ctx, (idx_new_rec, idx_new_lig)


class VecExpansion(nn.Module):
    def __init__(self, edge_channels):
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)
    
    def forward(self, edge_vector):
        edge_vector = edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True)+1e-7)
        expansion = self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1)
        return expansion
