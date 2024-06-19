
import torch
from torch import nn
from ..common import GaussianSmearing, MLP, outer_product, masked_softmax
from ..graph import *
import numpy as np 
from torch_scatter import scatter_softmax, scatter_sum

class H2XAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 r_max, num_r_gaussian, act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        self.distance_expansion = GaussianSmearing(0, r_max, 
                                            num_gaussians=num_r_gaussian)

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())


    def forward(self, x, h, edge_attr, edge_index, e_w):
        N = h.size(0)

        src, dst = edge_index
        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)
        dist_feat = self.distance_expansion(dist)
        dist_feat = outer_product(edge_attr, dist_feat)

        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([dist_feat, hi, hj], -1)
        if edge_attr is not None:
            kv_input = torch.cat([edge_attr, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(edge_index)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        
        return output.mean(1)
