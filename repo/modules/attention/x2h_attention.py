import torch
from torch import nn
from ..common import GaussianSmearing, MLP, outer_product, masked_softmax
from ..graph import *
import numpy as np 
from torch_scatter import scatter_softmax, scatter_sum

class X2HAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 r_max, num_r_gaussian, act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        self.distance_expansion = GaussianSmearing(0, r_max, 
                                                   num_gaussians=num_r_gaussian)

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, x, h, edge_attr, edge_index, e_w):
        N = h.size(0)
        
        src, dst = edge_index
        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)
        dist_feat = self.distance_expansion(dist)
        if dist_feat.dim() == 3:
            edge_attr = edge_attr.unsqueeze(1).repeat(1, dist_feat.shape[1], 1)
            if h.dim() == x.dim() - 1:
                h = h.unsqueeze(1).repeat(1, dist_feat.shape[1], 1)
        dist_feat = outer_product(edge_attr, dist_feat)

        L = dist_feat.shape[:-1]
        M = x.shape[:-1]
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([dist_feat, hi, hj], -1)
        if edge_attr is not None:
            kv_input = torch.cat([edge_attr, kv_input], -1)

        # compute k
        k = self.hk_func(kv_input).view(*L, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(dist_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(*L, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(*M, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(*M, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output
        
        

        
    



        