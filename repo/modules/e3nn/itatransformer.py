from torch import nn 
import torch 
from torch.nn import functional as F
from repo.utils.molecule.constants import *
from ..graph import *
from ..embs import get_dist_emb
from ..attention import X2HAttention, H2XAttention
from ..common import GaussianSmearing, outer_product, ShiftedSoftplus
from repo.utils.molecule.constants import *
from torch_geometric.nn import radius_graph, knn_graph
from repo.models.utils.geometry import *
from repo.models.utils.so3 import *

class ITATransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Build the network
        self.num_classes = cfg.get('num_classes', None)
        self.num_blocks = cfg.get('num_blocks', 1)
        self.num_layers = cfg.get('num_layers', 6)
        self.hidden_dim = cfg.get('node_feat_dim', 128)
        self.edge_hidden_dim = cfg.get('pair_feat_dim', 128)
        self.n_heads = cfg.get('n_heads', 16)
        self.edge_feat_dim = cfg.get('edge_feat_dim', 4)
        self.act_fn = cfg.get('act_fn', 'relu')
        self.norm = cfg.get('norm', True)
        # radius graph / knn graph
        self.cutoff_mode = cfg.get('cutoff_mode', 'knn')  # [radius, none]
        self.cut_off = cfg.get('k', 32)
        self.r_max = cfg.get('r_max', 10.0)
        self.ew_net_type = cfg.get('ew_type', 'global')  # [r, m, none]
        self.num_r_gaussian = cfg.get('num_r_gaussian', 20)

        self.num_x2h = cfg.get('num_x2h', 1)
        self.num_h2x = cfg.get('num_h2x', 1)
        self.num_init_x2h = cfg.get('num_init_x2h', 1)
        self.num_init_h2x = cfg.get('num_init_h2x', 0)

        self.x2h_out_fc = cfg.get('x2h_out_fc', False)

        self.dist_emb = get_dist_emb(cfg.get('dist_emb_type', 'gaussian_exp'), 
                                     self.num_r_gaussian, cut_off=self.r_max)

        self.blocks = self._build_share_blocks()

        if self.num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, self.num_classes),
            )
        else:
            self.classifier = None

        self.eps_rot_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim), nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 3)
        )

        self.eps_crd_net = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim), nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 3)
        )


    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'base block: {self.blocks.__repr__()} \n'


    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = InvAttentionLayer(
                self.hidden_dim, self.n_heads, self.edge_feat_dim, num_r_gaussian=self.num_r_gaussian,
                act_fn=self.act_fn, norm=self.norm, num_x2h=self.num_x2h, num_h2x=self.num_h2x, 
                r_max=self.r_max, ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, 
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=cut_off, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            cut_off = int(self.cut_off)
            edge_index = knn_graph(x, k=cut_off, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index
    
    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type


    def forward(self, x, o, h, batch_idx, lig_flag, gen_flag):

        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, lig_flag, batch_idx)
            src, dst = edge_index
            edge_type = self._build_edge_type(edge_index, lig_flag)

            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                logits = self.dist_emb(dist)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.blocks):
                h = layer(x, h, edge_type, edge_index, e_w=e_w, gen_flag=gen_flag)
        eps_rot = self.eps_rot_net(h)
        U_update = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        R_o = so3vec_to_rotation(o)
        R_next = R_o @ U_update
        o_next = rotation_to_so3vec(R_next)
        o_next = torch.where(gen_flag[:, None].expand_as(o_next), o_next, o)

        eps_crd = self.eps_crd_net(h)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R_o, eps_crd)  # (N, L, 3)
        eps_pos = torch.where(gen_flag[:, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        if self.classifier is not None:
            c = self.classifier(h)
            return eps_pos, h, o_next, R_next, c
        else:
            return eps_pos, h, o_next, R_next

class InvAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, edge_feat_dim, 
                act_fn, norm, num_x2h, num_h2x, num_r_gaussian=20, 
                r_max=10.0, ew_net_type='global', x2h_out_fc=False, 
                sequential_update=True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x

        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sequential_update = sequential_update
        self.num_r_gaussian = num_r_gaussian
        self.r_max = r_max

        self._init_x2h_layer()

    def _init_x2h_layer(self):
        self.x2h_layers = nn.ModuleList()
        for _ in range(self.num_x2h):
            self.x2h_layers.append(X2HAttention(self.hidden_dim, self.hidden_dim, self.hidden_dim, 
                                                self.n_heads, self.edge_feat_dim, r_feat_dim=self.num_r_gaussian*4,
                                                act_fn = self.act_fn, norm=self.norm, num_r_gaussian=self.num_r_gaussian,
                                                r_max=self.r_max, ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc))

    

    def forward(self, x, h, edge_type, edge_index, e_w, gen_flag):
        '''
        Use GNN message passing to update the node features; or the memory is not enough.
        '''
        x_in, h_in = x, h

        for x2h_layer in self.x2h_layers:
            h_out = x2h_layer(x, h_in, edge_type, edge_index, e_w)
            h_in = h_out
        
        return h_out
