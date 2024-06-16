from torch import nn 
from torch_geometric.nn import radius_graph, knn_graph
import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce
from repo.modules.common import GaussianSmearing, VecExpansion
from repo.modules.gvp.gvn import GVLinear, VNLeakyReLU, MessageModule, GVPerceptronVN
from torch_scatter import scatter_sum

class GVPTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Build the network
        self.num_classes = cfg.get('num_classes', None)
        self.num_layers = cfg.get('num_layers', 6)
        self.node_feat_dim = cfg.get('node_feat_dim', 128)
        self.vec_feat_dim = cfg.get('vec_feat_dim', 128)
        self.edge_feat_dim = cfg.get('edge_feat_dim', 4)
        self.num_edge_classes = cfg.get('num_edge_classes', None)

        self.cutoff_mode = cfg.get('cutoff_mode', 'knn')  # [radius, none]
        self.cut_off = cfg.get('k', 48)
        self.r_max = cfg.get('r_max', 10.0)
        if self.num_classes is not None:
            self.classifier = nn.Sequential(
            GVPerceptronVN(self.node_feat_dim,  self.vec_feat_dim , self.node_feat_dim,  self.vec_feat_dim ),
            GVLinear(self.node_feat_dim,  self.vec_feat_dim , self.num_classes, 1)
        )

        self._buid_blocks()
    
    @property
    def out_sca(self):
        return self.node_feat_dim[0]
    
    @property
    def out_vec(self):
        return self.vec_feat_dim[1]
    
    def _buid_blocks(self):
        self.interactions = nn.ModuleList()
        for _ in range(self.num_layers):
            block = AttentionInteractionBlockVN(
                hidden_channels=(self.node_feat_dim, self.vec_feat_dim),
                edge_channels=self.vec_feat_dim,
                num_edge_types=self.edge_feat_dim + 1,
                r_max = self.r_max
            )
            self.interactions.append(block)

    def _extend_edge_index(self, x, edge_index, edge_type, batch):
        if self.cutoff_mode == 'radius':
            edge_index_expand = radius_graph(x, r=cut_off, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            cut_off = int(self.cut_off)
            edge_index_expand = knn_graph(x, k=cut_off, batch=batch, flow='source_to_target')

        edge_type_expand = torch.zeros(edge_index_expand.size(1), dtype=torch.long, device=x.device)

        if edge_index is None:
            edge_index = torch.empty([2,0], dtype=torch.long).to(x.device)

        if edge_type is None:
            edge_type = torch.ones_like(edge_index[0,:]).long()

        if edge_index is not None:
            edge_index = torch.cat([edge_index, edge_index_expand], dim=1)
        
        edge_type = torch.cat([edge_type, edge_type_expand], dim=0)

        edge_index, edge_type = coalesce(edge_index, edge_attr=edge_type, reduce='max') # bond replace knn edge

        return edge_index, edge_type


    def forward(self, x, vec, h, batch_idx, edge_index=None, edge_type=None):
        edge_index, edge_type = self._extend_edge_index(x, edge_index, edge_type, batch_idx)
        edge_attr = F.one_hot(edge_type, num_classes=self.edge_feat_dim + 1).float()
        
        edge_vector = x[edge_index[0]] - x[edge_index[1]]
        # h_init, vec_init = h.clone(), vec.clone()

        for interaction in self.interactions:
            delta_h, delta_vec = interaction(h, vec, edge_index, edge_attr, edge_vector)
            h = h + delta_h
            vec = vec + delta_vec
        
        return h, vec


class AttentionInteractionBlockVN(nn.Module):
    def __init__(self, hidden_channels, edge_channels, num_edge_types, r_max=10.):
        super().__init__()
        # edge features
        self.distance_expansion = GaussianSmearing(stop=r_max, 
                                                   num_gaussians = edge_channels - num_edge_types,
                                                   fixed_offset=False)
        self.vector_expansion = VecExpansion(edge_channels) 
        ## compare encoder and classifier message passing

        # edge weigths and linear for values
        self.message_module = MessageModule(hidden_channels[0], hidden_channels[1], 
                                            edge_channels, edge_channels,
                                            hidden_channels[0], hidden_channels[1], r_max)

        # centroid nodes and finall linear
        self.centroid_lin = GVLinear(hidden_channels[0], hidden_channels[1], 
                                     hidden_channels[0], hidden_channels[1])
        self.act_sca = nn.LeakyReLU()
        self.act_vec = VNLeakyReLU(hidden_channels[1])
        self.out_transform = GVLinear(hidden_channels[0], hidden_channels[1], 
                                      hidden_channels[0], hidden_channels[1])

        self.layernorm_sca = nn.LayerNorm([hidden_channels[0]])
        self.layernorm_vec = nn.LayerNorm([hidden_channels[1], 3])

    def forward(self, node_h, node_vec, edge_index, edge_attr, edge_vector):
        """
        Args:
            x:  Node features: scalar features (N, feat), vector features(N, feat, 3)
            edge_index: (2, E).
            edge_attr:  (E, H).
            edge_vector: (E, 3).
        """
        scalar, vector = node_h, node_vec
        N = scalar.size(0)
        row, col = edge_index   # (E,) , (E,)

        # Compute edge features
        edge_dist = torch.norm(edge_vector, dim=-1, p=2, keepdim=True)
        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_attr], dim=-1)
        edge_vec_feat = self.vector_expansion(edge_vector) 

        msg_j_sca, msg_j_vec = self.message_module((scalar, vector), 
                                                   (edge_sca_feat, edge_vec_feat), 
                                                   col, edge_dist, annealing=True)

        # Aggregate messages
        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=N)  #.view(N, -1) # (N, heads*H_per_head)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=N)  #.view(N, -1, 3) # (N, heads*H_per_head, 3)
        x_out_sca, x_out_vec = self.centroid_lin((scalar, vector))
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out