from repo.modules.gvp.gvn import GVPerceptronVN, GVLinear, MessageModule
from torch_geometric.nn import radius_graph, knn
from repo.modules.common import GaussianSmearing, VecExpansion
import torch  
from torch import nn
from torch_scatter import scatter_add, scatter_softmax, scatter_sum
from torch.nn import functional as F
import math
GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)

class PositionPredictor(nn.Module):
    def __init__(self, in_sca, in_vec, num_filters, n_component):
        super().__init__()
        self.n_component = n_component
        self.gvp = nn.Sequential(
            GVPerceptronVN(in_sca, in_vec, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.mu_net = GVLinear(num_filters[0], num_filters[1], n_component, n_component)
        self.logsigma_net= GVLinear(num_filters[0], num_filters[1], n_component, n_component)
        self.pi_net = GVLinear(num_filters[0], num_filters[1], n_component, 1)

    def forward(self, h_compose, vec_compose, focal_index, pos_compose):

        h_focal = h_compose[focal_index]
        vec_compose = vec_compose[focal_index]

        pos_focal = pos_compose[focal_index]
        
        feat_focal = self.gvp((h_focal, vec_compose))
        relative_mu = self.mu_net(feat_focal)[1]  # (N_focal, n_component, 3)
        logsigma = self.logsigma_net(feat_focal)[1]  # (N_focal, n_component, 3)
        sigma = torch.exp(logsigma)
        pi = self.pi_net(feat_focal)[0]  # (N_focal, n_component)
        pi = F.softmax(pi, dim=1)
        
        abs_mu = relative_mu + pos_focal.unsqueeze(dim=1).expand_as(relative_mu)
        return relative_mu, abs_mu, sigma, pi

    def get_mdn_probability(self, mu, sigma, pi, pos_target):
        prob_gauss = self._get_gaussian_probability(mu, sigma, pos_target)
        prob_mdn = pi * prob_gauss
        prob_mdn = torch.sum(prob_mdn, dim=1)
        return prob_mdn


    def _get_gaussian_probability(self, mu, sigma, pos_target):
        """
        mu - (N, n_component, 3)
        sigma - (N, n_component, 3)
        pos_target - (N, 3)
        """
        target = pos_target.unsqueeze(1).expand_as(mu)
        errors = target - mu
        sigma = sigma + 1e-16
        p = GAUSSIAN_COEF * torch.exp(- 0.5 * (errors / sigma)**2) / sigma
        p = torch.prod(p, dim=2)
        return p # (N, n_component)

    def sample_batch(self, mu, sigma, pi, num):
        """sample from multiple mix gaussian
            mu - (N_batch, n_cat, 3)
            sigma - (N_batch, n_cat, 3)
            pi - (N_batch, n_cat)
        return
            (N_batch, num, 3)
        """
        index_cats = torch.multinomial(pi, num, replacement=True)  # (N_batch, num)
        # index_cats = index_cats.unsqueeze(-1)
        index_batch = torch.arange(len(mu)).unsqueeze(-1).expand(-1, num)  # (N_batch, num)
        mu_sample = mu[index_batch, index_cats]  # (N_batch, num, 3)
        sigma_sample = sigma[index_batch, index_cats]
        values = torch.normal(mu_sample, sigma_sample)  # (N_batch, num, 3)
        return values

    def get_maximum(self, mu, sigma, pi):
        """sample from multiple mix gaussian
            mu - (N_batch, n_cat, 3)
            sigma - (N_batch, n_cat, 3)
            pi - (N_batch, n_cat)
        return
            (N_batch, n_cat, 3)
        """
        return mu



class AttentionBias(nn.Module):

    def __init__(self, num_heads, hidden_channels, cutoff=10., num_edge_types=3): #TODO: change the cutoff
        super().__init__()

        self.num_edge_types = num_edge_types
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels[0]-num_edge_types, fixed_offset=False)  # minus 1 for self edges (e.g. edge 0-0)
        self.vector_expansion = VecExpansion(hidden_channels[1])  # Linear(in_features=1, out_features=hidden_channels[1], bias=False)
        self.gvlinear = GVLinear(hidden_channels[0], hidden_channels[1], num_heads, num_heads)

    def forward(self,  tri_edge_index, tri_edge_feat, pos_compose):
        node_a, node_b = tri_edge_index
        pos_a = pos_compose[node_a]
        pos_b = pos_compose[node_b]
        vector = pos_a - pos_b
        dist = torch.norm(vector, p=2, dim=-1).unsqueeze(-1)
        
        dist_feat = self.distance_expansion(dist)
        sca_feat = torch.cat([
            dist_feat,
            tri_edge_feat,
        ], dim=-1)
        vec_feat = self.vector_expansion(vector)
        output_sca, output_vec = self.gvlinear([sca_feat, vec_feat])
        output_vec = (output_vec * output_vec).sum(-1)
        return output_sca, output_vec

class AttentionEdges(nn.Module):

    def __init__(self, hidden_channels, key_channels, num_heads=1, num_edge_types=3):
        super().__init__()
        
        assert (hidden_channels[0] % num_heads == 0) and (hidden_channels[1] % num_heads == 0)
        assert (key_channels[0] % num_heads == 0) and (key_channels[1] % num_heads == 0)

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        # linear transformation for attention 
        self.q_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.k_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.v_lin = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])

        self.atten_bias_lin = AttentionBias(self.num_heads, hidden_channels, num_edge_types=num_edge_types)
        self.layernorm_sca = nn.LayerNorm([hidden_channels[0]])
        self.layernorm_vec = nn.LayerNorm([hidden_channels[1], 3])

    def forward(self, edge_attr, edge_index, pos_compose, 
                index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat,):
        """
        Args:
            x:  edge features: scalar features (N, feat), vector features(N, feat, 3)
            edge_attr:  (E, H)
            edge_index: (2, E). the row can be seen as batch_edge
        """
        scalar, vector = edge_attr
        N = scalar.size(0)
        row, col = edge_index   # (N,) 

        # Project to multiple key, query and value spaces
        h_queries = self.q_lin(edge_attr)
        h_queries = (h_queries[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_queries[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_keys = self.k_lin(edge_attr)
        h_keys = (h_keys[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_keys[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_values = self.v_lin(edge_attr)
        h_values = (h_values[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_values[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)

        index_edge_i_list, index_edge_j_list = index_real_cps_edge_for_atten

        # # get nodes of triangle edges

        atten_bias = self.atten_bias_lin(
            tri_edge_index,
            tri_edge_feat,
            pos_compose,
        )


        # query * key
        queries_i = [h_queries[0][index_edge_i_list], h_queries[1][index_edge_i_list]]
        keys_j = [h_keys[0][index_edge_j_list], h_keys[1][index_edge_j_list]]

        qk_ij = [
            (queries_i[0] * keys_j[0]).sum(-1),  # (N', heads)
            (queries_i[1] * keys_j[1]).sum(-1).sum(-1)  # (N', heads)
        ]

        alpha = [
            atten_bias[0] + qk_ij[0],
            atten_bias[1] + qk_ij[1]
        ]
        alpha = [
            scatter_softmax(alpha[0], index_edge_i_list, dim=0),  # (N', heads)
            scatter_softmax(alpha[1], index_edge_i_list, dim=0)  # (N', heads)
        ] 

        values_j = [h_values[0][index_edge_j_list], h_values[1][index_edge_j_list]]
        num_attens = len(index_edge_j_list)
        output =[
            scatter_sum((alpha[0].unsqueeze(-1) * values_j[0]).view(num_attens, -1), index_edge_i_list, dim=0, dim_size=N),   # (N, H, 3)
            scatter_sum((alpha[1].unsqueeze(-1).unsqueeze(-1) * values_j[1]).view(num_attens, -1, 3), index_edge_i_list, dim=0, dim_size=N)   # (N, H, 3)
        ]

        # output 
        output = [edge_attr[0] + output[0], edge_attr[1] + output[1]]
        output = [self.layernorm_sca(output[0]), self.layernorm_vec(output[1])]

        return output



class AtomEdgePredictor(nn.Module):
    def __init__(self, config, num_classes, num_edge_classes, num_heads=4) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.num_edge_classes = num_edge_classes
        self.k = config.get('k', 32)
        in_sca = config.node_feat_dim
        in_vec = config.vec_feat_dim
        edge_channels = config.edge_feat_dim
        num_filters = (config.node_feat_dim, config.vec_feat_dim)
        cutoff = config.get('r_max', 10.0)

        self.message_module = MessageModule(in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff)

        self.nn_edge_ij = nn.Sequential(
            GVPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        
        self.classifier = nn.Sequential(
            GVPerceptronVN(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_classes, 1)
        )

        self.edge_feat = nn.Sequential(
            GVPerceptronVN(num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, self.num_edge_classes)
        self.edge_pred = GVLinear(num_filters[0], num_filters[1], self.num_edge_classes, 1)
        
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels, fixed_offset=False)
        self.distance_expansion_3A = GaussianSmearing(stop=3., num_gaussians=edge_channels, fixed_offset=False)
        self.vector_expansion = VecExpansion(edge_channels)  
    
    def forward(self, x_context, h_context, vec_context, 
                x_target, context_batch_idx, target_batch_idx, 
                cross_edge_index = [], att_edge_index = [], 
                tri_edge_index = [], tri_edge_feat = []):
        
        knn_cross_edge_index = knn(x_context, x_target, k=self.k, 
                                   batch_x=context_batch_idx, batch_y=target_batch_idx)
        
        vec_ij = x_target[knn_cross_edge_index[0]] - x_context[knn_cross_edge_index[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (A, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)

        h_context_update, vec_context_update = self.message_module((h_context, vec_context), 
                                                                   edge_ij, knn_cross_edge_index[1], 
                                                                   dist_ij, annealing=True)
        y = [scatter_add(h_context_update, index=knn_cross_edge_index[0], dim=0, dim_size=x_target.size(0)), # (N_query, F)
             scatter_add(vec_context_update, index=knn_cross_edge_index[0], dim=0, dim_size=x_target.size(0))]

        y_cls, _ = self.classifier(y)

        if (len(cross_edge_index) != 0) and (cross_edge_index.size(1) > 0):
            idx_node_i = cross_edge_index[1]
            node_mol_i = [
                y[0][idx_node_i],
                y[1][idx_node_i]
            ]
            idx_node_j = cross_edge_index[0]
            node_mol_j = [
                h_context[idx_node_j],
                vec_context[idx_node_j]
            ]
            vec_ij = x_target[idx_node_i] - x_context[idx_node_j]
            dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (E, 1)

            edge_ij = self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij) 
            edge_feat = self.nn_edge_ij(edge_ij)  # (E, F)

            edge_attr = (torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),  # (E, F)
                         torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1))
            edge_attr = self.edge_feat(edge_attr)  # (E, N_edgetype)
            edge_attr = self.edge_atten(edge_attr, cross_edge_index, x_context, 
                                        att_edge_index, tri_edge_index, tri_edge_feat) #
            edge_pred, _ = self.edge_pred(edge_attr)

        else:
            edge_pred = torch.empty([0, self.num_edge_classes], device=x_target.device)

        return y_cls, edge_pred
