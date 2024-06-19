import torch
from ._base import register_transform
from repo.utils.molecule.constants import *
from easydict import EasyDict

@register_transform('contrastive_sample')
class ContrastiveSample(object):
    def __init__(self, num_contrast=50, pos_real_std=0.05, pos_fake_std=2.0, knn=32):
    # def __init__(self, knn=32, elements=None):
        super().__init__()
        self.num_contrast = num_contrast
        self.pos_real_std = pos_real_std
        self.pos_fake_std = pos_fake_std
        self.knn = knn

    @property
    def num_elements(self):
        return self.elements.size(0)
    
    def construct_contrast_edge(self, cross_edge_index, cross_edge_type, real_sample_idx):

        cross_edge_index_dst, cross_edge_index_src = cross_edge_index[0], cross_edge_index[1]
        real_ctx_edge_idx_0_list, real_ctx_edge_idx_1_list, real_ctx_edge_type_list = [], [], []

        for new_idx, real_node in enumerate(real_sample_idx):
            idx_edge = (cross_edge_index_dst == real_node)

            real_ctx_edge_idx_1 = cross_edge_index_src[idx_edge]  # get edges related to this node
            real_ctx_edge_type = cross_edge_type[idx_edge]
            real_ctx_edge_idx_0 = new_idx * torch.ones(idx_edge.sum(), dtype=torch.long)  # change to new node index
            real_ctx_edge_idx_0_list.append(real_ctx_edge_idx_0)
            real_ctx_edge_idx_1_list.append(real_ctx_edge_idx_1)
            real_ctx_edge_type_list.append(real_ctx_edge_type)

        real_ctx_edge_idx_0 = torch.cat(real_ctx_edge_idx_0_list, dim=-1)
        real_ctx_edge_idx_1 = torch.cat(real_ctx_edge_idx_1_list, dim=-1)
        real_edge_index = torch.stack([real_ctx_edge_idx_0, real_ctx_edge_idx_1], dim=0)
        real_edge_type = torch.cat(real_ctx_edge_type_list, dim=-1)
        return real_edge_index, real_edge_type
    
    def construct_tri_edge(self, cross_edge_index, context_edge_index, context_edge_type, real_sample_idx, num_context):
        row, col = cross_edge_index[0], cross_edge_index[1]
        acc_num_edges = 0
        index_real_cps_edge_i_list, index_real_cps_edge_j_list = [], []  # index of real-ctx edge (for attention)
        for node in torch.arange(len(real_sample_idx)):
            num_edges = (row == node).sum()
            index_edge_i = torch.arange(num_edges, dtype=torch.long, ) + acc_num_edges
            index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i, indexing='ij')
            index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
            index_real_cps_edge_i_list.append(index_edge_i)
            index_real_cps_edge_j_list.append(index_edge_j)
            acc_num_edges += num_edges
        index_real_cps_edge_i = torch.cat(index_real_cps_edge_i_list, dim=0)  # add len(real_compose_edge_index) in the dataloader for batch
        index_real_cps_edge_j = torch.cat(index_real_cps_edge_j_list, dim=0)

        node_a_cps_tri_edge = col[index_real_cps_edge_i]  # the node of tirangle edge for the edge attention (in the compose)
        node_b_cps_tri_edge = col[index_real_cps_edge_j]
        
        adj_mat = torch.zeros([num_context, num_context], dtype=torch.long) - torch.eye(num_context, dtype=torch.long)
        adj_mat[context_edge_index[0], context_edge_index[1]] = context_edge_type
        tri_edge_type = adj_mat[node_a_cps_tri_edge, node_b_cps_tri_edge]
        tri_edge_feat = (tri_edge_type.view([-1, 1]) == torch.tensor([[0, 1, 2, 3, 4]]).to(tri_edge_type)).long()
        tri_edge_index = torch.stack([node_a_cps_tri_edge, node_b_cps_tri_edge], dim=0)
        att_edge_index = torch.stack([index_real_cps_edge_i, index_real_cps_edge_j], dim=0)
        return tri_edge_index, tri_edge_feat, att_edge_index


    def __call__(self, data):
        context_idx = data.ligand.context_idx
        data.ligand.frontier = data.ligand_context.num_neighbors < data.ligand.num_neighbors[context_idx]

        # Positive samples
        pos_real_mode = data.ligand_masked.pos
        type_real = data.ligand_masked.atom_type
        # ind_real = data.ligand_masked_feature
        p = torch.zeros(len(pos_real_mode))
        p = torch.where(data.ligand_masked.gen_flag, torch.ones_like(p), p)
        real_sample_idx = np.random.choice(np.arange(pos_real_mode.size(0)), 
                                           size=self.num_contrast, 
                                           p=(p/p.sum()).numpy())
        
        data.ligand_masked_contrast = EasyDict()
        data.ligand_masked_contrast.pos_real = pos_real_mode[real_sample_idx]
        data.ligand_masked_contrast.pos_real += torch.randn_like(data.ligand_masked_contrast.pos_real) * self.pos_real_std
        data.ligand_masked_contrast.type_real = type_real[real_sample_idx]

        real_edge_index, real_edge_type = self.construct_contrast_edge(
            data.ligand_context_cross_ligand_masked.sampled_edge_index, 
            data.ligand_context_cross_ligand_masked.sampled_edge_type,
            real_sample_idx)

        data.ligand_context_cross_ligand_masked_contrast = EasyDict()
        data.ligand_context_cross_ligand_masked_contrast.real_edge_index = real_edge_index
        data.ligand_context_cross_ligand_masked_contrast.real_edge_type = real_edge_type

        # Negative samples
        if len(data.ligand_context.pos) != 0: # all mask
            pos_fake_mode = data.ligand_context.pos[data.ligand.frontier]
        else:
            pos_fake_mode = data.protein.pos[data.protein.focal_flag]

        fake_sample_idx = np.random.choice(np.arange(pos_fake_mode.size(0)), size=self.num_contrast)
        pos_fake = pos_fake_mode[fake_sample_idx]
        data.ligand_masked_contrast.pos_fake = pos_fake + torch.randn_like(pos_fake) * self.pos_fake_std / 2.
        data.ligand_masked_contrast.num_nodes = self.num_contrast

        tri_edge_index, tri_edge_feat, att_edge_index = self.construct_tri_edge(
            data.ligand_context_cross_ligand_masked_contrast.real_edge_index,
            data.ligand_context.bond_index,
            data.ligand_context.bond_type,
            real_sample_idx,
            data.ligand_context.num_nodes)
        
        data.edge_graph = EasyDict()
        data.edge_graph.num_nodes = len(data.ligand_context_cross_ligand_masked_contrast.real_edge_type)
        data.edge_graph.edge_attr = data.ligand_context_cross_ligand_masked_contrast.real_edge_type
        data.ligand_context.tri_edge_index = tri_edge_index
        data.ligand_context.tri_edge_feat = tri_edge_feat
        data.edge_graph.att_edge_index = att_edge_index

        return data
