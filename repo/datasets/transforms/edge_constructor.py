from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch
from ._base import register_transform
from torch_scatter import scatter_add
from torch_geometric.nn import knn
from easydict import EasyDict

@register_transform('count_bond_neighbors')
class CountBondNeighbors(object):
    def __init__(self, graph_name='ligand'):
        super().__init__()
        self.graph_name = graph_name

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __call__(self, data):
        data[self.graph_name].num_neighbors = self.count_neighbors(
            data[self.graph_name].bond_index, 
            symmetry=True,
            num_nodes=data[self.graph_name].element.size(0),
        )
        data[self.graph_name].atom_valence = self.count_neighbors(
            data[self.graph_name].bond_index, 
            symmetry=True, 
            valence=data[self.graph_name].bond_type,
            num_nodes=data[self.graph_name].element.size(0),
        )
        data[self.graph_name].num_neighbors_per_bond = torch.stack([
            self.count_neighbors(
                data[self.graph_name].bond_index, 
                symmetry=True, 
                valence=(data[self.graph_name].bond_type == i).long(),
                num_nodes=data[self.graph_name].element.size(0),
            ) for i in [1, 2, 3]
        ], dim = -1)
        return data


@register_transform('sample_edge_for_ligand')
class SampleEdge(object):
    def __init__(self, k) -> None:
        self.k = k
    
    def __call__(self, data):
        ligand_context_pos = data.ligand_context.pos
        ligand_masked_pos = data.ligand_masked.pos
        context_idx = data.ligand.context_idx
        masked_idx = data.ligand.masked_idx

        old_bond_index = data.ligand.bond_index
        old_bond_types = data.ligand.bond_type  
        
        # candidate edge: mask-contex edge
        idx_edge_index_candidate = [
            (context_node in context_idx) and (mask_node in masked_idx)
            for mask_node, context_node in zip(*old_bond_index)
        ]  # the mask-context order is right
        candidate_bond_index = old_bond_index[:, idx_edge_index_candidate]
        candidate_bond_types = old_bond_types[idx_edge_index_candidate]
        
        # index changer
        index_changer_masked = torch.zeros(masked_idx.max()+1, dtype=torch.int64)
        index_changer_masked[masked_idx] = torch.arange(len(masked_idx))

        has_unmask_atoms = len(context_idx) > 0
        if has_unmask_atoms:
            index_changer_context = torch.zeros(context_idx.max()+1, dtype=torch.int64)
            index_changer_context[context_idx] = torch.arange(len(context_idx))

            # new edge index (positive)
            new_edge_index_0 = index_changer_masked[candidate_bond_index[0]]
            new_edge_index_1 = index_changer_context[candidate_bond_index[1]]
            new_edge_index = torch.stack([new_edge_index_0, new_edge_index_1])
            new_edge_type = candidate_bond_types
            edge_index_knn = knn(ligand_context_pos, ligand_masked_pos, k=self.k, num_workers=16)
            dist = torch.norm(ligand_masked_pos[edge_index_knn[0]] - ligand_context_pos[edge_index_knn[1]], p=2, dim=-1)
            idx_sort = torch.argsort(dist)  #  choose negative edges as short as possible
            num_neg_edges = min(len(ligand_masked_pos) * (self.k // 2) + len(new_edge_index[0]), len(idx_sort))
            idx_sort = torch.unique(
                torch.cat([
                    idx_sort[:num_neg_edges],
                    torch.linspace(0, len(idx_sort), len(ligand_masked_pos)+1, dtype=torch.long)[:-1]  # each mask pos at least has one negative edge
                ], dim=0)
            )
            edge_index_knn = edge_index_knn[:, idx_sort]
            id_edge_knn = edge_index_knn[0] * len(context_idx) + edge_index_knn[1]  # delete false negative edges
            id_edge_new = new_edge_index[0] * len(context_idx) + new_edge_index[1]
            idx_real_edge_index = torch.tensor([id_ in id_edge_new for id_ in id_edge_knn])
            false_edge_index = edge_index_knn[:, ~idx_real_edge_index]
            false_edge_types = torch.zeros(len(false_edge_index[0]), dtype=torch.int64)

            new_edge_index = torch.cat([new_edge_index, false_edge_index], dim=-1)
            new_edge_type = torch.cat([new_edge_type, false_edge_types], dim=0)

        else:
            new_edge_index = torch.empty([2,0], dtype=torch.int64)
            new_edge_type = torch.empty([0], dtype=torch.int64)

        data.ligand_context_cross_ligand_masked.sampled_edge_index = new_edge_index
        data.ligand_context_cross_ligand_masked.sampled_edge_type = new_edge_type
        return data