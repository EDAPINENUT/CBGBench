import torch
from ._base import register_transform
from torch_geometric.nn import knn, radius
from easydict import EasyDict

@register_transform('build_focal_for_ligand')
class BuildFocal(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        protein_pos = data.protein.pos
        context_idx = data.ligand.context_idx
        masked_idx = data.ligand.masked_idx
        old_bond_index = data.ligand.bond_index

        has_unmask_atoms = len(context_idx) > 0
        data.protein_cross_ligand_masked = EasyDict() 
        data.ligand_context_cross_ligand_masked = EasyDict()

        if has_unmask_atoms:
            # # get bridge bond index (mask-context bond)
            ind_edge_index_candidate = [
                (context_node in context_idx) and (mask_node in masked_idx)
                for mask_node, context_node in zip(*old_bond_index)
            ]  # the mask-context order is right
            bridge_bond_index = old_bond_index[:, ind_edge_index_candidate]

            idx_generated_in_whole_ligand = bridge_bond_index[0]
            idx_focal_in_whole_ligand = bridge_bond_index[1]
            if len(masked_idx)==0:
                print('')
            
            index_changer_masked = torch.zeros(masked_idx.max()+1, dtype=torch.int64)
            index_changer_masked[masked_idx] = torch.arange(len(masked_idx))
            idx_generated_in_ligand_masked = index_changer_masked[idx_generated_in_whole_ligand]
            data.ligand_masked.gen_flag = torch.zeros_like(data.ligand_masked.element, dtype=torch.bool)
            data.ligand_masked.gen_flag[idx_generated_in_ligand_masked] = True

            index_changer_context = torch.zeros(context_idx.max()+1, dtype=torch.int64)
            index_changer_context[context_idx] = torch.arange(len(context_idx))
            idx_focal_in_ligand_context = index_changer_context[idx_focal_in_whole_ligand]
            
            data.ligand_context.focal_flag = torch.zeros_like(data.ligand_context.element, dtype=torch.bool)
            data.ligand_context.focal_flag[idx_focal_in_ligand_context] = True
            data.ligand_context.pred_flag = torch.ones_like(data.ligand_context.element, dtype=torch.bool)

            data.protein.focal_flag = torch.zeros_like(data.protein.element, dtype=torch.bool)
            data.protein.pred_flag = torch.zeros_like(data.protein.element, dtype=torch.bool)

            data.protein_cross_ligand_masked.edge_index = torch.empty((2, 0), dtype=torch.long)
            data.ligand_context_cross_ligand_masked.edge_index = torch.stack([idx_generated_in_ligand_masked,
                                                                              idx_focal_in_ligand_context], 
                                                                              dim=0)

        else:  # # the initial atom. surface atoms between ligand and protein
            ligand_masked_pos = data.ligand_masked.pos
            assign_index = radius(x=ligand_masked_pos, y=protein_pos, r=4., num_workers=16)
            
            if assign_index.size(1) == 0:
                dist = torch.norm(protein_pos.unsqueeze(1) - ligand_masked_pos.unsqueeze(0), p=2, dim=-1)
                assign_index = torch.nonzero(dist <= torch.min(dist)+1e-5)[0:1].transpose(0, 1)
            
            idx_focal_in_protein = assign_index[0]
            data.protein.focal_flag = torch.zeros_like(data.protein.element, dtype=torch.bool)  # no ligand context, so all composes are protein atoms
            data.protein.focal_flag[idx_focal_in_protein] = True
            data.protein.pred_flag = torch.ones_like(data.protein.element, dtype=torch.bool)

            data.ligand_context.focal_flag = torch.zeros_like(data.ligand_context.element, dtype=torch.bool)
            data.ligand_context.pred_flag = torch.zeros_like(data.ligand_context.element, dtype=torch.bool)

            data.ligand_masked.gen_flag = torch.zeros_like(data.ligand_masked.element, dtype=torch.bool)  # no ligand context, so all composes are protein atoms
            data.ligand_masked.gen_flag[assign_index[1]] = True

            data.protein_cross_ligand_masked.edge_index = torch.stack([assign_index[1], 
                                                                       idx_focal_in_protein], 
                                                                       dim=0)
            data.ligand_context_cross_ligand_masked.edge_index = torch.empty((2, 0), dtype=torch.long)

        return data