from ._base import register_transform, register_mode_transform, get_fg_type, get_index
import torch
import numpy as np 
import networkx as nx
from networkx.algorithms import tree
from easydict import EasyDict

@register_transform('reindex_atom_seq_graph')
class ReindexGenAndSeqGraph(object):
    def __init__(self) -> None:
        pass

    def _get_contact_ids(self, pos_gen, pos_ctx):
        num_atom_ctx = len(pos_ctx)

        dist_gen_to_ctx = torch.cdist(pos_gen, pos_ctx)
        dist_gen_to_gen = torch.cdist(pos_gen, pos_gen)

        min_index = torch.argmin(dist_gen_to_ctx)
        gen_contact_id = min_index // num_atom_ctx
        ctx_contact_id = min_index % num_atom_ctx
        ctx_noncontact_id = torch.argmax(torch.sum(dist_gen_to_ctx, dim=0))

        return dist_gen_to_ctx, dist_gen_to_gen, gen_contact_id, ctx_contact_id, ctx_noncontact_id
    
    def _reindex_attrs(self, dist_gen_to_gen):

        nx_graph = nx.from_numpy_array(dist_gen_to_gen.numpy())
        edges = list(tree.minimum_spanning_edges(nx_graph, algorithm='prim', data=False)) # return edges starts from the 0-th node (i.e., the contact node here) by default
        focus_node_id, target_node_id = zip(*edges)
        node_perm = torch.cat((torch.tensor([0]), torch.tensor(target_node_id)))

        return node_perm, focus_node_id

    def map_mol_index_to_gen_index(self, mol_index, gen_flag):
        gen_index = torch.arange(0, gen_flag.sum())
        gen_index_map = torch.zeros_like(gen_flag).long()
        gen_index_map[gen_flag] = gen_index
        return gen_index_map[mol_index]
    
    def map_gen_index_to_mol_index(self, gen_index, gen_flag):
        mol_index = torch.arange(0, len(gen_flag))
        mol_index_map = mol_index[gen_flag]
        return mol_index_map[gen_index]

    def __call__(self, data) -> torch.Any:
        lig = data['ligand']

        pos_lig = lig.pos
        mask_lig_ctx = lig.ctx_flag
        mask_lig_gen = lig.gen_flag
        atom_type_lig = lig.atom_type
        num_atom_gen = mask_lig_gen.sum()
        pos_gen = pos_lig[mask_lig_gen]
        atom_type_gen = atom_type_lig[mask_lig_gen] #[num_atom_gen]

        con_mat_gen = torch.zeros([num_atom_gen, num_atom_gen], dtype=int)
        if hasattr(lig, 'bond_index') and not hasattr(lig, 'gen_bond_index'):
            for bond_index, bond_type in zip(lig.bond_index.transpose(0,1), lig.bond_type):
                start, end = bond_index[0], bond_index[1]
                con_mat_gen[start, end] = bond_type
                con_mat_gen[end, start] = bond_type
        elif hasattr(lig, 'gen_bond_index'):
            gen_bond_index_0 = self.map_mol_index_to_gen_index(lig.gen_bond_index[0], mask_lig_gen)
            gen_bond_index_1 = self.map_mol_index_to_gen_index(lig.gen_bond_index[1], mask_lig_gen)
            gen_bond_index = torch.stack([gen_bond_index_0, gen_bond_index_1], dim=0)

            for bond_index, bond_type in zip(gen_bond_index.transpose(0,1), lig.gen_bond_type):
                start, end = bond_index[0], bond_index[1]
                con_mat_gen[start, end] = bond_type
                con_mat_gen[end, start] = bond_type

        gen_atom_bond_valency = torch.sum(con_mat_gen, axis=1) #[num_atom_gen]
        
        pos_ctx_merge = torch.concat([data['protein'].pos, data['ligand'].pos[mask_lig_ctx]])
        atom_type_ctx_merge = torch.concat([data['protein'].atom_type, data['ligand'].atom_type[mask_lig_ctx]])
        
        dist_gen_to_ctx, dist_gen_to_gen, gen_contact_id, ctx_contact_id, ctx_noncontact_id = self._get_contact_ids(pos_gen, pos_ctx_merge)
        perm = torch.arange(0, num_atom_gen, dtype=int)
        num_atom_ctx = len(pos_ctx_merge)

        perm[0] = gen_contact_id
        perm[gen_contact_id] = 0

        atom_type_gen, pos_gen = atom_type_gen[perm], pos_gen[perm]
        gen_atom_bond_valency, dist_gen_to_ctx = gen_atom_bond_valency[perm], dist_gen_to_ctx[perm]
        con_mat_gen, dist_gen_to_gen = con_mat_gen[perm][:, perm], dist_gen_to_gen[perm][:, perm]

        # Decide the order among lig nodes
        node_perm, focus_node_id = self._reindex_attrs(dist_gen_to_gen)
        atom_type_gen, pos_gen = atom_type_gen[node_perm], pos_gen[node_perm]
        gen_atom_bond_valency, dist_gen_to_ctx = gen_atom_bond_valency[node_perm], dist_gen_to_ctx[node_perm]
        con_mat_gen, dist_gen_to_gen = con_mat_gen[node_perm][:, node_perm], dist_gen_to_gen[node_perm][:, node_perm]

        # Prepare training data for sequential generation 
        focus_node_id = torch.tensor(focus_node_id)
        focus_ids = torch.nonzero(focus_node_id[:,None] == node_perm[None,:])[:,1] # focus_ids denotes the focus atom IDs that are indiced according to the order given by node_perm

        steps_cannot_focus = torch.empty([0,1], dtype=torch.float)
        idx_offsets = torch.cumsum(torch.arange(num_atom_gen), dim=0) #[M]
        idx_offsets_brought_by_ctx = num_atom_ctx * torch.arange(1, num_atom_gen) #[M-1]

        for i in range(num_atom_gen):
            if i==0:
                # In the 1st step, all we have is the rec. Note that contact classifier should be only applied for the 1st step in which we don't have any lig atoms
                steps_atom_type = atom_type_ctx_merge
                steps_ctx_mask = torch.ones([num_atom_ctx], dtype=torch.bool)
                contact_y_or_n = torch.tensor([ctx_contact_id, ctx_noncontact_id], dtype=int) # The atom IDs of contact node and the node that are furthest from lig. 
                cannot_contact = torch.tensor([0,1], dtype=torch.float) # The groundtruth for contact_y_or_n
                steps_position = pos_ctx_merge
                steps_batch = torch.tensor([i]).repeat(num_atom_ctx)
                steps_focus = torch.tensor([ctx_contact_id], dtype=int)

                dist_to_focus = torch.sum(torch.square(pos_ctx_merge[ctx_contact_id] - pos_ctx_merge), dim=-1)
                _, indices = torch.topk(dist_to_focus, 3, largest=False)
                one_step_c1, one_step_c2 = indices[1], indices[2]
                assert indices[0] == ctx_contact_id
                steps_c1_focus = torch.tensor([one_step_c1, ctx_contact_id], dtype=int).view(1,2)
                steps_c2_c1_focus = torch.tensor([one_step_c2, one_step_c1, ctx_contact_id], dtype=int).view(1,3)

                focus_pos, new_pos = pos_ctx_merge[ctx_contact_id], pos_gen[i]
                one_step_dis = torch.norm(new_pos - focus_pos)
                steps_dist = one_step_dis.view(1,1)

                c1_pos = pos_ctx_merge[one_step_c1]
                a = ((c1_pos - focus_pos) * (new_pos - focus_pos)).sum(dim=-1)
                b = torch.cross(c1_pos - focus_pos, new_pos - focus_pos).norm(dim=-1)
                one_step_angle = torch.atan2(b,a)
                steps_angle = one_step_angle.view(1,1)

                c2_pos = pos_ctx_merge[one_step_c2]
                plane1 = torch.cross(focus_pos - c1_pos, new_pos - c1_pos)
                plane2 = torch.cross(focus_pos - c1_pos, c2_pos - c1_pos)
                a = (plane1 * plane2).sum(dim=-1)
                b = (torch.cross(plane1, plane2) * (focus_pos - c1_pos)).sum(dim=-1) / torch.norm(focus_pos - c1_pos)
                one_step_torsion = torch.atan2(b, a)
                steps_torsion = one_step_torsion.view(1,1)


            else:
                one_step_atom_type = torch.cat((atom_type_gen[:i], atom_type_ctx_merge), dim=0)
                steps_atom_type = torch.cat((steps_atom_type, one_step_atom_type))
                one_step_ctx_mask = torch.cat((torch.zeros([i], dtype=torch.bool), torch.ones([num_atom_ctx], dtype=torch.bool)), dim=0)
                steps_ctx_mask = torch.cat((steps_ctx_mask, one_step_ctx_mask))
                one_step_position =  torch.cat((pos_gen[:i], pos_ctx_merge), dim=0)
                steps_position = torch.cat((steps_position, one_step_position))
                steps_batch = torch.cat((steps_batch, torch.tensor([i]).repeat(i + num_atom_ctx)))

                partial_con_mat_gen = con_mat_gen[:i, :i]
                bond_sum = partial_con_mat_gen.sum(dim=1, keepdim=True)
                steps_cannot_focus = torch.cat((steps_cannot_focus, (bond_sum == gen_atom_bond_valency[:i, None]).float()))

                focus_id = focus_ids[i-1]
                if i == 1: # c1, c2 must be in rec
                    dist_to_focus =  dist_gen_to_ctx[focus_id]
                    _, indices = torch.topk(dist_to_focus, 2, largest=False)
                    one_step_c1, one_step_c2 = indices[0], indices[1]
                    one_step_c1_focus = torch.tensor([one_step_c1+idx_offsets[i]+idx_offsets_brought_by_ctx[i-1], 
                                                      focus_id+idx_offsets_brought_by_ctx[i-1]], dtype=int).view(1,2)
                    steps_c1_focus = torch.cat((steps_c1_focus, one_step_c1_focus), dim=0)
                    one_step_c2_c1_focus = torch.tensor([one_step_c2+idx_offsets[i]+idx_offsets_brought_by_ctx[i-1],
                                                         one_step_c1+idx_offsets[i]+idx_offsets_brought_by_ctx[i-1], 
                                                         focus_id+idx_offsets_brought_by_ctx[i-1]], dtype=int).view(1,3)
                    steps_c2_c1_focus = torch.cat((steps_c2_c1_focus, one_step_c2_c1_focus), dim=0)

                    focus_pos, new_pos = pos_gen[focus_id], pos_gen[i]
                    one_step_dis = torch.norm(new_pos - focus_pos).view(1,1)
                    steps_dist = torch.cat((steps_dist, one_step_dis), dim=0)

                    c1_pos = pos_ctx_merge[one_step_c1]
                    a = ((c1_pos - focus_pos) * (new_pos - focus_pos)).sum(dim=-1)
                    b = torch.cross(c1_pos - focus_pos, new_pos - focus_pos).norm(dim=-1)
                    one_step_angle = torch.atan2(b,a).view(1,1)
                    steps_angle = torch.cat((steps_angle, one_step_angle), dim=0)

                    c2_pos = pos_ctx_merge[one_step_c2]
                    plane1 = torch.cross(focus_pos - c1_pos, new_pos - c1_pos)
                    plane2 = torch.cross(focus_pos - c1_pos, c2_pos - c1_pos)
                    a = (plane1 * plane2).sum(dim=-1)
                    b = (torch.cross(plane1, plane2) * (focus_pos - c1_pos)).sum(dim=-1) / torch.norm(focus_pos - c1_pos)
                    one_step_torsion = torch.atan2(b, a).view(1,1)
                    steps_torsion = torch.cat((steps_torsion, one_step_torsion), dim=0)

                else: #c1, c2 could be in both lig and rec
                    dist_to_focus = torch.cat((dist_gen_to_gen[focus_id, :i], dist_gen_to_ctx[focus_id]))
                    _, indices = torch.topk(dist_to_focus, 3, largest=False)
                    one_step_c1, one_step_c2 = indices[1], indices[2]

                    one_step_c1_focus = torch.tensor([one_step_c1+idx_offsets[i-1]+idx_offsets_brought_by_ctx[i-1], 
                                                      focus_id+idx_offsets[i-1]+idx_offsets_brought_by_ctx[i-1]], dtype=int).view(1,2)
                    one_step_c2_c1_focus = torch.tensor([one_step_c2+idx_offsets[i-1]+idx_offsets_brought_by_ctx[i-1], 
                                                         one_step_c1+idx_offsets[i-1]+idx_offsets_brought_by_ctx[i-1], 
                                                         focus_id+idx_offsets[i-1]+idx_offsets_brought_by_ctx[i-1]], dtype=int).view(1,3)
                    if one_step_c1 < i: # c1 in lig
                        c1_pos = pos_gen[one_step_c1]
                        if one_step_c2 < i: # c2 in lig
                            c2_pos = pos_gen[one_step_c2]
                        else: 
                            c2_pos = pos_ctx_merge[one_step_c2-i]
                    else: 
                        c1_pos = pos_ctx_merge[one_step_c1-i]
                        if one_step_c2 < i: # c2 in lig
                            c2_pos = pos_gen[one_step_c2]
                        else: # c2 in rec
                            c2_pos = pos_ctx_merge[one_step_c2-i]
                    steps_c1_focus = torch.cat((steps_c1_focus, one_step_c1_focus), dim=0)
                    steps_c2_c1_focus = torch.cat((steps_c2_c1_focus, one_step_c2_c1_focus), dim=0)

                    focus_pos, new_pos = pos_gen[focus_id], pos_gen[i]
#                     if i==3 or i==4: # Use for debug. We have verified the id offset is correct.
#                         print(new_pos)
                    one_step_dis = torch.norm(new_pos - focus_pos).view(1,1)
                    steps_dist = torch.cat((steps_dist, one_step_dis), dim=0)

                    a = ((c1_pos - focus_pos) * (new_pos - focus_pos)).sum(dim=-1)
                    b = torch.cross(c1_pos - focus_pos, new_pos - focus_pos).norm(dim=-1)
                    one_step_angle = torch.atan2(b,a).view(1,1)
                    steps_angle = torch.cat((steps_angle, one_step_angle), dim=0)

                    plane1 = torch.cross(focus_pos - c1_pos, new_pos - c1_pos)
                    plane2 = torch.cross(focus_pos - c1_pos, c2_pos - c1_pos)
                    a = (plane1 * plane2).sum(dim=-1)
                    b = (torch.cross(plane1, plane2) * (focus_pos - c1_pos)).sum(dim=-1) / torch.norm(focus_pos - c1_pos)
                    one_step_torsion = torch.atan2(b, a).view(1,1)
                    steps_torsion = torch.cat((steps_torsion, one_step_torsion), dim=0)


        steps_focus = torch.cat((steps_focus, 
                                 focus_ids+idx_offsets[:-1]+idx_offsets_brought_by_ctx), dim=0)
        steps_new_atom_type = atom_type_gen

        # For example, for a rec-lig pair, rec has N atoms and lig has M atoms
        data_batch = {}
        data_batch['atom_type'] = steps_atom_type # [N+(1+N)+(2+N)+...+(M-1+N)], which correspond to M steps
        data_batch['pos'] = steps_position # [N+(1+N)+(2+N)+...+(M-1+N), 3]
        data_batch['ctx_flag'] = steps_ctx_mask
        data_batch['gen_flag'] = torch.logical_not(steps_ctx_mask) # [N+(1+N)+(2+N)+...+(M-1+N)]
         # [N+(1+N)+(2+N)+...+(M-1+N)]
        data_batch['batch'] = steps_batch # [N+(1+N)+(2+N)+...+(M-1+N)]
        data_batch['contact_y_or_n'] = contact_y_or_n # [2]
        data_batch['cannot_contact'] = cannot_contact # [2]
        data_batch['new_atom_type'] = steps_new_atom_type # [M]

        data_batch['focus'] = steps_focus.view(-1)  # [M, 1]
        assert steps_focus.view(-1).max() <= steps_atom_type.shape[0]
        data_batch['c1_focus'] = steps_c1_focus # [M, 2]
        data_batch['c2_c1_focus'] = steps_c2_c1_focus # [M, 3]

        data_batch['new_dist'] = steps_dist # [M, 1]
        data_batch['new_angle'] = steps_angle # [M, 1]
        data_batch['new_torsion'] = steps_torsion # [M, 1]

        data_batch['cannot_focus'] = steps_cannot_focus.view(-1) 

        return EasyDict(data_batch)