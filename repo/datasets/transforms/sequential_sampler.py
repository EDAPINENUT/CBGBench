from typing import Any
from ._base import register_mode_transform, register_transform
import random
import torch
import networkx as nx
from networkx.algorithms import tree

@register_transform('seq_sample')
class SequentialSampler(object):
    def __init__(self) -> None:
        pass

    def __call__(self, data) -> Any:
        lig = data['ligand']
        mask_lig_gen = lig.gen_flag
        num_atom_gen = mask_lig_gen.sum()
        i = random.randint(0, num_atom_gen)
        ctx_contact_id, ctx_noncontact_id = data.ctx_contact_id, data.ctx_noncontact_id

        if i == 0:
            contact_index = torch.tensor([ctx_contact_id, ctx_noncontact_id], dtype=int)
            contact_label = torch.tensor([1,0], dtype=torch.float)

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

    def __call__(self, data) -> torch.Any:
        lig = data['ligand']

        pos_lig = lig.pos
        mask_lig_ctx = lig.ctx_flag
        mask_lig_gen = lig.gen_flag
        atom_type_lig = lig.atom_type
        num_atom_gen = mask_lig_gen.sum()
        pos_gen = pos_lig[mask_lig_gen]
        atom_type_gen = atom_type_lig[mask_lig_gen] #[gen_n_atoms]

        con_mat_gen = torch.zeros([num_atom_gen, num_atom_gen], dtype=int)
        for bond_index, bond_type in zip(lig.bond_index.transpose(0,1), lig.bond_type):
            start, end = bond_index[0], bond_index[1]
            con_mat_gen[start, end] = bond_type
            con_mat_gen[end, start] = bond_type
        gen_atom_bond_valency = torch.sum(gen_con_mat, axis=1) #[gen_n_atoms]
        
        pos_ctx_merge = torch.concat([data['protein'].pos, data['ligand'].pos[mask_lig_ctx]])
        atom_type_ctx_merge = torch.concat([data['protein'].atom_type, data['ligand'].atom_type[mask_lig_ctx]])
        
        dist_gen_to_ctx, dist_gen_to_gen, gen_contact_id, ctx_contact_id, ctx_noncontact_id = self._get_contact_ids(pos_gen, pos_ctx_merge)
        perm = torch.arange(0, num_atom_gen, dtype=int)


        perm[0] = gen_contact_id
        perm[gen_contact_id] = 0

        atom_type_gen, pos_gen = atom_type_gen[perm], pos_gen[perm]
        gen_atom_bond_valency, dist_gen_to_ctx = gen_atom_bond_valency[perm], dist_gen_to_ctx[perm]
        gen_con_mat, dist_gen_to_gen = gen_con_mat[perm][:, perm], dist_gen_to_gen[perm][:, perm]

        # Decide the order among lig nodes
        node_perm, focus_node_id = self._reindex_attrs(dist_gen_to_gen)
        atom_type_gen, pos_gen = atom_type_gen[node_perm], pos_gen[node_perm]
        gen_atom_bond_valency, dist_gen_to_ctx = gen_atom_bond_valency[node_perm], dist_gen_to_ctx[node_perm]
        gen_con_mat, dist_gen_to_gen = gen_con_mat[node_perm][:, node_perm], dist_gen_to_gen[node_perm][:, node_perm]

        # Prepare training data for sequential generation 
        focus_node_id = torch.tensor(focus_node_id)
        focus_ids = torch.nonzero(focus_node_id[:,None] == node_perm[None,:])[:,1] # focus_ids denotes the focus atom IDs that are indiced according to the order given by node_perm
