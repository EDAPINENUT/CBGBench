from ._base import register_transform
import random
import torch
import numpy as np
try:
    from torch_geometric.utils.subgraph import subgraph
except:
    from torch_geometric.utils._subgraph import subgraph
from repo.modules.common import _atom_index_select_data
from easydict import EasyDict
import copy
from repo.utils.chemutils import *
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from repo.utils.dihedutils import batch_dihedrals

def set_subgraph(data, idx):
    data_selected = EasyDict(_atom_index_select_data(data, idx))
    if data.bond_index.size(1) != 0:
        data_selected.bond_index, data_selected.bond_type = subgraph(
            idx,
            data.bond_index,
            edge_attr = data.bond_type,
            relabel_nodes = True,
            num_nodes = data.pos.shape[0]
        )
    else:
        data_selected.bond_index = torch.empty([2, 0], dtype=torch.long)
        data_selected.bond_type = torch.empty([0], dtype=torch.long)
    data_selected.num_nodes = len(idx)
    return data_selected

def enumerate_assemble(mol, idxs, current, next):
    ctr_mol = get_submol(mol, idxs, mark=current.clique)
    ground_truth = get_submol(mol, list(set(idxs) | set(next.clique)))
    # submol can also obtained with get_clique_mol, future exploration
    ground_truth_smiles = get_smiles(ground_truth)
    cand_smiles = []
    cand_mols = []
    cand_amap = enum_attach(ctr_mol, next.mol)
    for amap in cand_amap:
        try:
            cand_mol, _ = attach(ctr_mol, next.mol, amap)
            cand_mol = sanitize(cand_mol)
        except:
            continue
        if cand_mol is None:
            continue
        smiles = get_smiles(cand_mol)
        if smiles in cand_smiles or smiles == ground_truth_smiles:
            continue
        cand_smiles.append(smiles)
        cand_mols.append(cand_mol)
    if len(cand_mols) >= 1:
        cand_mols = sample(cand_mols, 1)
        cand_mols.append(ground_truth)
        labels = torch.tensor([0, 1])
    else:
        cand_mols = [ground_truth]
        labels = torch.tensor([1])

    return labels, cand_mols

@register_transform('mixed_mask')
class MixedMask(object):

    def __init__(self, 
                 mask_target='ligand',
                 min_ratio=0.0, 
                 max_ratio=1.2, 
                 min_num_masked=1, 
                 min_num_unmasked=0,
                 p_random=0.5, 
                 p_bfs=0.25, 
                 p_invbfs=0.25):
        super().__init__()
        self.mask_target = mask_target

        self.t = [
            RandomMask(mask_target, min_ratio, max_ratio, min_num_masked, min_num_unmasked),
            BFSMask(mask_target, min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=False),
            BFSMask(mask_target, min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=True),
        ]
        self.p = [p_random, p_bfs, p_invbfs]

    def __call__(self, data):
        f = random.choices(self.t, k=1, weights=self.p)[0]
        return f(data)

@register_transform('bfs_mask')
class BFSMask(object):

    def __init__(self, mask_target, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, inverse=False):
        super().__init__()
        self.mask_target = mask_target
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.inverse = inverse
    
    def _get_nbhd_list(self, data):
        nbhd_list = {i.item():[j.item() for k, j in enumerate(data.bond_index[1]) \
                               if data.bond_index[0, k].item() == i] for i in data.bond_index[0]}
        return nbhd_list
    
    @staticmethod
    def get_bfs_perm(nbhd_list):
        num_nodes = len(nbhd_list)
        num_neighbors = torch.LongTensor([len(nbhd_list[i]) for i in range(num_nodes)])

        bfs_queue = [random.randint(0, num_nodes-1)]
        bfs_perm = []
        num_remains = [num_neighbors.clone()]
        bfs_next_list = {}
        visited = {bfs_queue[0]}   

        num_nbh_remain = num_neighbors.clone()
        
        while len(bfs_queue) > 0:
            current = bfs_queue.pop(0)
            for nbh in nbhd_list[current]:
                num_nbh_remain[nbh] -= 1
            bfs_perm.append(current)
            num_remains.append(num_nbh_remain.clone())
            next_candid = []
            for nxt in nbhd_list[current]:
                if nxt in visited: continue
                next_candid.append(nxt)
                visited.add(nxt)
                
            random.shuffle(next_candid)
            bfs_queue += next_candid
            bfs_next_list[current] = copy.copy(bfs_queue)

        return torch.LongTensor(bfs_perm), bfs_next_list, num_remains

    def __call__(self, data):  
        bfs_perm, _, _ = self.get_bfs_perm(self._get_nbhd_list(data[self.mask_target]))

        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data[self.mask_target].element.size(0)
        num_masked = int(num_atoms * ratio)
        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        if self.inverse:
            masked_idx = bfs_perm[:num_masked]
            context_idx = bfs_perm[num_masked:]
        else:
            masked_idx = bfs_perm[-num_masked:]
            context_idx = bfs_perm[:-num_masked]

        if hasattr(data[self.mask_target], 'ctx_flag'):
            context_mask = torch.zeros(num_atoms, dtype=torch.bool)
            context_mask[context_idx] = True
            context_mask[data[self.mask_target].ctx_flag] = True
            context_idx = torch.arange(0, num_atoms)[context_mask]

            masked_mask = ~context_mask
            masked_idx = torch.arange(0, num_atoms)[masked_mask]

            if len(masked_idx) == 0:
                random_index = torch.randint(0, (~data[self.mask_target].ctx_flag).sum(), (1,))
                masked_idx = torch.arange(0, num_atoms)[~data[self.mask_target].ctx_flag][random_index]
                context_mask[masked_idx] = False
                context_idx = torch.arange(0, num_atoms)[context_mask]

        data[self.mask_target].context_idx = context_idx  # for change bond index
        data[self.mask_target].masked_idx = masked_idx

        data[self.mask_target + '_masked'] = set_subgraph(data[self.mask_target], masked_idx)
        # set context subgraph
        data[self.mask_target + '_context'] = set_subgraph(data[self.mask_target], context_idx)

        return data

@register_transform('prefixed_mask')
class PrefixedMask(object):

    def __init__(self, mask_target='ligand', 
                 prefixed_name=None):
        super().__init__()
        self.mask_target = mask_target
        self.prefixed_name = prefixed_name

    def __call__(self, data):
        num_atoms = data[self.mask_target].element.size(0)

        idx = np.arange(num_atoms)

        idx = torch.LongTensor(idx)
        prefixed_mask = data[self.mask_target].get(self.prefixed_name, 
                                                   torch.ones(num_atoms, dtype=torch.bool))
        masked_idx = idx[prefixed_mask]
        context_idx = idx[~prefixed_mask]

        data[self.mask_target].context_idx = context_idx
        data[self.mask_target].masked_idx = masked_idx

        data[self.mask_target + '_masked'] = set_subgraph(data[self.mask_target], masked_idx)
        data[self.mask_target + '_context'] = set_subgraph(data[self.mask_target], context_idx)
        return data

@register_transform('random_mask')
class RandomMask(object):

    def __init__(self, mask_target='ligand', 
                 min_ratio=0.0, max_ratio=1.2, 
                 min_num_masked=1, min_num_unmasked=0):
        super().__init__()
        self.mask_target = mask_target
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked

    def __call__(self, data):
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data[self.mask_target].element.size(0)
        num_masked = int(num_atoms * ratio)

        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        idx = torch.LongTensor(idx)
        masked_idx = idx[:num_masked]
        context_idx = idx[num_masked:]

        if hasattr(data[self.mask_target], 'ctx_flag'):
            context_mask = torch.zeros(num_atoms, dtype=torch.bool)
            context_mask[context_idx] = True
            context_mask[data[self.mask_target].ctx_flag] = True
            context_idx = torch.arange(0, num_atoms)[context_mask]

            masked_mask = ~context_mask
            masked_idx = torch.arange(0, num_atoms)[masked_mask]

            if len(masked_idx) == 0:
                random_index = torch.randint(0, (~data[self.mask_target].ctx_flag).sum(), (1,))
                masked_idx = torch.arange(0, num_atoms)[~data[self.mask_target].ctx_flag][random_index]
                context_mask[masked_idx] = False
                context_idx = torch.arange(0, num_atoms)[context_mask]

        data[self.mask_target].context_idx = context_idx  # for change bond index
        data[self.mask_target].masked_idx = masked_idx

        # masked ligand atom element/feature/pos.
        data[self.mask_target + '_masked'] = set_subgraph(data[self.mask_target], masked_idx)
        data[self.mask_target + '_context'] = set_subgraph(data[self.mask_target], context_idx)

        return data



@register_transform('bfs_motif_mask')
class LigandBFSMotifMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        from repo.utils.molecule.vocab import vocab
    
        self.vocab = vocab
        self.vocab_size = vocab.size()

    @staticmethod
    def get_bfs_perm_motif(moltree, vocab):
        for i, node in enumerate(moltree.nodes):
            node.nid = i
            node.wid = vocab.get_index(node.smiles)
        # num_motifs = len(moltree.nodes)
        bfs_queue = [0]
        bfs_perm = []
        bfs_focal = []
        visited = {bfs_queue[0]}
        while len(bfs_queue) > 0:
            current = bfs_queue.pop(0)
            bfs_perm.append(current)
            next_candid = []
            for motif in moltree.nodes[current].neighbors:
                if motif.nid in visited: continue
                next_candid.append(motif.nid)
                visited.add(motif.nid)
                bfs_focal.append(current)

            random.shuffle(next_candid)
            bfs_queue += next_candid

        return bfs_perm, bfs_focal
    
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
        bfs_perm, bfs_focal = self.get_bfs_perm_motif(data['ligand_moltree'], self.vocab)
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_motifs = len(bfs_perm)
        num_masked = int(num_motifs * ratio)
        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_motifs - num_masked) < self.min_num_unmasked:
            num_masked = num_motifs - self.min_num_unmasked
        num_unmasked = num_motifs - num_masked

        context_motif_ids = bfs_perm[:-num_masked]
        context_idx = set()
        for i in context_motif_ids:
            context_idx = context_idx | set(data['ligand_moltree'].nodes[i].clique)
        context_idx = torch.LongTensor(list(context_idx))

        if num_masked == num_motifs:
            data['current_wid'] = torch.tensor([self.vocab_size])
            data['current_atoms'] = torch.tensor([data['protein_contact_idx']])
            data['next_wid'] = torch.tensor([data['ligand_moltree'].nodes[bfs_perm[-num_masked]].wid])
        else:
            data['current_wid'] = torch.tensor([data['ligand_moltree'].nodes[bfs_focal[-num_masked]].wid])
            data['next_wid'] = torch.tensor([data['ligand_moltree'].nodes[bfs_perm[-num_masked]].wid])  # For Prediction
            current_atoms = data['ligand_moltree'].nodes[bfs_focal[-num_masked]].clique
            data['current_atoms'] = torch.cat([torch.where(context_idx == i)[0] for i in current_atoms]) + len(data['protein_pos'])

        data['ligand_context_element'] = data['ligand_element'][context_idx]
        data['ligand_context_atom_type'] = data['ligand_atom_type'][context_idx]  # For Input
        data['ligand_context_pos'] = data['ligand_pos'][context_idx]
        data['ligand_context_lig_flag'] = data['ligand_lig_flag'][context_idx]
        data['ligand_center'] = torch.mean(data['ligand_pos'], dim=0)
        data['num_atoms'] = torch.tensor([len(context_idx) + len(data['protein_pos'])])
        # distance matrix prediction
        if len(data['ligand_context_pos']) > 0:
            sample_idx = random.sample(data['ligand_moltree'].nodes[bfs_perm[0]].clique, 2)
            data['dm_ligand_idx'] = torch.cat([torch.where(context_idx == i)[0] for i in sample_idx])
            data['dm_protein_idx'] = torch.sort(torch.norm(data['protein_pos'] - data['ligand_context_pos'][data['dm_ligand_idx'][0]], dim=-1)).indices[:4]
            data['true_dm'] = torch.norm(data['protein_pos'][data['dm_protein_idx']].unsqueeze(1) - data['ligand_context_pos'][data['dm_ligand_idx']].unsqueeze(0), dim=-1).reshape(-1)
        else:
            data['true_dm'] = torch.tensor([])

        data['protein_alpha_carbon_index'] = torch.arange(len(data['protein_alpha_carbon_indicator']))[data['protein_alpha_carbon_indicator']]

        # assemble prediction
        if len(context_motif_ids) > 0:
            cand_labels, cand_mols = enumerate_assemble(data['ligand_moltree'].mol, context_idx.tolist(),
                                                        data['ligand_moltree'].nodes[bfs_focal[-num_masked]],
                                                        data['ligand_moltree'].nodes[bfs_perm[-num_masked]])
            data['cand_labels'] = cand_labels
            data['cand_mols'] = [mol_to_graph_data_obj_simple(mol) for mol in cand_mols]
        else:
            data['cand_labels'], data['cand_mols'] = torch.tensor([]), []

        data['ligand_context_bond_index'], data['ligand_context_bond_type'] = subgraph(
            context_idx,
            data['ligand_bond_index'],
            edge_attr=data['ligand_bond_type'],
            relabel_nodes=True,
        )
        data['ligand_context_num_neighbors'] = self.count_neighbors(
            data['ligand_context_bond_index'],
            symmetry=True,
            num_nodes=context_idx.size(0),
        )
        data['ligand_frontier'] = data['ligand_context_num_neighbors'] < data['ligand_num_neighbors'][context_idx]

        # find a rotatable bond as the current motif
        rotatable_ids = []
        for i, id in enumerate(bfs_focal):
            if data['ligand_moltree'].nodes[id].rotatable:
                rotatable_ids.append(i)
        if len(rotatable_ids) == 0:
            # assign empty tensor
            data['ligand_torsion_xy_index'] = torch.tensor([])
            data['dihedral_mask'] = torch.tensor([]).bool()
            data['ligand_element_torsion'] = torch.tensor([])
            data['ligand_pos_torsion'] = torch.tensor([])
            data['ligand_atom_type_torsion'] = torch.tensor([])
            data['ligand_lig_flag_torsion'] = torch.tensor([])
            data['true_sin'], data['true_cos'], data['true_three_hop'] = torch.tensor([]), torch.tensor([]), torch.tensor([])
            data['xn_pos'], data['yn_pos'], data['y_pos'] = torch.tensor([]), torch.tensor([]), torch.tensor([])
        else:
            num_unmasked = random.sample(rotatable_ids, 1)[0]
            current_idx = torch.LongTensor(data['ligand_moltree'].nodes[bfs_focal[num_unmasked]].clique)
            next_idx = torch.LongTensor(data['ligand_moltree'].nodes[bfs_perm[num_unmasked + 1]].clique)
            current_idx_set = set(data['ligand_moltree'].nodes[bfs_focal[num_unmasked]].clique)
            next_idx_set = set(data['ligand_moltree'].nodes[bfs_perm[num_unmasked + 1]].clique)
            all_idx = set()
            for i in bfs_perm[:num_unmasked + 2]:
                all_idx = all_idx | set(data['ligand_moltree'].nodes[i].clique)
            all_idx = list(all_idx)
            x_id = current_idx_set.intersection(next_idx_set).pop()
            y_id = (current_idx_set - {x_id}).pop()
            data['ligand_torsion_xy_index'] = torch.cat([torch.where(torch.LongTensor(all_idx) == i)[0] for i in [x_id, y_id]])

            x_pos, y_pos = deepcopy(data['ligand_pos'][x_id]), deepcopy(data['ligand_pos'][y_id])
            # remove x, y, and non-generated elements
            xn, yn = deepcopy(data['ligand_neighbors'][x_id]), deepcopy(data['ligand_neighbors'][y_id])
            xn.remove(y_id)
            yn.remove(x_id)
            xn, yn = xn[:3], yn[:3]
            # debug
            xn, yn = list_filter(xn, all_idx), list_filter(yn, all_idx)
            xn_pos, yn_pos = torch.zeros(3, 3), torch.zeros(3, 3)
            xn_pos[:len(xn)], yn_pos[:len(yn)] = deepcopy(data['ligand_pos'][xn]), deepcopy(data['ligand_pos'][yn])
            xn_idx, yn_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
            xn_idx = xn_idx.squeeze(-1)
            yn_idx = yn_idx.squeeze(-1)
            dihedral_x, dihedral_y = torch.zeros(3), torch.zeros(3)
            dihedral_x[:len(xn)] = 1
            dihedral_y[:len(yn)] = 1
            data['dihedral_mask'] = torch.matmul(dihedral_x.view(3, 1), dihedral_y.view(1, 3)).view(-1).bool()
            data['true_sin'], data['true_cos'] = batch_dihedrals(xn_pos[xn_idx], x_pos.repeat(9, 1), y_pos.repeat(9, 1),
                                                           yn_pos[yn_idx])
            data['true_three_hop'] = torch.linalg.norm(xn_pos[xn_idx] - yn_pos[yn_idx], dim=-1)[data['dihedral_mask']]

            # random rotate to simulate the inference situation
            dir = data['ligand_pos'][current_idx[0]] - data['ligand_pos'][current_idx[1]]
            ref = deepcopy(data['ligand_pos'][current_idx[0]])
            next_motif_pos = deepcopy(data['ligand_pos'][next_idx])
            data['ligand_pos'][next_idx] = rand_rotate(dir, ref, next_motif_pos)

            data['ligand_element_torsion'] = data['ligand_element'][all_idx]
            data['ligand_pos_torsion'] = data['ligand_pos'][all_idx]
            data['ligand_atom_type_torsion'] = data['ligand_atom_type'][all_idx]
            data['ligand_lig_flag_torsion'] = data['ligand_lig_flag'][all_idx]

            x_pos = deepcopy(data['ligand_pos'][x_id])
            data['y_pos'] = data['ligand_pos'][y_id] - x_pos
            data['xn_pos'], data['yn_pos'] = torch.zeros(3, 3), torch.zeros(3, 3)
            data['xn_pos'][:len(xn)], data['yn_pos'][:len(yn)] = data['ligand_pos'][xn] - x_pos, data['ligand_pos'][yn] - x_pos

        return data
