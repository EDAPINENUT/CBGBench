import torch.nn as nn
import torch
from repo.utils.molecule.vocab import vocab
from repo.modules.context_emb import get_context_embedder
from .._base import register_model
from repo.modules.e3nn import get_e3_gnn
from repo.modules.gnn.gnn import GNNPred
from repo.utils.protein.constants import atomic_numbers
from torch.nn.modules.loss import _WeightedLoss
from repo.modules.common import MLP
from torch_scatter import scatter_add, scatter_mean
from repo.modules.common import compose_context, unique
import torch.nn.functional as F 
from repo.modules.common import GaussianSmearing
from repo.utils.dihedutils import rotation_matrix_v2, batch_dihedrals, von_Mises_loss
from repo.utils.chemutils import *

from rdkit import Chem
from rdkit.Chem import AllChem
from repo.datasets.transforms._base import get_index

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss



@register_model('flag')
class FLAG(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.num_classes = cfg.num_atomtype

        cfg.embedder.num_atomtype = cfg.num_atomtype
        self.context_embedder = get_context_embedder(cfg.embedder)
        cfg.node_feat_dim = cfg.encoder.node_feat_dim
        cfg.dist_feat_dim = cfg.encoder.vec_feat_dim
        self.voc_embedding = nn.Embedding(vocab.size() + 1, cfg.node_feat_dim)
        self.W = nn.Linear(2 * cfg.node_feat_dim, cfg.node_feat_dim)
        self.W_o = nn.Linear(cfg.node_feat_dim, self.vocab.size())
        self.context_encoder = get_e3_gnn(cfg.encoder, num_classes = self.num_classes)

        self.comb_head = GNNPred(num_layer=3, emb_dim=cfg.node_feat_dim, num_tasks=1, JK='last',
                                drop_ratio=0.5, graph_pooling='mean', gnn_type='gin')
        
        self.alpha_mlp = MLP(in_dim=cfg.node_feat_dim * 3, hidden_dim=cfg.node_feat_dim * 3, out_dim=1)
        self.focal_mlp_ligand = MLP(in_dim=cfg.node_feat_dim, hidden_dim=cfg.node_feat_dim,out_dim=1)
        self.focal_mlp_protein = MLP(in_dim=cfg.node_feat_dim, hidden_dim=cfg.node_feat_dim, out_dim=1)
        self.dist_mlp = MLP(in_dim=2*cfg.node_feat_dim, out_dim=1, hidden_dim=cfg.node_feat_dim,)
        self.refine_protein = MLP(in_dim=cfg.node_feat_dim * 2 + cfg.encoder.vec_feat_dim,  
                                  hidden_dim=cfg.node_feat_dim*2, out_dim=1)
        self.refine_ligand = MLP(in_dim=cfg.node_feat_dim * 2 + cfg.encoder.vec_feat_dim, 
                                 hidden_dim=cfg.node_feat_dim*2, out_dim=1)
        
        self.distance_expand = GaussianSmearing(0., 10.0, num_gaussians=cfg.dist_feat_dim, fixed_offset=False)

        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction='mean', smoothing=0.1)
        self.pred_loss = nn.CrossEntropyLoss()
        self.comb_loss = nn.BCEWithLogitsLoss()
        self.three_hop_loss = torch.nn.MSELoss()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.dist_loss = torch.nn.MSELoss(reduction='mean')
    
    def remap_subgraph_index_into_composed(self, subgraph_index, idx_composed_ctx_new):
        if len (subgraph_index) == 0:
            return subgraph_index
        subgraph_index_new = idx_composed_ctx_new[..., subgraph_index]
        return subgraph_index_new
    
    def pred_motif(self, h_ctx_focal, current_wid, n_samples=1):
        node_hiddens = h_ctx_focal.sum(0, keepdim=True)
        motif_hiddens = self.voc_embedding(current_wid)
        pred_vecs = torch.cat([node_hiddens, motif_hiddens], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_scores = F.softmax(pred_scores, dim=-1)
        _, preds = torch.max(pred_scores, dim=1)
        # random select n_samples in topk
        k = 5 * n_samples
        select_pool = torch.topk(pred_scores, k, dim=1)[1]
        index = torch.randint(k, (select_pool.shape[0], n_samples))
        preds = torch.cat([select_pool[i][index[i]] for i in range(len(index))])

        idx_parent = torch.repeat_interleave(torch.arange(pred_scores.shape[0]), n_samples, dim=0).to(pred_scores.device)
        prob = pred_scores[idx_parent, preds]
        return preds, prob

    def get_atom_type(self, mol):
        ptable = Chem.GetPeriodicTable()
        Chem.SanitizeMol(mol)
        ligand_element = torch.tensor([ptable.GetAtomicNumber(atom.GetSymbol()) for atom in mol.GetAtoms()])
        x = [get_index(e, h, a, 'basic') for e, h, a in zip(ligand_element, 
                                                              torch.zeros_like(ligand_element), 
                                                              torch.zeros_like(ligand_element))]
        return torch.tensor(x)
    
    def find_reference(self, protein_pos, focal_id):
    # Select three reference protein atoms
        d = torch.norm(protein_pos - protein_pos[focal_id], dim=1)
        reference_idx = torch.topk(d, k=4, largest=False)[1]
        reference_pos = protein_pos[reference_idx]
        return reference_pos, reference_idx
    
    def get_new_ctx(self, batch):
        x_lig = batch['ligand_pos']
        x_rec = batch['protein_pos']
        v_lig = batch['ligand_atom_type']
        v_rec = batch['protein_atom_feature']
        aa_rec = batch['protein_aa_type']
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']

        x_lig_emb, x_rec_emb, h_lig_emb, h_rec_emb = self.context_embedder(
            x_lig, x_rec, v_lig, v_rec, aa_rec, 
            batch_idx_lig, batch_idx_rec, lig_flag, rec_flag
            )
        
        (context_composed, batch_idx_composed, 
        (idx_rec_ctx_new, idx_lig_ctx_new)) = compose_context({'x': x_lig, 'h': h_lig_emb, 'vec': x_lig_emb},
                                                              {'x': x_rec, 'h': h_rec_emb, 'vec': x_rec_emb},
                                                              batch_idx_lig, batch_idx_rec)
        
        h_composed, vec_composed = self.context_encoder(batch_idx = batch_idx_composed, **context_composed)
        return h_composed, idx_lig_ctx_new, idx_rec_ctx_new
    
    def rdinit_frag_conformer(self, mol):
        Chem.SanitizeMol(mol)
        Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
            print('UFF error')
        Chem.RemoveHs(mol)
        return mol
    
    def sample_init(self, batch):
        focal_pred, ligand_idx, protein_idx, h_ctx = self.pred_focal(batch)
        focal_protein = focal_pred[protein_idx]
        h_ctx_protein = h_ctx[protein_idx]
        focus_score = torch.sigmoid(focal_protein)

        device = h_ctx.device

        focal_id = torch.argmax(focus_score.reshape(-1).float()).item()
        h_ctx_focal = h_ctx_protein[focal_id].unsqueeze(0)
        current_wid = torch.tensor([vocab.size()]).to(device)
        next_motif_wid, _ = self.pred_motif(h_ctx_focal, current_wid)
        mol_start = [Chem.MolFromSmiles(vocab.get_smiles(id)) for id in next_motif_wid][0]
        mol_start = self.rdinit_frag_conformer(mol_start)

        batch['ligand_pos'] = torch.tensor(mol_start.GetConformer().GetPositions(), device=device).float()
        batch['ligand_atom_type'] = self.get_atom_type(mol_start).to(device).long()
        batch['ligand_lig_flag'] = torch.ones_like(batch['ligand_atom_type'])
        batch['ligand_element_batch'] = torch.zeros_like(batch['ligand_atom_type'])

        # set the initial positions with distance matrix
        reference_pos, reference_idx = self.find_reference(batch['protein_pos'], focal_id)
        ligand_pos = batch['ligand_pos'].clone()

        p_idx, l_idx = torch.cartesian_prod(torch.arange(4), torch.arange(len(ligand_pos))).chunk(2, dim=-1)
        p_idx = p_idx.squeeze(-1).to(device)
        l_idx = l_idx.squeeze(-1).to(device)
        h_ctx, idx_lig_cps, idx_rec_cps = self.get_new_ctx(batch)

        d_m = self.dist_mlp(torch.cat([h_ctx[idx_rec_cps][reference_idx[p_idx]], 
                                       h_ctx[idx_lig_cps][l_idx]], dim=-1)).reshape(4,len(ligand_pos))

        d_m = d_m ** 2
        p_d, l_d = self_square_dist(reference_pos), self_square_dist(ligand_pos)
        D = torch.cat([torch.cat([p_d, d_m], dim=1), torch.cat([d_m.permute(1, 0), l_d], dim=1)])
        coordinate = eig_coord_from_dist(D)
        new_pos, _, _ = kabsch_torch(coordinate[:len(reference_pos)], reference_pos,
                                        coordinate[len(reference_pos):])
        center = batch['protein_pos'].mean(0) # FIX THE LABEL LEAKY IN THE ORIGINAL CODE, where `center` is calculated by reference ligand.
        new_pos += (center - torch.mean(new_pos, dim=0)) * .8
        batch['ligand_pos'] = new_pos.float()
        atom_to_motif = {}
        motif_to_atoms = {}
        motif_wid = {}
        for k in range(mol_start.GetNumAtoms()):
            atom_to_motif[k] = 0
        motif_to_atoms[0] = list(np.arange(mol_start.GetNumAtoms()))
        motif_wid[0] = next_motif_wid.item()

        return batch, mol_start, atom_to_motif, motif_to_atoms, motif_wid

    def set_atom_num(self, mol, atoms):
        for atom in mol.GetAtoms():
            if atom.GetIdx() in atoms:
                atom.SetAtomMapNum(1)
            else:
                atom.SetAtomMapNum(0)
        return mol


    def forward_attach(self, mol, next_motif_smiles, device):
        cand_mols, cand_batch, new_atoms, one_atom_attach, intersection, attach_fail = assemble([mol], next_motif_smiles)
        graph_data = Batch.from_data_list([mol_to_graph_data_obj_simple(mol) for mol in cand_mols]).to(device)
        comb_pred = self.comb_head(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch).reshape(-1)
        slice_idx = torch.cat([torch.tensor([0]), torch.cumsum(cand_batch.bincount(), dim=0)], dim=0)
        select = [(torch.argmax(comb_pred[slice_idx[i]:slice_idx[i + 1]]) + slice_idx[i]).item() for i in
                  range(len(slice_idx) - 1)]
        '''
        select = []
        for k in range(len(slice_idx) - 1):
            id = torch.multinomial(torch.exp(comb_pred[slice_idx[k]:slice_idx[k + 1]]).reshape(-1).float(), 1)
            select.append((id+slice_idx[k]).item())'''

        select_mols = [cand_mols[i] for i in select]
        new_atoms = [new_atoms[i] for i in select]
        one_atom_attach = [one_atom_attach[i] for i in select]
        intersection = [intersection[i] for i in select]
        return select_mols[0], new_atoms[0], one_atom_attach[0], intersection[0], attach_fail[0]

    def sample(self, data, max_steps=12):
        finished = False               
        device = data['ligand_pos'].device

        data['ligand_pos'] = data['ligand_pos'].squeeze(0) 
        data['protein_pos'] = data['protein_pos'].squeeze(0) 
        data['ligand_atom_type'] = data['ligand_atom_type'].squeeze(0) 
        data['protein_atom_feature'] = data['protein_atom_feature'].squeeze(0) 
        data['protein_aa_type'] = data['protein_aa_type'].squeeze(0) 
        data['ligand_element_batch'] = torch.empty(0).to(device)
        data['protein_element_batch'] = torch.zeros_like(data['protein_lig_flag'], dtype=torch.long).squeeze(0) 
        data['ligand_lig_flag'] = data['ligand_lig_flag'].squeeze(0) 
        data['protein_lig_flag'] = data['protein_lig_flag'].squeeze(0) 
    
        data, mol, atom_to_motif, motif_to_atoms, motif_wid = self.sample_init(data)

        traj_result_inv = {0: (data['ligand_pos'].clone().cpu(), 
                               data['ligand_atom_type'].clone().cpu(),
                               torch.zeros_like(data['ligand_atom_type']).cpu())}

        for i in range(1, max_steps):
            try:
                ligand_pos = data['ligand_pos'].clone()

                focal_pred, ligand_idx, protein_idx, h_ctx = self.pred_focal(data)
                # structure refinement
                focal_ligand = focal_pred[ligand_idx]
                h_ctx_ligand = h_ctx[ligand_idx]
                focus_score = torch.sigmoid(focal_ligand)
                can_focus = focus_score > 0.
                focus = focus_score

                if torch.sum(can_focus) > 0 and ~finished:
                    sample_focal_atom = torch.multinomial(focus.reshape(-1).float(), 1)
                    focal_motif = atom_to_motif[sample_focal_atom.item()]
                    motif_id = focal_motif
                else:
                    finished = True

                current_atoms = (np.array(motif_to_atoms[motif_id])).tolist()
                current_atoms_batch = [0] * len(motif_to_atoms[motif_id])
                mol = self.set_atom_num(mol, motif_to_atoms[motif_id])

                # second step: next motif prediction
                current_wid = motif_wid[motif_id]
                next_motif_wid, motif_prob = self.pred_motif(h_ctx_ligand[torch.tensor(current_atoms)],
                                                            torch.tensor([current_wid]).to(device))

                # assemble
                next_motif_smiles = [vocab.get_smiles(id) for id in next_motif_wid]
                new_mol, new_atom, one_atom_attach, intersection, attach_fail = self.forward_attach(mol, next_motif_smiles, device)

                if ~finished and ~attach_fail:
                    # num_new_atoms
                    mol = new_mol

                rotatable = torch.logical_and(torch.tensor(current_atoms_batch).bincount() == 2, one_atom_attach)
                rotatable = torch.logical_and(rotatable, ~attach_fail)
                rotatable = torch.logical_and(rotatable, ~torch.tensor(finished)).to(device)
                # update motif2atoms and atom2motif
                if attach_fail or finished:
                    continue
                motif_to_atoms[i] = new_atom
                motif_wid[i] = next_motif_wid
                for k in new_atom:
                    atom_to_motif[k] = i

                # generate initial positions
                if attach_fail or finished:
                    continue
                anchor = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1]
                # positions = mol.GetConformer().GetPositions()
                anchor_pos = deepcopy(ligand_pos[anchor]).to(device)
                mol = self.rdinit_frag_conformer(mol)

                anchor_pos_new = mol.GetConformer(0).GetPositions()[anchor]
                new_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]
                '''
                R, T = kabsch(np.matrix(anchor_pos), np.matrix(anchor_pos_new))
                new_pos = R * np.matrix(mol.GetConformer().GetPositions()[new_idx]).T + np.tile(T, (1, len(new_idx)))
                new_pos = np.array(new_pos.T)'''
                new_pos = mol.GetConformer().GetPositions()[new_idx]
                new_pos, _, _ = kabsch_torch(torch.tensor(anchor_pos_new, device=device), anchor_pos, torch.tensor(new_pos, device=device))

                conf = mol.GetConformer()
                # update curated parameters
                ligand_pos = torch.cat([ligand_pos, new_pos])
                ligand_atom_type = self.get_atom_type(mol).to(device)
                
                for node in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(node, np.array(ligand_pos[node].cpu()))
                assert mol.GetNumAtoms() == len(ligand_pos)

                data['ligand_pos'] = ligand_pos.float()
                data['ligand_atom_type'] = ligand_atom_type.long()
                data['ligand_element_batch'] = torch.zeros_like(ligand_atom_type)
                data['ligand_lig_flag'] = torch.ones_like(ligand_atom_type)

                # predict alpha and rotate (only change the position)
                if torch.sum(rotatable) > 0 and i >= 2:

                    xy_index = (np.array(motif_to_atoms[motif_id])).tolist()

                    alpha = self.forward_alpha(data, torch.tensor(xy_index, device=device))

                    x_index = intersection 
                    y_index = list(set(xy_index) - set(x_index))

                    new_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]
                    positions = deepcopy(ligand_pos)

                    xn_pos = positions[new_idx].float()
                    dir = (positions[x_index] - positions[y_index]).reshape(-1)
                    ref = positions[x_index].reshape(-1)
                    xn_pos = rand_rotate(dir.to(device), ref.to(device), xn_pos.to(device), alpha, device=device)
                    if xn_pos.shape[0] > 0:
                        ligand_pos[-len(xn_pos):] = xn_pos
                    conf = mol.GetConformer()
                    for node in range(mol.GetNumAtoms()):
                        conf.SetAtomPosition(node, np.array(ligand_pos[node].cpu()))
                    assert mol.GetNumAtoms() == len(ligand_pos)

                    data['ligand_pos'] = ligand_pos.float()
                    data['ligand_atom_type'] = ligand_atom_type.long()
                    data['ligand_element_batch'] = torch.zeros_like(ligand_atom_type)
                    data['ligand_lig_flag'] = torch.ones_like(ligand_atom_type)


                traj_result_inv = {i: (data['ligand_pos'].clone().cpu(), 
                                        data['ligand_atom_type'].clone().cpu(),
                                        torch.zeros_like(data['ligand_atom_type']).cpu())}
                
            except:
                break

        key_list = list(traj_result_inv.keys())
        traj_result = {k: traj_result_inv[key_list[len(traj_result_inv) - k - 1]] for k in range(len(traj_result_inv))}
                    
        return traj_result

    
    def forward_alpha(self, batch, xy_index, random_alpha=False):
        batch_ligand = batch['ligand_element_batch']
        # encode again
        h_composed, idx_lig_ctx_new, idx_rec_ctx_new = self.get_new_ctx(batch)
        h_ctx_ligand = h_composed[idx_lig_ctx_new]
        hx, hy = h_ctx_ligand[xy_index[0]], h_ctx_ligand[xy_index[1]]

        h_mol = h_ctx_ligand.sum(dim=0)
        if random_alpha:
            rand_dist = torch.distributions.normal.Normal(loc=0, scale=1)
            rand_alpha = rand_dist.sample(hx.shape).to(hx.device)
            alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol, rand_alpha], dim=-1))
        else:
            alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol], dim=-1))
        return alpha



    def pred_focal(self, batch):
        x_lig = batch['ligand_pos']
        x_rec = batch['protein_pos']
        v_lig = batch['ligand_atom_type']
        v_rec = batch['protein_atom_feature']
        aa_rec = batch['protein_aa_type']
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']

        x_lig_emb, x_rec_emb, h_lig_emb, h_rec_emb = self.context_embedder(
            x_lig, x_rec, v_lig, v_rec, aa_rec, 
            batch_idx_lig, batch_idx_rec, lig_flag, rec_flag
            )
        
        (context_composed, batch_idx_composed, 
        (idx_rec_ctx_new, idx_lig_ctx_new)) = compose_context({'x': x_lig, 'h': h_lig_emb, 'vec': x_lig_emb},
                                                              {'x': x_rec, 'h': h_rec_emb, 'vec': x_rec_emb},
                                                              batch_idx_lig, batch_idx_rec)
        
        h_composed, vec_composed = self.context_encoder(batch_idx = batch_idx_composed, **context_composed)
        focal_pred = torch.cat([self.focal_mlp_protein(h_composed[idx_rec_ctx_new]), 
                                self.focal_mlp_ligand(h_composed[~idx_lig_ctx_new])], dim=0)
        protein_mask = torch.zeros_like(focal_pred[:,0], dtype=torch.bool)
        protein_mask[idx_rec_ctx_new] = True

        return focal_pred, idx_lig_ctx_new, idx_rec_ctx_new, h_composed


    def forward(self, batch):
        
        x_lig = batch['ligand_context_pos'] 
        x_rec = batch['protein_pos']
        v_lig = batch['ligand_context_atom_type']
        v_rec = batch['protein_atom_feature']
        aa_rec = batch['protein_aa_type']
        lig_flag = batch['ligand_context_lig_flag']
        rec_flag = batch['protein_lig_flag']
        lig_ts_flag = batch['ligand_lig_flag_torsion']
        x_lig_ts = batch['ligand_pos_torsion']
        v_lig_ts = batch['ligand_atom_type_torsion'].long()
        batch_idx_lig = batch['ligand_context_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        batch_idx_lig_ts = batch['ligand_element_torsion_batch']
        current_atom_batch = batch['current_atoms_batch']
        current_wid = batch['current_wid']
        next_wid = batch['next_wid']

        cand_labels = batch['cand_labels']
        ligand_frontier = batch['ligand_frontier']
        protein_contact = batch['protein_contact']

        true_dm = batch['true_dm']

        x_lig_emb, x_rec_emb, h_lig_emb, h_rec_emb = self.context_embedder(
            x_lig, x_rec, v_lig, v_rec, aa_rec, 
            batch_idx_lig, batch_idx_rec, lig_flag, rec_flag
            )
        
        (context_composed, batch_idx_composed, 
        (idx_rec_ctx_new, idx_lig_ctx_new)) = compose_context({'x': x_lig, 'h': h_lig_emb, 'vec': x_lig_emb},
                                                              {'x': x_rec, 'h': h_rec_emb, 'vec': x_rec_emb},
                                                              batch_idx_lig, batch_idx_rec)
        

        h_composed, vec_composed = self.context_encoder(batch_idx = batch_idx_composed, **context_composed)

        h_ctx_ligand = h_composed[idx_lig_ctx_new]
        h_ctx_protein = h_composed[idx_rec_ctx_new]
        h_ctx_focal = h_composed[batch['current_atoms']]

        # Encode for torsion prediction
        if len(batch['y_pos']) > 0:

            x_lig_ts_emb, x_rec_emb, h_lig_ts_emb, h_rec_emb = self.context_embedder(
                x_lig_ts, x_rec, v_lig_ts, v_rec, aa_rec, 
                batch_idx_lig_ts, batch_idx_rec, lig_ts_flag, rec_flag
                )
        
            (context_ts_composed, batch_idx_ts_composed, 
            (idx_rec_ts_ctx_new, idx_lig_ts_ctx_new)) = compose_context({'x': x_lig_ts, 'h': h_lig_ts_emb, 'vec': x_lig_ts_emb},
                                                                        {'x': x_rec, 'h': h_rec_emb, 'vec': x_rec_emb},
                                                                        batch_idx_lig_ts, batch_idx_rec)
            

            h_ts_composed, vec_composed = self.context_encoder(batch_idx = batch_idx_ts_composed, **context_ts_composed)
            h_ctx_ligand_torsion = h_ts_composed[idx_lig_ts_ctx_new]

        # next motif prediction

        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=current_atom_batch)
        motif_hiddens = self.voc_embedding(current_wid)
        pred_vecs = torch.cat([node_hiddens, motif_hiddens], dim=1)
        pred_vecs = nn.ReLU()(self.W(pred_vecs))
        pred_scores = self.W_o(pred_vecs)
        pred_loss = self.pred_loss(pred_scores, next_wid)

        # attachment prediction
        device = x_lig.device
        if len(cand_labels) > 0:
            cand_mols = batch['cand_mols'] 

            comb_pred = self.comb_head(cand_mols.x.to(device), 
                                       cand_mols.edge_index.to(device), 
                                       cand_mols.edge_attr.to(device), 
                                       cand_mols.batch.to(device))
            comb_loss = self.comb_loss(comb_pred, cand_labels.view(comb_pred.shape).float())
        else:
            comb_loss = 0

        # focal prediction
        focal_ligand_pred, focal_protein_pred = self.focal_mlp_ligand(h_ctx_ligand), self.focal_mlp_protein(h_ctx_protein)
        focal_loss = (self.focal_loss(focal_ligand_pred.reshape(-1), ligand_frontier.float()) 
                     +self.focal_loss(focal_protein_pred.reshape(-1), protein_contact.float()))

        # distance matrix prediction
        if len(true_dm) > 0:
            dm_protein_idx = batch['dm_protein_idx']
            dm_ligand_idx = batch['dm_ligand_idx']
            input = torch.cat([h_rec_emb[dm_protein_idx], h_lig_emb[dm_ligand_idx]], dim=-1)
            pred_dist = self.dist_mlp(input)
            dm_target = true_dm.unsqueeze(-1)
            dm_loss = self.dist_loss(pred_dist, dm_target)
        else:
            dm_loss = 0

        # structure refinement loss
        if len(true_dm) > 0:
            sr_ligand_idx = batch['sr_ligand_idx']
            sr_ligand_idx_0 = batch['sr_ligand_idx0']
            sr_ligand_idx_1 = batch['sr_ligand_idx1']
            sr_protein_idx = batch['sr_protein_idx']
            true_distance_alpha = torch.norm(x_lig[sr_ligand_idx] - x_rec[sr_protein_idx], dim=1)
            true_distance_intra = torch.norm(x_lig[sr_ligand_idx_0] - x_lig[sr_ligand_idx_1], dim=1)
            input_distance_alpha = x_lig[sr_ligand_idx] - x_rec[sr_protein_idx]
            input_distance_intra = x_lig[sr_ligand_idx_0] - x_lig[sr_ligand_idx_1]
            distance_emb1 = self.distance_expand(torch.norm(input_distance_alpha, dim=1, keepdim=True))
            distance_emb2 = self.distance_expand(torch.norm(input_distance_intra, dim=1, keepdim=True))
            input1 = torch.cat([h_ctx_ligand[sr_ligand_idx], h_ctx_protein[sr_protein_idx], distance_emb1], dim=-1)[true_distance_alpha<=10.0]
            input2 = torch.cat([h_ctx_ligand[sr_ligand_idx_0], h_ctx_ligand[sr_ligand_idx_1], distance_emb2], dim=-1)[true_distance_intra<=10.0]
            #distance cut_off
            norm_dir1 = F.normalize(input_distance_alpha, p=2, dim=1)[true_distance_alpha<=10.0]
            norm_dir2 = F.normalize(input_distance_intra, p=2, dim=1)[true_distance_intra<=10.0]
            force1 = scatter_mean(self.refine_protein(input1)*norm_dir1, dim=0, index=sr_ligand_idx[true_distance_alpha<=10.0], dim_size=x_lig.size(0))
            force2 = scatter_mean(self.refine_ligand(input2)*norm_dir2, dim=0, index=sr_ligand_idx_0[true_distance_intra<=10.0], dim_size=x_lig.size(0))
            new_ligand_pos = x_lig.clone()
            new_ligand_pos += force1
            new_ligand_pos += force2
            refine_dist1 = torch.norm(new_ligand_pos[sr_ligand_idx] - x_rec[sr_protein_idx], dim=1)
            refine_dist2 = torch.norm(new_ligand_pos[sr_ligand_idx_0] - new_ligand_pos[sr_ligand_idx_1], dim=1)
            sr_loss = (self.dist_loss(refine_dist1, true_distance_alpha) + self.dist_loss(refine_dist2, true_distance_intra))

        else:
            sr_loss = 0

        # torsion prediction
        if len(batch['y_pos']) > 0:
            Hx = rotation_matrix_v2(batch['y_pos'])
            xn_pos = torch.matmul(Hx, batch['xn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            yn_pos = torch.matmul(Hx, batch['yn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            y_pos = torch.matmul(Hx, batch['y_pos'].unsqueeze(1).permute(0, 2, 1)).squeeze(-1)

            hx, hy = h_ctx_ligand_torsion[batch['ligand_torsion_xy_index'][:, 0]], h_ctx_ligand_torsion[batch['ligand_torsion_xy_index'][:, 1]]
            h_mol = scatter_add(h_ctx_ligand_torsion, dim=0, index=batch['ligand_element_torsion_batch'])

            alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol], dim=-1))
            # rotate xn
            R_alpha = self.build_alpha_rotation(torch.sin(alpha).squeeze(-1), torch.cos(alpha).squeeze(-1))
            xn_pos = torch.matmul(R_alpha, xn_pos.permute(0, 2, 1)).permute(0, 2, 1)

            p_idx, q_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            pred_sin, pred_cos = batch_dihedrals(xn_pos[:, p_idx],
                                                 torch.zeros_like(y_pos).unsqueeze(1).repeat(1, 9, 1),
                                                 y_pos.unsqueeze(1).repeat(1, 9, 1),
                                                 yn_pos[:, q_idx])
            dihedral_loss = torch.mean(von_Mises_loss(batch['true_cos'], pred_cos.reshape(-1), batch['true_sin'], pred_cos.reshape(-1))[batch['dihedral_mask']])
            torsion_loss = -dihedral_loss
        else:
            torsion_loss = 0

        # dm: distance matrix
        loss_dict = {'type': pred_loss, 'comb': comb_loss, 'focal': focal_loss, 'dm':dm_loss, 'torsion': torsion_loss, 'sr': sr_loss}
        return loss_dict, {}



    def build_alpha_rotation(self, alpha, alpha_cos=None):
        """
        Builds the alpha rotation matrix

        :param alpha: predicted values of torsion parameter alpha (n_dihedral_pairs)
        :return: alpha rotation matrix (n_dihedral_pairs, 3, 3)
        """
        H_alpha = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(alpha.shape[0], 1, 1).to(alpha)

        if torch.is_tensor(alpha_cos):
            H_alpha[:, 1, 1] = alpha_cos
            H_alpha[:, 1, 2] = -alpha
            H_alpha[:, 2, 1] = alpha
            H_alpha[:, 2, 2] = alpha_cos
        else:
            H_alpha[:, 1, 1] = torch.cos(alpha)
            H_alpha[:, 1, 2] = -torch.sin(alpha)
            H_alpha[:, 2, 1] = torch.sin(alpha)
            H_alpha[:, 2, 2] = torch.cos(alpha)

        return H_alpha