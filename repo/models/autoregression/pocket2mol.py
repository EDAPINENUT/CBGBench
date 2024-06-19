from torch import nn 
from .._base import register_model
from repo.modules.context_emb import get_context_embedder
import torch.nn.functional as F
from repo.modules.common import compose_context, unique
from repo.modules.e3nn import get_e3_gnn
import torch
from repo.modules.gvp.gvn import GVPerceptronVN, GVLinear
from torch.nn.modules.loss import _WeightedLoss
from repo.modules.gvp.predict_heads import AtomEdgePredictor, PositionPredictor
from torch_scatter import scatter_add
import numpy as np
from torch_scatter import scatter_max
from torch_geometric.nn import radius_graph, knn

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

@register_model('pocket2mol')
class Pocket2Mol(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = config.encoder
        self.num_classes = config.num_atomtype

        config.embedder.num_atomtype = self.num_classes
        config.encoder.edge_feat_dim = config.num_bondtype
        self.context_embedder = get_context_embedder(config.embedder)
        self.context_encoder = get_e3_gnn(config.encoder)

        self.focal_head = nn.Sequential(
            GVPerceptronVN(config.encoder.node_feat_dim, 
                           config.encoder.vec_feat_dim, 
                           config.encoder.node_feat_dim // 2, 
                           config.encoder.vec_feat_dim // 2, ),
            GVLinear(config.encoder.node_feat_dim // 2, 
                     config.encoder.vec_feat_dim // 2, 1, 1)
        )

        self.pos_pred = PositionPredictor(config.encoder.node_feat_dim, 
                                          config.encoder.vec_feat_dim, 
                                          [config.encoder.node_feat_dim // 2] * 2,
                                          config.encoder.get('num_components', 3))

        self.atom_edge_pred = AtomEdgePredictor(config.encoder,
                                                num_classes=self.num_classes, 
                                                num_edge_classes=config.num_bondtype+1)
        
        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction='mean', smoothing=0.1)
        self.bceloss_with_logits = nn.BCEWithLogitsLoss()
    
    def remap_subgraph_index_into_composed(self, subgraph_index, idx_composed_ctx_new):
        if len (subgraph_index) == 0:
            return subgraph_index
        subgraph_index_new = idx_composed_ctx_new[..., subgraph_index]
        return subgraph_index_new
    

    def forward(self, data):
        ## 0. Prepare the required data
        x_lig_t = data['ligand_context']['pos']  
        x_rec_0 = data['protein']['pos']
        v_lig_t = data['ligand_context']['atom_type']
        v_rec_0 = data['protein']['atom_feature']
        aa_rec_0 = data['protein']['aa_type']
        batch_idx_lig = data['ligand_context']['batch']
        batch_idx_rec = data['protein']['batch']
        lig_flag = data['ligand_context']['lig_flag']
        rec_flag = data['protein']['lig_flag']
        lig_focal_mask = data['ligand_context']['focal_flag']
        lig_pred_mask = data['ligand_context']['pred_flag']
        rec_focal_mask = data['protein']['focal_flag']
        rec_pred_mask = data['protein']['pred_flag']

        lig_bond_index = data[('ligand_context', 'to', 'ligand_context')]['bond_index']
        lig_bond_type = data[('ligand_context', 'to', 'ligand_context')]['bond_type']
        tri_edge_index = data[('ligand_context', 'to', 'ligand_context')]['tri_edge_index']
        tri_edge_feat = data[('ligand_context', 'to', 'ligand_context')]['tri_edge_feat']

        att_edge_index = data[('edge_graph', 'to', 'edge_graph')]['att_edge_index']
        
        cross_contrast_edge_index = data[('ligand_context', 'to', 'ligand_masked_contrast')]['real_edge_index']
        cross_contrast_edge_type = data[('ligand_context', 'to', 'ligand_masked_contrast')]['real_edge_type']

        cross_rec_edge_index = data[('protein', 'to', 'ligand_masked')]['edge_index']
        cross_lig_edge_index = data[('ligand_context', 'to', 'ligand_masked')]['edge_index']

        x_ligand_masked = data['ligand_masked']['pos']

        x_lig_masked_real = data['ligand_masked_contrast']['pos_real']
        x_lig_masked_fake = data['ligand_masked_contrast']['pos_fake']
        v_lig_masked_real = data['ligand_masked_contrast']['type_real']

        x_lig_mask_batch_idx = data['ligand_masked_contrast']['batch']

        ## 1. Embed context and compose context and encode composed context
        ### 1.1 Embed context
        x_lig_emb, x_rec_emb, h_lig_emb, h_rec_emb = self.context_embedder(
            x_lig_t, x_rec_0, v_lig_t, v_rec_0, aa_rec_0, 
            batch_idx_lig, batch_idx_rec, lig_flag, rec_flag
            )
        
        ### 1.2 Compose context
        (context_composed, batch_idx_composed, 
         (idx_rec_ctx_new, idx_lig_ctx_new)) = compose_context({'x': x_lig_t, 'h': h_lig_emb, 'vec': x_lig_emb},
                                                               {'x': x_rec_0, 'h': h_rec_emb, 'vec': x_rec_emb},
                                                               batch_idx_lig, batch_idx_rec)
        x_composed = context_composed['x']

        ### 1.3 Encode composed context
        lig_bond_index_composed = self.remap_subgraph_index_into_composed(lig_bond_index, idx_lig_ctx_new)
        
        h_composed, vec_composed = self.context_encoder(edge_index = lig_bond_index_composed, edge_type = lig_bond_type,
                                                        batch_idx = batch_idx_composed, **context_composed)
        

        ## 2. Predict focal from protein and ligand context
        ### 2.1 Predict ligand focal
        lig_focal_idx = self.remap_subgraph_index_into_composed(torch.arange(0, len(lig_pred_mask)).to(lig_bond_index)[lig_pred_mask], 
                                                                idx_lig_ctx_new)
        if len(lig_focal_idx) > 0:
            h_composed_lig, vec_composed_lig = h_composed[lig_focal_idx], vec_composed[lig_focal_idx]
            lig_focal_pred, _ = self.focal_head((h_composed_lig, vec_composed_lig))

            loss_lig_focal = F.binary_cross_entropy_with_logits(
                input = lig_focal_pred,
                target = lig_focal_mask[lig_pred_mask].view(-1, 1).float()
            ).clamp_max(10.)
        else:
            loss_lig_focal = torch.tensor(0.).to(x_lig_t)

        ### 2.2 Predict rec focal
        rec_focal_idx = self.remap_subgraph_index_into_composed(torch.arange(0, len(rec_pred_mask)).to(lig_bond_index)[rec_pred_mask],
                                                                idx_rec_ctx_new)
        if len(rec_focal_idx) > 0: 
            h_composed_rec, vec_composed_rec = h_composed[rec_focal_idx], vec_composed[rec_focal_idx]
            rec_focal_pred, _ = self.focal_head((h_composed_rec, vec_composed_rec))
            loss_rec_focal = F.binary_cross_entropy_with_logits(
                input=rec_focal_pred,
                target=rec_focal_mask[rec_pred_mask].view(-1, 1).float()
            ).clamp_max(10.)
        else:
            loss_rec_focal = torch.tensor(0.).to(x_lig_t)

        ## 3. Predict masked position connected with focal
        ### 3.1 Predict masked position connected with ligand focal
        edge_from_lig_focal = self.remap_subgraph_index_into_composed(cross_lig_edge_index[0], idx_lig_ctx_new)
        if len(edge_from_lig_focal) > 0:
            edge_to_lig_masked = cross_lig_edge_index[1]

            _, abs_pos_mu_lig, pos_sigma_lig, pos_pi_lig  = self.pos_pred(
                h_composed, 
                vec_composed,
                edge_from_lig_focal,
                x_composed,
            )
            loss_pos_from_lig = -torch.log(
                self.pos_pred.get_mdn_probability(abs_pos_mu_lig, pos_sigma_lig, 
                                                  pos_pi_lig, x_ligand_masked[edge_to_lig_masked]) + 1e-16
            ).mean().clamp_max(10.)
        else:
            loss_pos_from_lig = torch.tensor(0.).to(x_lig_t)

        ### 3.2 Predicted masked position connected with rec focal
        edge_from_rec_focal = self.remap_subgraph_index_into_composed(cross_rec_edge_index[0], idx_rec_ctx_new)
        if len(edge_from_rec_focal) > 0:
            edge_to_lig_masked = cross_rec_edge_index[1]
            _, abs_pos_mu_rec, pos_sigma_rec, pos_pi_rec  = self.pos_pred(
                h_composed, 
                vec_composed,
                edge_from_rec_focal,
                x_composed,
            )
            loss_pos_from_rec = -torch.log(
                self.pos_pred.get_mdn_probability(abs_pos_mu_rec, pos_sigma_rec, 
                                                  pos_pi_rec, x_ligand_masked[edge_to_lig_masked]) + 1e-16
            ).mean().clamp_max(10.)
        else:
            loss_pos_from_rec = torch.tensor(0.).to(x_lig_t)

        
        ## 4. Predict atom type and edge type
        ligand_cross_index_composed = self.remap_subgraph_index_into_composed(cross_contrast_edge_index[0], 
                                                                              idx_lig_ctx_new)
        cross_contrast_edge_index_composed = torch.stack([ligand_cross_index_composed, 
                                                          cross_contrast_edge_index[1]], dim=0)
        tri_edge_index = self.remap_subgraph_index_into_composed(tri_edge_index, idx_lig_ctx_new)

        y_real_pred, edge_pred = self.atom_edge_pred(
            x_composed, h_composed, vec_composed, 
            x_target = x_lig_masked_real, 
            context_batch_idx = batch_idx_composed, 
            target_batch_idx = x_lig_mask_batch_idx, 
            cross_edge_index = cross_contrast_edge_index_composed, 
            att_edge_index = att_edge_index, 
            tri_edge_index = tri_edge_index, 
            tri_edge_feat = tri_edge_feat
        )   
        loss_atom = self.smooth_cross_entropy(y_real_pred, v_lig_masked_real).clamp_max(10.)    # Classes
        loss_edge = F.cross_entropy(edge_pred, cross_contrast_edge_type).clamp_max(10.)


        ## 5. Contrastive learning
        y_fake_pred, _ = self.atom_edge_pred(
            x_composed, h_composed, vec_composed,
            x_target = x_lig_masked_fake,
            context_batch_idx = batch_idx_composed, 
            target_batch_idx = x_lig_mask_batch_idx
        ) 
        energy_real = -1 * torch.logsumexp(y_real_pred, dim=-1)  # (N_real)
        energy_fake = -1 * torch.logsumexp(y_fake_pred, dim=-1)   # (N_fake)
        energy_real = torch.clamp_max(energy_real, 40)
        energy_fake = torch.clamp_min(energy_fake, -40)
        loss_real = self.bceloss_with_logits(-energy_real, torch.ones_like(energy_real)).clamp_max(10.)
        loss_fake = self.bceloss_with_logits(-energy_fake, torch.zeros_like(energy_fake)).clamp_max(10.)


        ## 6. Return the loss and results for training and evaluation
        loss_dict = {'atom': torch.nan_to_num(loss_atom), 'edge': torch.nan_to_num(loss_edge), 
                     'real': torch.nan_to_num(loss_real), 'fake': torch.nan_to_num(loss_fake),
                     'lig_focal': torch.nan_to_num(loss_lig_focal), 
                     'rec_focal': torch.nan_to_num(loss_rec_focal),
                     'pos_from_lig': torch.nan_to_num(loss_pos_from_lig), 
                     'pos_from_rec': torch.nan_to_num(loss_pos_from_rec)}
        results = {'type_true': v_lig_masked_real, 'type_pred': y_real_pred, 
                   'edge_true': cross_contrast_edge_type, 'edge_pred': edge_pred}

        return loss_dict, results


    def sample(self, data, n_samples_atom=-1, max_iter=15):

        data['ligand_context']['end_switch'] = torch.zeros(data['protein']['batch'].max()+1).to(data['protein']['batch']).bool()
        data['protein']['end_switch'] = torch.zeros(data['protein']['batch'].max()+1).to(data['protein']['batch']).bool()
        has_focal = True

        if len(data['ligand_context']['atom_type']) == 0:
            sample_result = self.sample_init(data, n_samples_atom)
            has_focal = sample_result['has_focal']
            data = self._add_results_to_context(sample_result, data)
        traj_result_inv = {0: (data['ligand_context']['pos'].clone().cpu(), data['ligand_context']['atom_type'].clone().cpu(), 
                               data[('ligand_context', 'to', 'ligand_context')]['bond_index'].clone().cpu(),
                               data[('ligand_context', 'to', 'ligand_context')]['bond_type'].clone().cpu(),
                               data['ligand_context']['batch'].clone().cpu())}
        i = 0
        while has_focal and (i <= max_iter):
            sample_result = self.sample_iter(data)
            has_focal = sample_result['has_focal']
            if has_focal:
                data = self._add_results_to_context(sample_result, data)
                i += 1
                traj_result_inv[i] = (data['ligand_context']['pos'].clone().cpu(), data['ligand_context']['atom_type'].clone().cpu(),
                                    data[('ligand_context', 'to', 'ligand_context')]['bond_index'].clone().cpu(),
                                    data[('ligand_context', 'to', 'ligand_context')]['bond_type'].clone().cpu(),
                                    data['ligand_context']['batch'].clone().cpu())
        traj_result = {k: traj_result_inv[len(traj_result_inv) - k - 1] for k in range(i)}
        return traj_result
        
    def sample_iter(self, data, n_samples_atom=5):
        x_lig_t = data['ligand_context']['pos']
        x_rec_0 = data['protein']['pos']
        v_lig_t = data['ligand_context']['atom_type']
        v_rec_0 = data['protein']['atom_feature']
        aa_rec_0 = data['protein']['aa_type']
        batch_idx_lig = data['ligand_context']['batch']
        batch_idx_rec = data['protein']['batch']
        lig_flag = data['ligand_context']['lig_flag']
        rec_flag = data['protein']['lig_flag']
        lig_bond_index = data[('ligand_context', 'to', 'ligand_context')]['bond_index']
        lig_bond_type = data[('ligand_context', 'to', 'ligand_context')]['bond_type']

        lig_context_lig_flag = data['ligand_context']['lig_flag']
        rec_lig_flag = data['protein']['lig_flag']

        lig_focal_pred_mask = torch.logical_and(lig_context_lig_flag, ~data['ligand_context']['end_switch'][batch_idx_lig])
        rec_focal_pred_mask = torch.logical_and(rec_lig_flag, ~data['protein']['end_switch'][batch_idx_rec])

        (has_focal, has_focal_batch, 
         idx_focal_lig, lig_focal_p, 
         (h_composed, vec_composed),
         (context_composed, batch_idx_composed, 
         (idx_rec_ctx_new, idx_lig_ctx_new))) = self.sample_focal(x_lig_t, x_rec_0, v_lig_t, v_rec_0, aa_rec_0, 
                                                                  lig_bond_index, lig_bond_type, 
                                                                  lig_focal_pred_mask, rec_focal_pred_mask,
                                                                  batch_idx_lig, batch_idx_rec, lig_flag, rec_flag)
        ### if there is no focal in the batch, the `end switch' will be open
        lig_end_pred_mask = torch.logical_not(has_focal_batch)
        rec_end_pred_mask = torch.zeros_like(has_focal_batch)
            
        if has_focal:
            x_composed = context_composed['x']

            idx_focal_lig_composed = self.remap_subgraph_index_into_composed(idx_focal_lig, idx_lig_ctx_new)
            pos_generated, pdf_pos, idx_parent, _, _, _ = self.sample_position(
                h_composed, vec_composed, context_composed, idx_focal_lig_composed
            )
            idx_focal_lig_composed, lig_focal_p = idx_focal_lig_composed[idx_parent], lig_focal_p[idx_parent]
            gen_batch_idx = batch_idx_composed[idx_focal_lig_composed]

            pred_type, prob_type, prob_has_atom, idx_parent, bond_index, bond_type, bond_prob = self.sample_element_and_bond(
                pos_generated, h_composed, vec_composed, x_composed, idx_lig_ctx_new,
                lig_bond_index, lig_bond_type, batch_idx_composed, gen_batch_idx
            )
            pos_generated, pdf_pos, lig_focal_p = pos_generated[idx_parent], pdf_pos[idx_parent], lig_focal_p[idx_parent]
            idx_focal_lig_composed = idx_focal_lig_composed[idx_parent]

            bond_index_composed = torch.stack([self.remap_subgraph_index_into_composed(bond_index[0], idx_lig_ctx_new),
                                               bond_index[1]], dim=0)
            mask_selected, flag_highest_prob, mask_bond_selected, flag_bond_highest_prob = self.filter_according_to_logprob(pdf_pos, prob_type, prob_has_atom, 
                                                                                                    lig_focal_p, idx_focal_lig_composed,
                                                                                                    bond_prob=bond_prob, 
                                                                                                    bond_index=bond_index_composed)
            result = {'has_focal': has_focal, 
                      'generated_batch_idx': gen_batch_idx[mask_selected][flag_highest_prob],
                      'pos_generated': pos_generated[mask_selected][flag_highest_prob], 
                      'pdf_pos': pdf_pos[mask_selected][flag_highest_prob], 
                      'focal_p': lig_focal_p[mask_selected][flag_highest_prob],
                      'pred_type': pred_type[mask_selected][flag_highest_prob],
                      'prob_type': prob_type[mask_selected][flag_highest_prob], 
                      'prob_has_atom': prob_has_atom[mask_selected][flag_highest_prob],
                      'lig_end_pred_mask': lig_end_pred_mask, 
                      'rec_end_pred_mask': rec_end_pred_mask,
                      'bond_index': bond_index[:, flag_bond_highest_prob],
                      'bond_type': bond_type[flag_bond_highest_prob],
                      'bond_prob': bond_prob[flag_bond_highest_prob]}
        else:
            result = {'has_focal': has_focal}

        return result
                
    def sample_element_and_bond(self, pos_generated, h_composed, vec_composed, x_composed, idx_lig_composed,
                                lig_bond_index, lig_bond_type, batch_idx_composed, batch_idx_query):
        n_query = len(pos_generated)
        batch_idx_query
        y_query_pred, edge_preds = self.query_position(pos_generated, h_composed, 
                                                       vec_composed, x_composed, 
                                                       batch_idx_query,
                                                       batch_idx_composed, 
                                                       idx_lig_composed,
                                                       lig_bond_index, lig_bond_type)
        
        has_atom_prob =  (1 - 1 / (1 + torch.exp(y_query_pred).sum(-1)))
        y_query_pred = F.softmax(y_query_pred, dim=-1)
        type_pred = y_query_pred.argmax(dim=-1)  # n_query * n_samples
        idx_parent = torch.repeat_interleave(torch.arange(n_query), 1, dim=0).to(batch_idx_query)
        type_prob = y_query_pred[idx_parent, type_pred]

        edge_pred, edge_index_pred = edge_preds
        edge_pred = F.softmax(edge_pred, dim=-1)  # (num_query, num_context, 4)

        edge_type_pred = edge_pred.argmax(dim=-1)  # (num_query * num_context, n_samples)

        bond_index = edge_index_pred[:,edge_type_pred>0]
        bond_type = edge_type_pred[edge_type_pred>0]
        bond_prob = edge_pred[edge_type_pred>0, bond_type]
        return type_pred, type_prob, has_atom_prob, idx_parent, bond_index, bond_type, bond_prob

    def query_position(self, pos_query, h_composed, vec_composed, x_composed, batch_idx_query,
                       batch_idx_composed, idx_lig_composed, lig_bond_index, lig_bond_type):

        #NOTE: Only one parent batch at a time (i.e. batch size = 1)
        edge_index_query = knn(x=x_composed[idx_lig_composed], y=pos_query, 
                               batch_x=batch_idx_composed[idx_lig_composed], 
                               batch_y=batch_idx_query,
                               k=100, num_workers=16)        

        att_edge_index, tri_edge_index, tri_edge_feat = self._get_tri_edges(
            edge_index_query = edge_index_query,
            batch_idx_query = batch_idx_query,
            batch_idx_lig = batch_idx_composed[idx_lig_composed],
            lig_bond_index = lig_bond_index,
            lig_bond_type = lig_bond_type
        )
        edge_index_query_composed = torch.stack([
            edge_index_query[0],
            self.remap_subgraph_index_into_composed(edge_index_query[1], idx_lig_composed)], 
            dim=0)
        tri_edge_index = self.remap_subgraph_index_into_composed(tri_edge_index, idx_lig_composed)

        y_real_pred, edge_pred = self.atom_edge_pred(
            x_composed, h_composed, vec_composed, 
            x_target = pos_query, 
            context_batch_idx = batch_idx_composed, 
            target_batch_idx = batch_idx_query, 
            cross_edge_index = edge_index_query_composed[[1,0]], 
            att_edge_index = att_edge_index, 
            tri_edge_index = tri_edge_index, 
            tri_edge_feat = tri_edge_feat
        )
        edge_index_query = edge_index_query[[1, 0]]
        return y_real_pred, (edge_pred, edge_index_query)
    
    def _get_tri_edges(self, edge_index_query,
                       batch_idx_query, batch_idx_lig, 
                       lig_bond_index, lig_bond_type):
        
        batch_size = batch_idx_lig.max().item() + 1
        num_query_per_batch = scatter_add(torch.ones_like(batch_idx_query), batch_idx_query, dim=0, dim_size=batch_size)
        num_lig_per_batch = scatter_add(torch.ones_like(batch_idx_lig), batch_idx_lig, dim=0, dim_size=batch_size)

        mark_query = torch.cat([torch.zeros((1,)).to(batch_idx_lig), torch.cumsum(num_query_per_batch, dim=0)], dim=0)
        mark_lig = torch.cat([torch.zeros((1,)).to(batch_idx_lig), torch.cumsum(num_lig_per_batch, dim=0)], dim=0)

        # split the attr into lists
        tri_edge_index_list = []
        tri_edge_type_list = []
        att_edge_index_list = []
        edge_graph_cumsum = 0
        lig_node_cumsum = 0
        for i in range(torch.unique(batch_idx_lig).size(0)):
            # print(i)
            real_sample_idx = torch.arange(0, len(batch_idx_query)).to(batch_idx_query)[batch_idx_query == i]
            num_context = batch_idx_lig[batch_idx_lig == i].size(0)
            edge_index_query_selected = torch.where(torch.logical_and(edge_index_query[1]>=mark_lig[i], 
                                                                  edge_index_query[1]<mark_lig[i+1]))[0]
            
            edge_index_current = edge_index_query[:, edge_index_query_selected]
            edge_index_current[0] = edge_index_current[0] - mark_query[i]
            edge_index_current[1] = edge_index_current[1] - mark_lig[i]

            lig_bond_index_selected = torch.where(torch.logical_and(lig_bond_index[1]>=mark_lig[i],
                                                                lig_bond_index[1]<mark_lig[i+1]))[0]
            lig_bond_index_current = lig_bond_index[:, lig_bond_index_selected]

            lig_bond_type_current = lig_bond_type[lig_bond_index_selected]
            lig_bond_index_current[0] = lig_bond_index_current[0] - mark_lig[i]
            lig_bond_index_current[1] = lig_bond_index_current[1] - mark_lig[i]

            tri_edge_index_current, tri_edge_feat_current, att_edge_index_current = self.construct_tri_edge(edge_index_current, lig_bond_index_current, lig_bond_type_current, 
                                                                                    real_sample_idx=real_sample_idx, num_context=num_context)
            tri_edge_index_list.append(tri_edge_index_current + lig_node_cumsum)
            tri_edge_type_list.append(tri_edge_feat_current)
            att_edge_index_list.append(att_edge_index_current + edge_graph_cumsum)

            edge_graph_cumsum = edge_graph_cumsum + edge_index_current.size(1)
            lig_node_cumsum = lig_node_cumsum + num_context

        tri_edge_index = torch.cat(tri_edge_index_list, dim=1)
        tri_edge_feat = torch.cat(tri_edge_type_list, dim=0)
        att_edge_index = torch.cat(att_edge_index_list, dim=1)
        return att_edge_index, tri_edge_index, tri_edge_feat
    

    def construct_tri_edge(self, cross_edge_index, context_edge_index, context_bond_type, real_sample_idx, num_context):
        row, col = cross_edge_index[0], cross_edge_index[1]
        acc_num_edges = 0
        index_real_cps_edge_i_list, index_real_cps_edge_j_list = [], []  # index of real-ctx edge (for attention)
        if len(real_sample_idx) == 0:
            return (torch.empty((2,0)).to(cross_edge_index), 
                    torch.empty((0,5)).to(cross_edge_index), 
                    torch.empty(2,0).to(cross_edge_index))
        
        for node in torch.arange(len(real_sample_idx)):
            num_edges = (row == node).sum()
            index_edge_i = torch.arange(num_edges, dtype=torch.long, device=cross_edge_index.device) + acc_num_edges
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
        adj_mat = adj_mat.to(cross_edge_index.device)
        adj_mat[context_edge_index[0], context_edge_index[1]] = context_bond_type
        tri_edge_type = adj_mat[node_a_cps_tri_edge, node_b_cps_tri_edge]
        tri_edge_feat = (tri_edge_type.view([-1, 1]) == torch.tensor([[0, 1, 2, 3, 4]]).to(tri_edge_type)).long()
        tri_edge_index = torch.stack([node_a_cps_tri_edge, node_b_cps_tri_edge], dim=0)
        att_edge_index = torch.stack([index_real_cps_edge_i, index_real_cps_edge_j], dim=0)
        return tri_edge_index, tri_edge_feat, att_edge_index


    def _add_results_to_context(self, sample_result, data):
        gen_batch_idx = sample_result['generated_batch_idx']
        pos_generated = sample_result['pos_generated']
        pred_type = sample_result['pred_type']
        if len(pred_type) == 0:
            data['ligand_context']['end_switch'] = torch.ones(data['protein']['batch'].max()+1).to(data['protein']['batch']).bool()
            data['protein']['end_switch'] = torch.ones(data['protein']['batch'].max()+1).to(data['protein']['batch']).bool()
            return data

        data['ligand_context']['atom_type'] = torch.cat([data['ligand_context']['atom_type'], pred_type], dim=0)
        data['ligand_context']['pos'] = torch.cat([data['ligand_context']['pos'], pos_generated], dim=0)
        data['ligand_context']['batch'] = torch.cat([data['ligand_context']['batch'], gen_batch_idx], dim=0)
        data['ligand_context']['lig_flag'] = torch.cat([data['ligand_context']['lig_flag'], 
                                                        torch.ones_like(gen_batch_idx, dtype=torch.bool)], dim=0)
        if 'end_switch' not in data['protein']:
            data['protein']['end_switch'] = sample_result['rec_end_pred_mask'] 
        else:
            data['protein']['end_switch'] = torch.logical_or(sample_result['rec_end_pred_mask'], data['protein']['end_switch'])
       
        if 'end_switch' not in data['lig_end_pred_mask']:
            data['ligand_context']['end_switch'] = sample_result['lig_end_pred_mask']
        else:
            data['ligand_context']['end_switch'] = torch.logical_or(sample_result['lig_end_pred_mask'], data['ligand_context']['end_switch'])

        if sample_result.get('bond_index') is not None:
            num_node_past = data['ligand_context'].num_nodes
            
            new_node_idx = num_node_past + torch.arange(0, len(pred_type)).to(pos_generated.device)
            new_bond_idx = sample_result['bond_index'][1].clone()
            assert sample_result['bond_index'].shape[1] >= pos_generated.shape[0]

            for idx,i in enumerate(sample_result['bond_index'][1].unique()):
                bond_mask = (sample_result['bond_index'][1] == i)
                new_bond_idx[bond_mask] = new_node_idx[idx]

            context_idx_added = torch.stack([sample_result['bond_index'][0], new_bond_idx])
            context_idx_added = torch.cat([context_idx_added, context_idx_added[[1,0]]], dim=1)
            bond_type_added = torch.cat([sample_result['bond_type'], sample_result['bond_type']], dim=0)

            bond_index = torch.cat([data[('ligand_context', 'to', 'ligand_context')]['bond_index'], 
                                    context_idx_added], dim=1)

            bond_type = torch.cat([data[('ligand_context', 'to', 'ligand_context')]['bond_type'], 
                                   bond_type_added], dim=0)
        
        # reindexing batch
        batch_idx_merged = data['ligand_context']['batch'].clone()
        sort_idx = torch.sort(batch_idx_merged, stable=True).indices
        data['ligand_context']['atom_type'] = data['ligand_context']['atom_type'][sort_idx]
        data['ligand_context']['pos'] = data['ligand_context']['pos'][sort_idx]
        data['ligand_context']['batch'] = data['ligand_context']['batch'][sort_idx]
        data['ligand_context']['lig_flag'] = data['ligand_context']['lig_flag'][sort_idx]

        if sample_result.get('bond_index') is not None:
            batch_idx_merged_bond = batch_idx_merged[bond_index[0]]
            sort_idx_bond = torch.sort(batch_idx_merged_bond, stable=True).indices
            reindex_0 = torch.where((bond_index[0].unsqueeze(0) == sort_idx.unsqueeze(1)))
            reindex_1 = torch.where((bond_index[1].unsqueeze(0) == sort_idx.unsqueeze(1)))
            bond_index = torch.stack([reindex_0[0][reindex_0[1].argsort()],
                                      reindex_1[0][reindex_1[1].argsort()]], dim=0)
            bond_index_reindexed = bond_index[:, sort_idx_bond]
            bond_type_reindexed = bond_type[sort_idx_bond]

            # #################
            # batch_idx_lig = data['ligand_context']['batch']
            # lig_bond_index = bond_index_reindexed
            # num_per_batch = scatter_add(torch.ones_like(batch_idx_lig), batch_idx_lig, dim=0, dim_size=batch_idx_lig.max()+1)
            # mark_lig = torch.cat([torch.zeros((1,)).to(batch_idx_lig), torch.cumsum(num_per_batch, dim=0)], dim=0)
            # for i in range(torch.unique(batch_idx_lig).size(0)):
            #     num_context = batch_idx_lig[batch_idx_lig == i].size(0)

            #     lig_bond_index_selected = torch.where(torch.logical_and(lig_bond_index[1]>=mark_lig[i],
            #                                                         lig_bond_index[1]<mark_lig[i+1]))[0]
            #     lig_bond_index_current = lig_bond_index[:, lig_bond_index_selected]

            #     lig_bond_index_current[0] = lig_bond_index_current[0] - mark_lig[i]
            #     lig_bond_index_current[1] = lig_bond_index_current[1] - mark_lig[i]
            #     if len(lig_bond_index_current[0]) > 0: 
            #         assert lig_bond_index_current[0].max() < num_per_batch[i] 
            #         assert lig_bond_index_current[1].max() < num_per_batch[i] 
                
            # ################
            data[('ligand_context', 'to', 'ligand_context')]['bond_index'] = bond_index_reindexed
            data[('ligand_context', 'to', 'ligand_context')]['bond_type'] = bond_type_reindexed
        data['ligand_context'].num_nodes = data['ligand_context']['atom_type'].size(0)
        return data
        
    def sample_focal(self, x_lig_t, x_rec_0, v_lig_t, v_rec_0, 
                     aa_rec_0, lig_bond_index, lig_bond_type,
                     lig_focal_pred_mask, rec_focal_pred_mask,
                     batch_idx_lig, batch_idx_rec, lig_flag, rec_flag,
                     focal_threshold=0):
        
        x_lig_emb, x_rec_emb, h_lig_emb, h_rec_emb = self.context_embedder(
            x_lig_t, x_rec_0, v_lig_t, v_rec_0, aa_rec_0, 
            batch_idx_lig, batch_idx_rec, lig_flag, rec_flag
        )
    
        ### 1 Compose context
        (context_composed, batch_idx_composed, 
        (idx_rec_ctx_new, idx_lig_ctx_new)) = compose_context({'x': x_lig_t, 'h': h_lig_emb, 'vec': x_lig_emb},
                                                              {'x': x_rec_0, 'h': h_rec_emb, 'vec': x_rec_emb},
                                                              batch_idx_lig, batch_idx_rec)
        lig_bond_index_composed = self.remap_subgraph_index_into_composed(lig_bond_index, idx_lig_ctx_new)
        
        ### 2 Encode composed context
        h_composed, vec_composed = self.context_encoder(edge_index = lig_bond_index_composed, edge_type = lig_bond_type,
                                                        batch_idx = batch_idx_composed, **context_composed)
        
        ### 3 Predict focal from protein context or ligand context
        lig_focal_idx = self.remap_subgraph_index_into_composed(torch.arange(0, len(lig_focal_pred_mask)).to(batch_idx_lig)[lig_focal_pred_mask],
                                                                idx_lig_ctx_new)
        rec_focal_idx = self.remap_subgraph_index_into_composed(torch.arange(0, len(rec_focal_pred_mask)).to(batch_idx_rec)[rec_focal_pred_mask],
                                                                idx_rec_ctx_new)
        if len(lig_focal_idx) > 0 and len(rec_focal_idx) == 0:
            h_composed_focal, vec_composed_focal = h_composed[lig_focal_idx], vec_composed[lig_focal_idx]
            batch_idx_focal = batch_idx_lig[lig_focal_pred_mask]
        elif len(lig_focal_idx) == 0 and len(rec_focal_idx) > 0:
            h_composed_focal, vec_composed_focal = h_composed[rec_focal_idx], vec_composed[rec_focal_idx]
            batch_idx_focal = batch_idx_rec[rec_focal_pred_mask]
        else:
            return False, torch.zeros(batch_idx_rec.max()+1).to(lig_focal_pred_mask), None, None, (None, None), (None, None, (None, None)), 
        
        focal_pred, _ = self.focal_head((h_composed_focal, vec_composed_focal))
        focal_pred = focal_pred.squeeze(-1)
        msk_focal = (focal_pred > focal_threshold)
        num_focal_batch = scatter_add(msk_focal.long(), batch_idx_focal, dim_size=batch_idx_rec.max()+1)
        has_focal = torch.sum(num_focal_batch) > 0
        has_focal_batch = (num_focal_batch > 0)
        idx_focal = torch.arange(0, len(focal_pred)).to(batch_idx_lig)[msk_focal]

        if has_focal:
            # get all focals as focal
            p_focal = torch.sigmoid(focal_pred[idx_focal])
        else:
            p_focal = torch.empty((0,), dtype=torch.float32)
        
        return (has_focal, has_focal_batch, 
                idx_focal, p_focal, 
                (h_composed, vec_composed),
                (context_composed, batch_idx_composed, 
                (idx_rec_ctx_new, idx_lig_ctx_new)))

    def sample_init(self, data,
                    n_samples_atom=5):
        
        x_lig_0 = data['ligand_context']['pos']
        x_rec_0 = data['protein']['pos']
        v_lig_0 = data['ligand_context']['atom_type']
        v_rec_0 = data['protein']['atom_feature']
        aa_rec_0 = data['protein']['aa_type']
        batch_idx_lig = data['ligand_context']['batch']
        batch_idx_rec = data['protein']['batch']
        lig_flag = data['ligand_context']['lig_flag']
        rec_flag = data['protein']['lig_flag']
        lig_bond_index = data[('ligand_context', 'to', 'ligand_context')]['bond_index']
        lig_bond_type = data[('ligand_context', 'to', 'ligand_context')]['bond_type']

        lig_context_lig_flag = data['ligand_context']['lig_flag']
        rec_lig_flag = data['protein']['lig_flag']

        lig_focal_pred_mask = torch.logical_not(lig_context_lig_flag)
        rec_focal_pred_mask = torch.logical_not(rec_lig_flag)

        (has_focal, has_focal_batch, 
         idx_focal_rec, rec_focal_p, 
         (h_composed, vec_composed),
         (context_composed, batch_idx_composed, 
         (idx_rec_ctx_new, idx_lig_ctx_new))) = self.sample_focal(x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0, 
                                                                  lig_bond_index, lig_bond_type, 
                                                                  lig_focal_pred_mask, rec_focal_pred_mask,
                                                                  batch_idx_lig, batch_idx_rec, lig_flag, rec_flag)
        x_composed = context_composed['x']
        ### if there is no focal in the batch, the `end switch' will be open
        lig_end_pred_mask = torch.logical_not(has_focal_batch)
        rec_end_pred_mask = torch.logical_not(has_focal_batch)
        
        if has_focal:
            idx_focal_rec_composed = self.remap_subgraph_index_into_composed(idx_focal_rec, idx_rec_ctx_new)
            pos_generated, pdf_pos, idx_parent, _, _, _ = self.sample_position(
                h_composed, vec_composed, context_composed, idx_focal_rec_composed
            )
            idx_focal_rec_composed, rec_focal_p = idx_focal_rec_composed[idx_parent], rec_focal_p[idx_parent]
            gen_batch_idx = batch_idx_composed[idx_focal_rec_composed]

            pred_type, prob_type, prob_has_atom, idx_parent = self.sample_init_element(
                pos_generated, h_composed, vec_composed, x_composed, 
                batch_idx_composed, gen_batch_idx, n_samples=n_samples_atom
            )
            pos_generated, pdf_pos, rec_focal_p = pos_generated[idx_parent], pdf_pos[idx_parent], rec_focal_p[idx_parent]
            idx_focal_rec_composed = idx_focal_rec_composed[idx_parent]

            mask_selected, flag_highest_prob, _, _ = self.filter_according_to_logprob(pdf_pos, prob_type, prob_has_atom, 
                                                                                      rec_focal_p, idx_focal_rec_composed)
            
            result = {'has_focal': has_focal, 
                      'generated_batch_idx': gen_batch_idx[mask_selected][flag_highest_prob],
                      'pos_generated': pos_generated[mask_selected][flag_highest_prob], 
                      'pdf_pos': pdf_pos[mask_selected][flag_highest_prob], 
                      'rec_focal_p': rec_focal_p[mask_selected][flag_highest_prob],
                      'pred_type': pred_type[mask_selected][flag_highest_prob],
                      'prob_type': prob_type[mask_selected][flag_highest_prob], 
                      'prob_has_atom': prob_has_atom[mask_selected][flag_highest_prob],
                      'lig_end_pred_mask': lig_end_pred_mask, 
                      'rec_end_pred_mask': ~rec_end_pred_mask}
            return result 

    def filter_according_to_logprob(self, pdf_pos, prob_type, prob_has_atom, prob_focal, 
                                    idx_focal_composed, 
                                    bond_prob=None, 
                                    bond_index=None,
                                    threshold_pos=np.log(0.25), 
                                    threshold_element=np.log(0.3), 
                                    threshod_has_atom=np.log(0.6), 
                                    threshold_focal=np.log(0.5),
                                    threshold_bond=np.log(0.4)):
        
        mask_pos = torch.log(pdf_pos) > threshold_pos
        mask_element = torch.log(prob_type) > threshold_element
        mask_has_atom = torch.log(prob_has_atom) > threshod_has_atom
        mask_focal = torch.log(prob_focal) > threshold_focal

        mask_selected = mask_pos & mask_element & mask_has_atom & mask_focal
        selected_idx = idx_focal_composed[mask_selected]
        selected_idx_in_mask = torch.arange(len(mask_pos)).to(mask_pos.device)[mask_selected]
        if bond_prob is not None:
            mask_bond_selected = torch.zeros_like(mask_selected, dtype=torch.bool)
            for idx, i in enumerate(selected_idx):
                index_bond_i = (bond_index[0, :] == i)
                if len(bond_prob[index_bond_i]) == 0:
                    continue
                logp_bond = torch.max(torch.log(bond_prob[index_bond_i]), dim=0)[0]
                if logp_bond > threshold_bond:
                    mask_bond_selected[selected_idx_in_mask[idx]] = True
            mask_selected = mask_selected & mask_bond_selected
            selected_idx = idx_focal_composed[mask_selected]

        mean_prob = torch.mean(torch.stack([torch.log(pdf_pos), 
                                            torch.log(prob_type), 
                                            torch.log(prob_has_atom), 
                                            torch.log(prob_focal)], dim=0), dim=0)
        mean_prob = mean_prob[mask_selected]
        flag_highest_prob = []
        if bond_prob is not None:
            bond_index_mask = torch.zeros_like(bond_prob, dtype=torch.bool)
        else: 
            bond_index_mask = torch.empty(0, dtype=torch.bool).to(mean_prob.device)

        selected_idx_gen = torch.arange(len(mask_selected)).to(selected_idx)[mask_selected]
        if bond_index is not None:
            bond_index_flag_highest_prob = torch.zeros_like(bond_index[0], dtype=torch.bool)
        else:
            bond_index_flag_highest_prob = torch.empty(0, dtype=torch.bool).to(mean_prob.device)

        for idx in selected_idx.unique():
            mask_prob = (selected_idx == idx)
            assert mask_prob.sum() > 0
            mean_prob_i = mean_prob[mask_prob]
            selected_idx_gen_i = selected_idx_gen[mask_prob]
            selected_flag = torch.zeros_like(mean_prob_i, dtype=torch.bool)
            max_prob_idx = torch.argmax(mean_prob_i)
            selected_flag[max_prob_idx] = True
            
            if bond_prob is not None:
                mask_bond_i = (bond_index[0, :] == idx)
                selected_bond_index = bond_index[:, mask_bond_i]
                selected_bond_index_mask = torch.zeros_like(selected_bond_index[0], dtype=torch.bool)
                selected_bond_index_flag_highest_prob = torch.zeros_like(selected_bond_index[0], dtype=torch.bool)
                if selected_idx_gen_i[max_prob_idx] not in selected_bond_index[1]:
                    mask_selected[selected_idx_gen_i] = False
                    continue
                for i, idx_gen in enumerate(selected_bond_index[1]):
                    if idx_gen in selected_idx_in_mask:
                        selected_bond_index_mask[i] = True
                    if idx_gen == selected_idx_gen_i[max_prob_idx]:
                        selected_bond_index_flag_highest_prob[i] = True

                bond_index_flag_highest_prob[mask_bond_i] = selected_bond_index_flag_highest_prob
                bond_index_mask[mask_bond_i] = selected_bond_index_mask
            flag_highest_prob.append(selected_flag)
        flag_highest_prob = torch.cat(flag_highest_prob, dim=0) if len(flag_highest_prob) > 0 else torch.empty((0,)).to(mask_pos)
        if bond_index is not None:
            assert bond_index_flag_highest_prob.sum() == flag_highest_prob.sum()
        return mask_selected, flag_highest_prob, bond_index_mask, bond_index_flag_highest_prob
        

    def sample_position(self, h_composed, vec_composed, context_composed, idx_focal, sample_num=10):
        _, abs_pos_mu, pos_sigma, pos_pi = self.pos_pred(
            h_composed, vec_composed, idx_focal, context_composed['x']
        )
        n_focals = len(idx_focal)
        pos_generated = self.pos_pred.sample_batch(abs_pos_mu, pos_sigma, pos_pi, sample_num)  # n_focals, n_per_pos, 3
        n_candidate_samples = pos_generated.size(1)
        pos_generated = torch.reshape(pos_generated, [-1, 3])
        pdf_pos = self.pos_pred.get_mdn_probability(
            mu=torch.repeat_interleave(abs_pos_mu, repeats=n_candidate_samples, dim=0),
            sigma=torch.repeat_interleave(pos_sigma, repeats=n_candidate_samples, dim=0),
            pi=torch.repeat_interleave(pos_pi, repeats=n_candidate_samples, dim=0),
            pos_target=pos_generated
        )

        idx_parent = torch.repeat_interleave(torch.arange(n_focals), 
                                             repeats=n_candidate_samples, 
                                             dim=0).to(idx_focal)

        return pos_generated, pdf_pos, idx_parent, abs_pos_mu, pos_sigma, pos_pi
    
    def sample_init_element(self, pos_generated, h_composed, vec_composed, x_composed, 
                            batch_idx_composed, gen_batch_idx, n_samples):
        n_query = len(pos_generated)
        y_real_pred, _ = self.atom_edge_pred(
            x_composed, h_composed, vec_composed, 
            x_target = pos_generated, 
            context_batch_idx = batch_idx_composed, 
            target_batch_idx = gen_batch_idx
        )   

        if n_samples < 0:
            # raise NotImplementedError('The following is not fixed')
            prob_has_atom =  1 - 1 / (1 + torch.exp(y_real_pred).sum(-1))
            y_real_pred = F.softmax(y_real_pred, dim=-1)
            pred_type = y_real_pred.argmax(dim=-1)
            prob_type = y_real_pred[torch.arange(len(y_real_pred)), pred_type]
            idx_parent = torch.arange(n_query).to(x_composed.device)
        else:
            prob_has_atom =  (1 - 1 / (1 + torch.exp(y_real_pred).sum(-1)))
            prob_has_atom = torch.repeat_interleave(prob_has_atom, n_samples, dim=0)  # n_query * n_samples
            y_real_pred = F.softmax(y_real_pred, dim=-1)
            pred_type = y_real_pred.multinomial(n_samples, replacement=True).reshape(-1)  # n_query, n_samples
            idx_parent = torch.repeat_interleave(torch.arange(n_query), n_samples, dim=0).to(x_composed.device)
            prob_type = y_real_pred[idx_parent, pred_type]
            # drop duplicates
            identifier = torch.stack([idx_parent, pred_type], dim=1)
            identifier, index_unique = unique(identifier, dim=0)

            pred_type, prob_type, prob_has_atom, idx_parent = pred_type[index_unique], prob_type[index_unique], prob_has_atom[index_unique], idx_parent[index_unique]

        return (pred_type, prob_type, prob_has_atom, idx_parent) # element