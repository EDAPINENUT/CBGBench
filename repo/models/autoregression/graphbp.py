from torch import nn 
from .._base import register_model
import torch
from repo.modules.common import MLP
from repo.modules.schnet.schnet import SchNet
from repo.modules.embs.dist_emb import get_dist_emb
from repo.modules.embs.angle_emb import get_angle_emb
import torch.nn.functional as F
from torch.distributions.normal import Normal


class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            print(self.weight)
            raise RuntimeError('Rescale factor has NaN entries')

        x = torch.exp(self.weight) * x
        return x

class ST_Net_Exp(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True):
        super(ST_Net_Exp, self).__init__()
        self.num_layers = num_layers  # unused
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim*2, bias=bias)
        self.rescale1 = Rescale()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)

    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        x = self.linear2(self.tanh(self.linear1(x)))
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.rescale1(torch.tanh(s))
        return s, t

def flow_forward(flow_layers, x, feat):
    for i in range(len(flow_layers)):
        s, t = flow_layers[i](feat)
        s = s.exp()
        x = (x + t) * s
        
        if i == 0:
            x_log_jacob = (torch.abs(s) + 1e-20).log()
        else:
            x_log_jacob += (torch.abs(s) + 1e-20).log()
    return x, x_log_jacob

def flow_reverse(flow_layers, latent, feat):
    for i in reversed(range(len(flow_layers))):
        s, t = flow_layers[i](feat)
        s = s.exp()
        latent = (latent / s) - t
    return latent


def dattoxyz(f, c1, c2, d, angle, torsion):
    c1c2 = c2 - c1
    c1f = f - c1
    c1c3 = c1f * torch.sum(c1c2 * c1f, dim=-1, keepdim=True) / torch.sum(c1f * c1f, dim=-1, keepdim=True)
    c3 = c1c3 + c1

    c3c2 = c2 - c3
    c3c4_1 = c3c2 * torch.cos(torsion[:, :, None])
    c3c4_2 = torch.cross(c3c2, c1f) / torch.norm(c1f, dim=-1, keepdim=True) * torch.sin(torsion[:, :, None])
    c3c4 = c3c4_1 + c3c4_2

    new_pos = -c1f / torch.norm(c1f, dim=-1, keepdim=True) * d[:, :, None] * torch.cos(angle[:, :, None])
    new_pos += c3c4 / torch.norm(c3c4, dim=-1, keepdim=True) * d[:, :, None] * torch.sin(angle[:, :, None])
    new_pos += f

    return new_pos


@register_model('graphbp')
class GraphBP(nn.Module):
    def __init__(self, config):

        super(GraphBP, self).__init__()
        self.config = config
        self.encoder = config.encoder
        
        self.num_classes = config.num_atomtype
        config.embedder.num_atomtype = self.num_classes

        self.context_embedder = SchNet(config.embedder.num_atomtype, 
                                       config.embedder.emb_dim, 
                                       num_interactions = config.embedder.num_layers)
        
        node_feat_dim = config.embedder.emb_dim 
        dist_feat_dim = config.embedder.emb_dim
        angle_feat_dim = config.embedder.emb_dim * 2
        torsion_feat_dim = config.embedder.emb_dim * 3

        self.node_flow_layers = nn.ModuleList([ST_Net_Exp(node_feat_dim, 
                                                          self.num_classes, 
                                                          hid_dim=config.encoder.node_feat_dim, 
                                                          bias=True) for _ in range(config.encoder.num_layers)])
        self.dist_flow_layers = nn.ModuleList([ST_Net_Exp(dist_feat_dim, 
                                                          1, 
                                                          hid_dim=config.encoder.node_feat_dim, 
                                                          bias=True) for _ in range(config.encoder.num_layers)])
        self.angle_flow_layers = nn.ModuleList([ST_Net_Exp(angle_feat_dim,
                                                           1, 
                                                           hid_dim=config.encoder.node_feat_dim, 
                                                           bias=True) for _ in range(config.encoder.num_layers)])
        self.torsion_flow_layers = nn.ModuleList([ST_Net_Exp(torsion_feat_dim, 
                                                             1, 
                                                             hid_dim=config.encoder.node_feat_dim, 
                                                             bias=True) for _ in range(config.encoder.num_layers)])
        

        self.focus_mlp = MLP(config.encoder.node_feat_dim, 1, config.encoder.node_feat_dim*2)
        self.contact_mlp = MLP(config.encoder.node_feat_dim, 1, config.encoder.node_feat_dim*2)
        self.deq_coeff = self.config.get('deq_coeff', 0.9)

        num_radial, num_spherical = config.embedder.get('num_radial', 6), config.embedder.get('num_spherical', 7)
        num_basis = config.embedder.get('num_basis', 32)
        
        self.dist_emb = get_dist_emb(config.embedder.dist.type, num_radial)
        self.angle_emb = get_angle_emb(config.embedder.angle.type, num_spherical, num_radial)
        
        self.dist_head = MLP(num_radial, node_feat_dim, num_basis)
        self.angle_head = MLP(num_spherical * num_radial, node_feat_dim, num_basis)

        self.focus_ce = torch.nn.BCELoss()
        self.contact_ce = torch.nn.BCELoss()
        

    def forward(self, data_batch):
        z, pos, batch = data_batch['atom_type'], data_batch['pos'], data_batch['batch']
        node_feat = self.context_embedder(z, pos, batch)
        focus_score = F.sigmoid(self.focus_mlp(node_feat[data_batch['gen_flag']]).squeeze(-1))
        contact_score = F.sigmoid(self.contact_mlp(node_feat[data_batch['contact_y_or_n']]).squeeze(-1))

        new_atom_type, focus = data_batch['new_atom_type'], data_batch['focus']
        x_z = F.one_hot(new_atom_type, num_classes=self.num_classes).float()
        x_z += self.deq_coeff * torch.rand(x_z.size(), device=x_z.device)

        local_node_type_feat = node_feat[focus]
        node_latent, node_log_jacob = flow_forward(self.node_flow_layers, x_z, local_node_type_feat)
        node_type_emb_block = self.context_embedder.embedding
        node_type_emb = node_type_emb_block(new_atom_type)
        node_emb = node_feat * node_type_emb[batch]
        
        c1_focus, c2_c1_focus = data_batch['c1_focus'], data_batch['c2_c1_focus']
        dist, angle, torsion = data_batch['new_dist'], data_batch['new_angle'], data_batch['new_torsion']

        local_dist_feat = node_emb[focus]
        dist_latent, dist_log_jacob = flow_forward(self.dist_flow_layers, dist, local_dist_feat)
        
        ### d --> theta
        dist_emb = self.dist_head(self.dist_emb(dist.squeeze()[batch].to(torch.float)))
        node_emb = node_emb * dist_emb # [N, hidden] * [N, hidden]. N is the total number of steps for all molecules in the batch
        

        node_emb_clone = node_emb.clone() # Avoid changing node_emb in-place --> cannot comput gradient otherwise
        local_angle_feat = torch.cat((node_emb_clone[c1_focus[:,1]], node_emb_clone[c1_focus[:,0]]), dim=1)
        angle_latent, angle_log_jacob = flow_forward(self.angle_flow_layers, angle, local_angle_feat)
        
        ###  d, theta --> phi
        dist_angle_emd = self.angle_head(self.angle_emb(dist.squeeze()[batch].to(torch.float), 
                                                        angle.squeeze()[batch].to(torch.float)))
        
        node_emb = node_emb * dist_angle_emd
        
        local_torsion_feat = torch.cat((node_emb[c2_c1_focus[:,2]], node_emb[c2_c1_focus[:,1]], node_emb[c2_c1_focus[:,0]]), dim=1)
        torsion_latent, torsion_log_jacob = flow_forward(self.torsion_flow_layers, torsion, local_torsion_feat)
        cannot_focus = data_batch['cannot_focus']
        cannot_contact = data_batch['cannot_contact']

        return self.get_loss(node_latent, node_log_jacob, 
                            dist_latent, dist_log_jacob, 
                            angle_latent, angle_log_jacob, 
                            torsion_latent, torsion_log_jacob,
                            focus_score, cannot_focus,
                            contact_score, cannot_contact)

    def get_loss(self, node_latent, node_log_jacob, 
                 dist_latent, dist_log_jacob, 
                 angle_latent, angle_log_jacob, 
                 torsion_latent, torsion_log_jacob,
                 focus_score, cannot_focus,
                 contact_score, cannot_contact):
        
        ll_node = torch.mean(1/2 * (node_latent ** 2) - node_log_jacob)
        ll_dist = torch.mean(1/2 * (dist_latent ** 2) - dist_log_jacob)
        ll_angle = torch.mean(1/2 * (angle_latent ** 2) - angle_log_jacob)
        ll_torsion = torch.mean(1/2 * (torsion_latent ** 2) - torsion_log_jacob)
        focus_ce = self.focus_ce(focus_score, cannot_focus)
        contact_ce = self.contact_ce(contact_score, cannot_contact)

        loss_dict = {'node': ll_node, 'dist': ll_dist, 'angle': ll_angle, 
                     'torsion': ll_torsion, 'focus': focus_ce, 'contact': contact_ce}
        
        loss_info = {'node_latent': node_latent, 'node_log_jacob': node_log_jacob, 
                     'dist_lantent': dist_latent, 'dist_log_jacob': dist_log_jacob, 
                     'angle_latent': angle_latent, 'angle_log_jacob': angle_log_jacob, 
                     'torsion_latent':torsion_latent, 'torsion_log_jacob': torsion_log_jacob,
                     'focus_score': focus_score, 'cannot_focus': cannot_focus,
                     'contact_scores': contact_score, 'cannot_contact': cannot_contact}
        
        return loss_dict, loss_info
    
    def sample(self, data, stds = [0.5, 0.3, 0.4, 0.1], max_atoms=45, min_atoms=12, 
               contact_prob=False, contact_th=0.5, focus_th=0.5, add_final=True):
        
        atom_type_protein = data.protein.atom_type
        pos_protein = data.protein.pos
        
        if hasattr(data, 'ligand_ctx'):
            atom_type_context = data.ligand_ctx.atom_type
            pos_context = data.ligand_ctx.pos
        else:
            atom_type_context = torch.empty([1,0]).to(atom_type_protein)
            pos_context = torch.empty([1,0,3]).to(pos_protein)

        if atom_type_protein.dim() == 2:
            atom_type_protein = atom_type_protein.squeeze(0)
        if atom_type_context.dim() == 2:
            atom_type_context = atom_type_context.squeeze(0)
        
        if pos_context.dim() == 3:
            pos_context = pos_context.squeeze(0)
        if pos_protein.dim() == 3:
            pos_protein = pos_protein.squeeze(0)

        ctx_atom_type = torch.concat([atom_type_context, atom_type_protein], dim=0)
        ctx_pos = torch.concat([pos_context, pos_protein], dim=0)
        lig_ctx_num = len(atom_type_context)

        z_lig = data.ligand.atom_type.squeeze(0)
        pos_lig = data.ligand.pos.squeeze(0)
        focuses = data.ligand.focuses.squeeze(0)
        num_gen = z_lig.shape[0]
        ctx_n_atoms = len(ctx_atom_type)
        node_type_emb_block = self.context_embedder.embedding

        prior_node = Normal(torch.zeros([self.num_classes]).to(pos_lig), stds[0] * torch.ones([self.num_classes]).to(pos_lig))
        prior_dist = Normal(torch.zeros([1]).to(pos_lig), stds[1] * torch.ones([1]).to(pos_lig))
        prior_angle = Normal(torch.zeros([1]).to(pos_lig), stds[2] * torch.ones([1]).to(pos_lig))
        prior_torsion = Normal(torch.zeros([1]).to(pos_lig), stds[3] * torch.ones([1]).to(pos_lig))

        feat_index = lambda node_id, f: f[torch.arange(num_gen), node_id]
        pos_index = lambda node_id, p: p[torch.arange(num_gen), node_id].view(num_gen,1,3)

        traj_result_inv = {0: (pos_lig.clone().cpu().squeeze(0), 
                               z_lig.clone().cpu().squeeze(0), 
                               torch.zeros_like(z_lig).long().cpu().squeeze(0))}

        for i in range(max_atoms):
            # print(i)
            batch = torch.arange(num_gen, device=z_lig.device).view(num_gen, 1).repeat(1, i+ctx_n_atoms)
            z = torch.cat((z_lig, ctx_atom_type.repeat(num_gen, 1)), dim=1)
            pos = torch.cat((pos_lig, ctx_pos.repeat(num_gen, 1, 1)), dim=1)
            node_feat = self.context_embedder(z.view(-1), pos.view(-1,3), batch.view(-1))
            
            if i == 0:
                contact_score = self.contact_mlp(node_feat).view(num_gen, ctx_n_atoms)
                if contact_prob: # The prob of selecting a atom is propotional to the predicted prob
                    contact_mask = contact_score > contact_th
                    can_contact = contact_score
                    can_contact[contact_mask] = 0
                else: # Contact atom is selected randomly from nodes with predicted score < contact_th
                    can_contact = contact_score < contact_th
                focus_node_id = torch.multinomial(can_contact.float(), 1).view(num_gen)
                
                node_feat = node_feat.view(num_gen, ctx_n_atoms, -1)

            else:
                rec_mask = torch.cat((torch.zeros([i], dtype=torch.bool), torch.ones([ctx_n_atoms], dtype=torch.bool))).repeat(num_gen)
                focus_score = self.focus_mlp(node_feat[~rec_mask]).view(num_gen, i)
                can_focus = (focus_score < focus_th)
                complete_mask = (can_focus.sum(dim=-1) == 0)
                if i > max(1, min_atoms-1-lig_ctx_num) and torch.sum(complete_mask) > 0:
                    traj_result_inv[i] = (pos_lig[complete_mask].view(-1, i, 3).clone().cpu().squeeze(0),
                                         z_lig[complete_mask].view(-1, i).clone().cpu().squeeze(0), 
                                         torch.zeros_like(z_lig).long().cpu().squeeze(0))
                    
                continue_mask = torch.logical_not(complete_mask)
                dirty_mask = torch.nonzero(torch.isnan(focus_score).sum(dim=-1))[:,0]
                if len(dirty_mask) > 0:
                    continue_mask[dirty_mask] = False
                dirty_mask = torch.nonzero(torch.isinf(focus_score).sum(dim=-1))[:,0]
                if len(dirty_mask) > 0:
                    continue_mask[dirty_mask] = False

                if torch.sum(continue_mask) == 0:
                    break
            
                node_feat = node_feat.view(num_gen, i+ctx_n_atoms, -1)
                num_gen = torch.sum(continue_mask).cpu().item()
                z, pos, can_focus, focuses = z[continue_mask], pos[continue_mask], can_focus[continue_mask], focuses[continue_mask]
                z_lig, pos_lig = z_lig[continue_mask], pos_lig[continue_mask]
                focus_node_id = torch.multinomial(can_focus.float(), 1).view(num_gen)
                node_feat = node_feat[continue_mask]

            latent_node = prior_node.sample([num_gen])
            
            local_node_type_feat = feat_index(focus_node_id, node_feat)
            
            latent_node = flow_reverse(self.node_flow_layers, latent_node, local_node_type_feat)
            node_type_id = torch.argmax(latent_node, dim=1)
            node_type_emb = node_type_emb_block(node_type_id)
            node_emb = node_feat * node_type_emb.view(num_gen, 1, -1)

            latent_dist = prior_dist.sample([num_gen])
            
            local_dist_feat = feat_index(focus_node_id, node_emb)
            
            dist = flow_reverse(self.dist_flow_layers, latent_dist, local_dist_feat)
            
            dist_emb = self.dist_head(self.dist_emb(dist.to(torch.float)))
            node_emb = node_emb * dist_emb.view(num_gen, 1, -1)
            
            # print(pos.shape)
            dist_to_focus = torch.sum(torch.square(pos - pos_index(focus_node_id, pos)), dim=-1)
            _, indices = torch.topk(dist_to_focus, 3, largest=False)
            c1_node_id, c2_node_id = indices[:,1], indices[:,2]


            latent_angle = prior_angle.sample([num_gen])
            local_angle_feat = torch.cat((feat_index(focus_node_id, node_emb), feat_index(c1_node_id, node_emb)), dim=1)

            angle = flow_reverse(self.angle_flow_layers, latent_angle, local_angle_feat)


            dist_angle_emd = self.angle_head(self.angle_emb(dist.to(torch.float), angle.to(torch.float)))
            node_emb = node_emb * dist_angle_emd.view(num_gen, 1, -1)

            latent_torsion = prior_torsion.sample([num_gen])

            local_torsion_feat = torch.cat((feat_index(focus_node_id, node_emb), 
                                            feat_index(c1_node_id, node_emb), 
                                            feat_index(c2_node_id, node_emb)), dim=1)

            torsion = flow_reverse(self.torsion_flow_layers, latent_torsion, local_torsion_feat)
            new_pos = dattoxyz(pos_index(focus_node_id, pos), pos_index(c1_node_id, pos), pos_index(c2_node_id, pos), dist, angle, torsion)


            # print(z_lig.shape)
            # print(node_type_id.shape)
            z_lig = torch.cat((z_lig, node_type_id[:, None]), dim=1)
            pos_lig = torch.cat((pos_lig, new_pos.view(num_gen, 1, 3)), dim=1)
            focuses = torch.cat((focuses, focus_node_id[:,None]), dim=1)

        if add_final and torch.sum(continue_mask) > 0:
            i = i + 1
            out_node_types = z_lig.view(-1,i).clone().cpu()
            out_pos = pos_lig.view(-1, i, 3).clone().cpu()
            traj_result_inv[i] = (out_pos.squeeze(0), 
                                  out_node_types.squeeze(0), 
                                  torch.zeros_like(out_node_types).long().cpu().squeeze(0))

        key_list = list(traj_result_inv.keys())
        traj_result = {}
        for k in range(len(key_list)):
            pos = traj_result_inv[key_list[len(traj_result_inv) - k - 1]][0]
            atom_type = traj_result_inv[key_list[len(traj_result_inv) - k - 1]][1]
            pos_merge = torch.cat([pos_context.clone().cpu(), pos], dim=0)
            atom_type_merge = torch.cat([atom_type_context.clone().cpu(), atom_type], dim=0)
            batch_index = torch.zeros_like(atom_type_merge)
            traj_result[k] = (pos_merge, atom_type_merge, batch_index)
        return traj_result