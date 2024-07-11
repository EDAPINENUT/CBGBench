from torch import nn 
import torch
from .diffusion_scheduler import VariationalScheduler
from repo.modules.e3nn import get_e3_gnn
from repo.modules.context_emb import get_context_embedder
from .._base import register_model
from repo.utils.molecule.constants import *
from repo.utils.protein.constants import *
import torch.nn.functional as F
from repo.modules.common import compose_context, get_dict_mean
from tqdm.auto import tqdm
from ._base import BaseDiff
from torch_scatter import scatter_mean, scatter_add
import math
from ..utils.categorical import index_to_log_onehot

def zero_com_translate(x_lig_pred, batch_idx_lig, x_composed, lig_flag_composed):
    noise_lig_pred = x_lig_pred - x_composed[lig_flag_composed]
    noise_lig_pred_mean = scatter_mean(noise_lig_pred, batch_idx_lig, dim=0)[batch_idx_lig]
    noise_lig_pred = noise_lig_pred - noise_lig_pred_mean
    return noise_lig_pred


@register_model('diffsbdd')
class DiffSBDD(BaseDiff):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.cfg = cfg
        pos_scheduler_cfg = cfg.generator.pos_schedule
        self.num_classes = cfg.num_atomtype
        
        self.pos_scheduler = VariationalScheduler(self.num_diffusion_timesteps, 
                                                  type = pos_scheduler_cfg.type)
        
        
        atom_scheduler_cfg = cfg.generator.atom_schedule
        self.type_scheduler = VariationalScheduler(self.num_diffusion_timesteps,
                                                   type = atom_scheduler_cfg.type) 
            
        cfg.embedder.num_atomtype = cfg.num_atomtype
        self.context_embedder = get_context_embedder(cfg.embedder)
        
        self.denoiser = get_e3_gnn(cfg.encoder, num_classes = self.num_classes)

        self.intersect_reg = cfg.get('intersect_reg', True)

    def forward(self, batch): 
        x_lig_0 = batch['ligand_pos']
        v_lig_0 = batch['ligand_atom_type']
        x_rec_0 = batch['protein_pos']
        v_rec_0 = batch['protein_atom_feature']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))

        N_lig, _ = x_lig_0.shape
        N_rec, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        if self.training:
            t = self.sample_time(B, device = x_lig_0.device, ctn=True)
            return self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                                 lig_flag, rec_flag, batch_idx_lig, batch_idx_rec, 
                                 gen_flag_lig, gen_flag_rec, t)
        
        else:
            loss_dicts = []
            results = []
            eval_times = np.linspace(0, 
                                     self.num_diffusion_timesteps - 1, 
                                     self.cfg.get('eval_interval', 10))
            for t in eval_times:
                t = torch.tensor([t] * B).long().to(x_lig_0.device) / self.num_diffusion_timesteps 
                loss_dict, result = self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                                                  lig_flag, rec_flag, batch_idx_lig, batch_idx_rec, 
                                                  gen_flag_lig, gen_flag_rec, t)
                loss_dicts.append(loss_dict)
                results.append(result)
            
            loss_dict_mean = get_dict_mean(loss_dicts)

            return loss_dict_mean, results

    def normalize_pos(self, pos, std=1, mean=0):
        return (pos - mean) / std

    def normalize_type(self, c, std=4, mean=0):
        return (c - mean) / std

    def get_loss(self, x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                  lig_flag, rec_flag, batch_idx_lig, batch_idx_rec, 
                  gen_flag_lig, gen_flag_rec, t):
        
        x_lig_0 = self.normalize_pos(x_lig_0)
        x_rec_0 = self.normalize_pos(x_rec_0)
        
        if self.denoise_structure:
            x_lig_t, pos_noise, _ = self.pos_scheduler.forward_add_noise(x_lig_0, t, batch_idx_lig, gen_flag_lig, zero_center=True)
        else:
            x_lig_t = x_lig_0

        # c_lig_0 = F.one_hot(v_lig_0, num_classes = self.num_classes).float()
        # c_lig_0 = self.normalize_type(c_lig_0)

        c_lig_0 = (index_to_log_onehot(v_lig_0, num_classes=self.num_classes))
        c_lig_0 = (c_lig_0 - c_lig_0.mean(-1, keepdim=True)) / c_lig_0.std(-1, keepdim=True)

        if self.denoise_atom:
            c_lig_t, type_noise = self.type_scheduler.forward_add_noise(c_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            c_lig_t = c_lig_0

        x_lig_t, x_rec_t, h_lig_t, h_rec_t = self.context_embedder(x_lig_t, x_rec_0, c_lig_t, v_rec_0, aa_rec_0, 
                                                                  batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
        
        context_composed, batch_idx, _ = compose_context({'x': x_lig_t, 'h': h_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                         {'x': x_rec_t, 'h': h_rec_t, 'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                         batch_idx_lig, batch_idx_rec)
        
        x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)
        x_lig_pred = x[context_composed['lig_flag']]
        c_lig_pred = v[context_composed['lig_flag']]

        x_composed, lig_flag_composed = context_composed['x'], context_composed['lig_flag']
        x_lig_pred = zero_com_translate(x_lig_pred, batch_idx_lig, x_composed, lig_flag_composed)
        if self.denoise_structure:    
            loss_pos, pos_info = self.pos_scheduler.get_score_loss(x_lig_pred, pos_noise, t, 
                                                                   gen_flag_lig, batch_idx_lig, 
                                                                   score_in=False, info_tag='pos')
        else:
            loss_pos, pos_info = torch.tensor(0).float(), {}

        if self.denoise_atom:
            loss_atom, atom_info = self.type_scheduler.get_score_loss(c_lig_pred, type_noise, t, 
                                                                      gen_flag_lig, batch_idx_lig, 
                                                                      score_in=False, info_tag='atom')
        else:
            loss_atom, atom_info = torch.tensor(0).float(), {}

        results = {}
        results.update(pos_info)
        results.update(atom_info)

        return {'pos': loss_pos, 'atom': loss_atom}, results
    
    def unnormalize_pos(self, x, std=1, mean=0):
        return x * std + mean
    
    def unnormalize_type(self, c, std=4, mean=0):
        return c * std + mean

    def zero_time_loss(self, c_lig_0, c_lig_t, pos_noise, x_lig_pred, t, gen_flag_lig, batch_idx_lig, epsilon=1e-7): 
        gamma = self.type_scheduler.gamma(t)[batch_idx_lig]
        sigma_0 = self.unnormalize_type(torch.sqrt(torch.sigmoid(gamma))) 

        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            (pos_noise - x_lig_pred) ** 2
        )

        ligand_onehot = self.unnormalize_type(c_lig_0)
        estimated_ligand_onehot = self.unnormalize_type(c_lig_t)
        centered_ligand_onehot = estimated_ligand_onehot - 1

        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian((centered_ligand_onehot + 0.5) / sigma_0.unsqueeze(-1))
            - self.cdf_standard_gaussian((centered_ligand_onehot - 0.5) / sigma_0.unsqueeze(-1))
            + epsilon
        )

        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1, keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z

        log_ph_given_z0_ligand = log_probabilities_ligand * ligand_onehot

        t_zero_mask = (t == 0)[batch_idx_lig]
        gen_mask = torch.logical_and(gen_flag_lig, t_zero_mask)

        if gen_mask.sum() > 0:
            loss_pos = - scatter_mean(log_p_x_given_z0_without_constants_ligand.sum(-1)[gen_mask],
                                      batch_idx_lig[gen_mask], dim=0)
            loss_atom = - scatter_mean(log_ph_given_z0_ligand.sum(-1)[gen_mask], 
                                       batch_idx_lig[gen_mask], dim=0)
        else:
            loss_pos = torch.zeros_like(log_p_x_given_z0_without_constants_ligand.sum(-1))
            loss_atom = torch.zeros_like(log_ph_given_z0_ligand.sum(-1))
        
        return loss_pos.mean(0), loss_atom.mean(0)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


    def sample(self, batch):
        x_lig_in = batch['ligand_pos']
        v_lig_in = batch['ligand_atom_type']
        x_rec_0 = batch['protein_pos']
        v_rec_0 = batch['protein_atom_feature']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))

        aa_rec_0 = F.one_hot(aa_rec_0, num_classes = len(aa_name_number)).float()
        c_lig_in = v_lig_in
        x_rec_0 = self.normalize_pos(x_rec_0)
        c_lig_in = torch.where(gen_flag_lig.unsqueeze(-1), c_lig_in, self.normalize_type(c_lig_in))

        time_seq = list(reversed(range(1, self.num_diffusion_timesteps)))
        N_lig, _ = x_lig_in.shape
        N_rec, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        traj = {self.num_diffusion_timesteps - 1: (x_lig_in, c_lig_in, batch_idx_lig)}

        for t_idx in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            x_lig, c_lig, _ = traj[t_idx]

            t = torch.full(size=(B,), fill_value=t_idx/self.num_diffusion_timesteps, device=x_lig_in.device)

            x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0, 
                                                              batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
        
            context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': gen_flag_lig, 'lig_flag':lig_flag},
                                                             {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                             batch_idx_lig, batch_idx_rec)
            
            x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

            x_lig_out = x[context_composed['lig_flag']]
            c_lig_out = v[context_composed['lig_flag']]
            x_composed, lig_flag_composed = context_composed['x'], context_composed['lig_flag']
            x_lig_pred = zero_com_translate(x_lig_out, batch_idx_lig, x_composed, lig_flag_composed)

            if self.denoise_structure:    
                x_lig_next = self.pos_scheduler.backward_remove_noise(x_lig, x_lig_pred, t, 
                                                                      batch_idx_lig, gen_flag_lig, zero_mean=True)
            else:
                x_lig_next = x_lig
                
            if self.denoise_atom:
                c_lig_next = self.type_scheduler.backward_remove_noise(c_lig, c_lig_out, t, 
                                                                       batch_idx_lig, gen_flag_lig, zero_mean=False)
            else:
                c_lig_next = c_lig
            
            traj[t_idx - 1] = (x_lig_next.clone(), c_lig_next.clone(), batch_idx_lig)
            traj[t_idx] = tuple(x.cpu() for x in traj[t_idx]) 

        x_lig, c_lig = self.sample_p_xh_given_z0(x_lig, c_lig, x_rec_0, v_rec_0, aa_rec_0, 
                                                 batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, 
                                                 gen_flag_lig, gen_flag_rec)
        # c_lig[:,1][gen_flag_lig] = c_lig[:,1][gen_flag_lig] + 500
        traj[0] = (x_lig.cpu(), c_lig.cpu(), batch_idx_lig.cpu())    
        return traj

    def sample_p_xh_given_z0(self, x_lig, c_lig, x_rec_0, v_rec_0, aa_rec_0, 
                             batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, 
                             gen_flag_lig, gen_flag_rec):
        B = batch_idx_lig.max() + 1
        t_zeros = torch.zeros(B).to(x_lig)
        
        gamma_0 = self.type_scheduler.gamma(t_zeros[batch_idx_lig]).unsqueeze(-1)
        sigma_0 = torch.exp(0.5 * gamma_0)

        x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0, 
                                                           batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_zeros)

        context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': gen_flag_lig, 'lig_flag':lig_flag},
                                                         {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                         batch_idx_lig, batch_idx_rec)
        
        x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

        x_lig_out = x[context_composed['lig_flag']]
        c_lig_out = v[context_composed['lig_flag']]
        x_composed, lig_flag_composed = context_composed['x'], context_composed['lig_flag']
        x_lig_pred = zero_com_translate(x_lig_out, batch_idx_lig, x_composed, lig_flag_composed)

        mu_x_lig = self.compute_pred(x_lig_pred, x_lig, gamma_0, gen_flag_lig)
        mu_c_lig = self.compute_pred(c_lig_out, c_lig, gamma_0, gen_flag_lig)

        x_lig = torch.where(gen_flag_lig.unsqueeze(-1), torch.randn_like(mu_x_lig) * sigma_0 + mu_x_lig, x_lig)
        c_lig = torch.where(gen_flag_lig.unsqueeze(-1), torch.randn_like(mu_c_lig) * sigma_0 + mu_c_lig, c_lig)

        x_lig = x_lig - scatter_mean(x_lig, batch_idx_lig, dim=0)[batch_idx_lig]

        x_lig = self.unnormalize_pos(x_lig)
        c_lig[:,1][gen_flag_lig] = c_lig[:,1][gen_flag_lig] + 0.5 # C type correction
        c_lig = self.unnormalize_type(c_lig)
        
        return x_lig, c_lig


    def compute_pred(self, net_out_lig, zt, gamma_t, gen_flag_lig):
        """Commputes x_pred, i.e. the most likely prediction of x."""

        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))

        eps_t = net_out_lig
        x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)

        return torch.where(gen_flag_lig.unsqueeze(-1), x_pred, zt)