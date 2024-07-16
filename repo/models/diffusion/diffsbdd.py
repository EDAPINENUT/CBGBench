from torch import nn 
import torch
from .diffusion_scheduler import VariationalScheduler, DiffsbddVariationalScheduler
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
        
        self.pos_scheduler = DiffsbddVariationalScheduler(self.num_diffusion_timesteps, 
                                                  type = pos_scheduler_cfg.type)
        
        
        atom_scheduler_cfg = cfg.generator.atom_schedule
        self.type_scheduler = DiffsbddVariationalScheduler(self.num_diffusion_timesteps,
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
            eval_times = np.linspace(1, 
                                     self.num_diffusion_timesteps, 
                                     self.cfg.get('eval_interval', 10))
            for t in eval_times:
                t = torch.tensor([t] * B).long().to(x_lig_0.device)
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

        c_lig_0 = F.one_hot(v_lig_0, self.num_classes)
        c_lig_0 = self.normalize_type(c_lig_0)
        v_rec_0 = self.normalize_type(v_rec_0)

        s_int = t - 1
        t_is_zero = (t == 0).float()
        t_is_not_zero = 1 - t_is_zero
        s = s_int / self.num_diffusion_timesteps
        t = t / self.num_diffusion_timesteps

        if self.denoise_structure:
            x_lig_0, x_rec_0 = self.pos_scheduler.remove_mean_batch(x_lig_0, x_rec_0, batch_idx_lig, batch_idx_rec)
            x_lig_t, pos_noise, x_rec_t = self.pos_scheduler.forward_pos_center_noise((x_lig_0, x_rec_0), t, (batch_idx_lig, batch_idx_rec), gen_flag_lig, zero_center=False)

        if self.denoise_atom:
            c_lig_t, type_noise = self.type_scheduler.forward_type_add_noise(c_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            c_lig_t = c_lig_0

        x_lig_t, x_rec_t, h_lig_t, h_rec_t = self.context_embedder(x_lig_t, x_rec_t, c_lig_t, v_rec_0, aa_rec_0, 
                                                                  batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
        
        context_composed, batch_idx, _ = compose_context({'x': x_lig_t, 'h': h_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                         {'x': x_rec_t, 'h': h_rec_t, 'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                         batch_idx_lig, batch_idx_rec)
        
        x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)
        x_lig_pred = x[context_composed['lig_flag']]
        c_lig_pred = v[context_composed['lig_flag']]

        x_lig_pred_non_training = None
        pos_noise_non_training = None
        c_lig_pred_non_training = None
        type_noise_non_training = None

        if not self.training:
            t_zeros = torch.zeros_like(s)
            if self.denoise_structure:
                x_lig_t, pos_noise_non_training, x_rec_t = self.pos_scheduler.forward_pos_center_noise((x_lig_0, x_rec_0), t_zeros, (batch_idx_lig, batch_idx_rec), gen_flag_lig, zero_center=False)

            if self.denoise_atom:
                c_lig_t, type_noise_non_training = self.type_scheduler.forward_type_add_noise(c_lig_0, t_zeros, batch_idx_lig, gen_flag_lig)
            else:
                c_lig_t = c_lig_0
            
            x_lig_t, x_rec_t, h_lig_t, h_rec_t = self.context_embedder(x_lig_t, x_rec_t, c_lig_t, v_rec_0, aa_rec_0, 
                                                                    batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_zeros)
            
            context_composed, batch_idx, _ = compose_context({'x': x_lig_t, 'h': h_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                            {'x': x_rec_t, 'h': h_rec_t, 'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                            batch_idx_lig, batch_idx_rec)
            
            x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)
            x_lig_pred_non_training = x[context_composed['lig_flag']]
            c_lig_pred_non_training = v[context_composed['lig_flag']]

        if self.denoise_structure:    
            loss_pos, pos_info = self.pos_scheduler.get_score_loss(x_lig_pred, pos_noise, t, 
                                                                   gen_flag_lig, batch_idx_lig, 
                                                                   score_in=False, info_tag='pos', 
                                                                   s=s, x_lig_0=x_lig_0, 
                                                                   compute_continus=True,
                                                                   t_is_zero=t_is_zero,
                                                                   t_is_not_zero=t_is_not_zero,
                                                                   x_pred_t_non_training=x_lig_pred_non_training,
                                                                   x_tgt_t_non_training=pos_noise_non_training,
                                                                   )
        else:
            loss_pos, pos_info = torch.tensor(0).float(), {}

        if self.denoise_atom:
            loss_atom, atom_info = self.type_scheduler.get_score_loss(c_lig_pred, type_noise, t, 
                                                                      gen_flag_lig, batch_idx_lig, 
                                                                      score_in=False, info_tag='atom',
                                                                      s=s, c_lig_0=c_lig_0, 
                                                                      c_lig_t=c_lig_t, compute_discrete=True,
                                                                      t_is_zero=t_is_zero,
                                                                      t_is_not_zero=t_is_not_zero,      
                                                                      c_pred_t_non_training=c_lig_pred_non_training,
                                                                      c_tgt_t_non_training=type_noise_non_training,
                                                                      )
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

        x_rec_0 = self.normalize_pos(x_rec_0)
        v_rec_0 = self.normalize_type(v_rec_0)

        n_samples = batch_idx_lig.max() + 1
        mu_lig_X = scatter_mean(x_rec_0, batch_idx_rec, dim=0)[batch_idx_lig]
        mu_lig_h = torch.zeros((n_samples, self.num_classes), device=x_rec_0.device)[batch_idx_lig]
        sigma = torch.ones_like(torch.bincount(batch_idx_rec)).unsqueeze(1)

        x_lig_in, x_rec_0 = self.pos_scheduler.sample_normal_zero_com(mu_lig_X, x_rec_0, sigma, batch_idx_lig, batch_idx_rec, com=True)

        v_lig_in = self.pos_scheduler.sample_normal_zero_com(mu_lig_h, v_rec_0, sigma, batch_idx_lig, batch_idx_rec)
        
        self.pos_scheduler.assert_mean_zero_with_mask(x_lig_in, batch_idx_lig)

        c_lig_in = v_lig_in

        time_seq = list(reversed(range(0, self.num_diffusion_timesteps)))

        N_lig, _ = x_lig_in.shape
        N_rec, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        traj = {self.num_diffusion_timesteps - 1: (x_lig_in, c_lig_in, batch_idx_lig)}

        for t_idx in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            x_lig, c_lig, _ = traj[t_idx]
            s_array = torch.full((n_samples,), fill_value=t_idx,
                                 device=x_lig.device)
            t_array = s_array + 1
            s_array = s_array / self.num_diffusion_timesteps
            t_array = t_array / self.num_diffusion_timesteps

            x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0, 
                                                              batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_array)
        
            context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': gen_flag_lig, 'lig_flag':lig_flag},
                                                             {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                             batch_idx_lig, batch_idx_rec)
            
            x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

            x_lig_pred = x[context_composed['lig_flag']]
            c_lig_out = v[context_composed['lig_flag']]

            if self.denoise_structure:
                x_lig_next, x_rec_0 = self.pos_scheduler.sample_p_zs_given_zt(
                            s_array, t_array, x_lig, x_rec_0, batch_idx_lig, batch_idx_rec, x_lig_pred, com=True)
            else:
                x_lig_next = x_lig

            if self.denoise_atom:
                c_lig_next, v_rec_0 = self.pos_scheduler.sample_p_zs_given_zt(
                            s_array, t_array, c_lig, v_rec_0, batch_idx_lig, batch_idx_rec, c_lig_out, com=False)
            else:
                c_lig_next = c_lig
            
            traj[t_idx - 1] = (x_lig_next.clone(), c_lig_next.clone(), batch_idx_lig)
            traj[t_idx] = tuple(x.cpu() for x in traj[t_idx]) 


        x_lig, c_lig = self.sample_p_xh_given_z0(x_lig_next, c_lig_next, x_rec_0, v_rec_0, aa_rec_0, 
                                                 batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, 
                                                 gen_flag_lig, gen_flag_rec)


        traj[0] = (x_lig.cpu(), c_lig.cpu(), batch_idx_lig.cpu())    
        return traj

    def sample_p_xh_given_z0(self, x_lig, c_lig, x_rec_0, v_rec_0, aa_rec_0, 
                             batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, 
                             gen_flag_lig, gen_flag_rec):
        B = batch_idx_lig.max() + 1
        t_zeros = torch.zeros(size=(B, )).to(x_lig)
        gamma_0 = self.pos_scheduler.gamma(t_zeros)
        sigma_0 = torch.exp(0.5 * gamma_0).unsqueeze(1)
        

        x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0, 
                                                           batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_zeros)

        context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': gen_flag_lig, 'lig_flag':lig_flag},
                                                         {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                         batch_idx_lig, batch_idx_rec)
        
        x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

        x_lig_pred = x[context_composed['lig_flag']]
        c_lig_out = v[context_composed['lig_flag']]

        mu_x_lig = self.compute_pred(x_lig_pred, x_lig, gamma_0, batch_idx_lig)
        mu_c_lig = self.compute_pred(c_lig_out, c_lig, gamma_0, batch_idx_lig)

        x_lig_in, _ = self.pos_scheduler.sample_normal_zero_com(mu_x_lig, x_rec_0, sigma_0, batch_idx_lig, batch_idx_rec, com=True)

        v_lig_in = self.pos_scheduler.sample_normal_zero_com(mu_c_lig, v_rec_0, sigma_0, batch_idx_lig, batch_idx_rec)        

        x_lig = self.unnormalize_pos(x_lig_in)
        c_lig = self.unnormalize_type(c_lig)
        
        return x_lig, c_lig


    def compute_pred(self, net_out_lig, zt, gamma_t, batch_idx_lig):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.pos_scheduler.sigma(gamma_t, target_tensor=net_out_lig)
        alpha_t = self.pos_scheduler.alpha(gamma_t, target_tensor=net_out_lig)
        eps_t = net_out_lig
        x_pred = 1. / alpha_t[batch_idx_lig] * (zt - sigma_t[batch_idx_lig] * eps_t)
        return x_pred