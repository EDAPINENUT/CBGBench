from torch import nn 
import torch
from .diffusion_scheduler import CTNVPScheduler, TypeVPScheduler, RotVPScheduler
from repo.modules.e3nn import get_e3_gnn
from repo.modules.context_emb import get_context_embedder
from .._base import register_model
from repo.utils.molecule.constants import *
from repo.utils.protein.constants import *
import torch.nn.functional as F
from repo.modules.common import compose_context, get_dict_mean
from tqdm.auto import tqdm
from ._base import BaseDiff
from ..utils.so3 import *
from torch_scatter import scatter_mean

def rotation_matrix_cosine_loss(R_pred, R_true, t, gen_flag, batch_idx):
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    cosine_loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    cosine_loss = cosine_loss.reshape(size + [3]).sum(dim=-1)
    loss = scatter_mean((cosine_loss[gen_flag]), batch_idx[gen_flag], dim=0)

    rot_info = {'R0': R_true, 'R_pred': R_pred, 'mask_gen': gen_flag}

    return loss.mean(), rot_info

@register_model('difffg')
class D3FG(BaseDiff):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.cfg = cfg
        self.num_classes = cfg.num_fgtype
        
        pos_scheduler_cfg = cfg.generator.pos_schedule
        self.pos_scheduler = CTNVPScheduler(self.num_diffusion_timesteps, 
                                            beta_start = pos_scheduler_cfg.beta_start, 
                                            beta_end = pos_scheduler_cfg.beta_end, 
                                            type = pos_scheduler_cfg.type)

        rot_scheduler_cfg = cfg.generator.rot_schedule
        self.rot_scheduler = RotVPScheduler(self.num_diffusion_timesteps,
                                            type = rot_scheduler_cfg.type,
                                            cosine_s = rot_scheduler_cfg.cosine_s)        

        
        fg_scheduler_cfg = cfg.generator.fg_schedule
        self.type_scheduler = TypeVPScheduler(self.num_diffusion_timesteps,
                                              num_classes = self.num_classes,
                                              type = fg_scheduler_cfg.type,
                                              cosine_s = fg_scheduler_cfg.cosine_s)
        
        cfg.embedder.num_fgtype = cfg.num_fgtype
        self.context_embedder = get_context_embedder(cfg.embedder)
        
        self.denoiser = get_e3_gnn(cfg.encoder, num_classes = self.num_classes)


    def forward(self, batch): 
        x_lig_0 = batch['ligand_pos_heavyatom']
        v_lig_0 = batch['ligand_type_fg']
        x_rec_0 = batch['protein_pos_heavyatom']
        v_rec_0 = batch['protein_type_fg']
        o_lig_0 = batch['ligand_o_fg']
        aa_rec_0 = batch['protein_aa']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_type_fg_batch']
        batch_idx_rec = batch['protein_type_fg_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))
        res_nb = batch['protein_res_nb']
        chain_nb = []
        chain_cumsum = batch['protein_num_chains'].cumsum(0)
        for i in batch_idx_rec.unique():
            chain_nb.append(batch['protein_chain_nb'][batch_idx_rec == i] + chain_cumsum[i] - 1)
        chain_nb = torch.cat(chain_nb)

        mask_atom_rec = batch['protein_mask_heavyatom']
        mask_atom_lig = batch['ligand_mask_heavyatom']

        N_lig, _, _ = x_lig_0.shape
        N_rec, _, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        if self.training:
            t = self.sample_time(B, device = x_lig_0.device)
            return self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, o_lig_0, aa_rec_0,
                                 lig_flag, rec_flag, res_nb, chain_nb, mask_atom_lig, mask_atom_rec,
                                 batch_idx_lig, batch_idx_rec, gen_flag_lig, gen_flag_rec, t)
        
        else:
            loss_dicts = []
            results = []
            eval_times = np.linspace(0, 
                                     self.num_diffusion_timesteps - 1, 
                                     self.cfg.get('eval_interval', 10))
            for t in eval_times:
                t = torch.tensor([t] * B).long().to(x_lig_0.device)
                loss_dict, result = self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, o_lig_0, aa_rec_0,
                                                  lig_flag, rec_flag, res_nb, chain_nb, mask_atom_lig, mask_atom_rec,
                                                  batch_idx_lig, batch_idx_rec, gen_flag_lig, gen_flag_rec, t)
                loss_dicts.append(loss_dict)
                results.append(result)
            
            loss_dict_mean = get_dict_mean(loss_dicts)

            return loss_dict_mean, results

    def get_loss(self, x_lig_0, x_rec_0, v_lig_0, v_rec_0, o_lig_0, aa_rec_0,
                  lig_flag, rec_flag, res_nb, chain_nb, mask_atom_lig, mask_atom_rec, 
                  batch_idx_lig, batch_idx_rec, gen_flag_lig, gen_flag_rec, t):
        
        xc_lig_0 = x_lig_0[:, BBHeavyAtom.CA]
        R_lig_0 = so3vec_to_rotation(o_lig_0)
        
        if self.denoise_structure:
            xc_lig_t, pos_noise = self.pos_scheduler.forward_add_noise(xc_lig_0, t, batch_idx_lig, gen_flag_lig)
            o_lig_t, _ = self.rot_scheduler.forward_add_noise(o_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            xc_lig_t = xc_lig_0
            o_lig_t = o_lig_0

        if self.denoise_atom:
            c_lig_t, v_lig_t = self.type_scheduler.forward_add_noise(v_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            c_lig_t = F.one_hot(v_lig_0, num_classes = self.num_classes).float()

        xc_lig_t, xc_rec_t, o_rec_t, h_lig_t, h_rec_t = self.context_embedder(xc_lig_t, x_rec_0, c_lig_t, v_rec_0, aa_rec_0,
                                                                              res_nb, chain_nb, mask_atom_lig, mask_atom_rec, 
                                                                              batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
        
        context_composed, batch_idx, _ = compose_context({'x': xc_lig_t, 'h': h_lig_t, 'o': o_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                         {'x': xc_rec_t, 'h': h_rec_t, 'o': o_rec_t,'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                         batch_idx_lig, batch_idx_rec)
        
        x, h, o, R, v = self.denoiser(batch_idx=batch_idx, **context_composed)
        x_lig_pred = x[context_composed['lig_flag']]
        c_lig_pred = v[context_composed['lig_flag']]
        R_lig_pred = R[context_composed['lig_flag']]


        if self.denoise_structure:    
            loss_pos, pos_info = self.pos_scheduler.get_score_loss(x_lig_pred, pos_noise, t, 
                                                                   gen_flag_lig, batch_idx_lig, 
                                                                   score_in=False)
            
            loss_rot, rot_info = rotation_matrix_cosine_loss(R_lig_pred, R_lig_0, t, gen_flag_lig, batch_idx_lig) # (N, L)
        else:
            loss_pos, pos_info = torch.tensor(0).float(), {}
            loss_rot, rot_info = torch.tensor(0).float(), {}

        if self.denoise_atom:
            loss_fg, fg_info = self.type_scheduler.get_loss(c_lig_pred, v_lig_0, v_lig_t, t, 
                                                                gen_flag_lig, batch_idx_lig, 
                                                                pred_logit=True)
        else:
            loss_fg, fg_info = torch.tensor(0).float(), {}

        results = {}
        results.update(pos_info)
        results.update(fg_info)
        results.update(rot_info)

        return {'pos': loss_pos, 'rot': loss_rot,  'fg': loss_fg}, results


    def sample(self, batch):
        xc_lig_0 = batch['ligand_pos_heavyatom'][:, BBHeavyAtom.CA]
        v_lig_0 = batch['ligand_type_fg']
        x_rec_0 = batch['protein_pos_heavyatom']
        v_rec_0 = batch['protein_type_fg']
        o_lig_0 = batch['ligand_o_fg']
        aa_rec_0 = batch['protein_aa']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_type_fg_batch']
        batch_idx_rec = batch['protein_type_fg_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))
        res_nb = batch['protein_res_nb']
        chain_nb = []
        chain_cumsum = batch['protein_num_chains'].cumsum(0)
        for i in batch_idx_rec.unique():
            chain_nb.append(batch['protein_chain_nb'][batch_idx_rec == i] + chain_cumsum[i] - 1)
        chain_nb = torch.cat(chain_nb)

        mask_atom_rec = batch['protein_mask_heavyatom']
        mask_atom_lig = batch.get('ligand_mask_heavyatom', None)

        c_lig_0 = F.one_hot(v_lig_0, num_classes = self.num_classes).float()

        time_seq = list(reversed(range(0, self.num_diffusion_timesteps)))
        N_lig, _ = xc_lig_0.shape
        N_rec, _, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        traj = {self.num_diffusion_timesteps - 1: (xc_lig_0, c_lig_0, o_lig_0, batch_idx_lig)}

        for t_idx in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            xc_lig, c_lig, o_lig, _ = traj[t_idx]

            t = torch.full(size=(B,), fill_value=t_idx, dtype=torch.long, device=xc_lig_0.device)

            xc_lig_t, xc_rec_t, o_rec_t, h_lig_t, h_rec_t = self.context_embedder(xc_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0,
                                                                                res_nb, chain_nb, mask_atom_lig, mask_atom_rec, 
                                                                                batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
            o_lig_t = o_lig.clone()

            context_composed, batch_idx, _ = compose_context({'x': xc_lig_t, 'h': h_lig_t, 'o': o_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                             {'x': xc_rec_t, 'h': h_rec_t, 'o': o_rec_t,'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                             batch_idx_lig, batch_idx_rec)
            
            x, h, o, R, v = self.denoiser(batch_idx=batch_idx, **context_composed)

            xc_lig_out = x[context_composed['lig_flag']]
            c_lig_out = v[context_composed['lig_flag']]
            R_lig_out = R[context_composed['lig_flag']]
            o_lig_out = o[context_composed['lig_flag']]

            if self.denoise_structure:    
                xc_lig_next = self.pos_scheduler.backward_remove_noise(xc_lig_out, xc_lig, t, 
                                                                       batch_idx_lig, gen_flag_lig)
                o_lig_next = self.rot_scheduler.backward_remove_noise(o_lig_out, o_lig_t, t, 
                                                                      batch_idx_lig, gen_flag_lig)
            else:
                xc_lig_next = xc_lig
                o_lig_next = o_lig
                
            if self.denoise_atom:
                c_lig_next, _ = self.type_scheduler.backward_remove_noise(c_lig_out, c_lig, t, 
                                                                          batch_idx_lig, gen_flag_lig, 
                                                                          pred_logit=True)
            else:
                c_lig_next = c_lig
            
            traj[t_idx - 1] = (xc_lig_next, c_lig_next, o_lig_next, batch_idx_lig)
            traj[t_idx] = tuple(x.cpu() for x in traj[t_idx]) 

        return traj



@register_model('difffg_v2')
class D3FG(BaseDiff):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.cfg = cfg
        self.num_classes = cfg.num_fgtype
        
        pos_scheduler_cfg = cfg.generator.pos_schedule
        self.pos_scheduler = CTNVPScheduler(self.num_diffusion_timesteps, 
                                            beta_start = pos_scheduler_cfg.beta_start, 
                                            beta_end = pos_scheduler_cfg.beta_end, 
                                            type = pos_scheduler_cfg.type)

        rot_scheduler_cfg = cfg.generator.rot_schedule
        self.rot_scheduler = RotVPScheduler(self.num_diffusion_timesteps,
                                            type = rot_scheduler_cfg.type,
                                            cosine_s = rot_scheduler_cfg.cosine_s)        

        
        fg_scheduler_cfg = cfg.generator.fg_schedule
        self.type_scheduler = TypeVPScheduler(self.num_diffusion_timesteps,
                                              num_classes = self.num_classes,
                                              type = fg_scheduler_cfg.type,
                                              cosine_s = fg_scheduler_cfg.cosine_s)
        
        cfg.embedder.num_fgtype = cfg.num_fgtype
        self.context_embedder = get_context_embedder(cfg.embedder)
        
        self.denoiser = get_e3_gnn(cfg.encoder, num_classes = self.num_classes)


    def forward(self, batch): 
        x_lig_0 = batch['ligand_pos_heavyatom']
        v_lig_0 = batch['ligand_type_fg']
        x_rec_0 = batch['protein_pos_heavyatom']
        v_rec_0 = batch['protein_type_fg']
        o_lig_0 = batch['ligand_o_fg']
        aa_rec_0 = batch['protein_aa']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_type_fg_batch']
        batch_idx_rec = batch['protein_type_fg_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))
        res_nb = batch['protein_res_nb']
        chain_nb = []
        chain_cumsum = batch['protein_num_chains'].cumsum(0)
        for i in batch_idx_rec.unique():
            chain_nb.append(batch['protein_chain_nb'][batch_idx_rec == i] + chain_cumsum[i] - 1)
        chain_nb = torch.cat(chain_nb)

        mask_atom_rec = batch['protein_mask_heavyatom']
        mask_atom_lig = batch['ligand_mask_heavyatom']

        N_lig, _, _ = x_lig_0.shape
        N_rec, _, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        if self.training:
            t = self.sample_time(B, device = x_lig_0.device)
            return self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, o_lig_0, aa_rec_0,
                                 lig_flag, rec_flag, res_nb, chain_nb, mask_atom_lig, mask_atom_rec,
                                 batch_idx_lig, batch_idx_rec, gen_flag_lig, gen_flag_rec, t)
        
        else:
            loss_dicts = []
            results = []
            eval_times = np.linspace(0, 
                                     self.num_diffusion_timesteps - 1, 
                                     self.cfg.get('eval_interval', 10))
            for t in eval_times:
                t = torch.tensor([t] * B).long().to(x_lig_0.device)
                loss_dict, result = self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, o_lig_0, aa_rec_0,
                                                  lig_flag, rec_flag, res_nb, chain_nb, mask_atom_lig, mask_atom_rec,
                                                  batch_idx_lig, batch_idx_rec, gen_flag_lig, gen_flag_rec, t)
                loss_dicts.append(loss_dict)
                results.append(result)
            
            loss_dict_mean = get_dict_mean(loss_dicts)

            return loss_dict_mean, results

    def get_loss(self, x_lig_0, x_rec_0, v_lig_0, v_rec_0, o_lig_0, aa_rec_0,
                  lig_flag, rec_flag, res_nb, chain_nb, mask_atom_lig, mask_atom_rec, 
                  batch_idx_lig, batch_idx_rec, gen_flag_lig, gen_flag_rec, t):
        
        xc_lig_0 = x_lig_0[:, BBHeavyAtom.CA]
        R_lig_0 = so3vec_to_rotation(o_lig_0)
        
        if self.denoise_structure:
            xc_lig_t, pos_noise = self.pos_scheduler.forward_add_noise(xc_lig_0, t, batch_idx_lig, gen_flag_lig)
            o_lig_t, _ = self.rot_scheduler.forward_add_noise(o_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            xc_lig_t = xc_lig_0
            o_lig_t = o_lig_0

        if self.denoise_atom:
            c_lig_t, v_lig_t = self.type_scheduler.forward_add_noise(v_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            c_lig_t = F.one_hot(v_lig_0, num_classes = self.num_classes).float()

        xc_lig_t, xc_rec_t, o_rec_t, h_lig_t, h_rec_t = self.context_embedder(xc_lig_t, x_rec_0, c_lig_t, v_rec_0, aa_rec_0,
                                                                              res_nb, chain_nb, mask_atom_lig, mask_atom_rec, 
                                                                              batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
        
        context_composed, batch_idx, _ = compose_context({'x': xc_lig_t, 'h': h_lig_t, 'o': o_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                         {'x': xc_rec_t, 'h': h_rec_t, 'o': o_rec_t,'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                         batch_idx_lig, batch_idx_rec)
        
        x, h, o, R, v = self.denoiser(batch_idx=batch_idx, **context_composed)
        x_lig_pred = x[context_composed['lig_flag']]
        c_lig_pred = v[context_composed['lig_flag']]
        R_lig_pred = R[context_composed['lig_flag']]


        if self.denoise_structure:    
            loss_pos, pos_info = self.pos_scheduler.get_loss(x_lig_pred, xc_lig_0, xc_lig_t, t, 
                                                             gen_flag_lig, batch_idx_lig, 
                                                             type='denoise')
            
            loss_rot, rot_info = rotation_matrix_cosine_loss(R_lig_pred, R_lig_0, t, gen_flag_lig, batch_idx_lig) # (N, L)
        else:
            loss_pos, pos_info = torch.tensor(0).float(), {}
            loss_rot, rot_info = torch.tensor(0).float(), {}

        if self.denoise_atom:
            loss_fg, fg_info = self.type_scheduler.get_loss(c_lig_pred, v_lig_0, v_lig_t, t, 
                                                                gen_flag_lig, batch_idx_lig, 
                                                                pred_logit=True)
        else:
            loss_fg, fg_info = torch.tensor(0).float(), {}

        results = {}
        results.update(pos_info)
        results.update(fg_info)
        results.update(rot_info)

        return {'pos': loss_pos, 'rot': loss_rot,  'fg': loss_fg}, results


    def sample(self, batch):
        xc_lig_0 = batch['ligand_pos_heavyatom'][:, BBHeavyAtom.CA]
        v_lig_0 = batch['ligand_type_fg']
        x_rec_0 = batch['protein_pos_heavyatom']
        v_rec_0 = batch['protein_type_fg']
        o_lig_0 = batch['ligand_o_fg']
        aa_rec_0 = batch['protein_aa']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_type_fg_batch']
        batch_idx_rec = batch['protein_type_fg_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))
        res_nb = batch['protein_res_nb']
        chain_nb = []
        chain_cumsum = batch['protein_num_chains'].cumsum(0)
        for i in batch_idx_rec.unique():
            chain_nb.append(batch['protein_chain_nb'][batch_idx_rec == i] + chain_cumsum[i] - 1)
        chain_nb = torch.cat(chain_nb)

        mask_atom_rec = batch['protein_mask_heavyatom']
        mask_atom_lig = batch.get('ligand_mask_heavyatom', None)

        c_lig_0 = F.one_hot(v_lig_0, num_classes = self.num_classes).float()

        time_seq = list(reversed(range(0, self.num_diffusion_timesteps)))
        N_lig, _ = xc_lig_0.shape
        N_rec, _, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        traj = {self.num_diffusion_timesteps - 1: (xc_lig_0, c_lig_0, o_lig_0, batch_idx_lig)}

        for t_idx in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            xc_lig, c_lig, o_lig, _ = traj[t_idx]

            t = torch.full(size=(B,), fill_value=t_idx, dtype=torch.long, device=xc_lig_0.device)

            xc_lig_t, xc_rec_t, o_rec_t, h_lig_t, h_rec_t = self.context_embedder(xc_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0,
                                                                                res_nb, chain_nb, mask_atom_lig, mask_atom_rec, 
                                                                                batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
            o_lig_t = o_lig.clone()

            context_composed, batch_idx, _ = compose_context({'x': xc_lig_t, 'h': h_lig_t, 'o': o_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                             {'x': xc_rec_t, 'h': h_rec_t, 'o': o_rec_t,'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                             batch_idx_lig, batch_idx_rec)
            
            x, h, o, R, v = self.denoiser(batch_idx=batch_idx, **context_composed)

            xc_lig_out = x[context_composed['lig_flag']]
            c_lig_out = v[context_composed['lig_flag']]
            R_lig_out = R[context_composed['lig_flag']]
            o_lig_out = o[context_composed['lig_flag']]

            if self.denoise_structure:    
                xc_lig_next = self.pos_scheduler.backward_remove_noise(xc_lig_out, xc_lig, t, 
                                                                       batch_idx_lig, gen_flag_lig)
                o_lig_next = self.rot_scheduler.backward_remove_noise(o_lig_out, o_lig_t, t, 
                                                                      batch_idx_lig, gen_flag_lig)
            else:
                xc_lig_next = xc_lig
                o_lig_next = o_lig
                
            if self.denoise_atom:
                c_lig_next, _ = self.type_scheduler.backward_remove_noise(c_lig_out, c_lig, t, 
                                                                          batch_idx_lig, gen_flag_lig, 
                                                                          pred_logit=True)
            else:
                c_lig_next = c_lig
            
            traj[t_idx - 1] = (xc_lig_next, c_lig_next, o_lig_next, batch_idx_lig)
            traj[t_idx] = tuple(x.cpu() for x in traj[t_idx]) 

        return traj

            



