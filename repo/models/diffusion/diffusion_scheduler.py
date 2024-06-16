from torch import nn
import torch 
import numpy as np 
from ..utils.register import register_from_numpy
from ..utils.categorical import *
from ..utils.continuous import *
from torch_scatter import scatter_sum, scatter_mean
from .schedule_utils import *
from ..utils.so3 import *

class VEScheduler(nn.Module):
    def __init__(self, num_timestep, sigma_min, sigma_max, type='log') -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_timestep = num_timestep
        if type == 'log':
            sigmas = torch.exp(
                torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), num_timestep + 1)
            )
        else:
            raise NotImplemented('No such VE Schedule Type')

        self.register_buffer('sigmas', sigmas)


class VPScheduler(nn.Module):
    def __init__(self, num_timestep, beta_start=1e-7, beta_end=2e-3, type='sigmoid', cosine_s=0.008) -> None:
        super().__init__()
    
        betas = self.init_betas(beta_start, beta_end, num_timestep, type, cosine_s)
        alphas = 1. - betas

        self.betas = register_from_numpy(betas)
        self.alphas = register_from_numpy(alphas)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod = register_from_numpy(alphas_cumprod)

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.alphas_cumprod_prev = register_from_numpy(alphas_cumprod_prev)

        self.sqrt_alphas_cumprod = register_from_numpy(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = register_from_numpy(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = register_from_numpy(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = register_from_numpy(np.sqrt(1. / alphas_cumprod - 1))
        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        self.posterior_mean_c0_coef = register_from_numpy(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = register_from_numpy(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.posterior_var = register_from_numpy(posterior_variance)
        self.posterior_logvar = register_from_numpy(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

    def init_betas(self, beta_start, beta_end, num_timestep, type, cosine_s):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if type == "quad":
            betas = (
                    np.linspace(
                        beta_start ** 0.5,
                        beta_end ** 0.5,
                        num_timestep,
                        dtype=np.float64,
                    )
                    ** 2
            )
        elif type == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_timestep, dtype=np.float64
            )
        elif type == "const":
            betas = beta_end * np.ones(num_timestep, dtype=np.float64)
        elif type == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(
                num_timestep, 1, num_timestep, dtype=np.float64
            )
        elif type == "sigmoid":
            betas = np.linspace(-6, 6, num_timestep)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif type == 'cosine':
            s = cosine_s
            steps = num_timestep + 1
            x = np.linspace(0, steps, steps)
            alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

            alphas = np.clip(alphas, a_min=0.001, a_max=1.)

            # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
            # Gaussian diffusion in Ho et al.
            alphas = np.sqrt(alphas)
            betas = 1. - alphas
        else:
            raise NotImplementedError(type)
        assert betas.shape == (num_timestep,)     
        return betas
    
    def forward_add_noise(self, *args, **kwargs):
        raise NotImplementedError()

    def backward_remove_noise(self, *args, **kwargs):
        raise NotImplementedError()

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError()


class CTNVPScheduler(VPScheduler):
    def __init__(self, num_timestep, beta_start=1e-7, beta_end=2e-3, type='sigmoid', cosine_s=0.008) -> None:
        super().__init__(num_timestep, beta_start, beta_end, type, cosine_s)
     # when the type is score, the output of network is '-eps', so that x_pred / sigma = score
        
    def forward_add_noise(self, x, t, batch_idx, gen_flag, noise=None, zero_center=False):
        mask_generate = gen_flag

        if noise is None:
            noise = torch.randn_like(x)
        if zero_center:
            com_noise = scatter_mean(noise, batch_idx, dim=0)[batch_idx]
            pos_noise = noise - scatter_mean(noise, batch_idx, dim=0)[batch_idx]
        
        a = self.alphas_cumprod.index_select(0, t)
        a_expand = a[batch_idx].unsqueeze(-1)

        x_noisy = a_expand.sqrt() * x + (1. - a_expand).sqrt() * noise
        
        if zero_center:
            return torch.where(mask_generate.unsqueeze(-1), x_noisy, x), pos_noise, com_noise
        
        return torch.where(mask_generate.unsqueeze(-1), x_noisy, x), noise

    
    def qxs_x0_xt(self, x0, xt, t, batch):
        # Compute the mean of the diffusion posterior q(x_s | x_t, x_0)
        # where s = t-1
        xs_mean = (self.posterior_mean_c0_coef[t][batch][:,None] * x0 + 
                  self.posterior_mean_ct_coef[t][batch][:,None] * xt )
        return xs_mean

    def backward_remove_noise(self, x_pred, x_noisy, t, batch_idx, gen_flag, type='score'):
        mask_generate = gen_flag
        a = self.alphas_cumprod.index_select(0, t)
        a_expand = a[:,None][batch_idx].expand_as(x_noisy)

        b = self.betas.index_select(0, t)
        b_expand = b[:,None][batch_idx].expand_as(x_noisy)

        nonzero_mask = (1 - (t == 0).float())[batch_idx].unsqueeze(-1)
        
        if type == 'score':
            sigma = (1 - a_expand).sqrt()
            score = - x_pred / sigma
            xs_denoised = (x_noisy + b_expand * score) / (1 - b_expand).sqrt() 
            xs_denoised += nonzero_mask * b_expand.sqrt() * torch.randn_like(x_noisy)
 
        else:
            xs_mean = self.qxs_x0_xt(x0=x_pred, xt=x_noisy, t=t, batch=batch_idx)
            pos_log_variance = self.posterior_logvar[t][batch_idx][:,None]
            xs_denoised = xs_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(x_noisy)

        return torch.where(mask_generate.unsqueeze(-1), xs_denoised, x_noisy)
    
    def xs_mean(self, x_pred, x_noisy, t, batch_idx, gen_flag, type='score'):
        mask_generate = gen_flag
        a = self.alphas_cumprod.index_select(0, t)
        a_expand = a[:,None][batch_idx].expand_as(x_noisy)

        b = self.betas.index_select(0, t)
        b_expand = b[:,None][batch_idx].expand_as(x_noisy)
        
        if type == 'score':
            sigma = (1 - a_expand).sqrt()
            score = - x_pred / sigma
            xs_mean = (x_noisy + b_expand * score) / (1 - b_expand).sqrt() 
            
        else:
            xs_mean = self.qxs_x0_xt(x0=x_pred, xt=x_noisy, t=t, batch=batch_idx)
        return torch.where(mask_generate.unsqueeze(-1), xs_mean, x_noisy)


    def get_loss(self, x_pred, x0, xt, t, gen_flag, batch_idx, type='score'):

        if type == 'score':
            a = self.alphas_cumprod.index_select(0, t)[batch_idx]
            a_expand = a[:,None].expand_as(x_pred)
            sigma = (1 - a_expand).sqrt()
            tgt = (x0 - xt) / sigma # score
            pred = x_pred 
        else:
            tgt = x0
            pred = x_pred

        mse = ((pred - tgt) ** 2).sum(-1)

        loss = scatter_mean((mse[gen_flag]), batch_idx[gen_flag], dim=0)
        pos_info = {'x0': x0, 'xt': xt, 'x_pred': x_pred, 'mask_gen': gen_flag}
        return loss.mean(), pos_info
    
    def get_score_loss(self, pred, tgt, t, gen_flag, batch_idx, score_in=False, info_tag=None):
        a = self.alphas_cumprod.index_select(0, t)[batch_idx]
        a_expand = a[:,None].expand_as(pred)
        sigma = (1 - a_expand).sqrt()
        if score_in:
            noise = tgt / sigma
        else:
            noise = tgt
        
        mse = ((pred - noise) ** 2).sum(-1)

        loss = scatter_mean((mse[gen_flag]), batch_idx[gen_flag], dim=0)
        pos_info = {'eps_0': noise, 'eps_pred': pred, 'score_0': noise * sigma, 
                    'score_pred': pred * sigma, 'mask_gen': gen_flag}
        if info_tag is not None:
            pos_info = {k + '_{}'.format(info_tag):v for k,v in pos_info.items()}
        return loss.mean(), pos_info


class CTNVEScheduler(VEScheduler):
    def __init__(self, num_timestep, sigma_min, sigma_max, type='log') -> None:
        super().__init__(num_timestep, sigma_min, sigma_max, type)
    
    def forward_add_noise(self, x, t, batch_idx, gen_flag, noise=None, zero_center=False):
        mask_generate = gen_flag

        if noise is None:
            noise = torch.randn_like(x)
        if zero_center:
            noise -= scatter_mean(noise, batch_idx)[batch_idx]
        
        sigma = self.sigmas.index_select(0, t)
        sigma_expand = sigma[batch_idx].unsqueeze(-1)

        x_noisy = x + (1. - sigma_expand).sqrt() * noise

        return torch.where(mask_generate.unsqueeze(-1), x_noisy, x)

    def forward_add_global_noise(self, x, t, batch_idx, gen_flag, noise=None):
        mask_generate = gen_flag
        
        if noise == None:
            noise = torch.randn((batch_idx.max() + 1, x.shape[-1])).to(x)
            noise_expand = noise[batch_idx]
        
        sigma = self.sigmas.index_select(0, t)
        sigma_expand = sigma[batch_idx].unsqueeze(-1)
        
        x_noisy = x + sigma_expand * noise_expand

        return torch.where(mask_generate.unsqueeze(-1), x_noisy, x), noise_expand

    def get_score_loss(self, pred, tgt, t, gen_flag, batch_idx, score_in=False, info_tag=None):

        sigma = self.sigmas.index_select(0, t)[batch_idx, None].expand_as(pred)
        if score_in:
            noise = tgt / sigma
        else:
            noise = tgt
        
        mse = ((pred - noise) ** 2).sum(-1)

        loss = scatter_mean((mse[gen_flag]), batch_idx[gen_flag], dim=0)
        pos_info = {'eps_0': noise, 'eps_pred': pred, 'score_0': noise * sigma, 
                    'score_pred': pred * sigma, 'mask_gen': gen_flag}
        if info_tag is not None:
            pos_info = {k + '_{}'.format(info_tag):v for k,v in pos_info.items()}
        return loss.mean(), pos_info
    
    def backward_remove_noise(self, x_pred, x_noisy, t, batch_idx, gen_flag, type='score'):
        mask_generate = gen_flag
        t = t[batch_idx]
        sigma_expand = self.sigmas[t].unsqueeze(-1)
        g = sigma_expand * torch.sqrt(2 * torch.tensor(self.sigma_max / self.sigma_min).log())
        score_norm = 1./sigma_expand

        d = 1 / self.num_timestep
        update = x_pred * score_norm * d * g ** 2

        z = torch.where(
            (t > 1)[:, None].expand_as(x_pred),
            torch.randn_like(x_pred),
            torch.zeros_like(x_pred),
        )
        
        if type == 'score':
            xs_denoised = x_noisy - update * z * np.sqrt(d) + g * z * np.sqrt(d)
 
        else:
            raise NotImplementedError()
        
        return torch.where(mask_generate.unsqueeze(-1), xs_denoised, x_noisy)
    
    def xs_mean(self, x_pred, x_noisy, t, batch_idx, gen_flag, type='score'):
        mask_generate = gen_flag
        t = t[batch_idx]
        sigma_expand = self.sigmas[t].unsqueeze(-1)
        g = sigma_expand * torch.sqrt(2 * torch.tensor(self.sigma_max / self.sigma_min).log())
        score_norm = 1./sigma_expand

        d = 1 / self.num_timestep
    
        update = x_pred * score_norm * d * g ** 2

        z = torch.where(
            (t > 1)[:, None].expand_as(x_pred),
            torch.randn_like(x_pred),
            torch.zeros_like(x_pred),
        )
        if type == 'score':
            xs_mean = x_noisy - update * z * np.sqrt(d)
        else:
            raise NotImplementedError()
        
        return torch.where(mask_generate.unsqueeze(-1), xs_mean, x_noisy)


class TypeVPScheduler(VPScheduler):

    def __init__(self, num_timestep, num_classes, beta_start=1e-7, 
                 beta_end=2e-3, type='sigmoid', cosine_s=0.008) -> None:
        super().__init__(num_timestep, beta_start, beta_end, type, cosine_s)
        
        self.num_classes = num_classes

        def log_1_min_a(a):
            return np.log(1 - np.exp(a) + 1e-40)

        alphas_v = self.alphas.cpu().numpy()
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = register_from_numpy(log_alphas_v)
        self.log_one_minus_alphas_v = register_from_numpy(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = register_from_numpy(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = register_from_numpy(log_1_min_a(log_alphas_cumprod_v))

    def forward_add_noise(self, v0, t, batch_idx, gen_flag):
        mask_generate = gen_flag
        log_c0 = index_to_log_onehot(v0, self.num_classes)
        v_noisy = self.qct_c0_sample(log_c0, t, batch_idx)
                
        v_noisy = torch.where(mask_generate, v_noisy, v0)
        c_noisy = F.one_hot(v_noisy, num_classes=self.num_classes).float()
        return c_noisy, v_noisy
        
    def get_loss(self, c_pred, v0, vt, t, gen_flag, batch_idx, pred_logit=True):
        
        log_c0 = index_to_log_onehot(v0, self.num_classes)
        log_ct = index_to_log_onehot(vt, self.num_classes)

        mask_generate = gen_flag
        if pred_logit:
            log_c_pred = F.log_softmax(c_pred, dim=-1)
        else:
            log_c_pred = torch.log(c_pred + 1e-8)

        log_v_pred_prob = self.q_v_posterior(log_c_pred, log_ct, t, batch_idx)
        log_v_true_prob = self.q_v_posterior(log_c0, log_ct, t, batch_idx)
        loss = self.compute_loss(log_v_pred_prob, log_v_true_prob,
                                 log_c0, t, batch_idx, mask_generate)
        
        type_info = {'v0': v0, 'vt': vt, 'c_pred': log_c_pred.exp(), 'mask_gen': mask_generate}
        return loss.mean(), type_info
    
    def backward_remove_noise(self, c_pred, ct, t, batch_idx, gen_flag, pred_logit=True):   
        if pred_logit:
            log_c_pred = F.log_softmax(c_pred, dim=-1)
        else:
            log_c_pred = torch.log(c_pred + 1e-8)

        log_ct = torch.log(ct + 1e-8)
        log_vs_prob = self.q_v_posterior(log_c_pred, log_ct, t, batch_idx)
        v_next = log_sample_categorical(log_vs_prob)
        v_next = torch.where(gen_flag, v_next, ct.argmax(-1))
        c_next = F.one_hot(v_next, num_classes=self.num_classes).float()
        return c_next, v_next
        
    def qct_c0_sample(self, log_c0, t, batch_idx):
        log_qct_c0 = self.qct_c0_pred(log_c0, t, batch_idx)
        vt = log_sample_categorical(log_qct_c0)
        return vt

    def qct_c0_pred(self, log_v0, t, batch_idx):
        # compute q(vt | v0)
        log_cumprod_alpha_t = self.log_alphas_cumprod_v.index_select(0, t)
        log_cumprod_alpha_t = log_cumprod_alpha_t[batch_idx].unsqueeze(-1)
        log_1_min_cumprod_alpha = self.log_one_minus_alphas_cumprod_v.index_select(0, t)
        log_1_min_cumprod_alpha = log_1_min_cumprod_alpha[batch_idx].unsqueeze(-1)

        log_qvt_v0 = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_qvt_v0

    def compute_loss(self, log_c_pred_prob, log_c_true_prob, log_v0, t, batch, mask_generate):
        kl_v = categorical_kl(log_c_true_prob, log_c_pred_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_c_pred_prob)
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean((mask * decoder_nll_v + (1. - mask) * kl_v)[mask_generate], 
                               batch[mask_generate], dim=0)
        return loss_v.mean()

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        log_qvs1_vt = self.q_v_pred_one_timestep(log_vt, t, batch)
        # make the padded type prediction prob to be zero
        if log_qvs1_vt.shape[-1] - log_qvt1_v0.shape[-1] == 1:
            log_qvt1_v0 = torch.cat([log_qvt1_v0, torch.zeros_like(log_qvt1_v0)[...,:1]], dim=-1)
        unnormed_logprobs = log_qvt1_v0 + log_qvs1_vt
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        return log_vt1_given_vt_v0
    
    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = self.log_alphas_cumprod_v[t][batch].unsqueeze(-1)
        log_1_min_cumprod_alpha = self.log_one_minus_alphas_cumprod_v[t][batch].unsqueeze(-1)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = self.log_alphas_v[t][batch].unsqueeze(-1)
        log_1_min_alpha_t = self.log_one_minus_alphas_v[t][batch].unsqueeze(-1)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs
    

class MaskTypeSchedule(nn.Module):
    def __init__(self, num_timestep, num_classes, absorbing_state, type='uniform') -> None:
        super().__init__()
        self.num_timestep = num_timestep
        self.num_classes = num_classes
        self.absorbing_state = absorbing_state
        self.schedule_type = type

    def forward_add_noise(self, v_0, t, batch_idx, gen_flag, eps=None):

        mask_gen = gen_flag.bool()
        v_t = v_0.clone()
        t = t[batch_idx]
        if eps is not None:
            mask_prob = eps  
        else:
            mask_prob = (
                (t.view(-1).float()).clamp(min=0.)
                / (self.num_timestep)
                ).to(v_0.device)

        diff_mask = (
            torch.rand_like(v_t.float()) < mask_prob
            )

        # if true, the element will be masked
        diff_mask = torch.logical_and(diff_mask, mask_gen)
        v_t = torch.where(diff_mask, self.absorbing_state, v_0)
        c_t = F.one_hot(v_t, num_classes=self.num_classes).float()
        return v_t, c_t, diff_mask
    
    def backward_remove_noise(self, c_pred, ct, t, batch_idx, gen_flag, pred_logit=True, fix_pred=True):
        if pred_logit:
            c_pred = F.softmax(c_pred, dim=-1)
        else:
            c_pred = c_pred
        
        vt = ct.argmax(-1) 
        t = t[batch_idx]
        prob = (
            (self.num_timestep - t) / (self.num_timestep)
            ).clamp(max=1., min=0.)
        change_flag = (
            torch.rand_like(vt.float()) 
            < prob
        )
        change_flag = torch.logical_and(change_flag, gen_flag)
        if fix_pred:
            change_flag = torch.logical_and(change_flag, vt==self.absorbing_state)
        v_pred = c_pred.argmax(-1)
        v_next = torch.where(change_flag, v_pred, vt)
        c_next = F.one_hot(v_next, num_classes=self.num_classes).float()
        return c_next, v_next


    def get_loss(self, c_pred, v0, vt, t, gen_flag, batch_idx, pred_logit=True):
        if pred_logit:
            c_pred = F.softmax(c_pred, dim=-1)
        else:
            c_pred = c_pred
        
        loss_v = F.cross_entropy(c_pred, v0, reduction='none')
        loss = scatter_mean(loss_v[gen_flag], 
                            batch_idx[gen_flag], dim=0)
        type_info = {'v0': v0, 'vt': vt, 'c_pred': c_pred, 'mask_gen': gen_flag}
        if len(loss) == 0:
            loss = torch.zeros_like(v0).float()
        return loss.mean(), type_info


class RotVPScheduler(VPScheduler):
    def __init__(self, num_timestep, beta_start=1e-7, beta_end=2e-3, type='sigmoid', cosine_s=0.008) -> None:
        super().__init__(num_timestep, beta_start, beta_end, type, cosine_s)
        
        c1 = torch.sqrt(1 - self.alphas_cumprod) # (T,).
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist())

        # Inverse (generate)
        sigmas = torch.zeros_like(self.betas)
        for i in range(1, self.betas.size(0)):
            sigmas[i] = ((1 - self.alphas_cumprod[i-1]) / (1 - self.alphas_cumprod[i])) * self.betas[i]
        self.sigmas = torch.sqrt(sigmas)
        sigma = self.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist())

        self.register_buffer('_dummy', torch.empty([0, ]))
    
    def forward_add_noise(self, o_0, t, batch_idx, mask_generate):
        """
        Args:
            v_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,).
        """

        t = t[batch_idx]
        alpha_bar = self.alphas_cumprod[t]
        c0 = torch.sqrt(alpha_bar).unsqueeze(-1)
        c1 = torch.sqrt(1 - alpha_bar).unsqueeze(-1)

        # Noise rotation
        e_scaled = random_normal_so3(t, self.angular_distrib_fwd, device=t.device)    # (N, L, 3)
        e_normal = e_scaled / (c1 + 1e-8)
        E_scaled = so3vec_to_rotation(e_scaled)   # (N, L, 3, 3)

        # Scaled true rotation
        R0_scaled = so3vec_to_rotation(c0 * o_0)  # (N, L, 3, 3)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(o_0), v_noisy, o_0)

        return v_noisy, e_scaled

    def backward_remove_noise(self, o_pred, ot, t, batch_idx, gen_flag):
        mask_generate = gen_flag
        N = gen_flag.size()[0]
        t = t[batch_idx]
        e = random_normal_so3(t, self.angular_distrib_inv, device = t.device) # (N, L, 3)
        e = torch.where(
            (t > 1)[:, None].expand(N, 3),
            e, 
            torch.zeros_like(e) # Simply denoise and don't add noise at the last step
        )
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(o_pred)
        o_next = rotation_to_so3vec(R_next)
        o_next = torch.where(mask_generate[..., None].expand_as(o_next), o_next, ot)

        return o_next
    

class VariationalScheduler(nn.Module):
    def __init__(self, num_timestep, type='polynomial_2') -> None:
        super().__init__()
        self.num_timestep = num_timestep
        if type == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(type,
                                                 timesteps=num_timestep,
                                                 precision=5e-4)

    def forward_add_noise(self, x, t, batch_idx, gen_flag, noise=None, zero_center=False):
        mask_generate = gen_flag
        gamma_t = self.gamma(t)[batch_idx].unsqueeze(-1)

        if noise is None:
            noise = torch.randn_like(x)
        if zero_center:
            com_noise = scatter_mean(noise, batch_idx, dim=0)[batch_idx]
            pos_noise = noise - scatter_mean(noise, batch_idx, dim=0)[batch_idx]
        
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))

        x_noisy = alpha_t * x + sigma_t * noise
        
        if zero_center:
            return torch.where(mask_generate.unsqueeze(-1), x_noisy, x), pos_noise, com_noise
        
        return torch.where(mask_generate.unsqueeze(-1), x_noisy, x), noise
    
    def get_score_loss(self, pred, tgt, t, gen_flag, batch_idx, score_in=False, info_tag=None):
        gamma_t = self.gamma(t)[batch_idx].unsqueeze(-1)
        
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))

        if score_in:
            noise = tgt / sigma_t
        else:
            noise = tgt
        
        mse = ((pred - noise) ** 2).sum(-1)
        if gen_flag.sum() > 0:
            loss = scatter_mean((mse[gen_flag]), batch_idx[gen_flag], dim=0)
        else:
            loss = torch.zeros_like(mse)

        pos_info = {'eps_0': noise, 'eps_pred': pred, 'score_0': noise * sigma_t, 
                    'score_pred': pred * sigma_t, 'mask_gen': gen_flag}
        if info_tag is not None:
            pos_info = {k + '_{}'.format(info_tag):v for k,v in pos_info.items()}
        return loss.mean(), pos_info
    
    def backward_remove_noise(self, x, x_pred, t, batch_idx, gen_flag, zero_mean=False):
        gamma_t = self.gamma(t)[batch_idx].unsqueeze(-1)
        gamma_s = self.gamma(t-1/self.num_timestep)[batch_idx].unsqueeze(-1)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)

        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        sigma_s = torch.sqrt(torch.sigmoid(gamma_s))

        mu_lig = x / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * x_pred
        sigma = sigma_t_given_s * sigma_s / sigma_t
        if zero_mean:
            zero_mean_noise = scatter_mean(torch.randn_like(mu_lig) * sigma, batch_idx, dim=0)[batch_idx]
        else:
            zero_mean_noise = torch.rand_like(mu_lig) * sigma
        return torch.where(gen_flag.unsqueeze(-1), mu_lig + zero_mean_noise, x)
        
        
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor,
                                  gamma_s: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))
        
        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s