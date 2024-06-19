import numpy as np
import torch

from .protein.constants import chi_pi_periodic, AA
from repo.utils.misc import BlackHole


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    elif cfg.type is None:
        return BlackHole()
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def log_losses(loss, loss_dict, scalar_dict, it, tag, logger=BlackHole(), writer=BlackHole()):
    logstr = '[%s] Iter %05d' % (tag, it)
    logstr += ' | loss %.4f' % loss.item()
    for k, v in loss_dict.items():
        logstr += ' | loss(%s) %.4f' % (k, v.item())
    for k, v in scalar_dict.items():
        logstr += ' | %s %.6f' % (k, v.item() if isinstance(v, torch.Tensor) else v)
    logger.info(logstr)

    writer.add_scalar('%s/loss' % tag, loss, it)
    for k, v in loss_dict.items():
        writer.add_scalar('%s/loss_%s' % (tag, k), v, it)
    for k, v in scalar_dict.items():
        writer.add_scalar('%s/%s' % (tag, k), v, it)
    writer.flush()


class ScalarMetricAccumulator(object):

    def __init__(self):
        super().__init__()
        self.accum_dict = {}
        self.count_dict = {}

    @torch.no_grad()
    def add(self, name, value, batchsize=None, mode=None):
        assert mode is None or mode in ('mean', 'sum')

        if mode is None:
            delta = value.sum()
            count = value.size(0)
        elif mode == 'mean':
            delta = value * batchsize
            count = batchsize
        elif mode == 'sum':
            delta = value
            count = batchsize
        delta = delta.item() if isinstance(delta, torch.Tensor) else delta

        if name not in self.accum_dict:
            self.accum_dict[name] = 0
            self.count_dict[name] = 0
        self.accum_dict[name] += delta
        self.count_dict[name] += count

    def log(self, it, tag, logger=BlackHole(), writer=BlackHole()):
        summary = {k: self.accum_dict[k] / self.count_dict[k] for k in self.accum_dict}
        logstr = '[%s] Iter %05d' % (tag, it)
        for k, v in summary.items():
            logstr += ' | %s %.4f' % (k, v)
            writer.add_scalar('%s/%s' % (tag, k), v, it)
        logger.info(logstr)

    def get_average(self, name):
        return self.accum_dict[name] / self.count_dict[name]


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def aggregate_sidechain_accuracy(aa, chi_pred, chi_native, chi_mask):
    aa = aa.reshape(-1)
    chi_mask = chi_mask.reshape(-1, 4)
    diff = torch.min(
        (chi_pred - chi_native) % (2 * np.pi),
        (chi_native - chi_pred) % (2 * np.pi),
    )   # (N, L, 4)
    diff = torch.rad2deg(diff)
    diff = diff.reshape(-1, 4)

    diff_flip = torch.min(
        ( (chi_pred + np.pi) - chi_native) % (2 * np.pi),
        (chi_native - (chi_pred + np.pi) ) % (2 * np.pi),
    )
    diff_flip = torch.rad2deg(diff_flip)
    diff_flip = diff_flip.reshape(-1, 4)
    
    acc = [{j:[] for j in range(1, 4+1)} for i in range(20)]
    for i in range(aa.size(0)):
        for j in range(4):
            chi_number = j+1
            if not chi_mask[i, j].item(): continue
            if chi_pi_periodic[AA(aa[i].item())][chi_number-1]:
                diff_this = min(diff[i, j].item(), diff_flip[i, j].item())
            else:
                diff_this = diff[i, j].item()
            acc[aa[i].item()][chi_number].append(diff_this)
    
    table = np.full((20, 4), np.nan)
    for i in range(20):
        for j in range(1, 4+1):
            if len(acc[i][j]) > 0:
                table[i, j-1] = np.mean(acc[i][j])
    return table



def load_model_from_checkpoint(ckpt_path, return_ckpt=False):
    from repo.models import get_model
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = get_model(ckpt['config'].model)
    model.load_state_dict(ckpt['model'])
    if return_ckpt:
        return model, ckpt
    else:
        return model
    

class CrossValidation(object):

    def __init__(self, model_factory, config, num_cvfolds):
        super().__init__()
        self.num_cvfolds = num_cvfolds
        self.models = [
            model_factory(config.model)
            for _ in range(num_cvfolds)
        ]

        self.optimizers = []
        self.schedulers = []
        for model in self.models:
            optimizer = get_optimizer(config.train.optimizer, model)
            self.optimizers.append(optimizer)
            self.schedulers.append(get_scheduler(config.train.scheduler, optimizer))

    def get(self, fold):
        return self.models[fold], self.optimizers[fold], self.schedulers[fold]

    def to(self, device):
        for m in self.models:
            m.to(device)
        return self

    def state_dict(self):
        return {
            'models': [m.state_dict() for m in self.models],
            'optimizers': [o.state_dict() for o in self.optimizers],
            'schedulers': [s.state_dict() for s in self.schedulers],
        }

    def load_state_dict(self, state_dict):
        for sd, obj in zip(state_dict['models'], self.models):
            obj.load_state_dict(sd)
        for sd, obj in zip(state_dict['optimizers'], self.optimizers):
            obj.load_state_dict(sd)
        for sd, obj in zip(state_dict['schedulers'], self.schedulers):
            obj.load_state_dict(sd)


