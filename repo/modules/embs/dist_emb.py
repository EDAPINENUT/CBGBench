import torch
from torch import nn
from math import sqrt, pi as PI
from ..common import GaussianSmearing, MLP

def get_dist_emb(type, emb_dim, cut_off=20.0):
    if type == 'gaussian_exp':
        embeder = nn.Sequential(GaussianSmearing(0., cut_off, num_gaussians=emb_dim),
                                MLP(emb_dim, 1, emb_dim * 8))
    elif type == 'power':
        embeder = PowerDistEmb(emb_dim)
    else:
        raise ValueError(f'Unknown distance embedding type: {type}')
    return embeder


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class PowerDistEmb(torch.nn.Module):
    def __init__(self, num_radial, cutoff=10.0, envelope_exponent=5):
        super(PowerDistEmb, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()
