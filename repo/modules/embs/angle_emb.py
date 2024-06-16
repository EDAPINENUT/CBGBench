import torch
from torch import nn

import numpy as np
from scipy.optimize import brentq
from scipy import special as sp
import torch
from math import sqrt, pi as PI
from repo.models.utils.sphere import *


def get_angle_emb(type, num_spherical, num_radial, cut_off=10.0):
    if type == 'cos':
        embeder = AngularEncoding(num_spherical)
    elif type == 'spherical':
        embeder = SphericalAngularEncoding(num_spherical, num_radial, cut_off)
    else:
        raise ValueError(f'Unknown distance embedding type: {type}')
    return embeder


class AngularEncoding(nn.Module):

    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', torch.FloatTensor(
            [i+1 for i in range(num_funcs)] + [1./(i+1) for i in range(num_funcs)]
        ))

    def get_out_dim(self, in_dim):
        return in_dim * (1 + 2*2*self.num_funcs)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code


class SphericalAngularEncoding(nn.Module):
    def __init__(self, num_spherical, num_radial, cut_off=5.0):
        super(SphericalAngularEncoding, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cut_off
        # self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj=None):
        dist = dist / self.cutoff
        
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        # rbf = self.envelope(dist).unsqueeze(-1) * rbf 
       
        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        
        n, k = self.num_spherical, self.num_radial
        if idx_kj is None: # Use for encoding in generative modeling
            out = (rbf.view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        else: # Use for SphereNet physical representation
            out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out
