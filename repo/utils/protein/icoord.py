import torch
import numpy as np
from Bio.PDB.Residue import Residue

from .constants import AA, chi_angles_atoms, chi_pi_periodic


def get_chi_angles(restype: AA, res: Residue):
    ic = res.internal_coord
    chi_angles = torch.zeros([4, ])
    chi_angles_alt = torch.zeros([4, ],)
    chi_angles_mask = torch.zeros([4, ], dtype=torch.bool)
    count_chi_angles = len(chi_angles_atoms[restype])
    if ic is not None:
        for i in range(count_chi_angles):
            angle_name = 'chi%d' % (i+1)
            if ic.get_angle(angle_name) is not None:
                angle = np.deg2rad(ic.get_angle(angle_name))
                chi_angles[i] = angle
                chi_angles_mask[i] = True

                if chi_pi_periodic[restype][i]:
                    if angle >= 0:
                        angle_alt = angle - np.pi
                    else:
                        angle_alt = angle + np.pi
                    chi_angles_alt[i] = angle_alt
                else:
                    chi_angles_alt[i] = angle
            
    chi_complete = (count_chi_angles == chi_angles_mask.sum().item())
    return chi_angles, chi_angles_alt, chi_angles_mask, chi_complete


def get_backbone_torsions(res: Residue):
    ic = res.internal_coord
    if ic is None:
        return None, None, None
    phi, psi, omega = ic.get_angle('phi'), ic.get_angle('psi'), ic.get_angle('omega')
    if phi is not None: phi = np.deg2rad(phi)
    if psi is not None: psi = np.deg2rad(psi)
    if omega is not None: omega = np.deg2rad(omega)
    return phi, psi, omega
