import torch
from rdkit import Chem
import numpy as np

DUMMY_ATOM = '*'

_FRAME_FUNC_DICT = {}

def register_fg_frame_func(name):
    def decorator(cls):
        _FRAME_FUNC_DICT[name] = cls
        return cls
    return decorator

def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center    # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center    # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)    # (N, L, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat

def log_rotation(R):
    trace = R[..., range(3), range(3)].sum(-1)
    if torch.is_grad_enabled():
        # The derivative of acos at -1.0 is -inf, so to stablize the gradient, we use -0.9999
        min_cos = -0.999
    else:
        min_cos = -1.0
    cos_theta = ( (trace-1) / 2 ).clamp_min(min=min_cos)
    sin_theta = torch.sqrt(1 - cos_theta**2)
    theta = torch.acos(cos_theta)
    coef = ((theta+1e-8)/(2*sin_theta+2e-8))[..., None, None]
    logR = coef * (R - R.transpose(-1, -2))
    return logR

def skewsym_to_so3vec(S):
    x = S[..., 1, 2]
    y = S[..., 2, 0]
    z = S[..., 0, 1]
    w = torch.stack([x,y,z], dim=-1)
    return w

def rotation_to_so3vec(R):
    logR = log_rotation(R)
    w = skewsym_to_so3vec(logR)
    return w

def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e

def conf_with_smiles(smiles, positions):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (positions[i][0], positions[i][1], positions[i][2]))
    mol.AddConformer(conf)
    return mol

#c1ccccc1
@register_fg_frame_func('c1ccccc1')
def frame_benzene(smiles, positions):
    assert(smiles == 'c1ccccc1')

    frame_idx = [1,0,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

# NC=O
@register_fg_frame_func('NC=O')
def frame_isocyanate(smiles, positions):
    assert(smiles == 'NC=O')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

# O=CO
@register_fg_frame_func('O=CO')
def frame_carboxylic(smiles, positions):
    assert(smiles == 'O=CO')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

# c1ccncc1
@register_fg_frame_func('c1ccncc1')
def frame_pyridine(smiles, positions):
    assert(smiles == 'c1ccncc1')

    frame_idx = [2,3,4]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#c1ncc2nc[nH]c2n1
@register_fg_frame_func('c1ncc2nc[nH]c2n1')
def frame_purine(smiles, positions):
    assert(smiles == 'c1ncc2nc[nH]c2n1')

    frame_idx = [7,3,6]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#NS(=O)=O
@register_fg_frame_func('NS(=O)=O')
def frame_methanesulfonamide(smiles, positions):
    assert(smiles == 'NS(=O)=O')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#O=P(O)(O)O
@register_fg_frame_func('O=P(O)(O)O')
def frame_phosphorusquad(smiles, positions):
    assert(smiles == 'O=P(O)(O)O')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

# OCO
@register_fg_frame_func('OCO')
def frame_oco(smiles, positions):
    assert(smiles == 'OCO')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#c1cncnc1
@register_fg_frame_func('c1cncnc1')
def frame_pyrimidine(smiles, positions):
    assert(smiles == 'c1cncnc1')

    frame_idx = [2,3,4]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

# c1cn[nH]c1
@register_fg_frame_func('c1cn[nH]c1')
def frame_c1cnnc1(smiles, positions):
    assert(smiles == 'c1cn[nH]c1')

    frame_idx = [3,2,1]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#O=P(O)O
@register_fg_frame_func('O=P(O)O')
def frame_phosphorustri(smiles, positions):
    assert(smiles == 'O=P(O)O')
    
    frame_idx = [0,1,4]
    dummy_pos = (np.array(positions[2])+np.array(positions[3]))/2
    smiles = smiles + DUMMY_ATOM
    positions = torch.from_numpy(np.concatenate([np.array(positions), [dummy_pos]], 0))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]])) 
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#c1ccc2ccccc2c1
@register_fg_frame_func('c1ccc2ccccc2c1')
def frame_decalin(smiles, positions):
    assert(smiles == 'c1ccc2ccccc2c1')

    frame_idx = [2,3,4]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#c1ccsc1
@register_fg_frame_func('c1ccsc1')
def frame_thiophene(smiles, positions):
    assert(smiles == 'c1ccsc1')

    frame_idx = [2,3,4]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#N=CN
@register_fg_frame_func('N=CN')
def frame_ncn(smiles, positions):
    assert(smiles == 'N=CN')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#NC(N)=O
@register_fg_frame_func('NC(N)=O')
def frame_ncno(smiles, positions):
    assert(smiles == 'NC(N)=O')

    frame_idx = [1,2,3]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#O=c1cc[nH]c(=O)[nH]1
@register_fg_frame_func('O=c1cc[nH]c(=O)[nH]1')
def frame_occcncon(smiles, positions):
    assert(smiles == 'O=c1cc[nH]c(=O)[nH]1')

    frame_idx = [1,7,5]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#c1ccc2ncccc2c1
@register_fg_frame_func('c1ccc2ncccc2c1')
def frame_quinoline(smiles, positions):
    assert(smiles == 'c1ccc2ncccc2c1')

    frame_idx = [2,3,4]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#c1cscn1
@register_fg_frame_func('c1cscn1')
def frame_thiazole(smiles, positions):
    assert(smiles == 'c1cscn1')

    frame_idx = [3,2,4]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#c1ccc2[nH]cnc2c1
@register_fg_frame_func('c1ccc2[nH]cnc2c1')
def frame_benzimidazole(smiles, positions):
    assert(smiles == 'c1ccc2[nH]cnc2c1')

    frame_idx = [4,5,6]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#c1c[nH]cn1
@register_fg_frame_func('c1c[nH]cn1')
def frame_imidazole(smiles, positions):
    assert(smiles == 'c1c[nH]cn1')

    frame_idx = [2,3,4]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

#O=[N+][O-]
@register_fg_frame_func('O=[N+][O-]')
def frame_ono(smiles, positions):
    assert(smiles == 'O=[N+][O-]')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

@register_fg_frame_func('O=CNO')
def frame_ocno(smiles, positions):
    assert(smiles == 'O=CNO')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

@register_fg_frame_func('NC(=O)O')
def frame_ncoo(smiles, positions):
    assert(smiles == 'NC(=O)O')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

@register_fg_frame_func('O=S=O')
def frame_sulphurdioxide(smiles, positions):
    assert(smiles == 'O=S=O')

    frame_idx = [0,1,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

@register_fg_frame_func('c1ccc2[nH]ccc2c1')
def frame_benzpyrole(smiles, positions):
    assert(smiles == 'c1ccc2[nH]ccc2c1')

    frame_idx = [4,3,2]
    positions = torch.from_numpy(np.array(positions))
    center = torch.from_numpy(np.array(positions[frame_idx[1]]))
    p1 = torch.from_numpy(np.array(positions[frame_idx[0]]))
    p2 = torch.from_numpy(np.array(positions[frame_idx[2]]))
    R = construct_3d_basis(center, p1, p2)
    v = rotation_to_so3vec(R)
    local_pos = positions - center
    local_pos = torch.matmul(local_pos, R)
    framed_mol = conf_with_smiles(smiles, local_pos.numpy().tolist())
    _, idx_re = reindex(frame_idx, len(positions))
    positions_reindexed = positions[idx_re]
    return center.numpy(), R.numpy(), v.numpy(), local_pos.numpy(), framed_mol, positions_reindexed, idx_re

def reindex(frame_idx, fg_size):
    idx_raw = np.arange(fg_size)
    idx_re = []
    for idx in frame_idx:
        idx_re.append(idx)
    for idx in idx_raw:
        if idx not in frame_idx:
            idx_re.append(idx)
    return frame_idx, idx_re


def transform_into_fg_data(smiles, positions):
    if smiles in _FRAME_FUNC_DICT.keys():
        return _FRAME_FUNC_DICT[smiles](smiles, positions=positions)
    else:
        raise NotImplementedError('The functional group is not defined.')