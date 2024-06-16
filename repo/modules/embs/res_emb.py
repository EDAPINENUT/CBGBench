from torch import nn 
import torch 
from repo.utils.protein.constants import *
from repo.models.utils.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles
def get_res_embedding_func(type, 
                            emb_dim, 
                            input_dim = num_aa_types):
    if type == 'linear':
        return nn.Linear(input_dim, emb_dim)
    elif type == 'frame':
        return PerResidueEncoder(input_dim, emb_dim)
    else:
        raise ValueError(f'Unknown time embedding type: {type}')


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


class PerResidueEncoder(nn.Module):

    def __init__(self, input_dim, feat_dim, max_num_atoms=15, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()
        infeat_dim = feat_dim + self.max_aa_types*self.max_num_atoms*3 + self.dihed_embed.get_out_dim(3) # Phi, Psi, Chi1-4
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, res_nb, chain_nb, pos_atoms, mask_atoms):
        """
        Args:
            aa: (N, L)
            phi, phi_mask: (N, L)
            psi, psi_mask: (N, L)
            chi, chi_mask: (N, L, 4)
            mask_residue: (N, L)
        """
        if aa.dim() == 2:
            aa = aa.argmax(-1)
        N = aa.size()[0]
        # Amino acid identity features
        aa_feat = self.aatype_embed(aa) # (N, L, feat)
        mask_residue = mask_atoms[:, BBHeavyAtom.CA] # (N, L)
        R = construct_3d_basis(
            pos_atoms[:, BBHeavyAtom.CA], 
            pos_atoms[:, BBHeavyAtom.C], 
            pos_atoms[:, BBHeavyAtom.N]
        )

        trans = pos_atoms[:, BBHeavyAtom.CA]
        crd = global_to_local(R, trans, pos_atoms)    # (N, L, A, 3)
        crd_mask = mask_atoms[:, :, None].expand_as(crd)
        crd = torch.where(crd_mask, crd, torch.zeros_like(crd))

        aa_expand  = aa[:, None, None, None].expand(N, self.max_aa_types, self.max_num_atoms, 3)
        rng_expand = torch.arange(0, self.max_aa_types)[None, :, None, None].expand(N, self.max_aa_types, self.max_num_atoms, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd[:, None, :, :].expand(N, self.max_aa_types, self.max_num_atoms, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, self.max_aa_types*self.max_num_atoms*3)

        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, None]) * mask_bb_dihed[:, :, None]  # (N, L, 3, dihed/3)
        dihed_feat = dihed_feat.reshape(N, -1)

        out_feat = self.mlp(torch.cat([aa_feat, crd_feat, dihed_feat], dim=-1)) # (N, L, F)
        out_feat = out_feat * mask_residue[:, None]

        return out_feat
