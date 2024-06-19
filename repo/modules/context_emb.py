import torch.nn as nn 
import torch 
import torch.nn.functional as F
from .embs.time_emb import get_time_embedding_func
from .embs.atom_emb import get_atom_embedding_func
from .embs.fg_emb import get_fg_embedding_func
from .embs.res_emb import get_res_embedding_func
from .embs.vec_emb import get_vec_embedding_func
from repo.utils.molecule.constants import *
from repo.utils.protein.constants import *
from repo.models.utils.geometry import *
from repo.models.utils.so3 import *


def get_context_embedder(cfg):
    embeder_type = cfg.get('type', 'fa')
    if embeder_type == 'fa':
        return PLContextEmbedder(cfg)
    elif embeder_type == 'fg':
        return FGContextEmbedder(cfg)



class FGContextEmbedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embeding_cfg = cfg
        self.num_classes = cfg.get('num_fgtype', 50) + num_aa_types
        emb_dim = embeding_cfg.get('emb_dim', 128)
        self.emb_dim = emb_dim

        time_cfg = embeding_cfg.get('time', None)
        if time_cfg is not None:
            self.time_emb = get_time_embedding_func(time_cfg.type, emb_dim)
        else:
            self.time_emb = None
        
        fg_fg = embeding_cfg.get('fg', None)
        if fg_fg is not None:
            self.ligand_fg_emb = get_fg_embedding_func(fg_fg.type, 
                                                       emb_dim, 
                                                       input_dim = self.num_classes)
            self.protein_fg_emb = get_fg_embedding_func(fg_fg.type,
                                                        emb_dim,
                                                        input_dim = self.num_classes)
        else:
            self.ligand_atom_emb = None
            self.protein_atom_emb = None

        residue_cfg = embeding_cfg.get('residue', None)
        if residue_cfg is not None:
            self.residue_emb = get_res_embedding_func(residue_cfg.type, 
                                                      emb_dim,
                                                      input_dim = len(aa_name_number))
        else:
            self.residue_emb = None

        vec_cfg = embeding_cfg.get('vec', None)
        if vec_cfg is not None:
            self.vec_emb = get_vec_embedding_func(vec_cfg.type, vec_cfg.vec_emb_dim)
        else:
            self.vec_emb = None

        self.ligand_indicator = nn.Linear(1, emb_dim)

    def forward(self, xc_lig, x_rec, v_lig, v_rec, aa_rec, res_nb, chain_nb, 
                mask_atom_lig, mask_atom_rec, batch_idx_lig, batch_idx_rec,
                lig_flag, rec_flag, t=None):
        
        N_lig, _ = xc_lig.shape
        N_rec, _, _ = x_rec.shape

        xc_rec = x_rec[:, BBHeavyAtom.CA]
        R_rec = construct_3d_basis(
            x_rec[:, BBHeavyAtom.CA], 
            x_rec[:, BBHeavyAtom.C], 
            x_rec[:, BBHeavyAtom.N]
            )
        o_rec = rotation_to_so3vec(R_rec) 

        if t is not None:
            t_lig = t[batch_idx_lig].unsqueeze(-1).float()
            t_rec = t[batch_idx_rec].unsqueeze(-1).float()
        else:
            t_lig = torch.zeros(N_lig, 1).to(xc_lig.device)
            t_rec = torch.zeros(N_rec, 1).to(xc_rec.device)

        if self.time_emb is not None:
            t_emb_lig = self.time_emb(t_lig)
            t_emb_rec = self.time_emb(t_rec)
        else:
            t_emb_lig = torch.zeros_like(t_lig).repeat(1, self.emb_dim)
            t_emb_rec = torch.zeros_like(t_rec).repeat(1, self.emb_dim)
        
        if self.ligand_fg_emb is not None:
            if v_lig.dim() == 1:
                v_lig = F.one_hot(v_lig, num_classes = self.num_classes).float()
            if v_lig.shape[-1] != self.num_classes:
                v_lig = v_lig.argmax(-1)
                v_lig = F.one_hot(v_lig, num_classes = self.num_classes).float()

            h_lig = self.ligand_fg_emb(v_lig)
        else:
            h_lig = torch.zeros_like(v_lig)

        if self.protein_fg_emb is not None:
            if v_rec.dim() == 1:
                v_rec = F.one_hot(v_rec, num_classes = self.num_classes).float()
            if v_rec.shape[-1] != self.num_classes:
                v_rec = v_rec.argmax(-1)
                v_rec = F.one_hot(v_rec, num_classes = self.num_classes).float()

            h_rec = self.protein_fg_emb(v_rec)
        else:
            h_rec = torch.zeros_like(v_rec)
                       
        if self.residue_emb is not None:
            if aa_rec.dim() == 1:
                aa_rec = F.one_hot(aa_rec, num_classes = len(aa_name_number)).float()

            h_aa = self.residue_emb(aa_rec, res_nb, chain_nb, x_rec, mask_atom_rec)
        else:
            h_aa = torch.zeros_like(aa_rec)
        
        bias_lig = self.ligand_indicator(lig_flag.float().unsqueeze(-1))
        bias_rec = self.ligand_indicator(rec_flag.float().unsqueeze(-1))

        h_lig = h_lig + t_emb_lig + bias_lig
        h_rec = h_rec + t_emb_rec + h_aa + bias_rec

        if self.vec_emb is not None:
            xc_lig = self.vec_emb(xc_lig)
            xc_rec = self.vec_emb(xc_rec)

        return xc_lig, xc_rec, o_rec, h_lig, h_rec

class PLContextEmbedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embeding_cfg = cfg
        self.num_classes = cfg.get('num_atomtype', 14)
        emb_dim = embeding_cfg.get('emb_dim', 128)
        self.emb_dim = emb_dim

        time_cfg = embeding_cfg.get('time', None)
        if time_cfg is not None:
            self.time_emb = get_time_embedding_func(time_cfg.type, emb_dim)
        else:
            self.time_emb = None
        
        atom_cfg = embeding_cfg.get('atom', None)
        if atom_cfg is not None:
            self.ligand_atom_emb = get_atom_embedding_func(atom_cfg.type, 
                                                           emb_dim, 
                                                           input_dim = self.num_classes)
            self.protein_atom_emb = get_atom_embedding_func(atom_cfg.type,
                                                            emb_dim,
                                                            input_dim = len(atomic_numbers) + 1) # +1 means is_backbone
        else:
            self.ligand_atom_emb = None
            self.protein_atom_emb = None

        residue_cfg = embeding_cfg.get('residue', None)
        if residue_cfg is not None:
            self.residue_emb = get_res_embedding_func(residue_cfg.type, 
                                                      emb_dim,
                                                      input_dim = len(aa_name_number))
        else:
            self.residue_emb = None

        vec_cfg = embeding_cfg.get('vec', None)
        if vec_cfg is not None:
            self.vec_emb = get_vec_embedding_func(vec_cfg.type, vec_cfg.vec_emb_dim)
        else:
            self.vec_emb = None

        self.ligand_indicator = nn.Linear(1, emb_dim)

    def forward(self, x_lig, x_rec, v_lig, v_rec, aa_rec, batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t=None):
        N_lig, _ = x_lig.shape
        N_rec, _ = x_rec.shape

        if t is not None:
            t_lig = t[batch_idx_lig].unsqueeze(-1).float()
            t_rec = t[batch_idx_rec].unsqueeze(-1).float()
        else:
            t_lig = torch.zeros(N_lig, 1).to(x_lig.device)
            t_rec = torch.zeros(N_rec, 1).to(x_rec.device)

        if self.time_emb is not None:
            t_emb_lig = self.time_emb(t_lig)
            t_emb_rec = self.time_emb(t_rec)
        else:
            t_emb_lig = torch.zeros_like(t_lig).repeat(1, self.emb_dim)
            t_emb_rec = torch.zeros_like(t_rec).repeat(1, self.emb_dim)
        
        if self.ligand_atom_emb is not None:
            if v_lig.dim() == 1:
                v_lig = F.one_hot(v_lig, num_classes = self.num_classes).float()

            h_lig = self.ligand_atom_emb(v_lig)
        else:
            h_lig = torch.zeros_like(v_lig)

        if self.protein_atom_emb is not None:
            if v_rec.dim() == 1:
                v_rec = F.one_hot(v_rec.long(), num_classes = len(atomic_numbers) + 1).float()

            h_rec = self.protein_atom_emb(v_rec)
        else:
            h_rec = torch.zeros_like(v_rec)
        
        if self.residue_emb is not None:
            if aa_rec.dim() == 1:
                aa_rec = F.one_hot(aa_rec, num_classes = len(aa_name_number)).float()

            h_aa = self.residue_emb(aa_rec)
        else:
            h_aa = torch.zeros_like(aa_rec)
        
        bias_lig = self.ligand_indicator(lig_flag.float().unsqueeze(-1))
        bias_rec = self.ligand_indicator(rec_flag.float().unsqueeze(-1))

        h_lig = h_lig + t_emb_lig + bias_lig
        h_rec = h_rec + t_emb_rec + h_aa + bias_rec

        if self.vec_emb is not None:
            x_lig = self.vec_emb(x_lig)
            x_rec = self.vec_emb(x_rec)

        return x_lig, x_rec, h_lig, h_rec