import torch.nn as nn 
import torch 
import torch.nn.functional as F
from .embs.time_emb import get_time_embedding_func
from .embs.atom_emb import get_atom_embedding_func
from .embs.res_emb import get_res_embedding_func
from repo.utils.molecule.constants import *
from repo.utils.protein.constants import *


def get_context_encoder(cfg):
    if cfg.get('type', 'merge') == 'none':
        return MergeContextEncoder(cfg)
    else:
        return PLContextEncoder(cfg)

class MergeContextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pass

class PLContextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embeding_cfg = cfg
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
                                                           input_dim = len(map_atom_type_aromatic_to_index))
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

        self.ligand_indicator = nn.Linear(1, emb_dim)

    def forward(self, x_lig, x_rec, v_lig, v_rec, aa_rec, batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t):
        N_lig, _ = x_lig.shape
        N_rec, _ = x_rec.shape

        t_lig = t[batch_idx_lig].unsqueeze(-1).float()
        t_rec = t[batch_idx_rec].unsqueeze(-1).float()

        if self.time_emb is not None:
            t_emb_lig = self.time_emb(t_lig)
            t_emb_rec = self.time_emb(t_rec)
        else:
            t_emb_lig = torch.zeros_like(t_lig).repeat(1, self.emb_dim)
            t_emb_rec = torch.zeros_like(t_rec).repeat(1, self.emb_dim)
        
        if self.ligand_atom_emb is not None:
            h_lig = self.ligand_atom_emb(v_lig)
        else:
            h_lig = torch.zeros_like(v_lig)

        if self.protein_atom_emb is not None:
            h_rec = self.protein_atom_emb(v_rec)
        else:
            h_rec = torch.zeros_like(v_rec)
        
        if self.residue_emb is not None:
            h_aa = self.residue_emb(aa_rec)
        else:
            h_aa = torch.zeros_like(aa_rec)
        
        bias_lig = self.ligand_indicator(lig_flag.float().unsqueeze(-1))
        bias_rec = self.ligand_indicator(rec_flag.float().unsqueeze(-1))

        h_lig = h_lig + t_emb_lig + bias_lig
        h_rec = h_rec + t_emb_rec + h_aa + bias_rec

        return x_lig, x_rec, h_lig, h_rec