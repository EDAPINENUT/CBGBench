from ._base import register_transform
import torch
from repo.utils.protein.constants import BBHeavyAtom

@register_transform('center_pos')
class CenterPos(object):
    def __init__(self, center_flag, mask_flag=None):
        self.center_flag = center_flag
        self.mask_flag = mask_flag

    def __call__(self, data):
        data_flag = data[self.center_flag]
        if self.mask_flag is not None and data_flag[self.mask_flag].sum() > 0:
            data_center = data_flag.pos[data_flag[self.mask_flag]].mean(dim=0, keepdim=True)
        else:
            data_center = data_flag.pos.mean(dim=0, keepdim=True)
            
        data.protein.pos = data.protein.pos - data_center
        data.protein.translation = data_center.expand(data.protein.pos.size(0), -1)
        if hasattr(data, 'ligand'):
            if hasattr(data.ligand, 'pos'):
                data.ligand.pos = data.ligand.pos - data_center
                data.ligand.translation = data_center.expand(data.ligand.pos.size(0), -1)

        return data
    
@register_transform('center_whole_pos')
class CenterPos(object):
    def __init__(self):
        pass 

    def __call__(self, data):
        protein_pos = data['protein'].pos
        
        if hasattr(data, 'ligand') and hasattr(data.ligand, 'pos'):
            ligand_pos = data['ligand'].pos
            data_center = (ligand_pos.sum(0) + protein_pos.sum(0)) / (len(ligand_pos) + len(protein_pos))
        else:
            data_center = protein_pos.mean(0)
        
        data_center = data_center.unsqueeze(0)
        
        data['protein'].pos = data['protein'].pos - data_center
        data['protein'].translation = data_center.expand(data['protein'].pos.size(0), -1)
        
        if hasattr(data, 'ligand') and hasattr(data.ligand, 'pos'):
            data['ligand'].pos = data['ligand'].pos - data_center
            data['ligand'].translation = data_center.expand(data['ligand'].pos.size(0), -1)

        return data


@register_transform('center_frame_pos')
class CenterFramePos(object):
    def __init__(self, center_flag):
        self.center_flag = center_flag

    def __call__(self, data):
        data_flag = data[self.center_flag]
        data_frame_pos = data_flag.pos_heavyatom[:, BBHeavyAtom.CA]

        data_center = data_frame_pos.mean(dim=0, keepdim=True)
        data.protein.pos_heavyatom = (data.protein.pos_heavyatom - data_center) * data.protein.mask_heavyatom.unsqueeze(-1)

        data.protein.translation = data_center.expand(data.protein.pos_heavyatom.size(0), -1)

        if hasattr(data, 'ligand'):
            if hasattr(data.ligand, 'pos_heavyatom'):
                data.ligand.pos_heavyatom = (data.ligand.pos_heavyatom - data_center) * data.ligand.mask_heavyatom.unsqueeze(-1)
                data.ligand.translation = data_center.expand(data.ligand.pos_heavyatom.size(0), -1)

        return data
    
@register_transform('add_pos_noise')
class AddPosNoise(object):

    def __init__(self, noise_std, graph_name='protein', frame_mode=False):
        self.noise_std = noise_std
        self.graph_name = graph_name
        self.frame_mode = frame_mode

    def __call__(self, data):
        if self.frame_mode:
            mask_heavyatom = data[self.graph_name].mask_heavyatom
            pos_heavyatom = data[self.graph_name].pos_heavyatom
            data[self.graph_name].pos_heavyatom = pos_heavyatom + torch.randn_like(pos_heavyatom) * self.noise_std * mask_heavyatom.unsqueeze(-1)
        else:
            data[self.graph_name].pos = data[self.graph_name].pos + torch.randn_like(data[self.graph_name].pos) * self.noise_std
        return data

