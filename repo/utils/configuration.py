from .molecule.constants import *
from .protein.constants import *
from repo.datasets.transforms import TRANSFORM_WITH_MODE    
from repo.utils.molecule.fg_constants import *

type_num_dict = {'basic': len(map_atom_type_only_to_index),
                 'add_aromatic': len(map_atom_type_aromatic_to_index),
                 'add_aromatic_hybrid': len(map_atom_type_full_to_index),
                 'protein_fg_merge': num_fg_types + len(ressymb_to_resindex),
                 'fg_only': num_fg_types}


def set_num_atom_type(config, num_type = None):
    if num_type is not None:
        config.model.num_atomtype = num_type
        return config
    
    elif hasattr(config.data, 'test'): 
        cfg_tsfm = config.data.test.transform
    
    elif hasattr(config.data, 'train'): 
        cfg_tsfm = config.data.train.transform
    
    else: 
        raise ValueError('no mode can be detected, please specific it.')
    
    for tsfm_dict in cfg_tsfm:
        for mode_transform in TRANSFORM_WITH_MODE:
            if tsfm_dict.type == mode_transform:
                mode = tsfm_dict.mode
                break
    try:
        config.model.num_atomtype = type_num_dict[mode]
        config.mode = mode
    except NameError:
        raise ValueError('the mode cannot be inferred automatically, please specific it.')
    
    return config


def set_num_bond_type(config):
    config.model.num_bondtype = 4
    return config

def set_num_fg_type(config, num_type=None):
    if num_type is not None:
        config.model.num_fgtype = num_type
        return config
    
    elif hasattr(config.data, 'test'): 
        cfg_tsfm = config.data.test.transform
    
    elif hasattr(config.data, 'train'): 
        cfg_tsfm = config.data.train.transform
    
    else: 
        raise ValueError('no mode can be detected, please specific it.')
    
    for tsfm_dict in cfg_tsfm:
        for mode_transform in TRANSFORM_WITH_MODE:
            if tsfm_dict.type == mode_transform:
                mode = tsfm_dict.mode
                break
    try:
        config.model.num_fgtype = type_num_dict[mode]
        config.mode = mode
    except NameError:
        raise ValueError('the mode cannot be inferred automatically, please specific it.')
    
    return config