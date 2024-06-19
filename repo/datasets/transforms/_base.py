from repo.utils.molecule.constants import *

TRANSFORM_DICT = {}
def register_transform(name):
    def decorator(cls):
        TRANSFORM_DICT[name] = cls
        return cls
    return decorator

TRANSFORM_WITH_MODE = []
def register_mode_transform(name):
    def decorator(cls):
        TRANSFORM_WITH_MODE.append(name)
        return cls
    return decorator

def get_fg_type(idx, mode='protein_fg_merge', is_aa=True):
    from repo.utils.molecule.fg_constants import num_fg_types

    if mode == 'protein_fg_merge':
        if is_aa:
            idx = idx + num_fg_types
            return idx
        else:
            return idx
    else:
        return idx
    

def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == 'basic':
        return map_atom_type_only_to_index[int(atom_num)]
    elif mode == 'add_aromatic':
        if (int(atom_num), bool(is_aromatic)) in map_atom_type_aromatic_to_index:
            return map_atom_type_aromatic_to_index[int(atom_num), bool(is_aromatic)]
        else:
            print(int(atom_num), bool(is_aromatic))
            return map_atom_type_aromatic_to_index[(1, False)]
    else:
        return map_atom_type_full_to_index[(int(atom_num), str(hybridization), bool(is_aromatic))]
