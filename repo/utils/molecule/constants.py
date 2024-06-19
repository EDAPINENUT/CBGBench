import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

absorbing_state = 0 # set H element type as absorbing type because the model only generate heavy atoms

atom_families = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
atom_families_id = {s: i for i, s in enumerate(atom_families)}
bond_types = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
bond_names = {v: str(k) for k, v in bond_types.items()}
hybridization_type = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
hybridization_type_id = {s: i for i, s in enumerate(hybridization_type)}


aromatic_feat_map_idx = atom_families_id['Aromatic']

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
map_atom_type_full_to_index = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

map_atom_type_only_to_index = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
}

map_atom_type_aromatic_to_index = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

map_index_to_atom_type_only = {v: k for k, v in map_atom_type_only_to_index.items()}
map_index_to_atom_type_aromatic = {v: k for k, v in map_atom_type_aromatic_to_index.items()}
map_index_to_atom_type_full = {v: k for k, v in map_atom_type_full_to_index.items()}

def get_atomic_number_from_index(index, mode):
    if mode == 'basic':
        atomic_number = [map_index_to_atom_type_only[i] for i in index.tolist()]
    elif mode == 'add_aromatic':
        atomic_number = [map_index_to_atom_type_aromatic[i][0] for i in index.tolist()]
    elif mode == 'full':
        atomic_number = [map_index_to_atom_type_full[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number


def is_aromatic_from_index(index, mode):
    if mode == 'add_aromatic':
        is_aromatic = [map_index_to_atom_type_aromatic[i][1] for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [map_index_to_atom_type_full[i][2] for i in index.tolist()]
    elif mode == 'basic':
        is_aromatic = None
    else:
        raise ValueError
    return is_aromatic


single_atom = ['O', 'C', 'N', 'F', 'Cl', 'Br', 'S', 'I', 'P', 'B']
map_atom_symbol_to_atomic_number= {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'P': 15, 'S':16, 'Cl': 17}
map_atomic_number_to_atom_symbol = {v:k for k,v in map_atom_symbol_to_atomic_number.items()}
