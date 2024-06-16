from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem, TorsionFingerprints
# from .eval_bond_angle_config import *
from scipy import spatial as sci_spatial
from repo.utils.molecule.constants import *
from tqdm.auto import tqdm
import pickle
from repo.tools.geometry.eval_bond_angle_config import EMPIRICAL_DISTRIBUTIONS

from itertools import combinations
from typing import Tuple, Sequence, Dict, Optional
import numpy as np
import collections

AngleType = Tuple[int, int, int, int, int]  # (atomic_num, bond_type, atomic_num, bond_type, atomic_num)

def get_bond_type(bond):
    return bond_types[bond.GetBondType()]

def get_bond_str(bond_type):
    if bond_type == 1:
        return '-'
    elif bond_type == 2:
        return '='
    elif bond_type == 3:
        return '#'
    else:
        return '?'

BondAngleData = Tuple[AngleType, float] 

def get_distribution(angles: Sequence[float], bins=np.arange(0, 180, 2)) -> np.ndarray:

    bin_counts = collections.Counter(np.searchsorted(bins, angles))
    bin_counts = [bin_counts[i] if i in bin_counts else 0 for i in range(len(bins) + 1)]
    bin_counts = np.array(bin_counts) / np.sum(bin_counts)
    return bin_counts

BondAngleProfile = Dict[AngleType, np.ndarray] 

def get_bond_angle_profile(bond_angles: Sequence[BondAngleData]) -> BondAngleProfile:
    bond_angle_profile = collections.defaultdict(list)
    for bond_type, bond_angle in bond_angles:
        bond_angle_profile[bond_type].append(bond_angle)
    bond_angle_profile = {k: get_distribution(v) for k, v in bond_angle_profile.items()}
    return bond_angle_profile

def _angle_type_str(bond_type: AngleType) -> str:
    atom1, bond12, atom2, bond23, atom3 = bond_type
    return f'{atom1}{get_bond_str(bond12)}{atom2}{get_bond_str(bond23)}{atom3}'

def eval_bond_angle_profile(bond_angle_profile: BondAngleProfile) -> Dict[str, Optional[float]]:
    metrics = {}

    # Jensen-Shannon distances
    for angle_type, gt_distribution in EMPIRICAL_DISTRIBUTIONS.items():
        if angle_type not in bond_angle_profile:
            metrics[f'JSD_{_angle_type_str(angle_type)}'] = None
        else:
            metrics[f'JSD_{_angle_type_str(angle_type)}'] = sci_spatial.distance.jensenshannon(gt_distribution,
                                                                                               bond_angle_profile[
                                                                                               angle_type])

    return metrics

def bond_angle_from_mol(mol):
    angles = []
    types = [] # double counting
    conf =  mol.GetConformer(id=0)
    for atom in mol.GetAtoms():
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 2:  # At least two neighbors are required to form an angle
            continue
        for a1, a2 in combinations(neighbors, 2):
            angle = rdMolTransforms.GetAngleDeg(conf, a1, atom.GetIdx(), a2)
            tup = (mol.GetAtomWithIdx(a1).GetAtomicNum(), 
                   get_bond_type(mol.GetBondBetweenAtoms(a1, atom.GetIdx())), 
                   atom.GetAtomicNum(), 
                   get_bond_type(mol.GetBondBetweenAtoms(atom.GetIdx(), a2)), 
                   mol.GetAtomWithIdx(a2).GetAtomicNum())
            idx = (a1, atom.GetIdx(), a2)
            angles.append(angle)
            types.append(tup)

            tup_rev = (mol.GetAtomWithIdx(a2).GetAtomicNum(), 
                       get_bond_type(mol.GetBondBetweenAtoms(a2, atom.GetIdx())), 
                       atom.GetAtomicNum(), 
                       get_bond_type(mol.GetBondBetweenAtoms(atom.GetIdx(), a1)), 
                       mol.GetAtomWithIdx(a1).GetAtomicNum())
            types.append(tup_rev)
            angles.append(angle)

    return [(type, angle) for type, angle in zip(types, angles)]

def statis_angles(mols):
    statis = {}
    for mol in mols:
        types_angles = bond_angle_from_mol(mol)
        for i, tetra_type in enumerate(types_angles):
            if tetra_type[0] not in statis:
                statis[tetra_type[0]] = [tetra_type[1]]
            else:
                statis[tetra_type[0]].append(tetra_type[1])
    return statis
    

if __name__ == '__main__':
    import torch 
    raw_path = '/usr/commondata/public/conformation_generation/data/crossdocked_v1.1_rmsd1.0_pocket10'
    index = torch.load('/usr/commondata/public/conformation_generation/CGBBench/data/split_by_name_10m.pt')
    index = index['train'][:]
    mols = []
    for i, (pocket_fn, ligand_fn) in enumerate(tqdm(index, dynamic_ncols=True, desc='Get mol from sdf')):
        ligand_path = os.path.join(raw_path, ligand_fn)
        try:
            if ligand_path.endswith('.sdf'):
                rdmol = Chem.MolFromMolFile(ligand_path, sanitize=False)
            elif ligand_path.endswith('.mol2'):
                rdmol = Chem.MolFromMol2File(ligand_path, sanitize=False)
            rdmol = Chem.RemoveHs(rdmol)
            mols.append(rdmol)
        except:
            pass

    statis_saved = statis_angles(mols)
    dist_angles = {}
    for key, val in statis_saved.items():
        distribution = get_distribution(val)
        dist_angles[key] = distribution

    dist_angles_filtered = {}

    def get_bond_num(k):
        return [k[1], k[3]]
    def get_atomic_num(k):
        return [k[0], k[2], k[4]]

    for k,v in dist_angles.items():
        is_in = True
        for bond_num in get_bond_num(k):
            if bond_num not in [1,2,3]:
                is_in = False
        for atomic_num in get_atomic_num(k):
            if atomic_num not in [1, 6, 7, 8, 9, 15, 16, 17]:
                is_in = False
        if is_in:
            dist_angles_filtered[k] = np.array(v)


    np.save('dist_angles', dist_angles_filtered)
    # import shutil
    # for i,  (pocket_fn, ligand_fn) in enumerate(index):
    #     shutil.copy(os.path.join(raw_path, pocket_fn), '/usr/commondata/public/conformation_generation/CGBBench/test_pdb/')