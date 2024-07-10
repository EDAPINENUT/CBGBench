from rdkit import Chem
from repo.datasets.parsers.protein_parser import PDBProteinFA
import numpy as np
import numpy as np
from scipy.spatial.distance import cdist

default_vdw_radii = {
    1: 1.2,  # Hydrogen
    6: 1.7,  # Carbon
    7: 1.55, # Nitrogen
    8: 1.52, # Oxygen
    9: 1.47, # Fluorine
    15: 1.8, # Phosphorus
    16: 1.8, # Sulfur
    17: 2.27, # Chlorine
    35: 1.85 # Bromine
}


def read_sdf(file_path):
    supplier = Chem.SDMolSupplier(file_path)
    molecules = []
    for mol in supplier:
        if mol is not None:
            molecules.append(mol)
    return molecules

def parse_sdf_file(input_mol):
    if type(input_mol) == str:
        mol = read_sdf(input_mol)[0]
    else:
        mol = input_mol
    mol_info = {}
    atomic_type = []
    atomic_number = []
    atomic_coords = []
    # Iterate through each atom in the molecule
    for atom in mol.GetAtoms():
        atomic_type.append(atom.GetSymbol())
        atomic_number.append(atom.GetAtomicNum())
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        atomic_coords.append((pos.x, pos.y, pos.z))

    mol_info['atom_name'] = atomic_type
    mol_info['element'] = np.array(atomic_number)
    mol_info['pos'] = np.array(atomic_coords)
    mol_info['bond_adj'] = np.zeros((len(atomic_type), len(atomic_type)), dtype=bool)
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        mol_info['bond_adj'][start_idx, end_idx] = True
        mol_info['bond_adj'][end_idx, start_idx] = True
    return mol_info

def eval_steric_clash(mol, pdb_file, vdw_radii=default_vdw_radii, tolerance=0.4):
    '''
    This function detects steric clashes between a ligand and a protein based on VdW radii.
    'A clash is considered to occur when the van der Waals radii overlap by ≥ 0.4 Å between non-bound atoms'
                                        -- from Proteopedia: https://proteopedia.org/wiki/index.php/Clashes
    Input:
        mol: RDKit molecule object or path to SDF file containing ligand
        pdb_file: path to PDB file containing protein
        vdw_radii: dictionary of atomic numbers and corresponding VdW radii
        tolerance: tolerance for steric clash detection
    Output:
        clash_detected: boolean indicating whether a steric clash is detected
        additional_info: dictionary containing additional information
    Example:
        clash_detected, additional_info = steric_clash('ligand.sdf', 'protein.pdb')
    '''
    ligand_info = parse_sdf_file(mol)
    protein_info = PDBProteinFA(pdb_file).to_dict_atom()
    protein_pos = np.array(protein_info['pos'])
    ligand_pos = np.array(ligand_info['pos'])
    protein_elements = protein_info['element']
    ligand_elements = ligand_info['element']
    ligand_intra_mask = (~ligand_info['bond_adj']) ^ np.eye(len(ligand_pos), dtype=bool)

    lig_pro_clash_detected, lig_pro_clash_info = detect_clash(ligand_pos, protein_pos, ligand_elements, protein_elements, 
                                                              pair_mask=None, vdw_radii=vdw_radii, tolerance=tolerance)
    lig_lig_clash_detected, lig_lig_clash_info = detect_clash(ligand_pos, ligand_pos, ligand_elements, ligand_elements, 
                                                              pair_mask=ligand_intra_mask, vdw_radii=vdw_radii, tolerance=tolerance)
    
    clash_detected = lig_pro_clash_detected or lig_lig_clash_detected
    additional_info = {
        'lig_pro_clash_detected': lig_pro_clash_detected,
        'lig_lig_clash_detected': lig_lig_clash_detected,
        'lig_pro_clash': lig_pro_clash_info,
        'lig_lig_clash': lig_lig_clash_info
    }
    return clash_detected, additional_info

def detect_clash(pos_dst, 
                 pos_src, 
                 element_dst, 
                 element_src, 
                 pair_mask = None, 
                 vdw_radii = default_vdw_radii, 
                 tolerance=0.4):
    distances = cdist(pos_dst, pos_src)
    if pair_mask is None:
        pair_mask = np.ones((len(pos_dst), len(pos_src)), dtype=bool)

    src_vdw_radii = np.array([vdw_radii[a] for a in element_src])
    dst_vdw_radii = np.array([vdw_radii[a] for a in element_dst])

    vdw_sums_with_tolerance = dst_vdw_radii[:, np.newaxis] + src_vdw_radii - tolerance
    clashes = (distances < vdw_sums_with_tolerance) * pair_mask
    clash_detected = np.any(clashes)

    # Get indices where clashes occur
    clash_indices = np.where(clashes)
    clashed_distances = distances[clash_indices]
    clashed_vdw_sums = vdw_sums_with_tolerance[clash_indices]

    # Additional information to return
    additional_info = {
        'clashed_indices': clash_indices[0],
        'clashed_distances': clashed_distances,
        'clashed_vdw_sums': clashed_vdw_sums,
        'clash_atom_num': len(np.unique(clash_indices[0])),
        'atom_num': len(pos_dst)
    }
    return clash_detected, additional_info





if __name__ == '__main__':
    pdb_file = '1a2g_A_rec.pdb'
    sdf_file = '1a2g_A_rec_4jmv_1ly_lig_tt_min_0.sdf'
    clash_detected, additional_info = eval_steric_clash(sdf_file, pdb_file)
    print(clash_detected)
    print(additional_info)