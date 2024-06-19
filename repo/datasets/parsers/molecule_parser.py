from rdkit import Chem
import os
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from repo.utils.molecule.constants import *
import numpy as np
from .frag import *
from .frame_fg import *
from EFGs import mol2frag
# % pip install --use-pep517 efgs
from repo.utils.molecule.fg_constants import *
from torch_geometric.utils import coalesce
from .mol_tree import *

ALIGNED_RMSD = 0.3

def parse_sdf_file(path):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    # Remove Hydrogens.
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(atom_families)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), atom_families_id[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        # if atom_num == 35:
        #     print(atom_num)
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int64)

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_types[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.longlong)
    edge_type = np.array(edge_type, dtype=np.longlong)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization
    }
    return data

def parse_sdf_file_moltree(path):

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)
    moltree = MolTree(rdmol)

    # Remove Hydrogens.
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(atom_families)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), atom_families_id[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        # if atom_num == 35:
        #     print(atom_num)
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int64)

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_types[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.longlong)
    edge_type = np.array(edge_type, dtype=np.longlong)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    neighbor_list = []

    #used in rotation angle prediction
    for i, atom in enumerate(rdmol.GetAtoms()):
        neighbor_list.append([n.GetIdx() for n in atom.GetNeighbors()])

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization,
        'moltree': moltree,
        'neighbor_list': neighbor_list
    }
    return data

def get_substructure_bond_info(mol, sub_node_idx):
    edge_index = []
    edge_type = []
    for bond in mol.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        e_sym = bond.GetEndAtom().GetAtomicNum()
        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_types[bond.GetBondType()]

        if s_idx in sub_node_idx and e_idx in sub_node_idx:
            edge_index.append([s_idx, e_idx])
            edge_type.append(bond_type)
            edge_index.append([e_idx, s_idx])
            edge_type.append(bond_type)
    edge_index, edge_type = coalesce(torch.tensor(edge_index).transpose(0,1),torch.tensor(edge_type))
    return edge_index.numpy(), edge_type.numpy()

def get_cross_bond_info(mol, sub_node_idx1, sub_node_idx2):
    edge_index = []
    edge_type = []
    for bond in mol.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        e_sym = bond.GetEndAtom().GetAtomicNum()
        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_types[bond.GetBondType()]

        if s_idx in sub_node_idx1 and e_idx in sub_node_idx2:
            edge_index.append([s_idx, e_idx])
            edge_type.append(bond_type)
        if s_idx in sub_node_idx2 and e_idx in sub_node_idx1:
            edge_index.append([s_idx, e_idx])
            edge_type.append(bond_type)
    edge_index, edge_type = coalesce(torch.tensor(edge_index).transpose(0,1),torch.tensor(edge_type))
    return edge_index.numpy(), edge_type.numpy()


def linker_decomp(mol):
    '''
    Input: mol
    Output: decomp_infos
    '''
    Chem.SanitizeMol(mol)
    fragmentations = linkerize_mol(mol)
    fragmentations = check_linkers(fragmentations)
    decomp_infos = []
    for frag_idx, fragmentation in enumerate(fragmentations):
        try:
            linker, frags = fragmentation
            frag_mols = qsmis([frags])[0]
            frag_mols, _ = remove_dummys_mol(frag_mols)
            frag_mols, _ = remove_dummys_mol(frag_mols)
            frag_3d = transfer_coord(frag_mols, mol)
            anchor_in_frag, anchor_in_mol = find_anchor_indices_3d(mol, frag_3d)
            # frag1_id, frag2_id = anchor_in_frag 
            # frag_3d.SetProp('anchor_ids',f'{frag1_id}_{frag2_id}')
            frag_atom_mapping = get_atom_map_3d(mol, frag_3d)
            frag_in_mol_index = [i[1] for i in frag_atom_mapping]
            frag_atoms = np.array([frag_3d.GetAtomWithIdx(i).GetAtomicNum() for i in range(frag_3d.GetNumAtoms())])
            frag_pos = get_mol_coord(frag_3d)

            linker_mol = qsmis([linker])[0]
            linker_mol, _ = remove_dummys_mol(linker_mol)
            linker_mol, _ = remove_dummys_mol(linker_mol)
            linker_mol = rm_radical(linker_mol)
            linker_3d = transfer_coord(linker_mol, mol)
            linker_atom_mapping = get_atom_map_3d(mol, linker_3d)
            linker_in_mol_index = [i[1] for i in linker_atom_mapping]
            linker_atoms = np.array([linker_3d.GetAtomWithIdx(i).GetAtomicNum() for i in range(linker_3d.GetNumAtoms())])
            linker_pos = get_mol_coord(linker_3d)
            frag_edge_info = get_substructure_bond_info(mol, frag_in_mol_index)
            linker_edge_info = get_substructure_bond_info(mol, linker_in_mol_index)
            cross_edge_info = get_cross_bond_info(mol, frag_in_mol_index, linker_in_mol_index)
            decomp_info = {
                'gen_mol': linker_3d,
                'gen_atoms': linker_atoms,
                'gen_pos': linker_pos,
                'gen_index_in_mol': linker_in_mol_index,
                'ctx_mol': frag_3d,
                'ctx_atoms': frag_atoms,
                'ctx_pos': frag_pos,
                'ctx_index_in_mol': frag_in_mol_index,
                'anchor_id': anchor_in_mol,
                'ctx_edge_info': frag_edge_info,
                'gen_edge_info': linker_edge_info,
                'cross_edge_info': cross_edge_info # fragment is the condition
            }
            decomp_infos.append(decomp_info)
        except Exception as e:
            print(e)
    return decomp_infos


def fragment_decomp(mol):
    Chem.SanitizeMol(mol)
    fragmentations = fragmentize_mol(mol)
    fragmentations = check_frags(fragmentations)
    decomp_infos = []
    for frag_idx, fragmentation in enumerate(fragmentations):
        try:
            frag_small, frag_large = frag2mols(fragmentation)
            frag_small_3d = transfer_coord(frag_small, mol)
            anchor_in_frag, anchor_in_mol = find_anchor_indices_3d(mol, frag_small_3d)
            frag_small_atom_mapping = get_atom_map_3d(mol, frag_small_3d)
            frag_small_in_mol_index = [i[1] for i in frag_small_atom_mapping]
            frag_small_atoms = torch.tensor([frag_small_3d.GetAtomWithIdx(i).GetAtomicNum() for i in range(frag_small_3d.GetNumAtoms())])
            frag_small_pos = get_mol_coord(frag_small_3d)

            frag_large_3d = transfer_coord(frag_large, mol)
            frag_large_atom_mapping = get_atom_map_3d(mol, frag_large_3d)
            frag_large_in_mol_index = [i[1] for i in frag_large_atom_mapping]
            frag_large_atoms = torch.tensor([frag_large_3d.GetAtomWithIdx(i).GetAtomicNum() for i in range(frag_large_3d.GetNumAtoms())])
            frag_large_pos = get_mol_coord(frag_large_3d)

            frag_samll_edge_info = get_substructure_bond_info(mol, frag_small_in_mol_index)
            frag_large_edge_info = get_substructure_bond_info(mol, frag_large_in_mol_index)
            cross_edge_info = get_cross_bond_info(mol, frag_small_in_mol_index, frag_large_in_mol_index)

            decomp_info = {
                'gen_mol': frag_small_3d,
                'gen_atoms': frag_small_atoms,
                'gen_pos': frag_small_pos,
                'gen_index_in_mol': frag_small_in_mol_index,
                'ctx_mol': frag_large_3d,
                'ctx_atoms': frag_large_atoms,
                'ctx_pos': frag_large_pos,
                'ctx_index_in_mol': frag_large_in_mol_index,
                'anchor_id': anchor_in_mol,
                'ctx_edge_info': frag_large_edge_info,
                'gen_edge_info': frag_samll_edge_info,
                'cross_edge_info': cross_edge_info # fragment is the condition
            }
            decomp_infos.append(decomp_info)
        except Exception as e:
            print(e)
    return decomp_infos


def scaffold_decomp(mol):
    Chem.SanitizeMol(mol)
    scaffold, _ = Murcko_decompose_anchor(mol)
    scaffold = rm_radical(scaffold)
    scaffold_mol_mapping_list = get_atom_map_3d(mol, scaffold)
    scaffold_in_mol_index = [i[1] for i in scaffold_mol_mapping_list]
    scaffold_atoms = torch.tensor([scaffold.GetAtomWithIdx(i).GetAtomicNum() for i in range(scaffold.GetNumAtoms())])
    scaffold_pos = get_mol_coord(scaffold)

    side_chains, anchor_idx = remove_substructure(mol, sub_mol=scaffold)
    anchor_in_frag, anchor_in_mol = find_anchor_indices_3d(mol, side_chains)
    side_chains_mol_mapping_list = get_atom_map_3d(mol, side_chains)
    side_chains_in_mol_index = [i[1] for i in side_chains_mol_mapping_list]
    side_chains_atoms = torch.tensor([side_chains.GetAtomWithIdx(i).GetAtomicNum() for i in range(side_chains.GetNumAtoms())])
    side_chains_pos = get_mol_coord(side_chains)

    scaffold_edge_info = get_substructure_bond_info(mol, scaffold_in_mol_index)
    side_chains_edge_info = get_substructure_bond_info(mol, side_chains_in_mol_index)
    cross_edge_info = get_cross_bond_info(mol, scaffold_in_mol_index, side_chains_in_mol_index)

    decomp_info = {
        'gen_atoms': scaffold_atoms,
        'gen_pos': scaffold_pos,
        'gen_index_in_mol': scaffold_in_mol_index,
        'ctx_atoms': side_chains_atoms,
        'ctx_pos': side_chains_pos,
        'ctx_index_in_mol': side_chains_in_mol_index,
        'anchor_id': anchor_in_mol,
        'ctx_edge_info': side_chains_edge_info,
        'gen_edge_info': scaffold_edge_info,
        'cross_edge_info': cross_edge_info # fragment is the condition
    }
    return [decomp_info] # only one scaffold decomposition


def sidechain_decomp(mol):
    Chem.SanitizeMol(mol)
    scaffold, _ = Murcko_decompose_anchor(mol)
    scaffold = rm_radical(scaffold)
    scaffold_mol_mapping_list = get_atom_map_3d(mol, scaffold)
    scaffold_in_mol_index = [i[1] for i in scaffold_mol_mapping_list]
    scaffold_atoms = torch.tensor([scaffold.GetAtomWithIdx(i).GetAtomicNum() for i in range(scaffold.GetNumAtoms())])
    scaffold_pos = get_mol_coord(scaffold)

    side_chains, anchor_idx = remove_substructure(mol, sub_mol=scaffold)
    anchor_in_frag, anchor_in_mol = find_anchor_indices_3d(mol, scaffold)
    side_chains_mol_mapping_list = get_atom_map_3d(mol, side_chains)
    side_chains_in_mol_index = [i[1] for i in side_chains_mol_mapping_list]
    side_chains_atoms = torch.tensor([side_chains.GetAtomWithIdx(i).GetAtomicNum() for i in range(side_chains.GetNumAtoms())])
    side_chains_pos = get_mol_coord(side_chains)

    scaffold_edge_info = get_substructure_bond_info(mol, scaffold_in_mol_index)
    side_chains_edge_info = get_substructure_bond_info(mol, side_chains_in_mol_index)
    cross_edge_info = get_cross_bond_info(mol, side_chains_in_mol_index, scaffold_in_mol_index)

    decomp_info = {
        'ctx_atoms': scaffold_atoms,
        'ctx_pos': scaffold_pos,
        'ctx_index_in_mol': scaffold_in_mol_index,
        'gen_atoms': side_chains_atoms,
        'gen_pos': side_chains_pos,
        'gen_index_in_mol': side_chains_in_mol_index,
        'anchor_id': anchor_in_mol,
        'gen_edge_info': side_chains_edge_info,
        'ctx_edge_info': scaffold_edge_info,
        'cross_edge_info': cross_edge_info # fragment is the condition
    }
    return [decomp_info] # only one scaffold decomposition


def decomp_parse_sdf_file(path, decomp_type='linker'):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(atom_families)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), atom_families_id[feat.GetFamily()]] = 1

    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        # if atom_num == 35:
        #     print(atom_num)
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int64)

    if decomp_type == 'linker':
        decomp_infos = linker_decomp(mol=rdmol)
    elif decomp_type == 'frag':
        decomp_infos = fragment_decomp(mol=rdmol)
    elif decomp_type == 'sidechain':
        decomp_infos = sidechain_decomp(mol=rdmol)
    elif decomp_type == 'scaffold':
        decomp_infos = scaffold_decomp(mol=rdmol)
    else:
        raise NotImplementedError()
    
    if len(decomp_infos) == 0:
        return None
    
    gen_index_list = []
    ctx_index_list = []
    gen_bond_index_list = []
    ctx_bond_index_list = []
    cross_bond_index_list = []
    gen_bond_type_list = []
    ctx_bond_type_list = []
    cross_bond_type_list = []

    for decomp_info in decomp_infos:
        gen_index = np.array(decomp_info['gen_index_in_mol'])
        ctx_index = np.array(decomp_info['ctx_index_in_mol'])
        gen_bond_index, gen_bond_type = decomp_info['gen_edge_info']
        ctx_bond_index, ctx_bond_type = decomp_info['ctx_edge_info']
        cross_bond_index, cross_bond_type = decomp_info['cross_edge_info']
        gen_index_list.append(gen_index)
        ctx_index_list.append(ctx_index)
        gen_bond_index_list.append(np.array(gen_bond_index))
        gen_bond_type_list.append(np.array(gen_bond_type))
        ctx_bond_index_list.append(np.array(ctx_bond_index))
        ctx_bond_type_list.append(np.array(ctx_bond_type))
        cross_bond_index_list.append(np.array(cross_bond_index))
        cross_bond_type_list.append(np.array(cross_bond_type))
        
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_types[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.longlong)
    edge_type = np.array(edge_type, dtype=np.longlong)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'gen_index': gen_index_list,
        'ctx_index': ctx_index_list,
        'gen_bond_index': gen_bond_index_list,
        'gen_bond_type': gen_bond_type_list,
        'ctx_bond_index': ctx_bond_index_list,
        'ctx_bond_type': ctx_bond_type_list,
        'cross_bond_index': cross_bond_index_list,
        'cross_bond_type': cross_bond_type_list,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization
        }
    return data

def parse_sdf_file_to_functional_group_linker(path):

    data = {}

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    fg, single_c, fg_idx, single_c_idx = mol2frag(rdmol, returnidx=True)

    
    fg = fg + single_c
    fg_idx = fg_idx + single_c_idx

    fg_center_pos = []
    v_fgs = []
    type_fgs = []
    pos_fgs = []
    center_fgs = []

    linker_idx = []
    ctx_idx = []

    for fg_smile, idx in zip(fg, fg_idx):    

        fg_pos, atomic_num, atom_type = get_fg_pos_and_type(rdmol, idx)

        if fg_smile in merge_c:
            fg_smile = 'C'
        
        if (fg_smile in fragment_factory):
            fg_frame_vec, fg_type, fg_center_pos, fg_frame_pos = prepare_fg_atom(fg_smile, fg_pos)
            v_fgs.append(fg_frame_vec)
            type_fgs.append(fg_type)
            center_fgs.append(fg_center_pos)
            pos_fgs.append(fg_frame_pos)

        if fg_smile in fragment_factory:
            ctx_idx.append(idx)
        else:
            linker_idx.append(idx)

    pos_fg_pad = []
    mask_fg_pad = []

    for pos in pos_fgs:
        fg_atom_num = len(pos)

        pos_pad = np.zeros([max_num_heavyatoms, 3],dtype=np.float32)
        mask_pad = np.zeros([max_num_heavyatoms, ], dtype=np.bool_)
    
        pos_pad[:fg_atom_num] = pos
        mask_pad[:fg_atom_num] = True  

        pos_fg_pad.append(pos_pad)
        mask_fg_pad.append(mask_pad)

    fg_type_encode = [fg2class_dict[fg_smile] for fg_smile in type_fgs]

    assert (
        len(center_fgs) 
        == len(pos_fg_pad) 
        == len(mask_fg_pad)
        )
    
    if len(center_fgs) == 0:
        data['fg'] = {
            'pos_center': np.array(center_fgs, dtype=np.float32), 
            'pos_heavyatom': np.array(pos_fg_pad),
            'mask_heavyatom': np.array(mask_fg_pad, dtype=np.bool_),
            'type_fg': np.array(fg_type_encode, dtype=np.int64), 
            'o_fg': np.array(v_fgs, dtype=np.bool_)
        }
    else:
        data['fg'] = {
            'pos_center': np.array(center_fgs, dtype=np.float32), 
            'pos_heavyatom': np.stack(pos_fg_pad, axis=0),
            'mask_heavyatom': np.stack(mask_fg_pad, axis=0, dtype=np.bool_),
            'type_fg': np.array(fg_type_encode, dtype=np.int64), 
            'o_fg': np.array(v_fgs, dtype=np.float32)
            }

    ctx_idx = np.concatenate(ctx_idx, axis=0)
    linker_idx = np.concatenate(linker_idx, axis=0)
    ptable = Chem.GetPeriodicTable()

    rd_num_atoms = rdmol.GetNumAtoms()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int64)
    
    gen_index_list = []
    ctx_index_list = []
    gen_bond_index_list = []
    ctx_bond_index_list = []
    cross_bond_index_list = []
    gen_bond_type_list = []
    ctx_bond_type_list = []
    cross_bond_type_list = []

    gen_index = linker_idx
    ctx_index = ctx_idx
    gen_bond_index, gen_bond_type = get_substructure_bond_info(rdmol, gen_index)
    ctx_bond_index, ctx_bond_type = get_substructure_bond_info(rdmol, ctx_index)
    cross_bond_index, cross_bond_type = get_cross_bond_info(rdmol, gen_index, ctx_index)
    gen_index_list.append(np.array(gen_index))
    ctx_index_list.append(np.array(ctx_index))
    gen_bond_index_list.append(np.array(gen_bond_index))
    gen_bond_type_list.append(np.array(gen_bond_type))
    ctx_bond_index_list.append(np.array(ctx_bond_index))
    ctx_bond_type_list.append(np.array(ctx_bond_type))
    cross_bond_index_list.append(np.array(cross_bond_index))
    cross_bond_type_list.append(np.array(cross_bond_type))

    feat_mat = np.zeros([rd_num_atoms, len(atom_families)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), atom_families_id[feat.GetFamily()]] = 1

    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    data['linker'] = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'gen_index': gen_index_list,
        'ctx_index': ctx_index_list,
        'gen_bond_index': gen_bond_index_list,
        'gen_bond_type': gen_bond_type_list,
        'ctx_bond_index': ctx_bond_index_list,
        'ctx_bond_type': ctx_bond_type_list,
        'cross_bond_index': cross_bond_index_list,
        'cross_bond_type': cross_bond_type_list,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization
        }
    

    other_ratio =  len([fg_smile for fg_smile in type_fgs if fg_smile == 'Others'])/len(type_fgs) if len(type_fgs) > 0 else 1.0

    if other_ratio > 0.5:
        return None
        
    return data

def get_fg_pos_and_type(rdmol, idx):
    fg_pos = []
    atomic_nums = []
    atom_types = []
    c = rdmol.GetConformer()
    for atom_idx in idx:
        atomic_num = rdmol.GetAtomWithIdx(atom_idx).GetAtomicNum()
        atom_type = rdmol.GetAtomWithIdx(atom_idx).GetSymbol() 
        pos = c.GetAtomPosition(atom_idx)
        fg_pos.append([pos.x, pos.y, pos.z])
        atomic_nums.append(atomic_num)
        atom_types.append(atom_type)
    return fg_pos, np.array(atomic_nums), np.array(atom_types)

def prepare_single_atom(fg_smile, fg_pos):
    v = [0,0,0]
    type = fg_smile
    center_pos = fg_pos[0]
    frame_vec = v
    frame_pos = [[0,0,0],fg_pos[0]]
    return type, center_pos, frame_vec, frame_pos

def prepare_fg_atom(fg_smile, fg_pos):
    center, R, v, local_pos, framed_mol, rearrange_global_pos, idx_re = transform_into_fg_data(fg_smile, fg_pos)
    if fg_smile == 'NS(=O)=O':
        rmsd = Chem.rdMolAlign.CalcRMS(framed_mol, ref_nso2_c1)
        if rmsd <= ALIGNED_RMSD:
            fg_frame_vec = v
            fg_type = nso2_chirality1
            fg_center_pos = center
            fg_frame_pos= rearrange_global_pos
        else:
            fg_frame_vec = v
            fg_type = nso2_chirality2
            fg_center_pos = center
            fg_frame_pos = rearrange_global_pos

    elif fg_smile == 'O=CNO':
        rmsd = Chem.rdMolAlign.CalcRMS(framed_mol, ref_ocno_c1)
        if rmsd <= ALIGNED_RMSD:
            fg_frame_vec = v
            fg_type = ocno_chirality1
            fg_center_pos = center
            fg_frame_pos = rearrange_global_pos

        else:
            fg_frame_vec = v
            fg_type = ocno_chirality2
            fg_center_pos = center     
            fg_frame_pos = rearrange_global_pos
    
    else:   
        fg_frame_vec = v
        fg_type = fg_smile
        fg_center_pos = center
        fg_frame_pos = rearrange_global_pos        
    return fg_frame_vec, fg_type, fg_center_pos, fg_frame_pos

def conf_with_smiles(smiles, positions):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (positions[i][0], positions[i][1], positions[i][2]))
    mol.AddConformer(conf)
    return mol

def prepare_linker_atom(fg_smile, fg_pos):
    rdmol = conf_with_smiles(fg_smile, fg_pos)
    positions = rdmol.GetConformer(0).GetPositions()
    atom_types = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    atomic_numbers = [atom.GetAtomicNum() for atom in rdmol.GetAtoms()]
    return positions, atom_types, atomic_numbers