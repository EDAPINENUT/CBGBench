from rdkit import Chem
from rdkit.Chem import AllChem 
from rdkit.Chem.rdMolAlign import CalcRMS
import numpy as np
import copy
from rdkit import Chem, Geometry
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition
import pickle
import os.path as osp
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def read_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file)
    mols_list = [i for i in supp]
    return mols_list

def write_sdf(mol_list,file):
    writer = Chem.SDWriter(file)
    for i in mol_list:
        writer.write(i)
    writer.close()

def read_pkl(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(list,file):
    with open(file,'wb') as f:
        pickle.dump(list,f)
        print('pkl file saved at {}'.format(file))

def combine_mols(mols):
    ref_mol = mols[0]
    for add_mol in mols[1:]:
        ref_mol = Chem.CombineMols(ref_mol,add_mol)
    return ref_mol
    
def Murcko_decompose(mol, visualize=False):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    decompose = rdRGroupDecomposition.RGroupDecompose([scaffold], [mol])
    side_chains = []
    decompose = list(decompose[0][0].values())
    for i, rgroup in enumerate(decompose):
        if i >0:
            if visualize:
                side_chains.append(rgroup)
            else:
                rgroup, id = remove_mark_mol(rgroup)
                side_chains.append(rgroup)

    if visualize:
        scaffold = decompose[0]

    return scaffold, side_chains

def Murcko_decompose_anchor(mol):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    decompose = rdRGroupDecomposition.RGroupDecompose([scaffold], [mol])
    scaffold = decompose[0][0]['Core']
    num_attch = len(get_mark(scaffold))
    attch_ids = []
    for i in range(num_attch):
        scaffold, attch_idx = remove_mark_mol(scaffold)
        attch_ids.append(attch_idx)
    return scaffold, attch_ids

def remove_substructure(mol, sub_mol):
    # Create a molecule object for the substructure
    substructure = sub_mol
    if type(sub_mol) == str:
        substructure = Chem.MolFromSmiles(sub_mol)
    
    # Find occurrences of the substructure in the molecule
    matches = mol.GetSubstructMatches(substructure)
    
    # Convert the molecule to an editable molecule
    emol = Chem.EditableMol(mol)
    
    # Record the atoms to be removed
    atoms_to_remove = set()
    for match in matches:
        for atom_idx in match:
            atoms_to_remove.add(atom_idx)
    
    # Remove atoms and create mapping
    old_to_new_idx = {}
    new_idx = 0
    breaking_points_in_modified_mol = []
    for old_idx in range(mol.GetNumAtoms()):
        if old_idx in atoms_to_remove:
            continue  # Skip atoms that are removed
        old_to_new_idx[old_idx] = new_idx
        new_idx += 1

    # Identify the breaking points in terms of the modified molecule
    for atom_idx in atoms_to_remove:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in atoms_to_remove:  # Only consider atoms outside the substructure
                breaking_points_in_modified_mol.append(old_to_new_idx[neighbor_idx])
    
    # Remove atoms
    for atom_idx in sorted(atoms_to_remove, reverse=True):
        emol.RemoveAtom(atom_idx)
    
    # Get the modified molecule
    modified_mol = emol.GetMol()
    
    return modified_mol, breaking_points_in_modified_mol



def HeriS_scaffold(mol):
    import scaffoldgraph as sg #pip install scaffoldgraph 
    network = sg.HierS.from_sdf('example.sdf', progress=True)
    scaffold = list(network.get_scaffold_nodes())
    return scaffold

def qsmis(smis):
    return [Chem.MolFromSmiles(i) for i in smis]

def fragmentize_mol(mol, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]", asmol=False):
    fragmentations = rdMMPA.FragmentMol(mol, minCuts=1, maxCuts=1, maxCutBonds=100, pattern=pattern, resultsAsMols=asmol)
    return fragmentations #(linker, frags) * n 

def linkerize_mol(mol, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]",asmol=False):
    fragmentations = rdMMPA.FragmentMol(mol, minCuts=2, maxCuts=2, maxCutBonds=100, pattern=pattern, resultsAsMols=asmol)
    return fragmentations #core, chains

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    tmp_mol = Chem.Mol(mol)
    for idx in range(atoms):
        tmp_mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(tmp_mol.GetAtomWithIdx(idx).GetIdx()))
    return tmp_mol

def check_atom_type(mol):
    flag=True
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        # The Protein
        if int(atomic_number) not in [6,7,8,9,15,16,17]:
            flag=False
    return flag
    
def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer


def transfer_conformers(frag, mol):
    """
    Computes coordinates from molecule to fragment (for all matchings)
    """
    matches = mol.GetSubstructMatches(frag)
    if len(matches) < 1:
        raise Exception('Could not find fragment or linker matches')

    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf[match] = frag_conformer

    return match2conf

def transfer_coord(frag, mol):
    matches = mol.GetSubstructMatches(frag)
    if len(matches) < 1:
        raise Exception('Could not find fragment or linker matches')

    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf[remove_mark_mol] = frag_conformer
    new_frag = copy.deepcopy(frag)
    new_frag.AddConformer(frag_conformer)
    return new_frag

def check_linker(fragmentation, verbose=False, linker_min=2,min_path_length=2,fragment_min=5):
    linker, frags = fragmentation
    if type(linker) == str:
        linker = Chem.MolFromSmiles(linker)
        frags = Chem.MolFromSmiles(frags)

    frag1, frag2 = Chem.GetMolFrags(frags, asMols=True)
    if min(frag1.GetNumHeavyAtoms(), frag2.GetNumHeavyAtoms()) < fragment_min:
        if verbose:
            print('These Fragments are too small')
        return False
    if linker.GetNumHeavyAtoms()< linker_min:
        if verbose:
            print('This linker are too small')
        return False
    dummy_atom_idxs = [atom.GetIdx() for atom in linker.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atom_idxs) != 2:
        if verbose:
            print('This linker is not the middle linker')
        return False
    path_length = len(Chem.rdmolops.GetShortestPath(linker, dummy_atom_idxs[0], dummy_atom_idxs[1]))-2
    if path_length < min_path_length:
        if verbose:
            print('This linker is too short')
        return False
    return True

def check_linkers(fragmentations):
    filter_fragmentations = []
    for fragmentation in fragmentations:
        if check_linker(fragmentation):
            filter_fragmentations.append(fragmentation)
    return filter_fragmentations

def check_frag(fragmentation, fragment_min=5, verbose=False):
    '''
    fragmentations = fragmentize_mol(mol,asmol=False)
    '''

    if type(fragmentation[1]) == str:
        smis = fragmentation[1].split('.')
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
    else:
        frags = fragmentation[1]
        mols = Chem.GetMolFrags(frags, asMols=True)
    if mols[1].GetNumHeavyAtoms() < mols[0].GetNumHeavyAtoms(): 
        # first one should be smaller, is the one that to be masked
        new_mols = [mols[1], mols[0]]
        mols = new_mols
    frag1, frag_id1 = remove_dummys_mol(mols[0]) 
    frag2, frag_id2 = remove_dummys_mol(mols[1]) 
    if frag1.GetNumHeavyAtoms() < fragment_min:
        if verbose:
            print('This fragment is too small')
        return False
    elif frag2.GetNumHeavyAtoms() < frag1.GetNumHeavyAtoms() * 2:
        if verbose:
            print('The kept fragment is too small')
        return False
    else:
        return True

def check_frags(fragmentations):
    filter_fragmentations = []
    for fragmentation in fragmentations:
        if check_frag(fragmentation):
            filter_fragmentations.append(fragmentation)
    return filter_fragmentations

def frag2mols(fragmentation):
    '''
    Input:
    ('',''CN(CC[C@H](N)CC(=O)N[C@H]1CC[C@@H]([*:1])')
    Output:
    frag1, frag2
    '''
    smis = fragmentation[1].split('.')
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    if mols[1].GetNumHeavyAtoms() < mols[0].GetNumHeavyAtoms(): 
        # first one should be smaller, is the one that to be masked
        new_mols = [mols[1], mols[0]]
        mols = new_mols
    frag1, frag1_id = remove_dummys_mol(mols[0])
    frag2, frag2_id = remove_dummys_mol(mols[1])
    frag1.SetProp('anchor_idx',f'{frag1_id}')
    frag2.SetProp('anchor_idx',f'{frag2_id}')

    return frag1, frag2

def get_exits(mol):
    """
    Returns atoms marked as exits in DeLinker data
    """
    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits

def get_mark(mol):
    '''
    The R Group Mark Finder
    '''
    marks = []
    for atom in mol.GetAtoms():
        atomicnum = atom.GetAtomicNum()
        if atomicnum == 0:
            marks.append(atom)
    return marks

def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        if atom.GetProp('_Anchor') == '1':
            anchors_idx.append(atom.GetIdx())

    return anchors_idx

def remove_mark_mol(molecule):
    '''
    Input: mol / str containing dummy atom
    Return: Removed mol, anchor_idx
    '''
    if type(molecule) == str:
        dum_mol = Chem.MolFromSmiles(molecule)
    else:
        dum_mol = molecule
    Chem.SanitizeMol(dum_mol)
    marks = get_mark(dum_mol)
    mark = marks[0]
    bonds = mark.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]
    mark_idx = mark.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == mark_idx else target_idx
    efragment = Chem.EditableMol(dum_mol)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(mark_idx)

    return efragment.GetMol(), anchor_idx

def remove_dummys_mol(molecule):
    '''
    Input: mol / str containing dummy atom
    Return: Removed mol, anchor_idx
    '''
    if type(molecule) == str:
        dum_mol = Chem.MolFromSmiles(molecule)
    else:
        dum_mol = molecule
    Chem.SanitizeMol(dum_mol)
    exits = get_exits(dum_mol)
    exit = exits[0]
    bonds = exit.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]
    exit_idx = exit.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    efragment = Chem.EditableMol(dum_mol)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(exit_idx)

    return efragment.GetMol(), anchor_idx


def dockedpdb2sdf(ref_mol, dockedpdbqt):
    '''
    Correct Bond Orders of Docked PDBFile
    '''
    if type(ref_mol) == str:
        mols = read_sdf(ref_mol)

def rmradical(mol):
    for atom in mol.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    return mol

def docked_rmsd(ref_mol, docked_mols):
    rmsd_list  =[]
    for mol in docked_mols:
        clean_mol = rmradical(mol)
        rightref = AllChem.AssignBondOrdersFromTemplate(clean_mol, ref_mol) #(template, mol)
        rmsd = CalcRMS(rightref,clean_mol)
        rmsd_list.append(rmsd)
    return rmsd_list

from rdkit import Chem
from rdkit.Chem import rdMMPA
from rdkit.Chem import AllChem
import sys

def fragment_mol(smi, cid, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]"):
    mol = Chem.MolFromSmiles(smi)

    #different cuts can give the same fragments
    #to use outlines to remove them
    outlines = set()

    if (mol == None):
        sys.stderr.write("Can't generate mol for: %s\n" % (smi))
    else:
        frags = rdMMPA.FragmentMol(mol, minCuts=2, maxCuts=2, maxCutBonds=100, pattern=pattern, resultsAsMols=False)
        for core, chains in frags:
            output = '%s,%s,%s,%s' % (smi, cid, core, chains)
            if (not (output in outlines)):
                outlines.add(output)
        if not outlines:
            # for molecules with no cuts, output the parent molecule itself
            outlines.add('%s,%s,,' % (smi,cid))

    return outlines

def fragment_dataset(smiles, linker_min=3, fragment_min=5, min_path_length=2, linker_leq_frags=True, verbose=False):
    successes = []

    for count, smi in enumerate(smiles):
        smi = smi.rstrip()
        smiles = smi
        cmpd_id = smi

        # Fragment smi
        o = fragment_mol(smiles, cmpd_id)

        # Checks if suitable fragmentation
        for l in o:
            smiles = l.replace('.',',').split(',')
            mols = [Chem.MolFromSmiles(smi) for smi in smiles[1:]]
            add = True
            fragment_sizes = []
            for i, mol in enumerate(mols):
                # Linker
                if i == 1:
                    linker_size = mol.GetNumHeavyAtoms()
                    # Check linker at least than minimum size
                    if linker_size < linker_min:
                        add = False
                        break
                    # Check path between the fragments at least minimum
                    dummy_atom_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
                    if len(dummy_atom_idxs) != 2:
                        print("Error")
                        add = False
                        break
                    else:
                        path_length = len(Chem.rdmolops.GetShortestPath(mol, dummy_atom_idxs[0], dummy_atom_idxs[1]))-2
                        if path_length < min_path_length:
                            add = False
                            break
                # Fragments
                elif i > 1:
                    fragment_sizes.append(mol.GetNumHeavyAtoms())
                    min_fragment_size = min(fragment_sizes)
                    # Check fragment at least than minimum size
                    if mol.GetNumHeavyAtoms() < fragment_min:
                        add = False
                        break
                    # Check linker not bigger than fragments
                    if linker_leq_frags:
                        if min_fragment_size < linker_size:
                            add = False
                            break
            if add == True:
                successes.append(l)
        
        if verbose:
            # Progress
            if count % 1000 == 0:
                print("\rProcessed smiles: " + str(count), end='')
    
    # Reformat output
    fragmentations = []
    for suc in successes:
        fragmentations.append(suc.replace('.',',').split(',')[1:])
    
    return fragmentations


# ####################################### MMPA ####################################### #
from rdkit.Chem.rdMMPA import FragmentMol

def check_mmpa_linker(linker_smi, min_size):
    mol = Chem.MolFromSmiles(linker_smi)
    num_exits = linker_smi.count('*:')
    return (mol.GetNumAtoms() - num_exits) >= min_size


def check_mmpa_fragment(fragment_smi, min_size):
    mol = Chem.MolFromSmiles(fragment_smi)
    num_exits = fragment_smi.count('*:')
    return (mol.GetNumAtoms() - num_exits) >= min_size


def check_mmpa_fragments(fragments_smi, min_size):
    for fragment_smi in fragments_smi.split('.'):
        if not check_mmpa_fragment(fragment_smi, min_size):
            return False
    return True


def fragment_by_mmpa(mol, mol_name, mol_smiles, min_cuts=2, max_cuts=2, min_frag_size=5, min_link_size=3):
    mmpa_results = []
    for i in range(min_cuts, max_cuts + 1):
        mmpa_results += FragmentMol(
            mol,
            minCuts=i,
            maxCuts=i,
            maxCutBonds=100,
            pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]",
            resultsAsMols=False
        )

    filtered_mmpa_results = []
    for linker_smiles, fragments_smiles in mmpa_results:
        if check_mmpa_linker(linker_smiles, min_link_size) and check_mmpa_fragments(fragments_smiles, min_frag_size):
            filtered_mmpa_results.append([mol_name, mol_smiles, linker_smiles, fragments_smiles, 'mmpa'])
    return filtered_mmpa_results

def get_exits(mol):
    """
    Returns atoms marked as exits in DeLinker data
    """
    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits

def set_anchor_flags(mol, anchor_idx):
    """
    Sets property _Anchor to all atoms in a molecule
    """
    for atom in mol.GetAtoms():
        if atom.GetIdx() == anchor_idx:
            atom.SetProp('_Anchor', '1')
        else:
            atom.SetProp('_Anchor', '0')
            
def update_fragment(frag):
    """
    Removes exit atoms with corresponding bonds and sets _Anchor property
    """
    exits = get_exits(frag)
    if len(exits) > 1:
        raise Exception('Found more than one exits in fragment')
    exit = exits[0]

    bonds = exit.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]

    exit_idx = exit.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    set_anchor_flags(frag, anchor_idx)

    efragment = Chem.EditableMol(frag)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(exit_idx)

    return efragment.GetMol()


def update_linker(linker):
    """
    Removes exit atoms with corresponding bonds
    """
    exits = get_exits(linker)
    if len(exits) > 2:
        raise Exception('Found more than two exits in linker')

    # Sort exit atoms by id for further correct deletion
    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)
    elinker = Chem.EditableMol(linker)

    # Remove exit bonds
    for exit in exits:
        bonds = exit.GetBonds()
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        elinker.RemoveBond(source_idx, target_idx)

    # Remove exit atoms
    for exit in exits:
        elinker.RemoveAtom(exit.GetIdx())

    return elinker.GetMol()

def pocket_trunction(pdb_file, threshold=10, outname=None, sdf_file=None, centroid=None):
    from pdb_parser import PDBProtein
    pdb_parser = PDBProtein(pdb_file)
    if centroid is None:
        centroid = sdf2centroid(sdf_file)
    else:
        centroid = centroid
    residues = pdb_parser.query_residues_radius(centroid,threshold)
    residue_block = pdb_parser.residues_to_pdb_block(residues)
    if outname is None:
        outname = pdb_file[:-4]+f'_pocket{threshold}.pdb'
    f = open(outname,'w')
    f.write(residue_block)
    f.close()
    return outname
def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:,0].mean()
    centroid_y = lig_xyz[:,1].mean()
    centroid_z = lig_xyz[:,2].mean()
    return centroid_x, centroid_y, centroid_z


from rdkit import DataStructs
def compute_subsim(mol1, mol2, shared):
    if type(mol1) == str:
        mol1 = Chem.MolFromSmiles(mol1)
    if type(mol2) == str:
        mol2 = Chem.MolFromSmiles(mol2)
    if type(shared) == str:
        shared = Chem.MolFromSmiles(shared)
    frag1 = Chem.DeleteSubstructs(mol1, shared)
    frag2 = Chem.DeleteSubstructs(mol2, shared)
    frag1_fp = Chem.RDKFingerprint(frag1)
    frag2_fp = Chem.RDKFingerprint(frag2)
    sim = DataStructs.TanimotoSimilarity(frag1_fp, frag2_fp)
    return sim

def compute_sim(ref, gen, source='mol'):
    if source =='mol':
        fp_refmol = Chem.RDKFingerprint(ref)
        fp_genmol = Chem.RDKFingerprint(gen)
        sim = DataStructs.TanimotoSimilarity(fp_refmol, fp_genmol)
    elif source == 'fp':
        sim = DataStructs.TanimotoSimilarity(ref, gen)
    else:
        raise NotImplementedError('Error: you must choose the mol or fp to compute the similariy')
    return sim


def compute_fps(mols):
    fps = [Chem.RDKFingerprint(i) for i in mols]
    return fps


def compute_sims(gen_mols, ref_mols):
    gen_fps = compute_fps(gen_mols)
    ref_fps = compute_fps(ref_mols)
    sim_mat = np.zeros([len(gen_fps), len(ref_fps)])
    for gen_id, gen_fp in enumerate(gen_fps):
        for ref_id, ref_fp in enumerate(ref_fps):
            sim_mat[gen_id][ref_id] = DataStructs.TanimotoSimilarity(gen_fp, ref_fp)
    return sim_mat


def rm_shared(mols, shared):
    '''
    remove the shared structures for a mol list
    e.g.: a series of generated scaffold-constrained molecules, to get the generated part
    '''
    return [Chem.DeleteSubstructs(i, shared) for i in mols]


def anchorfinder(mol, frag):
    '''
    Checking the bound bonds and find where is anchor nodes
    '''
    anchor_idx = []
    anchor_bonds = []
    match = mol.GetSubstructMatch(frag)
    for atom_idx in match:
        atom = mol.GetAtomWithIdx(atom_idx)
        bonds = atom.GetBonds()
        tmp_idx = []
        for bond in bonds:
            src = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            tmp_idx.append(src)
            tmp_idx.append(end)
            if (src not in match) & (end in match):
                anchor_bonds.append(bond.GetIdx())
            if (end not in match) & (src in match):
                anchor_bonds.append(bond.GetIdx())
        tmp_idx = set(tmp_idx)
        if not tmp_idx.issubset(set(match)):
            anchor_idx.append(atom_idx)
    return anchor_idx, anchor_bonds

def find_linker_from_indices(mol, anchor_idx):
    '''
    Using the topological search to find the linked substructure
    '''
    path = Chem.GetShortestPath(mol,anchor_idx[0],anchor_idx[1])
    return path # I need to discover how to get the substructure according to the path tuple

def find_linker_from_bonds(mol, bond_indices):
    '''
    Using the bond_indices to fragmentation the mol and get the linker with two dymmy atoms
    '''
    frags = Chem.FragmentOnSomeBonds(mol, bond_indices, numToBreak=2)[0]
    frags = Chem.GetMolFrags(frags, asMols=True)
    for frag in frags:
        dummy_atom_idxs = [atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy_atom_idxs) == 2:
            linker = frag
    return linker, frags

# def find_genpart(mol, frag, return_large=True):
#     '''
#     Delete fragment in mol, return the residue substructs (generated part)
#     Optional: 
#         return_max: return the largest frag in the fragments
#     '''
#     ress = Chem.DeleteSubstructs(mol,frag)
#     ress = Chem.GetMolFrags(ress, asMols=True)
#     if return_large:
#         ress_num = [i.GetNumAtoms() for i in ress]
#         max_id = np.argmax(ress_num)
#         return ress[max_id]
#     else:
#         return ress
    

from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
fdefName = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
fdef = AllChem.BuildFeatureFactory(fdefName)
fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 
        'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')
def get_FeatureMapScore(query_mol, ref_mol):
    featLists = []
    for m in [query_mol, ref_mol]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode=FeatMaps.FeatMapScoreMode.Best
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))
    
    return fm_score

def calc_SC_RDKit_score(query_mol, ref_mol):
    fm_score = get_FeatureMapScore(query_mol, ref_mol)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
            allowReordering=False)
    SC_RDKit_score = 0.5*fm_score + 0.5*(1 - protrude_dist)
    #SC_RDKit_score = (1 - protrude_dist)
    return SC_RDKit_score

import torch
def get_mol_coord(mol):
    return  torch.tensor(np.array(mol.GetConformer().GetPositions()), dtype=torch.float32)
def get_atom_map_3d(mol, frag, verbose=True):
    """
    Find the frag-mol atomic maps.
    """
    coords_mol = get_mol_coord(mol)
    coords_frag = get_mol_coord(frag)

    # Calculate pairwise distance matrix using broadcasting
    # Expand dims to (n_atoms_frag, 1, 3) and (1, n_atoms_mol, 3) to enable broadcasting
    distance_matrix = torch.norm(coords_frag[:, None, :] - coords_mol[None, :, :], dim=2)

    # Define epsilon (ε) - threshold for considering distances as close
    epsilon = 0.01

    # Find atom mappings based on the distance threshold
    # Using a masked array to find indices where distance is less than epsilon
    atom_mapping = torch.nonzero(distance_matrix < epsilon, as_tuple=True)

    # Convert atom_mapping from tensor to list of tuples for easier interpretation
    atom_mapping_list = list(zip(atom_mapping[0].tolist(), atom_mapping[1].tolist()))
    if len(atom_mapping_list) != frag.GetNumAtoms():
        if verbose:
            print('frag do not exactly match the total molecule')
    return atom_mapping_list


def find_anchor_indices_3d(mol, frag):
    """
    Finds anchor indices in a fragment based on 3D geometry overlap.
    
    Returns:
    - List of anchor indices within the fragment / mol.
    """
    # Convert atom_mapping_list to a dictionary for easier access
    atom_mapping_list = get_atom_map_3d(mol, frag)
    mapping_dict = dict(atom_mapping_list)
    
    # Find the bonds in mol that connect mapped atoms to unmapped atoms
    # These will help us identify the anchor points in frag
    anchor_indices_in_frag = set()
    anchor_indices_in_mol = set()
    for frag_atom_idx, mol_atom_idx in atom_mapping_list:
        mol_atom = mol.GetAtomWithIdx(mol_atom_idx)
        for neighbor in mol_atom.GetNeighbors():
            # Check if the neighbor atom is not in the mapping (i.e., it's outside the frag mapping)
            if neighbor.GetIdx() not in mapping_dict.values():
                # If the neighbor is outside, then the current mol_atom (and corresponding frag_atom)
                # is connected to an atom outside the frag, making it an anchor point
                anchor_indices_in_frag.add(frag_atom_idx)
                anchor_indices_in_mol.add(mol_atom_idx)
                break  # Move to the next mapped atom since we've found an anchor

    # Convert set to list and return
    return list(anchor_indices_in_frag), list(anchor_indices_in_mol)

import copy
import numpy as np
def set_mol_position(mol, pos):
    mol = copy.deepcopy(mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol 

def rm_radical(mol):
    mol = copy.deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    return mol

def generalize(core):
    query_params = Chem.AdjustQueryParameters()
    query_params.makeBondsGeneric = True
    query_params.aromatizeIfPossible = False
    query_params.adjustDegree = False
    query_params.adjustHeavyDegree = False
    generic_core = Chem.AdjustQueryProperties(core,query_params)
    return generic_core

def transfer_coord_generic(frag, mol, match=None):
    """
    Computes coordinates from molecule to fragment (for all matchings)
    """
    frag_generic = generalize(frag)
    match = mol.GetSubstructMatch(frag_generic)
    mol_coords = mol.GetConformer().GetPositions()
    frag_coords = mol_coords[np.array(match)]
    new_frag = set_mol_position(frag, frag_coords)
    return new_frag