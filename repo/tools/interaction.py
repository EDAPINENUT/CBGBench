import shutil
import pickle
import xml.etree.ElementTree as ET
#from plip.structure.preparation import PDBComplex
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import os.path as osp
from tqdm import tqdm
import numpy as np
from glob import glob
import os
import sys
sys.path.append("..")
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from repo.datasets.parsers.protein_parser import PDBProteinFA
from .rdkit_utils import uff_relax

def merge_lig_pkt(pdb_file, sdf_file, out_name, mol=None):
    '''
    pdb_file = './1A1C_MALDO_2_433_0/1m4n_A_rec_1m7y_ppg_lig_tt_min_0_pocket10.pdb'
    sdf_file = './1A1C_MALDO_2_433_0/1m4n_A_rec_1m7y_ppg_lig_tt_min_0.sdf'
    '''
    protein = Chem.MolFromPDBFile(pdb_file)
    if mol == None:
        ligand = read_sdf(sdf_file)[0]
    else:
        ligand = mol
    complex = Chem.CombineMols(protein,ligand)
    Chem.MolToPDBFile(complex, out_name)


import argparse

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}
import tempfile
TMPDIR = tempfile.TemporaryDirectory().name

class InteractionAnalyzer(object):
    def __init__(self, protein_file, ligand_file, tmp_dir='./tmp'):
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile()
        tmp_file_name = os.path.join(tmp_dir, os.path.basename(tmp.name))
        self.merged_pdb_file_path = tmp_file_name + '_merged.pdb'
        merge_lig_pkt(protein_file, ligand_file, self.merged_pdb_file_path)
    
    
    def plip_parser(self, xml_file):
        xml_tree = ET.parse(xml_file)
        report = xml_tree.getroot()
        interaction_ele = report.findall('bindingsite/interactions')
        if len(interaction_ele) == 0:
            return None 
        else:
            interaction_ele = interaction_ele[0]
        result = {}
        for interaction in interaction_ele:
            result['num_hydrophobic'] = len(interaction_ele.findall('hydrophobic_interactions/*'))
            result['num_hydrogen'] = len(interaction_ele.findall('hydrogen_bonds/*'))
            result['num_wb'] = len(interaction_ele.findall('water_bridges/*'))
            result['num_pi_stack'] = len(interaction_ele.findall('pi_stacks/*'))
            result['num_pi_cation'] = len(interaction_ele.findall('pi_cation_interactions/*'))
            result['num_halogen'] = len(interaction_ele.findall('halogen_bonds/*'))
            result['num_metal'] = len(interaction_ele.findall('metal_complexes/*'))
        return result
    
        
    def plip_analysis(self, out_dir):
        '''
        out_dir 
        '''
        out_dir_tmp = os.path.join(out_dir, os.path.basename(self.merged_pdb_file_path)[:-4])
        os.makedirs(out_dir_tmp, exist_ok=True)
        command = 'plip -f {pdb_file} -o {out_dir} -x'.format(pdb_file=self.merged_pdb_file_path,
                                                              out_dir = out_dir_tmp)
        proc = subprocess.Popen(
                command, 
                shell=True, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        proc.communicate()
        return os.path.join(out_dir_tmp, 'report.xml')

    def plip_analysis_visual(self, out_dir):
        '''
        out_dir 
        '''
        command = 'plip -f {pdb_file} -o {out_dir} -tpy'.format(pdb_file=self.merged_pdb_file_path,
                                                                out_dir = out_dir)
        proc = subprocess.Popen(
                command, 
                shell=True, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        proc.communicate()
        return out_dir + '/report.xml'

    def interact_analysis(self, results_pkl, pkt_file, sdf_file, k=10):
        '''
        Designed for a bunch of interaction analysis performed on results file
        results_pkl contained the score and docked poses
        pkt_file contained the .pdb file 
        sdf_file contained the original ligand
        '''
        results = read_pkl(results_pkl)
        scores = []
        mols = []
        for i in range(len(results)):
            try:
                scores.append(results[i][0]['affinity'])
                mols.append(results[i][0]['rdmol'])
            except:
                scores.append(0)
                mols.append(0)
        scores_zip = zip(np.sort(scores),np.argsort(scores))
        scores = np.sort(scores)
        scores_idx = np.argsort(scores)
        sorted_mols = [mols[i] for i in scores_idx]
        truncted_file = pkt_file.split('/')[-1][:-4] + '_pocket10.pdb'
        truncted_file = pocket_trunction(pkt_file, outname=f'./tmp/{truncted_file}',sdf_file=sdf_file)
        if k == 'all':
            k = len(sorted_mols)
        
        gen_report = []
        for i in range(min(k,len(sorted_mols))):
            try:
                merge_lig_pkt(truncted_file, None, f'./tmp/{i}.pdb', mol=sorted_mols[i])
                report = self.plip_parser(self.plip_analysis(f'./tmp/{i}.pdb','./tmp'))
                gen_report.append(report)
            except:
                #print(i,'failed')
                ...
        clear_plip_file('./tmp/')
        return gen_report, sdf_file.split('/')[-1]
    
        
    def patter_analysis(self, ori_report, gen_report):
        compare = {}
        num_ori = 0
        num_gen = 0
        patterns = ['num_hydrophobic','num_hydrogen','num_wb','num_pi_stack','num_pi_cation','num_halogen','num_metal']
        for pattern in patterns:
            if (ori_report[pattern] == 0)&(gen_report[pattern]==0):
                continue
            num_ori += ori_report[pattern]
            num_gen += gen_report[pattern]
            #compare[pattern] = max(ori_report[pattern] - gen_report[pattern],0)
            try:
                compare[pattern] = min(gen_report[pattern]/ori_report[pattern],1)
            except:
                compare[pattern] = None

        return compare, num_ori, num_gen



def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    return pdb_id


def parse_sdf_file(path):
    mol = Chem.MolFromMolFile(path, sanitize=True)
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.int_)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    with open(path, 'r') as f:
        sdf = f.read()

    sdf = sdf.splitlines()
    num_atoms, num_bonds = map(int, [sdf[3][0:3], sdf[3][3:6]])
    assert num_atoms == rd_num_atoms

    ptable = Chem.GetPeriodicTable()
    element, pos = [], []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    for atom_line in map(lambda x:x.split(), sdf[4:4+num_atoms]):
        x, y, z = map(float, atom_line[:3])
        symb = atom_line[3]
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())
        element.append(atomic_number)
        pos.append([x, y, z])
        
        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight

    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)

    element = np.array(element, dtype=np.int_)
    pos = np.array(pos, dtype=np.float32)

    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    bond_type_map = {
        1: BOND_TYPES[BondType.SINGLE],
        2: BOND_TYPES[BondType.DOUBLE],
        3: BOND_TYPES[BondType.TRIPLE],
        4: BOND_TYPES[BondType.AROMATIC],
    }
    row, col, edge_type = [], [], []
    for bond_line in sdf[4+num_atoms:4+num_atoms+num_bonds]:
        start, end = int(bond_line[0:3])-1, int(bond_line[3:6])-1
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_type_map[int(bond_line[6:9])]]

    edge_index = np.array([row, col], dtype=np.int_)
    edge_type = np.array(edge_type, dtype=np.int_)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    neighbor_dict = {}

    #used in rotation angle prediction
    for i, atom in enumerate(mol.GetAtoms()):
        neighbor_dict[i] = [n.GetIdx() for n in atom.GetNeighbors()]

    data = {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'neighbors': neighbor_dict
    }
    return data


def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:,0].mean()
    centroid_y = lig_xyz[:,1].mean()
    centroid_z = lig_xyz[:,2].mean()
    return centroid_x, centroid_y, centroid_z

def pocket_trunction(pdb_file, threshold=10, outname=None, sdf_file=None, centroid=None):
    pdb_parser = PDBProteinFA(pdb_file)
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

def clear_plip_file(dir):
    files = glob(dir+'/plip*')
    for i in range(len(files)):
        os.remove(files[i])

def read_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(data_list, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_list, f)


def read_sdf(file):
    supp = Chem.SDMolSupplier(file)
    return [i for i in supp]

def merge_lig_pkt(pdb_file, sdf_file, out_name, mol=None, uff_relax_iter=500):
    '''
    pdb_file = './1A1C_MALDO_2_433_0/1m4n_A_rec_1m7y_ppg_lig_tt_min_0_pocket10.pdb'
    sdf_file = './1A1C_MALDO_2_433_0/1m4n_A_rec_1m7y_ppg_lig_tt_min_0.sdf'
    '''
    protein = Chem.MolFromPDBFile(pdb_file)
    if mol == None:
        ligand = read_sdf(sdf_file)[0]
    else:
        ligand = mol
    ligand = uff_relax(ligand, relax_iter=uff_relax_iter)
    
    complex = Chem.CombineMols(protein,ligand)
    Chem.MolToPDBFile(complex, out_name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', type=str, default='./dataset/ppbench2024')
    parser.add_argument('--ref_dir', type=str, default='./dataset/ppbench2024')
    parser.add_argument('--save_path', type=str, default='./results/interaction')
    args = parser.parse_args()

    reports = {'num_hydrophobic': 0, 'num_hydrogen': 0, 'num_wb': 0, 'num_pi_stack': 0, 'num_pi_cation': 0, 'num_halogen': 0, 'num_metal': 0}
    interaction_detected = 0

    for ref_name in args.ref_dir:
        protein_file = os.path.join('./dataset/ppbench2024', ref_name, 'receptor.pdb')
        ligand_file = os.path.join('./dataset/ppbench2024', ref_name, 'peptide.pdb')

        interaction_analyzer = InteractionAnalyzer(protein_file, ligand_file)
        report_path = interaction_analyzer.plip_analysis('interaction')
        report_dict = interaction_analyzer(report_path)

        # if report is not None:
        #     print(report)
        #     for k, v in report.items():
        #         if v >0:
        #             reports[k] += 1
        #     interaction_detected += 1
        #     for k, v in reports.items():
        #         logger.info(f'{k}, {v/interaction_detected}')