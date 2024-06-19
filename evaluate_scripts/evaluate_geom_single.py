import argparse
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm.auto import tqdm
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(repo_dir)

from repo.tools.geometry import eval_bond_length, eval_stability, eval_bond_angle, eval_steric_clash
from repo.utils import misc
import torch

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--result_path', type=str, default='../results/denovo/diffbp/selftrain/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10')
    parser.add_argument('--pdb_path', type=str, default='../data/crossdocked_test/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10.pdb')

    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--center', type=float, nargs=3, default=None,
                        help='Center of the pocket bounding box, in format x,y,z')
    args = parser.parse_args()

    receptor_name = args.pdb_path.split('/')[-1].split('.')[0]
    result_path = args.result_path

    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    n_eval_success = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_clash_mol, intra_clash_atom_num, inter_clash_atom_num, atom_num_list = 0, [], [], []

    results = []
    all_bond_dist, all_bond_angle = [], []
    success_pair_dist, success_atom_types = [], Counter()
    file_list = os.listdir(result_path)
    file_list = [file_name for file_name in file_list if file_name.endswith('.sdf')]

    for file_name in file_list:
        try:         
            mol_path = os.path.join(result_path, file_name)
            mol = Chem.SDMolSupplier(mol_path)[0]
            atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            pos = mol.GetConformer().GetPositions()

            # eval bond_length
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            # eval bond_angle
            bond_angle = eval_bond_angle.bond_angle_from_mol(mol)
            all_bond_angle += bond_angle

            # eval stability            
            r_stable = eval_stability.check_stability(pos, atom_types)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            #eval clash
            clash_detected, clash_info = eval_steric_clash.eval_steric_clash(mol, pdb_file=args.pdb_path)
            if clash_detected: n_clash_mol += 1
            
            intra_clash_atom_num.append(clash_info['lig_lig_clash']['clash_atom_num'])
            inter_clash_atom_num.append(clash_info['lig_pro_clash']['clash_atom_num'])


            result = {'file_name': file_name, 'mol_path': mol_path, 
                      'bond_dist': bond_dist, 'bond_angle': bond_angle, 
                      'mol_stable': r_stable[0], 'atom_stable_num': r_stable[1], 
                      'atom_num': r_stable[2], 'clash_detected': clash_info['lig_pro_clash_detected'],
                      'intra_clash_atom_num': clash_info['lig_lig_clash']['clash_atom_num'],
                      'inter_clash_atom_num': clash_info['lig_pro_clash']['clash_atom_num']}
            
            results.append(result)

            n_eval_success += 1

        except:
           if args.verbose:
               logger.warning('Evaluation failed for %s' % f'{mol_path}')
           continue
## 10 x 100 -> 10 resutls (100 result)
# 1000 result bond_dist concat
# 
    logger.info(f'Evaluate done! {n_eval_success} samples in total.')

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    c_bond_angle_profile = eval_bond_angle.get_bond_angle_profile(all_bond_angle)
    c_bond_angle_dict = eval_bond_angle.eval_bond_angle_profile(c_bond_angle_profile)
    logger.info('JS bond angles of complete mols: ')
    print_dict(c_bond_angle_dict, logger)


    fraction_mol_stable = all_mol_stable / n_eval_success
    fraction_atm_stable = all_atom_stable / all_n_atom
    intra_clash_atom_ratio = np.sum(intra_clash_atom_num) / all_n_atom
    inter_clash_atom_ratio = np.sum(inter_clash_atom_num) / all_n_atom
    clash_mol_ratio = n_clash_mol / n_eval_success

    torch.save(results, os.path.join(result_path, 'geom_eval_results.pt'))
