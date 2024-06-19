import argparse
import os, sys

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(repo_dir)

from repo.tools.docking_vina import VinaDockingTask
from repo.tools import scoring
from repo.utils import misc
import pandas as pd

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


def eval_single_mol(mol_path, save_path):

    mol = Chem.SDMolSupplier(mol_path)[0]
    smiles = Chem.MolToSmiles(mol)
    chem_results = scoring.get_chem(mol)

    vina_task = VinaDockingTask.from_generated_mol(
        mol, mol_path, protein_path=args.pdb_path, center=args.center)
    
    score_only_results = vina_task.run(mode='score_only', 
                                        exhaustiveness=args.exhaustiveness, 
                                        save_dir=save_path)
    minimize_results = vina_task.run(mode='minimize', 
                                        exhaustiveness=args.exhaustiveness,
                                        save_dir=save_path)
    docking_results = vina_task.run(mode='dock', 
                                    exhaustiveness=args.exhaustiveness,
                                    save_dir=save_path)
    
    vina_results = {
        'score_only': score_only_results,
        'minimize': minimize_results,
        'dock': docking_results
    }

    return {
            'mol': mol,
            'smiles': smiles,
            'ligand_filename': mol_path,
            'chem_results': chem_results,
            'vina': vina_results,
            'num_atoms': mol.GetNumAtoms()
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--result_path', type=str, default='../data/case_study/adrb1_act/')
    parser.add_argument('--pdb_path', type=str, default='../data/case_study/adrb1_act/adrb1_active_pocket10.pdb')
    parser.add_argument('--eval_ref', type=bool, default=True)
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--center', type=float, nargs=3, default=None,
                        help='Center of the pocket bounding box, in format x,y,z') # [4.35 , 3.75, 3.16] for adrb1  [1.30, -3.75, -1.90] for drd3
    args = parser.parse_args()

    receptor_name = args.pdb_path.split('/')[-1].split('.')[0]
    result_path = args.result_path

    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    num_samples = 0
    n_eval_success = 0
    results = []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    file_list = os.listdir(result_path)
    file_list = sorted([file_name for file_name in file_list if file_name.endswith('.sdf')])

    for file_name in file_list:
        try:
            dock_result_path = os.path.join(result_path, 'docking_results')
            os.makedirs(dock_result_path, exist_ok=True)

            mol_path = os.path.join(result_path, file_name)
            result = eval_single_mol(mol_path, dock_result_path)

            n_eval_success += 1
            results.append(result)

        except:
            if args.verbose:
                logger.warning('Evaluation failed for %s' % f'{mol_path}')
            continue


    logger.info(f'Evaluate done! {n_eval_success} samples in total.')

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))

    vina_score_only = [r['vina']['score_only']['affinity'] for r in results]
    vina_min = [r['vina']['minimize']['affinity'] for r in results]
    logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
    logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
    vina_dock = [r['vina']['dock']['affinity'] for r in results]
    logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))
    
    result_filter = [result for result in results if result['vina']['dock']['affinity'] < 0]
    vina_dock = [r['vina']['dock']['affinity'] for r in result_filter]
    vina_dock_idx = np.argsort(vina_dock)

    file_names = [result_filter[i]['ligand_filename'] for i in vina_dock_idx]
    chem_results = [result_filter[i]['chem_results'] for i in vina_dock_idx]
    vina_results = [result_filter[i]['vina']['dock']['affinity'] for i in vina_dock_idx]
    vina_min_results = [result_filter[i]['vina']['minimize']['affinity'] for i in vina_dock_idx]
    vina_score_only_results = [result_filter[i]['vina']['score_only']['affinity'] for i in vina_dock_idx]

    smiles = [result_filter[i]['smiles'] for i in vina_dock_idx]

    df = pd.DataFrame({'file_names': ['docking_results/docked_' + os.path.split(filename)[1] for filename in file_names],
                       'smiles': smiles,
                       'vina_dock_result': vina_results,
                       'vina_min_result': vina_min_results,
                       'vina_score_result': vina_score_only_results,
                       'qed': [chem['qed'] for chem in chem_results], 'sa': [chem['sa'] for chem in chem_results],
                       'logp': [chem['logp'] for chem in chem_results], 'lipinski': [chem['lipinski'] for chem in chem_results],}, 
                       )

    
    df.to_csv(os.path.join(result_path, 'molecule_properties.csv'), index=False)
    torch.save(results, os.path.join(result_path, 'chem_eval_results.pt'))

    if args.eval_ref:
        ref_mol_path = os.path.join(os.path.dirname(args.pdb_path), '_'.join(os.path.basename(args.pdb_path).split('_')[:-1]) + '.sdf')
        ref_result = eval_single_mol(ref_mol_path, dock_result_path)
        torch.save(ref_result, os.path.join(result_path, 'chem_reference_results.pt'))
        logger.info('Reference ligand evaluation done!')

        df = pd.read_csv(os.path.join(result_path, 'molecule_properties.csv'),ignore_index=True)
        df = df.append({'file_names': 'reference', 'smiles': ref_result['smiles'],
                        'vina_dock_result': ref_result['vina']['dock']['affinity'],
                        'vina_min_result': ref_result['vina']['minimize']['affinity'],
                        'vina_score_result': ref_result['vina']['score_only']['affinity'],
                        'qed': ref_result['chem_results']['qed'], 'sa': ref_result['chem_results']['sa'],
                        'logp': ref_result['chem_results']['logp'], 'lipinski': ref_result['chem_results']['lipinski']}, ignore_index=True)
        df.to_csv(os.path.join(result_path, 'molecule_properties.csv'), index=False)
