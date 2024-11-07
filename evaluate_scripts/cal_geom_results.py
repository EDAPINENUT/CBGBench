import os
import torch
import numpy as np
from collections import Counter
import sys
import os
import argparse
import pandas as pd 
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(repo_dir)
from repo.tools.geometry import eval_bond_length, eval_bond_angle
import re

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            print(f'{k}:\t{v:.4f}')
        else:
            # print(f'{k}:\tNone')
            pass

def aggregate_results(base_result_path):
    all_bond_dist, all_bond_angle = [], []
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    intra_clash_atom_num, inter_clash_atom_num = [], []
    n_clash_mol, n_eval_success = 0, 0

    for root, _, files in os.walk(base_result_path):
        if 'geom_eval_results.pt' in files:
            results_path = os.path.join(root, 'geom_eval_results.pt')
            results = torch.load(results_path)
            for result in results:
                filtered_bond_dist = [bond for bond in result['bond_dist'] if all(atom in [6, 7, 8] for atom in bond[0][:2]) and bond[0][2] in [1, 2, 3]]

                all_bond_dist += filtered_bond_dist

                filtered_bond_angle = [angle for angle in result['bond_angle'] if all(atom in [6, 7, 8] for atom in (angle[0][0], angle[0][2], angle[0][4])) and angle[0][1] in [1, 2, 3] and angle[0][3] in [1, 2, 3]]

                all_bond_angle += filtered_bond_angle

                all_mol_stable += result['mol_stable']
                all_atom_stable += result['atom_stable_num']
                all_n_atom += result['atom_num']
                n_clash_mol += 1 if result['clash_detected'] else 0
                intra_clash_atom_num.append(result['intra_clash_atom_num'])
                inter_clash_atom_num.append(result['inter_clash_atom_num'])
                n_eval_success += 1

    return {
        'all_bond_dist': all_bond_dist,
        'all_bond_angle': all_bond_angle,
        'all_mol_stable': all_mol_stable,
        'all_atom_stable': all_atom_stable,
        'all_n_atom': all_n_atom,
        'n_clash_mol': n_clash_mol,
        'intra_clash_atom_num': intra_clash_atom_num,
        'inter_clash_atom_num': inter_clash_atom_num,
        'n_eval_success': n_eval_success
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_result_path', type=str, default='../results/denovo/diffbp/selftrain', help="Base result path to traverse")
    args = parser.parse_args()


    results = aggregate_results(args.base_result_path)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(results['all_bond_dist'])
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    print('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, None)

    c_bond_angle_profile = eval_bond_angle.get_bond_angle_profile(results['all_bond_angle'])
    c_bond_angle_dict = eval_bond_angle.eval_bond_angle_profile(c_bond_angle_profile)
    print('JS bond angles of complete mols: ')
    print_dict(c_bond_angle_dict, None)

    filtered_c_bond_length_dict = {k: round(v, 4) for k, v in c_bond_length_dict.items() if v is not None}
    filtered_c_bond_angle_dict = {k: round(v, 4) for k, v in c_bond_angle_dict.items() if v is not None}
    filtered_c_bond_length_dict['JSD_BL'] = round(np.mean([v for _, v in filtered_c_bond_length_dict.items()]), 4)
    filtered_c_bond_angle_dict['JSD_BA'] = round(np.mean([v for _, v in filtered_c_bond_angle_dict.items()]), 4)

    df_bond_length = pd.DataFrame.from_dict(filtered_c_bond_length_dict, orient='index', columns=['value']).reset_index()
    df_bond_length.columns = ['key', 'value']

    df_bond_angle = pd.DataFrame.from_dict(filtered_c_bond_angle_dict, orient='index', columns=['value']).reset_index()
    df_bond_angle.columns = ['key', 'value']

    df_bond_length_sorted = df_bond_length.sort_values(by='key').reset_index(drop=True)
    df_bond_angle_sorted = df_bond_angle.sort_values(by='key').reset_index(drop=True)

    method_name_parts = args.base_result_path.split('/') 
    method_name = '_'.join(method_name_parts[-1:]) 

    match = re.search(r'\/results\/([^\/]+\/[^\/]+)', args.base_result_path)
    if match:
        sub_path_parts = match.group(1) + f'_{method_name}'
        output_sub_dir = os.path.join('evaluate_script', sub_path_parts)
    else:
        raise ValueError("The base_result_path does not contain the expected pattern after 'results/'")



    fraction_mol_stable = results['all_mol_stable'] / results['n_eval_success']
    fraction_atm_stable = results['all_atom_stable'] / results['all_n_atom']
    intra_clash_atom_ratio = np.sum(results['intra_clash_atom_num']) / results['all_n_atom']
    inter_clash_atom_ratio = np.sum(results['inter_clash_atom_num']) / results['all_n_atom']
    clash_mol_ratio = results['n_clash_mol'] / results['n_eval_success']

    print(f'fraction_mol_stable: {fraction_mol_stable:.4f}')
    print(f'fraction_atm_stable: {fraction_atm_stable:.4f}')
    print(f'intra_clash_atom_ratio: {intra_clash_atom_ratio:.4f}')
    print(f'inter_clash_atom_ratio: {inter_clash_atom_ratio:.4f}')
    print(f'clash_mol_ratio: {clash_mol_ratio:.4f}')


    summary_data = {
        'key': ['fraction_mol_stable', 'fraction_atm_stable', 'intra_clash_atom_ratio', 'inter_clash_atom_ratio', 'clash_mol_ratio'],
        'value': [round(fraction_mol_stable, 4), round(fraction_atm_stable, 4), round(intra_clash_atom_ratio, 4), round(inter_clash_atom_ratio, 4), round(clash_mol_ratio, 4)]
    }

    df_summary = pd.DataFrame(summary_data)
    os.makedirs(os.path.join('./', sub_path_parts.split('/')[0]), exist_ok=True)
    output_csv_path = os.path.join('./', f'{sub_path_parts}_geom.csv')

    def add_separator(df, separator_label='---'):
        separator = pd.DataFrame([{'key': separator_label, 'value': ''}])
        return pd.concat([df, separator], ignore_index=True)

    df_bond_length_sorted_sep = add_separator(df_bond_length_sorted)
    df_bond_angle_sorted_sep = add_separator(df_bond_angle_sorted)

    with open(output_csv_path, 'w') as f:
        df_bond_length_sorted_sep.to_csv(f, index=False)
        df_bond_angle_sorted_sep.to_csv(f, index=False, header=False)
        df_summary.to_csv(f, index=False, header=False)
