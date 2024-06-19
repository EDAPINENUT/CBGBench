import os
import torch
import argparse
import pandas as pd
from collections import Counter
import numpy as np
import re, sys
from rdkit import Chem
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(repo_dir)
from repo.tools import eval_fg_type, eval_atom_type, eval_ring_type

def print_dict(d):
    for k, v in d.items():
        if v is not None:
            print(f'{k}:\t{v:.4f}')
        else:
            pass

def aggregate_results(base_result_path):
    all_atom_types, all_fg_types, all_ring_types = Counter(), Counter(), Counter()
    total_success_num = 0

    for root, _, files in os.walk(base_result_path):
        if 'substruct_result.pt' in files:
            results_path = os.path.join(root, 'substruct_result.pt')
            results = torch.load(results_path)

            all_atom_types += results['atom_type']
            all_fg_types += results['fg_type']
            all_ring_types += results['ring_type']
            total_success_num += results['success_num']

    return {
        'all_atom_types': all_atom_types,
        'all_fg_types': all_fg_types,
        'all_ring_types': all_ring_types,
        'total_success_num': total_success_num
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_result_path', type=str, default='../results/denovo/diffbp/selftrain', help="Base result path to traverse")
    args = parser.parse_args()

    results = aggregate_results(args.base_result_path)

    atom_type_js, pred_atom_distribution = eval_atom_type.eval_atom_type_distribution(results['all_atom_types'])
    atom_type_mae, pred_atom_ratio = eval_atom_type.eval_atom_type_ratio(results['all_atom_types'], results['total_success_num'])
    print('Atom type JS: %.4f' % atom_type_js)
    print('Atom type MAE: %.4f' % atom_type_mae)
    print(pred_atom_distribution)
    print(pred_atom_ratio)

    fg_type_js, pred_fg_distribution = eval_fg_type.eval_fg_type_distribution(results['all_fg_types'])
    fg_type_mae, pred_fg_ratio = eval_fg_type.eval_fg_type_ratio(results['all_fg_types'], results['total_success_num'])
    print('Fg type JS: %.4f' % fg_type_js)
    print('Fg type MAE: %.4f' % fg_type_mae)
    print(pred_fg_distribution)
    print(pred_fg_ratio)

    ring_type_js, pred_ring_distribution = eval_ring_type.eval_ring_type_distribution(results['all_ring_types'])
    ring_type_mae, pred_ring_ratio = eval_ring_type.eval_ring_type_ratio(results['all_ring_types'], results['total_success_num'])
    print('Ring type JS: %.4f' % ring_type_js)
    print('Ring type MAE: %.4f' % ring_type_mae)
    print(pred_ring_distribution)
    print(pred_ring_ratio)

    summary_data = {
        'Metric': ['Atom type JS', 'Atom type MAE', 'Fg type JS', 'Fg type MAE', 'Ring type JS', 'Ring type MAE'],
        'Value': [round(atom_type_js, 4), round(atom_type_mae, 4), round(fg_type_js, 4), round(fg_type_mae, 4), round(ring_type_js, 4), round(ring_type_mae, 4)]
    }

    df_summary = pd.DataFrame(summary_data)
    
    method_name_parts = args.base_result_path.split('/') 
    method_name = '_'.join(method_name_parts[-1:]) 

    match = re.search(r'\/results\/([^\/]+\/[^\/]+)', args.base_result_path)
    if match:
        sub_path_parts = match.group(1) + f'_{method_name}'
        output_csv_path = os.path.join('./', f'{sub_path_parts}_substruct_summary.csv')
    else:
        output_csv_path = os.path.join('./', 'substruct_summary.csv')

    os.makedirs(os.path.join('./', sub_path_parts.split('/')[0]), exist_ok=True)
    pred_atom_distribution_converted = {Chem.GetPeriodicTable().GetElementSymbol(k): v for k, v in pred_atom_distribution.items()}
    df_pred_atom_distribution = pd.DataFrame(list(pred_atom_distribution_converted.items()), columns=['Element', 'Frequency'])
    df_pred_fg_distribution = pd.DataFrame(list(pred_fg_distribution.items()), columns=['Functional Group', 'Frequency'])
    df_pred_ring_distribution = pd.DataFrame(list(pred_ring_distribution.items()), columns=['Ring Size', 'Frequency'])
    df_pred_atom_distribution.to_csv(output_csv_path.split('.csv')[0] + "atoms.csv", index=False)
    df_pred_fg_distribution.to_csv(output_csv_path.split('.csv')[0] + "fg.csv", index=False)
    df_pred_ring_distribution.to_csv(output_csv_path.split('.csv')[0] + "ring.csv", index=False)

    df_summary.to_csv(output_csv_path, index=False)

    print("Summary results saved to", output_csv_path)