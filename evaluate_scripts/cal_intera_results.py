import os
import torch
import argparse
import numpy as np
import pandas as pd
from scipy import spatial as sci_spatial
from sklearn.metrics import mean_absolute_error
import re

def aggregate_interact_metrics(base_result_path):
    num_success = 0
    jsds = []
    maes = []
    num_interactions = []
    num_interactions_ref = []
    gen_files = []
    ref_files = []

    for root, _, files in os.walk(base_result_path):
        if 'interact_gen_results.pt' in files and 'interact_ref_results.pt' in files:
            gen_file_path = os.path.join(root, 'interact_gen_results.pt')
            ref_file_path = os.path.join(root, 'interact_ref_results.pt')
            gen_files.append(torch.load(gen_file_path))
            ref_files.append(torch.load(ref_file_path))

    for file_gen, file_ref in zip(gen_files, ref_files):
        dist_gen = file_gen['dist']
        dist_ref = file_ref['dist']
        dist_gen_array = np.array([v for _, v in dist_gen.items()])
        dist_ref_array = np.array([v for _, v in dist_ref.items()])
        
        jsd = sci_spatial.distance.jensenshannon(dist_ref_array, dist_gen_array)
        
        if not np.isnan(jsd):
            jsds.append(jsd)
            ratio_gen = file_gen['ratio']
            ratio_ref = file_ref['ratio']
            mae = mean_absolute_error([v for _, v in ratio_ref.items()], [v for _, v in ratio_gen.items()])
            maes.append(mae)

            n_eval_success = file_gen['n_eval_success']

            num_interact_per_struct = np.array([v for _, v in ratio_gen.items()]) * n_eval_success
            num_success += n_eval_success

            num_interactions.append(num_interact_per_struct)
            num_interactions_ref.append(np.array([v for _, v in ratio_ref.items()]))
        else:
            print("JSD is NaN for the following distributions:")
            print("dist_ref:", dist_ref_array)
            print("dist_gen:", dist_gen_array)

    num_all_interact = np.sum(np.stack(num_interactions, axis=0), axis=0)
    ratio_all_interact = num_all_interact / num_success
    dist_all_interact = num_all_interact / num_all_interact.sum()

    num_ref_interact = np.sum(np.stack(num_interactions_ref, axis=0), axis=0)
    ratio_ref_interact = num_ref_interact / len(num_interactions_ref)
    dist_ref_interact = num_ref_interact / num_ref_interact.sum()

    overall_jsd = sci_spatial.distance.jensenshannon(dist_ref_interact, dist_all_interact)
    overall_mae = mean_absolute_error(ratio_ref_interact, ratio_all_interact)

    print(f"Average JSD (per structure): {np.mean(jsds):.4f}")
    print(f"Average MAE (per structure): {np.mean(maes):.4f}")
    print(f"Overall JSD: {overall_jsd:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")
    print('ratio_all_interact', ratio_all_interact)
    print('dist_all_interact', dist_all_interact)

    return np.mean(jsds), np.mean(maes), overall_jsd, overall_mae, dist_all_interact, ratio_all_interact

def save_metrics_to_csv(metrics, dist_keys, base_result_path):
    method_name_parts = args.base_result_path.split('/') 
    method_name = '_'.join(method_name_parts[-1:]) 
    match = re.search(r'/results/([^/]+/[^/]+)', base_result_path)
    if match:
        sub_path_parts = match.group(1) + f'_{method_name}'
        output_sub_dir = os.path.join('evaluate_script', sub_path_parts)
    else:
        raise ValueError("The base_result_path does not contain the expected pattern after 'results/'")
    
    os.makedirs(os.path.join('./', sub_path_parts.split('/')[0]), exist_ok=True)
    output_csv_path = os.path.join('./', f'{sub_path_parts}_interact_metrics.csv')
    
    ratio_all_interact = metrics[5]
    dist_all_interact = metrics[4]

    df_ratio = pd.DataFrame([ratio_all_interact], columns=dist_keys, index=['overall_interact_ratio'])
    df_dist = pd.DataFrame([dist_all_interact], columns=dist_keys, index=['overall_ineract_dist'])
    
    df_metrics = pd.DataFrame({
        'overall_error': ['%overall_mae', '%overall_jsd'],
        'overall_struct_error': [metrics[3], metrics[2]]
    })

    df_metrics_1 = pd.DataFrame({
        'per_error': ['%per_struct_mae', '%per_struct_jsd'],
        'per_struct_error': [metrics[1], metrics[0]]
    })

    with open(output_csv_path, 'w') as f:
        df_ratio.to_csv(f)
        df_dist.to_csv(f, header=False)
        df_metrics.to_csv(f, header=False, index=False)
        df_metrics_1.to_csv(f, header=False, index=False)

    print(f"Metrics saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_result_path', type=str, default='../results/denovo/diffbp/selftrain', help="Base result path to traverse")
    args = parser.parse_args()

    avg_jsd, avg_mae, overall_jsd, overall_mae, dist_all_interact, ratio_all_interact = aggregate_interact_metrics(args.base_result_path)

    dist_keys = ['num_hydrophobic', 'num_hydrogen', 'num_wb', 'num_pi_stack', 'num_pi_cation', 'num_halogen', 'num_metal']
    save_metrics_to_csv((avg_jsd, avg_mae, overall_jsd, overall_mae, dist_all_interact, ratio_all_interact), dist_keys, args.base_result_path)
