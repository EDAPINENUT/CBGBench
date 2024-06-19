import os
import torch
import numpy as np
import glob
import json
import argparse
import re 

def collect_chem_results(root_dir):
    all_qed = []
    all_logp = []
    all_sa = []
    all_lipinski = []
    ref_list = []
    result_list = []
    validity_list = []

    for subdir, dirs, files in os.walk(root_dir):
        chem_eval_file = os.path.join(subdir, 'chem_eval_results.pt')
        chem_reference_file = os.path.join(subdir, 'chem_reference_results.pt')

        if os.path.isfile(chem_eval_file) and os.path.isfile(chem_reference_file):
            try:
                result = torch.load(chem_eval_file)
                for res in result:
                    all_qed.append(res['chem_results']['qed'])
                    all_logp.append(res['chem_results']['logp'])
                    all_sa.append(res['chem_results']['sa'])
                    all_lipinski.append(res['chem_results']['lipinski'])
                result_list.append(result)
                validity_list.append(len(result) / 200)

                ref = torch.load(chem_reference_file)
                ref_list.append(ref)
            except Exception as e:
                print(f"Error loading {chem_eval_file} or {chem_reference_file}: {e}")

    return all_qed, all_logp, all_sa, all_lipinski, result_list, ref_list, validity_list


def calculate_means(all_qed, all_logp, all_sa, all_lipinski, validity_list):
    qed_mean = np.mean(all_qed)
    logp_mean = np.mean(all_logp)
    sa_mean = np.mean(all_sa)
    lipinski_mean = np.mean(all_lipinski)

    validity_mean = np.mean(validity_list)

    return qed_mean, logp_mean, sa_mean, lipinski_mean, validity_mean

def calculate_vina_metrics(result, ref, key):
    vina = np.array([res['vina'][key]['affinity'] for res in result])
    atom_num = np.array([res['num_atoms'] for res in result])
    ref_vina = ref['vina'][key]['affinity']
    
    if ref_vina > 0:
        print(f"Skipping {key} with positive ref_vina: {ref_vina}")
        return None

    imp_vina = (vina - ref_vina) / ref_vina 
    imp_vina_mean = np.mean(imp_vina) * 100
    delta_binding = ((vina < ref_vina).sum() / len(vina)) * 100
    mean_vina = np.mean(vina)
    lig_eff = vina / atom_num
    lig_eff_mean = np.mean(lig_eff)

    return imp_vina_mean, delta_binding, mean_vina, lig_eff_mean

def main(root_directory):
    all_qed, all_logp, all_sa, all_lipinski, result_list, ref_list, validity_list = collect_chem_results(root_directory)

    qed_mean, logp_mean, sa_mean, lipinski_mean, validity_mean = calculate_means(all_qed, all_logp, all_sa, all_lipinski, validity_list)

    print(f"QED mean: {qed_mean}")
    print(f"logP mean: {logp_mean}")
    print(f"SA mean: {sa_mean}")
    print(f"Lipinski mean: {lipinski_mean}")
    print(f"Validity mean: {validity_mean}")

    final_results = {
        "QED mean": qed_mean,
        "logP mean": logp_mean,
        "SA mean": sa_mean,
        "Lipinski mean": lipinski_mean,
        "Validity mean": validity_mean
    }

    score_metrics = []
    minimize_metrics = []
    dock_metrics = []

    for result, ref in zip(result_list, ref_list):
        score_metric = calculate_vina_metrics(result, ref, 'score_only')
        if score_metric:
            score_metrics.append(score_metric)

        minimize_metric = calculate_vina_metrics(result, ref, 'minimize')
        if minimize_metric:
            minimize_metrics.append(minimize_metric)

        dock_metric = calculate_vina_metrics(result, ref, 'dock')
        if dock_metric:
            dock_metrics.append(dock_metric)
    
    score_means = np.nanmean(score_metrics, axis=0)
    minimize_means = np.nanmean(minimize_metrics, axis=0)
    dock_means = np.nanmean(dock_metrics, axis=0)

    # Print results
    print('Score Only:')
    print(f"Improvement (%) in Vina Mean: {score_means[0]}")
    print(f"Delta Binding (%): {score_means[1]}")
    print(f"Mean Vina: {score_means[2]}")
    print(f"Ligand Efficiency Mean: {score_means[3]}")

    print('Minimize:')
    print(f"Improvement (%) in Vina Mean: {minimize_means[0]}")
    print(f"Delta Binding (%): {minimize_means[1]}")
    print(f"Mean Vina: {minimize_means[2]}")
    print(f"Ligand Efficiency Mean: {minimize_means[3]}")

    print('Dock:')
    print(f"Improvement (%) in Vina Mean: {dock_means[0]}")
    print(f"Delta Binding (%): {dock_means[1]}")
    print(f"Mean Vina: {dock_means[2]}")
    print(f"Ligand Efficiency Mean: {dock_means[3]}")

    final_results.update({
        "Score Only": {
            "Improvement (%) in Vina Mean": score_means[0],
            "Delta Binding (%)": score_means[1],
            "Mean Vina": score_means[2],
            "Ligand Efficiency Mean": score_means[3]
        },
        "Minimize": {
            "Improvement (%) in Vina Mean": minimize_means[0],
            "Delta Binding (%)": minimize_means[1],
            "Mean Vina": minimize_means[2],
            "Ligand Efficiency Mean": minimize_means[3]
        },
        "Dock": {
            "Improvement (%) in Vina Mean": dock_means[0],
            "Delta Binding (%)": dock_means[1],
            "Mean Vina": dock_means[2],
            "Ligand Efficiency Mean": dock_means[3]
        }
    })

    method_name_parts = args.root_directory.split('/')
    method_name = '_'.join(method_name_parts[-1:]) 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    match = re.search(r'\/results\/([^\/]+\/[^\/]+)', args.root_directory)
    if match:
        sub_path_parts = match.group(1) + f'_{method_name}'
        output_sub_dir = '/chem_eval_'.join(sub_path_parts.split('/'))
    else:
        raise ValueError("The base_result_path does not contain the expected pattern after 'results/'")

    json_dir = './'
    os.makedirs(json_dir, exist_ok=True)
    json_file_name = os.path.join(json_dir, f"{output_sub_dir}.json")
    os.makedirs("/".join(json_file_name.split('/')[:2]), exist_ok=True)
    
    with open(json_file_name, 'w') as json_file:
        json.dump(final_results, json_file, indent=4)

    print(f"Results saved to {json_file_name}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean chemical properties from chem_eval_results.pt and collect chem_reference_results.pt files.")
    parser.add_argument("--root_directory", type=str, default='../results/denovo/diffbp/selftrain', help="Root directory containing the method folders")

    args = parser.parse_args()
    main(args.root_directory)
