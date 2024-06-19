import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(repo_dir)

from repo.tools.interaction import *
from repo.utils import misc
import argparse
import torch
from scipy import spatial as sci_spatial

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='../results/denovo/diffbp/selftrain/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10')
    parser.add_argument('--pdb_path', type=str, default='../data/crossdocked_test/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10.pdb')
    parser.add_argument('--save_path', type=str, default='./tmp/interaction')
    args = parser.parse_args()
    
    logger = misc.get_logger('evaluate_interaction', log_dir=args.result_path)

    reports = {'num_hydrophobic': 0, 'num_hydrogen': 0, 'num_wb': 0, 'num_pi_stack': 0, 'num_pi_cation': 0, 'num_halogen': 0, 'num_metal': 0}
    interaction_detected = 0
    result_path = args.result_path
    file_list = os.listdir(result_path)
    file_list = [file_name for file_name in file_list if file_name.endswith('.sdf')]
    
    report_dicts = []
    n_eval_success = 0

    for file_name in file_list:
        try:
            protein_file = args.pdb_path
            ligand_file = os.path.join(result_path, file_name)

            interaction_analyzer = InteractionAnalyzer(protein_file, ligand_file)
            report_path = interaction_analyzer.plip_analysis(args.save_path)
            report_dict = interaction_analyzer.plip_parser(report_path)
            report_dicts.append(report_dict)

            n_eval_success += 1

        except:
            pass

    interact_ratio_per_mol = {key: [] for key in report_dicts[0].keys()}
    for report_dict in report_dicts:
        for key, val in report_dict.items():
            interact_ratio_per_mol[key].append(val)
    for key, val in interact_ratio_per_mol.items():
        interact_ratio_per_mol[key] = np.sum(val) / n_eval_success
    
    interact_distribution = {key: [] for key in report_dicts[0].keys()}
    interact_distribution_sum = []
    for report_dict in report_dicts:
        for key, val in report_dict.items():
            interact_distribution[key].append(val)
            interact_distribution_sum.append(val)
    interact_distribution_sum = np.sum(interact_distribution_sum)
    for key, val in interact_distribution.items():
        interact_distribution[key] = np.sum(interact_distribution[key]) / interact_distribution_sum

    gen_dict = {'dist': interact_distribution, 'ratio': interact_ratio_per_mol, 'n_eval_success': n_eval_success}

    ref_mol_path = os.path.join(os.path.dirname(args.pdb_path), '_'.join(os.path.basename(args.pdb_path).split('_')[:-1]) + '.sdf')
    interaction_analyzer = InteractionAnalyzer(protein_file, ligand_file)
    report_path = interaction_analyzer.plip_analysis(args.save_path)
    report_ref_dict = interaction_analyzer.plip_parser(report_path)
    num_interact = np.sum([v for k,v in report_dict.items()])
    report_ref_dist_dict = {k: v/num_interact for k,v in report_dict.items()}
    ref_dict = {'dist': interact_distribution, 'ratio': report_ref_dict}
    
    torch.save(gen_dict, os.path.join(result_path, 'interact_gen_results.pt'))
    torch.save(ref_dict, os.path.join(result_path, 'interact_ref_results.pt'))

    jsd = sci_spatial.distance.jensenshannon([v for _, v in ref_dict['dist'].items()], 
                                             [v for _, v in gen_dict['dist'].items()]) # 100 jsd - > mean
    mae = np.abs([v for _, v in ref_dict['ratio'].items()] - [v for _, v in gen_dict['ratio'].items()]).mean()

    print('jsd: ', jsd, 'mae: ', mae)

    # num_success = 0
    # jsds = []
    # maes = []
    # num_interactions = []
    # num_interactions_ref = []
    # for file_gen, file_ref in files:
    #     dist_gen = file_gen['dist']
    #     dist_ref = file_ref['dist']
    #     jsd = sci_spatial.distance.jensenshannon([v for _, v in dist_ref.items()], 
    #                                              [v for _, v in dist_gen.items()])
    #     jsds.append(jsd)

    #     ratio_gen = file_gen['ratio']
    #     ratio_ref = file_ref['ratio']
    #     mae =  ([v for _, v in ratio_gen.items()] - [v for _, v in ratio_ref.items()]).abs().mean() 
    #     maes.append(mae)

    #     n_eval_success = file_gen['n_eval_success']

    #     num_interact_per_struct = np.array([v for _,v in ratio_gen.items()]) * n_eval_success
    #     num_success += n_eval_success

    #     num_interactions.append(num_interact_per_struct)
    #     num_interactions_ref.append(ratio_ref)

    
    # num_all_interact = np.stack(num_interactions, dim=0).sum(0)
    # ratio_all_interact = num_all_interact / num_success
    # dist_all_interact = num_all_interact / num_all_interact.sum()

    # num_ref_interact = np.stack(num_interactions_ref, dim=0).sum(0)
    # ratio_ref_interact = num_ref_interact / len(num_interactions_ref)
    # dist_ref_interact = num_ref_interact / num_ref_interact.sum()

    # jsd_overall = sci_spatial.distance.jensenshannon(dist_ref_interact, dist_all_interact)
    # mae_overall = compute_mae(ratio_ref_interact, ratio_all_interact)

    # print(maes.mean(), jsds.mean())

    