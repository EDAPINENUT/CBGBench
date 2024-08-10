import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(repo_dir)

from repo.tools.interaction import *
from repo.tools import eval_fg_type, eval_atom_type, eval_ring_type
from repo.utils import misc
from collections import Counter
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='../results/denovo/flag/selftrain/AK1BA_HUMAN_1_316_0/5liu_X_rec_4gq0_qap_lig_tt_min_0_pocket10')
    parser.add_argument('--pdb_path', type=str, default='../data/crossdocked_test/AK1BA_HUMAN_1_316_0/5liu_X_rec_4gq0_qap_lig_tt_min_0_pocket10.pdb')
    args = parser.parse_args()
    
    logger = misc.get_logger('evaluate_substruct', log_dir=args.result_path)

    result_path = args.result_path
    file_list = os.listdir(result_path)

    file_list = [file_name for file_name in file_list if file_name.endswith('.sdf')]
    success_atom_types, success_fg_types, success_ring_types = Counter(), Counter(), Counter()
    success_num = 0
    for file_name in file_list:
        try:
            protein_file = args.pdb_path
            ligand_file = os.path.join(result_path, file_name)

            mol = Chem.SDMolSupplier(ligand_file)[0]
            if mol is None:
                continue
            
            fg_type = eval_fg_type.get_func_group_from_mol(mol)
            success_fg_types += Counter(fg_type)

            atom_type = eval_atom_type.get_atom_from_mol_num(mol)
            success_atom_types += Counter(atom_type)

            ring_type = eval_ring_type.ring_type_from_mol(mol)
            success_ring_types += Counter(ring_type)

            success_num += 1

        except:
            pass

    result = {'atom_type': success_atom_types, 'ring_type': success_ring_types, 'fg_type': success_fg_types, 'success_num': success_num}
    torch.save(result, os.path.join(result_path, 'substruct_result.pt'))

    atom_type_js, pred_atom_distribution = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    atom_type_mae, pred_atom_ratio = eval_atom_type.eval_atom_type_ratio(success_atom_types, success_num)
    print('Atom type JS: %.4f' % atom_type_js)
    print('Atom type MAE: %.4f' % atom_type_mae)
    print(pred_atom_distribution)
    print(pred_atom_ratio)

    fg_type_js, pred_fg_distribution = eval_fg_type.eval_fg_type_distribution(success_fg_types)
    fg_type_mae, ped_fg_ratio = eval_fg_type.eval_fg_type_ratio(success_fg_types, success_num)

    print('Fg type JS: %.4f' % fg_type_js)
    print('Fg type MAE: %.4f' % fg_type_mae)
    print(pred_fg_distribution)
    print(ped_fg_ratio)

    ring_type_js, pred_ring_distribution = eval_ring_type.eval_ring_type_distribution(success_ring_types)
    ring_type_mae, ped_ring_ratio = eval_ring_type.eval_ring_type_ratio(success_ring_types, success_num)

    print('ring type JS: %.4f' % ring_type_js)
    print('ring type MAE: %.4f' % ring_type_mae)
    print(pred_ring_distribution)
    print(ped_ring_ratio)
    
