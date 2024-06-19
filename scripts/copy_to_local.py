import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader
import torch
from repo.datasets.pl import get_pl_dataset
from repo.models import get_model
from repo.utils.misc import *
from repo.utils.molecule.constants import *
import os 
from repo.tools.rdkit_utils import reconstruct_mol, evaluate_validity, save_mol, atom_from_fg
from repo.utils.data import recursive_to 
import shutil

def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('-c', '--config', type=str, default='./configs/denovo/test/targetdiff.yml')
    parser.add_argument('-raw_pth', '--raw_path', type=str, default='./raw_data/crossdocked_v1.1_rmsd1.0_pocket10/')

    parser.add_argument('-o', '--out_root', type=str, default='./data/crossdocked_test')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=2024)

    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    args = parser.parse_args()

    config, config_name = load_config(args.config)

    datasets = get_pl_dataset(config.data.test)
    dataset = datasets['test']

    for i in range(0, len(dataset)):
        get_raw_structure = lambda: dataset.dataset.get_raw(dataset.indices[i])
        raw_strcuture_ = get_raw_structure()

        structure_id = raw_strcuture_['entry'][0][:-4]
        sdf_id = raw_strcuture_['entry'][1]
        pdb_path = os.path.join(args.raw_path, structure_id + '.pdb')
        ref_sdf_path = os.path.join(args.raw_path, sdf_id)
        target_dir = os.path.join(args.out_root, structure_id.split('/')[0])
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(pdb_path, target_dir)
        shutil.copy(ref_sdf_path, target_dir)


def main2():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default='./configs/denovo/test/targetdiff.yml')
    args = parser.parse_args()
    
    config, config_name = load_config(args.config)

    datasets = get_pl_dataset(config.data.test)
    dataset = datasets['test']

    all_pocket_names = []
    for i in range(0, len(dataset)):
        get_raw_structure = lambda: dataset.dataset.get_raw(dataset.indices[i])
        raw_strcuture_ = get_raw_structure()

        structure_id = raw_strcuture_['entry'][0][:-4]
        all_pocket_names.append(structure_id)
    raw_path = '/linhaitao/CGBBench/results/denovo/ligan_raw'
    dest_path = '/linhaitao/CGBBench/results/denovo/ligan/20240528'
    ligand_names = os.listdir(raw_path)
    for ligand_name in ligand_names:
        for all_pocket_name in all_pocket_names:
            if ligand_name in all_pocket_name:
                new_dir = os.path.join(dest_path, all_pocket_name)
                os.makedirs(new_dir, exist_ok=True)
                raw_dir = os.path.join(raw_path, ligand_name, 'SDF')
                for file in os.listdir(raw_dir):
                    shutil.copy(os.path.join(raw_dir, file),  new_dir)



if __name__ == '__main__':
    main1()
