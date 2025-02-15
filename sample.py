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
from repo.tools.rdkit_utils import reconstruct_mol, evaluate_validity, save_mol, atom_from_fg, obabel_recover_bond
from repo.utils.data import recursive_to

def split_batch_into_samples(batch, mode='add_aromatic'):
    batch_idx = batch[-1]
    if batch_idx.numel() == 0:
        return []
    B = batch_idx.max() + 1
    batch_split = []
    for i in range(B):
        idx = (batch_idx == i)
        sample = {}
        sample['pos'] = batch[0].cpu()[idx].tolist()
        sample['type'] = batch[1].cpu()[idx].numpy()
        if len(sample['type'].shape) == 2:
            sample['type'] = sample['type'].argmax(axis=-1)
        sample['atom'] = get_atomic_number_from_index(sample['type'], mode)
        sample['aromatic'] = is_aromatic_from_index(sample['type'], mode)
        batch_split.append(sample)
    return batch_split

def split_batch_into_samples_fg(batch, mode=None):
    batch_idx = batch[-1]
    B = batch_idx.max() + 1
    batch_split = []
    for i in range(B):
        idx = (batch_idx == i)
        sample = {}
        sample['pos_center'] = batch[0].cpu()[idx].tolist()
        sample['fg_type'] = batch[1].cpu()[idx].numpy()
        if len(sample['fg_type'].shape) == 2:
            sample['fg_type'] = sample['fg_type'].argmax(axis=-1)
        sample['orientation'] = batch[2].cpu()[idx].numpy()
        batch_split.append(sample)
    return batch_split

def translate(result, translation):
    result_pos = result[0].cpu()
    result_pos += translation.cpu()
    return [result_pos] + [result[k+1] for k in range(len(result) - 1)]

# python sample.py --config ./configs/denovo/test/d3fg_fg.yml --out_root ./results/denovo/ --tag context
# python sample.py --config ./configs/denovo/test/d3fg_linker.yml --out_root ./results/denovo/
# python sample.py --config ./configs/denovo/test/diffbp.yml --out_root ./results/denovo/
# python sample.py --config ./configs/denovo/test/diffsbdd.yml --out_root ./results/denovo/
# python sample.py --config ./configs/denovo/test/targetdiff.yml --out_root ./results/denovo/
# python sample.py --config ./configs/denovo/test/flag.yml --out_root ./results/denovo/
# python sample.py --config ./configs/denovo/test/graphbp.yml --out_root ./results/denovo/
# python sample.py --config ./configs/denovo/test/pocket2mol.yml --out_root ./results/denovo/


# python sample.py --config ./configs/frag/test/pocket2mol.yml --out_root ./results/frag/ 
# python sample.py --config ./configs/frag/test/graphbp.yml --out_root ./results/frag/ 
# python sample.py --config ./configs/frag/test/targetdiff.yml --out_root ./results/frag/ 
# python sample.py --config ./configs/frag/test/diffbp.yml --out_root ./results/frag/ 
# python sample.py --config ./configs/frag/test/diffsbdd.yml --out_root ./results/frag/ 


# python sample.py --config ./configs/linker/test/pocket2mol.yml --out_root ./results/linker/ 
# python sample.py --config ./configs/linker/test/graphbp.yml --out_root ./results/linker/ 
# python sample.py --config ./configs/linker/test/targetdiff.yml --out_root ./results/linker/ 
# python sample.py --config ./configs/linker/test/diffbp.yml --out_root ./results/linker/ 
# python sample.py --config ./configs/linker/test/diffsbdd.yml --out_root ./results/linker/ 


# python sample.py --config ./configs/scaffold/test/pocket2mol.yml --out_root ./results/scaffold/ 
# python sample.py --config ./configs/scaffold/test/graphbp.yml --out_root ./results/scaffold/ 
# python sample.py --config ./configs/scaffold/test/targetdiff.yml --out_root ./results/scaffold/ 
# python sample.py --config ./configs/scaffold/test/diffbp.yml --out_root ./results/scaffold/ 
# python sample.py --config ./configs/scaffold/test/diffsbdd.yml --out_root ./results/scaffold/ 


# python sample.py --config ./configs/sidechain/test/pocket2mol.yml --out_root ./results/sidechain/ 
# python sample.py --config ./configs/sidechain/test/graphbp.yml --out_root ./results/sidechain/ 
# python sample.py --config ./configs/sidechain/test/targetdiff.yml --out_root ./results/sidechain/ 
# python sample.py --config ./configs/sidechain/test/diffbp.yml --out_root ./results/sidechain/ 
# python sample.py --config ./configs/sidechain/test/diffsbdd.yml --out_root ./results/sidechain/ 

# python sample.py --config ./configs/denovo/casestudy/targetdiff.yml --out_root ./results/denovo/ --tag casestudy
# python sample.py --config ./configs/denovo/casestudy/diffbp.yml --out_root ./results/denovo/ --tag casestudy
# python sample.py --config ./configs/denovo/casestudy/diffsbdd.yml --out_root ./results/denovo/ --tag casestudy
# python sample.py --config ./configs/denovo/casestudy/flag.yml --out_root ./results/denovo/ --tag casestudy
# python sample.py --config ./configs/denovo/casestudy/graphbp.yml --out_root ./results/denovo/ --tag casestudy
# python sample.py --config ./configs/denovo/casestudy/pocket2mol.yml --out_root ./results/denovo/ --tag casestudy
# python sample.py --config ./configs/denovo/casestudy/d3fg_fg.yml --out_root ./results/denovo/ --tag casestudy_context
# python sample.py --config ./configs/denovo/casestudy/d3fg_linker.yml --out_root ./results/denovo/ --tag casestudy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('-c', '--config', type=str, default='./configs/denovo/test/diffsbdd.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results/denovo/')
    parser.add_argument('-t', '--tag', type=str, default='selftrain')
    parser.add_argument('-s', '--seed', type=int, default=2024)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-ckpt', '--checkpoint', type=str, default='pretrained')
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--threshold_ratio', type=float, default=0.8)

    args = parser.parse_args()
 
    # Load configs
    config, config_name = load_config(args.config)
    if args.checkpoint is not None:
        config.model.checkpoint = os.path.join(
            "/".join(config.model.checkpoint.split('/')[:4]), 
            args.tag, 
            'checkpoints',
            args.checkpoint + '.pt'
        )

    if 'fg' not in config.model.type:
        from repo.utils.configuration import set_num_atom_type, set_num_bond_type
        set_num_atom_type(config)
        set_num_bond_type(config)
    else:
        from repo.utils.configuration import set_num_fg_type
        set_num_fg_type(config)

    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Testset 
    datasets = get_pl_dataset(config.data.test)
    dataset = datasets['test']

    dr = os.path.join(args.out_root, config_name)

    if not os.path.exists(dr):

        os.makedirs(dr, exist_ok=True)

    mark = 0
    
    log_dir = get_new_log_dir(dr, prefix='', tag=args.tag)

    logger = get_logger('sample', log_dir)

    # Load checkpoint and model
    logger.info('Loading model config and checkpoints: %s' % (config.model.checkpoint))
    ckpt = torch.load(config.model.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    logger.info(str(lsd))

    for i in range(mark, len(dataset)):

        args.index = i
        get_structure = lambda: dataset[args.index]
        get_raw_structure = lambda: dataset.dataset.get_raw(dataset.indices[args.index])

        # Logging
        raw_strcuture_ = get_raw_structure()

        structure_id = raw_strcuture_['entry'][0][:-4]
        
        save_dir = os.path.join(log_dir, '%s' % (structure_id))
        os.makedirs(save_dir, exist_ok=True)

        logger.info('Data ID: %s' % structure_id)
        
        data_native = {'entry': raw_strcuture_['entry']}
    
        data_list_repeat = [get_structure() for _ in range(config.sampling.num_samples)]
        batch_size = args.batch_size if config.data.get('batch_size', None) is None else config.data.batch_size
        
        loader = DataLoader(PygDatasetFromList(data_list_repeat), 
                            batch_size=batch_size, 
                            shuffle=False,
                            follow_batch=config.data.get('follow_batch', []))
        count = 0
        mol_part_list = []
        for batch in tqdm(loader, desc=structure_id, dynamic_ncols=True):
            torch.set_grad_enabled(False)
            model.eval()
            try:
                batch = batch.to(args.device)
            except:
                batch = recursive_to(batch, args.device)
            
            traj_batch = model.sample(batch)
            
            if len(traj_batch) == 0: continue

            if config.sampling.translate:
                result_batch = translate(traj_batch[0], batch.protein_translation[:1])
            else:
                result_batch = traj_batch[0]      

            if 'fg' in config.model.type:
                result_split = split_batch_into_samples_fg(result_batch, mode=config.mode) 
            else:
                result_split = split_batch_into_samples(result_batch, mode=config.mode)

            if config.get('reconstruct', None) is not None:
                
                for result in result_split:
                    try:
                        try:
                            mol = reconstruct_mol(result['pos'], 
                                                  result['atom'], 
                                                  result['aromatic'], 
                                                  basic_mode=config.reconstruct.basic_mode)
                        except:
                            mol = obabel_recover_bond(result['pos'], 
                                                      result['atom'])
                            
                        mol, success = evaluate_validity(mol, args.threshold, args.threshold_ratio)
                        if success:
                            count += 1
                            data = {'pos': np.array(result['pos']),
                                    'atom': np.array(result['atom']),
                                    'entry': data_native['entry']}
                            torch.save(data, os.path.join(save_dir, 'sample_%04d.pt' % count))
                            save_mol(mol, os.path.join(save_dir, 'sample_%04d.sdf' % count))
                    except:
                        continue
            
            elif config.get('fg2mol', None) is not None:
                for result in result_split:
                    part_mol = atom_from_fg(result['pos_center'],
                                            result['orientation'],
                                            result['fg_type'])
                    mol_part_list.append(part_mol)
    
        if config.get('fg2mol', None) is not None:
            torch.save(mol_part_list, os.path.join(save_dir, 'gen_ctx_pool_%04d.pt' % len(mol_part_list)))
            torch.save(mol_part_list, os.path.join(save_dir, 'gen_ctx_pool_raw.pt'))


if __name__ == '__main__':
    main()
