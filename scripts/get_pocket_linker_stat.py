
import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from repo.utils.loader import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
from repo.utils.misc import *
from repo.utils.train import *
from repo.datasets.pl import get_pl_dataset
from repo.utils.evaluate import *
import numpy as np 
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/denovo/train/d3fg_linker.yml', type=str)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--resume', type=str, default=None)
args = parser.parse_args()

config, config_name = load_config(args.config)

datasets = get_pl_dataset(config.data.train)
train_dataset, val_dataset = datasets['train'], datasets['val']


train_loader = DataLoader(train_dataset, 
                        batch_size=1, 
                        shuffle=True, 
                        num_workers=args.num_workers,
                        follow_batch=config.data.get('follow_batch', []),
                        exclude_keys=config.data.get('exclude_keys', [])
                        )
dists = []
num_linkers = []
for i, batch in enumerate(tqdm(train_loader, desc='get_size', dynamic_ncols=True)):
    # Prepare data
    batch = batch.to(args.device)
    x_ca = batch['protein_pos']
    aa_dist = torch.pdist(x_ca)
    aa_dist = torch.sort(aa_dist, descending=True)[0]
    aa_dist_mean = torch.median(aa_dist[:10])

    num_linker = batch['ligand_gen_flag'].sum().item()

    dists.append(aa_dist_mean)
    num_linkers.append(num_linker)

min_dist = torch.tensor(dists).min().item()
max_dist = torch.tensor(dists).max().item()

dists_list = torch.tensor(dists).numpy()
num_linkers = np.array(num_linkers)

bounds = np.linspace(min_dist, max_dist, 20)
num_linker_stat = [[] for i in range(len(bounds))]

for dist,num_fg in zip(dists_list, num_linkers):
    for i, bound in enumerate(bounds):
        if bound > dist:
            num_linker_stat[i].append(num_fg) 


bounds = bounds[1:]
num_fg_stat = num_linker_stat[1:]
bins = [dict(Counter(i)) for i in num_fg_stat]
bins_tuple = []
for i, dict_stat in enumerate(bins):
    keys = []
    vals = []
    for key, val in dict_stat.items():
        keys.append(key)
        vals.append(val / len(num_fg_stat[i]))
    bins_tuple.append((keys, vals)) 

linker_stat = {'bounds': bounds, 
               'bins': bins_tuple}   
np.save('linker_stat', linker_stat)

