import os
import argparse
import multiprocessing as mp
import pickle
import shutil
from functools import partial

from tqdm.auto import tqdm

from repo.datasets.parsers import parse_sdf_file, PDBProteinFA


def load_item(item, path):
    pdb_path = os.path.join(path, item[0])
    sdf_path = os.path.join(path, item[1])
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    with open(sdf_path, 'r') as f:
        sdf_block = f.read()
    return pdb_block, sdf_block


def process_item(item, args):
    # try:
        pdb_block, sdf_block = load_item(item, args.source)
        protein = PDBProteinFA(pdb_block)
        ligand = parse_sdf_file(os.path.join(args.source, item[1]))

        pdb_block_pocket = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand, args.radius)
        )
        
        ligand_fn = item[1]
        pocket_fn = ligand_fn[:-4] + '_pocket%d.pdb' % args.radius
        ligand_dest = os.path.join(args.dest, ligand_fn)
        pocket_dest = os.path.join(args.dest, pocket_fn)
        os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)
        if not os.path.exists(os.path.join(args.dest, ligand_fn)):
            shutil.copyfile(
                src=os.path.join(args.source, ligand_fn),
                dst=os.path.join(args.dest, ligand_fn)
            )
        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)
        return pocket_fn, ligand_fn, item[0], item[2]  # item[0]: original protein filename; item[2]: rmsd.
    # except Exception:
    #     print('Exception occurred.', item)
    #     return None, item[1], item[0], item[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./raw_data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--dest', type=str, default='./raw_data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    try:
        with open(os.path.join(args.source, 'index_raw.pkl'), 'rb') as f:
            index = pickle.load(f)
    except:
        with open(os.path.join(args.source, 'index.pkl'), 'rb') as f:
            index = pickle.load(f)

    pool = mp.Pool(args.num_workers)
    index_pocket = []
    if args.num_workers > 1:
        for item_pocket in tqdm(pool.imap_unordered(partial(process_item, args=args), index), total=len(index)):
            index_pocket.append(item_pocket)
    else:
        for item in index:
            item_pocket = process_item(item=item, args=args)
            index_pocket.append(item_pocket)

    # index_pocket = pool.map(partial(process_item, args=args), index)
    pool.close()

    index_path = os.path.join(args.dest, 'index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(index_pocket, f)

    print('Done. %d protein-ligand pairs in total.' % len(index))
    