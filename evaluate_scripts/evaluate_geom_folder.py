import os
import argparse
import subprocess
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count

def get_all_deepest_subfolders(base_path):
    deepest_subfolders = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  
            deepest_subfolders.append(root)
    return deepest_subfolders

def run_evaluation(args):
    result_path, base_result_path, base_pdb_path, exhaustiveness, verbose = args
    try:
        relative_path = os.path.relpath(result_path, base_result_path)
        pdb_sub_path = os.path.join(base_pdb_path, relative_path + ".pdb")
        if os.path.exists(pdb_sub_path) and '/'.join(result_path.split('/')[-2:]) == relative_path:
            print(f"Processing {result_path} with PDB {pdb_sub_path}")

            cmd = [
                "python", "evaluate_geom_single.py",
                "--result_path", result_path,
                "--pdb_path", pdb_sub_path,
                "--exhaustiveness", str(exhaustiveness),
                "--verbose", str(verbose)
            ]
            subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error processing {result_path}: {e}")

def main(base_result_path, base_pdb_path, exhaustiveness, verbose):
    deepest_subfolders = get_all_deepest_subfolders(base_result_path)

    nthreads = cpu_count()
    print("Number of CPU cores:", nthreads)

    args_list = []
    for result_path in deepest_subfolders:
        if 'docking_results' in result_path:
            result_path = os.path.dirname(result_path)
        args_list.append((result_path, base_result_path, base_pdb_path, exhaustiveness, verbose))

    with Pool(processes=nthreads) as pool:
        for _ in tqdm(pool.imap(run_evaluation, args_list), total=len(args_list)):
            pass

    print('Evaluation done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_result_path', type=str, default='../results/denovo/diffbp/selftrain', help="Base result path to traverse")
    parser.add_argument('--base_pdb_path', type=str, default='../data/crossdocked_test/',  help="Base PDB path for constructing pdb_path")
    parser.add_argument('--exhaustiveness', type=int, default=16, help="Exhaustiveness parameter for Vina docking")
    parser.add_argument('--verbose', type=eval, default=False, help="Verbose output")
    args = parser.parse_args()

    main(args.base_result_path, args.base_pdb_path, args.exhaustiveness, args.verbose)
