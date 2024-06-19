import pickle
import os
from rdkit import Chem
path = '/linhaitao/CGBBench/data/geomdrug_random'
file_names = [file for file in os.listdir(path) if file.endswith('pickle')]
save_path = '/linhaitao/CGBBench/data/geomdrug_random/geomdrug_random_sdf'
for file_name in file_names:
    file_path = os.path.join(path, file_name)
    f = open(file_path, 'rb')
    pkl = pickle.load(f)
    rdmol = pkl['conformers'][0]['rd_mol']
    w = Chem.SDWriter(os.path.join(save_path, file_name.split('.')[0] + '.sdf'))  
    w.write(rdmol)
