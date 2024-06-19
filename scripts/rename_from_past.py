import os
import shutil

past_dir = '/linhaitao/CGBBench/results/denovo/diffsbdd/diffsbdd_nouff'
new_dir = '/linhaitao/CGBBench/results/denovo/diffsbdd/2024_05_30__16_00_10/'
protein_names = os.listdir(past_dir)
for protein_name in protein_names:
    sdf_names = os.listdir(os.path.join(past_dir, protein_name))
    pocke_name = '_'.join(sdf_names[0].split('_')[:-1]) + '_pocket10'
    os.makedirs(os.path.join(new_dir, protein_name, pocke_name), exist_ok=True)
    for i, name in enumerate(sdf_names):
        shutil.copyfile(os.path.join(past_dir, protein_name, name), os.path.join(new_dir, protein_name, pocke_name, str(i)+'.sdf'))
