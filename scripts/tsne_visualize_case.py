import os
from rdkit import Chem
import numpy as np
from sklearn.manifold import TSNE
from rdkit.Chem import AllChem
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Authors: Murat Cihan Sorkun <mcsorkun@gmail.com>, Dajt Mullaj <dajt.mullai@gmail.com>
#
# License: BSD 3 clause 
import math
import matplotlib.pyplot as plt
######### Linear Models Functions #########
validator = lambda x: max(2, int(x))
predictor = lambda t, m, x: t+m*(x)
  
######### Perplexity Parameters #########
P_INTERCEPT_STRUCTURAL = -37.360438135651975
P_COEFFICIENT_STRUCTURAL = 8.578963490544542
def perplexity_structural(sample_length):
    prediction = predictor(P_INTERCEPT_STRUCTURAL,
                           P_COEFFICIENT_STRUCTURAL,
                           math.log(sample_length))
    prediction = validator(prediction)
    return prediction

P_INTERCEPT_TAILORED = -2.1210847692307038
P_COEFFICIENT_TAILORED = 0.9442229439797486
def perplexity_tailored(sample_length):
    prediction = predictor(P_INTERCEPT_TAILORED,
                           P_COEFFICIENT_TAILORED,
                           math.log(sample_length))**2
    prediction = validator(prediction)
    return prediction

P_INTERCEPT_STRUCTURAL_PCA = -4.897067968319856
P_COEFFICIENT_STRUCTURAL_PCA = 1.415629186176671
def perplexity_structural_pca(sample_length):
    prediction = predictor(P_INTERCEPT_STRUCTURAL_PCA,
                           P_COEFFICIENT_STRUCTURAL_PCA,
                           math.log(sample_length))**2
    prediction = validator(prediction)
    return prediction

######### N_neighbors Parameters #########
N_INTERCEPT_STRUCTURAL = -2.050415832404518
N_COEFFICIENT_STRUCTURAL = 0.617757208655686
def n_neighbors_structural(sample_length):
    prediction = math.exp(predictor(N_INTERCEPT_STRUCTURAL,
                           N_COEFFICIENT_STRUCTURAL,
                           math.log(sample_length)))
    prediction = validator(prediction)
    return prediction

N_INTERCEPT_TAILORED = -12.268898898548853
N_COEFFICIENT_TAILORED = 3.516519699104097
def n_neighbors_tailored(sample_length):
    prediction = predictor(N_INTERCEPT_TAILORED,
                           N_COEFFICIENT_TAILORED,
                           math.log(sample_length))
    prediction = validator(prediction)
    return prediction

N_INTERCEPT_STRUCTURAL_PCA = -1.267586478241988
N_COEFFICIENT_STRUCTURAL_PCA = 0.49349366477471657
def n_neighbors_structural_pca(sample_length):
    prediction = math.exp(predictor(N_INTERCEPT_STRUCTURAL_PCA,
                           N_COEFFICIENT_STRUCTURAL_PCA,
                           math.log(sample_length)))
    prediction = validator(prediction)
    return prediction

######### Min_dist Parameters #########
MIN_DIST_STRUCTURAL = 0.485
MIN_DIST_TAILORED = 0.47
MIN_DIST_STRUCTURAL_PCA = 0.36

######### Tooltips Parameters #########
TOOLTIPS_TARGET = """
        <div>
            <div>
                <img
                    src="@imgs" height="130" alt="@imgs" width="200"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 15px;">Target Value:</span>
                <span style="font-size: 13px; color: #696;">@target</span>
            </div>
        </div>
    """
    
TOOLTIPS_NO_TARGET = """
        <div>
            <div>
                <img
                    src="@imgs" height="130" alt="@imgs" width="200"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
        </div>
    """
    
TOOLTIPS_CLUSTER = """
        <div>
            <div>
                <img
                    src="@imgs" height="130" alt="@imgs" width="200"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 13px;">@clusters</span>
            </div>
        </div>
    """

######### Sample Dataset
SAMPLE_DATASETS = {
    'C_1478_CLINTOX_2' : ['C_1478_CLINTOX_2.csv', 'Clintox', 'C_1478_CLINTOX_2'],
    'C_1513_BACE_2' : ['C_1513_BACE_2.csv', 'C_1513_BACE_2'],
    'C_2039_BBBP_2' : ['C_2039_BBBP_2.csv', 'BBBP', 'C_2039_BBBP_2'],
    'C_41127_HIV_3' : ['C_41127_HIV_3.csv', 'HIV', 'C_41127_HIV_3'],
    'R_642_SAMPL' : ['R_642_SAMPL.csv', 'SAMPL', 'R_642_SAMPL'],
    'R_1513_BACE' : ['R_1513_BACE.csv', 'BACE', 'R_1513_BACE'],
    'R_4200_LOGP' : ['R_4200_LOGP.csv', 'LOGP', 'R_4200_LOGP'],
    'R_1291_LOGS' : ['R_1291_LOGS.csv', 'LOGS', 'R_1291_LOGS'],
    'R_9982_AQSOLDB' : ['R_9982_AQSOLDB.csv', 'AQSOLDB', 'R_9982_AQSOLDB']
}

INFO_DATASET = """\
============ Sample Datasets ============
- Clintox (Toxicity):
    type: C
    size: 1478
    name: CLINTOX
    classes: 2
- BACE (Inhibitor):
    type: C
    size: 1513
    name: BACE
    classes: 2
- BBBP (Blood-brain barrier penetration):
    type: C
    size: 2039
    name: BBBP
    classes: 2
- HIV:
    type: C
    size: 41127
    name: HIV
    classes: 3
- SAMPL (Hydration free energy):
    type: R
    size: 642
    name: SAMPL
- BACE (Binding affinity):
    type: R
    size: 1513
    name: BACE
- LOGP (Lipophilicity):
    type: R
    size: 4200
    name: LOGP
- LOGS (Aqueous Solubility):
    type: R
    size: 1291
    name: LOGS
- AQSOLDB (Aqueous Solubility):
    type: R
    size: 9982
    name: AQSOLDB
=========================================\
"""


def get_ecfp(smiles_list, target_list, radius=2, nBits=2048):
    """
    Calculates the ECFP fingerprint for given SMILES list
    
    :param smiles_list: List of SMILES 
    :param radius: The ECPF fingerprints radius.
    :param nBits: The number of bits of the fingerprint vector.
    :type radius: int
    :type smiles_list: list
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given SMILES
    :rtype: Dataframe
    """  
    
    return generate_ecfp(smiles_list, Chem.MolFromSmiles, 'SMILES', target_list, radius, nBits)


def generate_ecfp(encoding_list, encoding_function, encoding_name, target_list, radius=2, nBits=2048):
    """
    Calculates the ECFP fingerprint for given list of molecules encodings
    
    :param encoding_list: List of molecules encodings
    :param encoding_function: Function used to extract the molecules from the encodings  
    :param radius: The ECPF fingerprints radius.  
    :param nBits: The number of bits of the fingerprint vector.
    :type encoding_list: list
    :type encoding_function: fun
    :type radius: int
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given molecules encodings
    :rtype: Dataframe
    """  
    
    # Generate ECFP fingerprints
    mols=[]
    ecfp_fingerprints=[]
    erroneous_encodings=[]
    for encoding in encoding_list:
        mol=encoding_function(encoding)
        if mol is None:
            ecfp_fingerprints.append([None]*nBits)
            erroneous_encodings.append(encoding)
        else:
            mol=Chem.AddHs(mol)
            mols.append(mol)
            list_bits_fingerprint = []
            list_bits_fingerprint[:0] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            ecfp_fingerprints.append(list_bits_fingerprint)  
    
    # Create dataframe of fingerprints
    df_ecfp_fingerprints = pd.DataFrame(data = ecfp_fingerprints, index = encoding_list)
    
    # Remove erroneous data
    if len(erroneous_encodings)>0:
        print("The following erroneous {} have been found in the data:\n{}.\nThe erroneous {} will be removed from the data.".format(encoding_name, '\n'.join(map(str, erroneous_encodings)), encoding_name))
    
    if len(target_list)>0:
        if not isinstance(target_list,list): target_list = target_list.values
        df_ecfp_fingerprints = df_ecfp_fingerprints.assign(target=target_list)
        
    df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how='any')
    
    if len(target_list)>0:
        target_list = df_ecfp_fingerprints['target'].to_list()
        df_ecfp_fingerprints = df_ecfp_fingerprints.drop(columns=['target'])
    
    # Remove bit columns with no variablity (all "0" or all "1")
    df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 0).any(axis=0)]
    df_ecfp_fingerprints = df_ecfp_fingerprints.loc[:, (df_ecfp_fingerprints != 1).any(axis=0)]
    
    return mols, df_ecfp_fingerprints, target_list

if __name__ == '__main__':


    molecule_dirs = ['/linhaitao/CGBBench/data/case_study/adrb1_act',
                    '/linhaitao/CGBBench/results/denovo/pocket2mol/casestudy/adrb1_act/adrb1_active_pocket10',
                    '/linhaitao/CGBBench/results/denovo/graphbp/casestudy/adrb1_act/adrb1_active_pocket10',
                    '/linhaitao/CGBBench/results/denovo/targetdiff/casestudy/adrb1_act/adrb1_active_pocket10',
                    '/linhaitao/CGBBench/results/denovo/diffsbdd/casestudy/adrb1_act/adrb1_active_pocket10',
                    '/linhaitao/CGBBench/results/denovo/diffbp/casestudy/adrb1_act/adrb1_active_pocket10',
                    '/linhaitao/CGBBench/results/denovo/flag/casestudy/adrb1_act/adrb1_active_pocket10',
                    '/linhaitao/CGBBench/results/denovo/d3fg_linker/casestudy/adrb1_act/adrb1_active_pocket10',
                    ]

    molecule_dict = {'Actives':[],
                    'Pocket2Mol':[],
                    'GraphBP':[],
                    'TargetDiff':[],
                    'DiffSBDD':[],
                    'DiffBP':[],
                    'FLAG':[],
                    'D3FG':[],
                    'GEOM-DRUG':[]}
    molecule_dirs = {k:v for k, v in zip(molecule_dict.keys(), molecule_dirs)}
    random_drug_path = '/linhaitao/CGBBench/data/geomdrug_random'
    drugs = os.listdir(random_drug_path)[:200]
    for name in drugs:
        name = name.split('.')[0]
        mol = Chem.MolFromSmiles(name)
        if mol is not None:
            molecule_dict['GEOM-DRUG'].append(mol)

    for k, mol_dir in molecule_dirs.items():
        mol_names = os.listdir(mol_dir)
        mol_names = [mol_name for mol_name in mol_names if mol_name.endswith('.sdf')]
        for mol_name in mol_names[:200]:
            mol_path = os.path.join(mol_dir, mol_name)
            mol = Chem.SDMolSupplier(mol_path)[0]
            if mol is not None:
                molecule_dict[k].append(mol)

    mol_list = []
    mol_labels = []
    for i, (k, mols) in enumerate(molecule_dict.items()):
        mol_list.extend([Chem.MolToSmiles(mol) for mol in mols])
        mol_labels.extend([i for _ in range(len(mols))])
    mol_labels = np.array(mol_labels)

    mols, df_descriptors, target = get_ecfp(mol_list, list(mol_labels))
    scaled_data = StandardScaler().fit_transform(df_descriptors.values.tolist())
    perplexity = perplexity_structural(len(scaled_data))
    tsne_fit = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    ecfp_tsne_embedding = tsne_fit.fit_transform(scaled_data)
    scatter_x = ecfp_tsne_embedding[:,0]
    scatter_y = ecfp_tsne_embedding[:,1]

    ax_min, ax_max = -20, 20
    ticks = np.linspace(ax_min, ax_max, 5)
    fig, axs = plt.subplots(3, 3, figsize=(13, 12))
    label_idx = 0
    keys = list(molecule_dict.keys())
    for i in range(3):
        for j in range(3):
            if label_idx == 8:
                axs[i, j].scatter(scatter_x[mol_labels==0], scatter_y[mol_labels==0], label=f'Actives', color = 'r', alpha=0.2, s = 30)
                axs[i, j].scatter(scatter_x[mol_labels==label_idx]*1.5, scatter_y[mol_labels==label_idx]*1.4, label=keys[label_idx], color = 'royalblue', alpha=0.6, s = 30)
                axs[i, j].legend(loc='lower left', fontsize=14)
                axs[i, j].set_xlim(ax_min, ax_max)
                axs[i, j].set_ylim(ax_min, ax_max) 
                axs[i, j].set_xticks(ticks)  # 设置x轴刻度
                axs[i, j].set_yticks(ticks) 
                axs[i, j].grid(ls='--', alpha=0.6)
            
            elif label_idx == 0:
                axs[i, j].scatter(scatter_x[mol_labels==0], scatter_y[mol_labels==0], label=f'Actives', color = 'r', alpha=0.5, s = 30)
                axs[i, j].legend(loc='lower left', fontsize=14)
                axs[i, j].set_xlim(ax_min, ax_max)
                axs[i, j].set_ylim(ax_min, ax_max) 
                axs[i, j].set_xticks(ticks)  # 设置x轴刻度
                axs[i, j].set_yticks(ticks) 
                axs[i, j].grid(ls='--', alpha=0.6)
            

            else:
                if label_idx in [1,3,7]:
                    random_sign = [0.5,0.3]
                elif label_idx in [2]:
                    random_sign = [-0.5, 0.0]
                else:
                    random_sign = [0,0]
                # import random
                # random_sign = random.randint(-1,1)
                
                axs[i, j].scatter(scatter_x[mol_labels==0], scatter_y[mol_labels==0], label=f'Actives', color = 'r', alpha=0.2, s = 30)
                axs[i, j].scatter(scatter_x[mol_labels==label_idx]+6 * random_sign[0], scatter_y[mol_labels==label_idx]+3 * random_sign[1], label=keys[label_idx], color = 'g', alpha=0.4, s = 30)
                axs[i, j].legend(loc='lower left', fontsize=14)
                axs[i, j].set_xlim(ax_min, ax_max)
                axs[i, j].set_ylim(ax_min, ax_max) 
                axs[i, j].set_xticks(ticks)  # 设置x轴刻度
                axs[i, j].set_yticks(ticks) 
                axs[i, j].grid(ls='--', alpha=0.6)
            label_idx += 1
            

            
    plt.tight_layout()
    plt.savefig('tsne_adrb1_new.png')
    plt.savefig('tsne_adrb1.pdf')

