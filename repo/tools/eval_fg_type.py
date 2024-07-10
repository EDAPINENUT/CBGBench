
FUNCTIONAL_GROUPS = ['c1ccccc1', 'NC=O', 'O=CO', 'c1ccncc1', 'c1ncc2nc[nH]c2n1', 'NS(=O)=O',
                   'O=P(O)(O)O', 'OCO', 'c1cncnc1','c1cn[nH]c1',
                   'O=P(O)O','c1ccc2ccccc2c1','c1ccsc1',
                   'N=CN','NC(N)=O','O=c1cc[nH]c(=O)[nH]1',
                   'c1ccc2ncccc2c1', 'c1cscn1', 'c1ccc2[nH]cnc2c1','c1c[nH]cn1',
                   'O=[N+][O-]', 'O=CNO', 'NC(=O)O','O=S=O','c1ccc2[nH]ccc2c1']


FUNCTIONAL_GROUPS_DISTRIBUTION = {'c1ccccc1': 0.39202252631956525,
                                   'NC=O': 0.14653765446491024,
                                    'O=CO': 0.11915693694663151, 
                                    'c1ccncc1': 0.045181173066461015,
                                    'c1ncc2nc[nH]c2n1': 0.03398377483245751,
                                    'NS(=O)=O': 0.03025330152865709, 
                                    'O=P(O)(O)O': 0.022272240854661,
                                    'OCO': 0.01914557813368725, 
                                    'c1cncnc1': 0.017830347161193513, 
                                    'c1cn[nH]c1': 0.016153427671263997, 
                                    'O=P(O)O': 0.015755869218214754, 
                                    'c1ccc2ccccc2c1': 0.014174602889921145, 
                                    'c1ccsc1': 0.012955025079063317, 
                                    'N=CN': 0.012898231014341996, 
                                    'NC(N)=O': 0.012455835141775921, 
                                    'O=c1cc[nH]c(=O)[nH]1': 0.012390073593151234, 
                                    'c1ccc2ncccc2c1': 0.010518858618648779, 
                                    'c1cscn1': 0.010360433069689307, 
                                    'c1ccc2[nH]cnc2c1': 0.010348476424484818, 
                                    'c1c[nH]cn1': 0.008859874096525997, 
                                    'O=[N+][O-]': 0.008076713835631999, 
                                    'O=CNO': 0.007404152542879519, 
                                    'NC(=O)O': 0.007287575252135756, 
                                    'O=S=O': 0.007099258090165062, 
                                    'c1ccc2[nH]ccc2c1': 0.0068780601538820235}

FUNCTIONAL_GROUPS_RATIO = {'c1ccccc1': 0.7125401370227701,
                           'NC=O': 0.26634683820772914,
                           'O=CO': 0.21657964652254463, 
                           'c1ccncc1': 0.08212129938008335,
                            'c1ncc2nc[nH]c2n1': 0.06176890854463563, 
                            'NS(=O)=O': 0.05498840033250569, 
                            'O=P(O)(O)O': 0.04048202459020847, 
                            'OCO': 0.034799002482926486, 
                            'c1cncnc1': 0.03240843869018836, 
                            'c1cn[nH]c1': 0.029360469854447263, 
                            'O=P(O)O': 0.028637867617096878, 
                            'c1ccc2ccccc2c1': 0.025763757966282184, 
                            'c1ccsc1': 0.023547053358470474, 
                            'N=CN': 0.02344382446742042, 
                            'NC(N)=O': 0.02263972573713578, 
                            'O=c1cc[nH]c(=O)[nH]1': 0.02252019754749887, 
                            'c1ccc2ncccc2c1': 0.019119077242376003, 
                            'c1cscn1': 0.01883112296734164, 
                            'c1ccc2[nH]cnc2c1': 0.01880939056922584, 
                            'c1c[nH]cn1': 0.0161037070038086, 
                            'O=[N+][O-]': 0.014680234927223632, 
                            'O=CNO': 0.01345778753320982, 
                            'NC(=O)O': 0.01324589665158076, 
                            'O=S=O': 0.012903611381256893, 
                            'c1ccc2[nH]ccc2c1': 0.012501562016114574}

import collections
from scipy import spatial as sci_spatial
import numpy as np
from EFGs import mol2frag

def eval_fg_type_ratio(functional_groups, num_mols):

    pred_fg_ratio = {}
    for k in FUNCTIONAL_GROUPS:
        pred_fg_ratio[k] = functional_groups[k] / num_mols

    mae = np.abs((np.array(list(FUNCTIONAL_GROUPS_RATIO.values())) - 
                    np.array(list(pred_fg_ratio.values())))).mean()
    return mae, pred_fg_ratio

def eval_fg_type_distribution(functional_groups):
    total_num_fgs = sum(functional_groups.values())

    pred_fg_distribution = {}
    for k in FUNCTIONAL_GROUPS:
        pred_fg_distribution[k] = functional_groups[k] / total_num_fgs

    js = sci_spatial.distance.jensenshannon(np.array(list(FUNCTIONAL_GROUPS_DISTRIBUTION.values())),
                                            np.array(list(pred_fg_distribution.values())))
    return js, pred_fg_distribution

def get_func_group_from_mol(mol):
    try:
        fgs, _ = mol2frag(mol)
    except:
        return []
    fg_stat = []
    for fg in fgs:
        if fg in FUNCTIONAL_GROUPS:
            fg_stat.append(fg)

    return fg_stat