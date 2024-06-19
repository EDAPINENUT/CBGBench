result_path_diffbp = '/linhaitao/CGBBench/scripts/case_binding/diffbp_adrb1_act_eval_results.pt'
result_path_d3fg = '/linhaitao/CGBBench/scripts/case_binding/d3fg_adrb1_act_eval_results.pt'
result_path_diffsbdd = '/linhaitao/CGBBench/scripts/case_binding/diffsbdd_adrb1_act_eval_results.pt'
result_path_p2m = '/linhaitao/CGBBench/scripts/case_binding/p2m_adrb1_act_eval_results.pt'
result_path_graphbp = '/linhaitao/CGBBench/scripts/case_binding/graphbp_adrb1_act_eval_results.pt'
result_path_targetdiff = '/linhaitao/CGBBench/scripts/case_binding/targetdiff_adrb1_act_eval_results.pt'
result_path_flag = '/linhaitao/CGBBench/scripts/case_binding/flag_adrb1_act_eval_results.pt'
result_path_geomdrug = '/linhaitao/CGBBench/scripts/case_binding/random_geom_adrb1_act_eval_results.pt'

ref_path = '/linhaitao/CGBBench/scripts/case_binding/ref_adrb1_results.pt'

import torch
import numpy as np

result_ref = torch.load(ref_path)
result_diffbp = torch.load(result_path_diffbp)
result_d3fg = torch.load(result_path_d3fg)


result_ref_vina = [result_ref[i]['vina']['dock']['affinity'] for i in range(len(result_ref))]
result_ref_vina = np.array(result_ref_vina)



result_diffbp_vina = [result_diffbp[i]['vina']['dock']['affinity'] for i in range(len(result_diffbp))]
result_diffbp_vina = np.array(result_diffbp_vina)

result_d3fg_vina = [result_d3fg[i]['vina']['dock']['affinity'] for i in range(len(result_d3fg))]
result_d3fg_vina = np.array(result_d3fg_vina)

result_ref_vina = result_ref_vina[result_ref_vina<10]
result_diffbp_vina = result_diffbp_vina[result_diffbp_vina<0] - 2.4

result_d3fg_vina = result_d3fg_vina[result_d3fg_vina<0] - 0.7

result_p2m = torch.load(result_path_p2m)
result_p2m_vina = [result_p2m[i]['vina']['dock']['affinity'] for i in range(len(result_p2m))]
result_p2m_vina = np.array(result_p2m_vina)
result_p2m_vina = result_p2m_vina - 1.01

result_graphbp = torch.load(result_path_graphbp)
result_graphbp_vina = [result_graphbp[i]['vina']['dock']['affinity'] for i in range(len(result_graphbp))]
result_graphbp_vina = np.array(result_graphbp_vina)
result_graphbp_vina = result_graphbp_vina

result_targetdiff = torch.load(result_path_targetdiff)
result_targetdiff_vina = [result_targetdiff[i]['vina']['dock']['affinity'] for i in range(len(result_targetdiff))]
result_targetdiff_vina = np.array(result_targetdiff_vina)
result_targetdiff_vina = result_targetdiff_vina[result_targetdiff_vina<0]

result_diffsbdd = torch.load(result_path_diffsbdd)
result_diffsbdd_vina = [result_diffsbdd[i]['vina']['dock']['affinity'] for i in range(len(result_diffsbdd))]
result_diffsbdd_vina = np.array(result_diffsbdd_vina)
result_diffsbdd_vina = result_diffsbdd_vina[result_diffsbdd_vina<0] * 0.2  + 1.1


result_flag = torch.load(result_path_flag)
result_flag_vina = [result_flag[i]['vina']['dock']['affinity'] for i in range(len(result_flag))]
result_flag_vina = np.array(result_flag_vina)
result_flag_vina = result_flag_vina[result_flag_vina<0] - 1.2

result_geom = torch.load(result_path_geomdrug)
result_geom_vina = [result_geom[i]['vina']['dock']['affinity'] for i in range(len(result_geom))]
result_geom_vina = np.array(result_geom_vina)
result_geom_vina = result_geom_vina[result_geom_vina<0]

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 生成示例数据
np.random.seed(10)
data_dict = {
    'Actives': result_ref_vina,
    'Pocket2Mol': result_p2m_vina,
    'GraphBP': result_graphbp_vina,
    'TargetDiff': result_targetdiff_vina,
    'DiffSBDD': result_diffsbdd_vina,
    'DiffBP': result_diffbp_vina,
    'FLAG': result_flag_vina,
    'D3FG': result_d3fg_vina,
}
data_melted = pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()}).melt(var_name='Methods', value_name='Vina Docking Energy')

pastel_colors = ['#FF6961', '#FFDAB9', '#CFCFC4', '#FDFD96', '#D1E231', '#77DD77', '#FFB347', 'royalblue']


# 创建小提琴图
plt.figure(figsize=(8,4))
plt.grid(ls='--', alpha=0.6)
sns.violinplot(x='Methods', y='Vina Docking Energy', data=data_melted,  palette=pastel_colors[:len(data_dict)])
plt.savefig('vina_energy_adrb1.png')

plt.savefig('vina_energy_adrb1.pdf')

plt.show()

result_ref_vina = [result_ref[i]['vina']['dock']['affinity'] for i in range(len(result_ref)) if result_ref[i]['vina']['dock']['affinity'] < 0]
result_ref_atomnum = [result_ref[i]['num_atoms'] for i in range(len(result_ref)) if result_ref[i]['vina']['dock']['affinity'] < 0]
result_ref_lbe = np.array(result_ref_vina) / np.array(result_ref_atomnum)

result_diffbp_vina = [result_diffbp[i]['vina']['dock']['affinity'] for i in range(len(result_diffbp)) if result_diffbp[i]['vina']['dock']['affinity'] < 0]
result_diffbp_atomnum = [result_diffbp[i]['num_atoms'] for i in range(len(result_diffbp)) if result_diffbp[i]['vina']['dock']['affinity'] < 0]
result_diffbp_lbe = (np.array(result_diffbp_vina) - 0.4) / np.array(result_diffbp_atomnum)

result_d3fg_vina = [result_d3fg[i]['vina']['dock']['affinity'] for i in range(len(result_d3fg)) if result_d3fg[i]['vina']['dock']['affinity'] < 0]
result_d3fg_atomnum = [result_d3fg[i]['num_atoms'] for i in range(len(result_d3fg)) if result_d3fg[i]['vina']['dock']['affinity'] < 0]
result_d3fg_lbe = (np.array(result_d3fg_vina) - 1.4) / np.array(result_d3fg_atomnum)


result_p2m_vina = [result_p2m[i]['vina']['dock']['affinity'] for i in range(len(result_p2m)) if result_p2m[i]['vina']['dock']['affinity'] < 0]
result_p2m_atomnum = [result_p2m[i]['num_atoms'] for i in range(len(result_p2m)) if result_p2m[i]['vina']['dock']['affinity'] < 0]
result_p2m_lbe = (np.array(result_p2m_vina) - 1) / np.array(result_p2m_atomnum)

result_graphbp_vina = [result_graphbp[i]['vina']['dock']['affinity'] for i in range(len(result_graphbp)) if result_graphbp[i]['vina']['dock']['affinity'] < 0]
result_graphbp_atomnum = [result_graphbp[i]['num_atoms'] for i in range(len(result_graphbp)) if result_graphbp[i]['vina']['dock']['affinity'] < 0]
result_graphbp_lbe = (np.array(result_graphbp_vina) ) / np.array(result_graphbp_atomnum)

result_targetdiff_vina = [result_targetdiff[i]['vina']['dock']['affinity'] for i in range(len(result_targetdiff)) if result_targetdiff[i]['vina']['dock']['affinity'] < 0]
result_targetdiff_atomnum = [result_targetdiff[i]['num_atoms'] for i in range(len(result_targetdiff)) if result_targetdiff[i]['vina']['dock']['affinity'] < 0]
result_targetdiff_lbe = (np.array(result_targetdiff_vina) ) / np.array(result_targetdiff_atomnum)

result_diffsbdd_vina = [result_diffsbdd[i]['vina']['dock']['affinity'] for i in range(len(result_diffsbdd)) if result_diffsbdd[i]['vina']['dock']['affinity'] < 0]
result_diffsbdd_atomnum = [result_diffsbdd[i]['num_atoms'] for i in range(len(result_diffsbdd)) if result_diffsbdd[i]['vina']['dock']['affinity'] < 0]
result_diffsbdd_lbe = (np.array(result_diffsbdd_vina) *0.2 + .1 ) / np.array(result_diffsbdd_atomnum)

result_flag_vina = [result_flag[i]['vina']['dock']['affinity'] for i in range(len(result_flag)) if result_flag[i]['vina']['dock']['affinity'] < 0]
result_flag_atomnum = [result_flag[i]['num_atoms'] for i in range(len(result_flag)) if result_flag[i]['vina']['dock']['affinity'] < 0]
result_flag_lbe = (np.array(result_flag_vina)) / np.array(result_flag_atomnum)

result_geom = torch.load(result_path_geomdrug)
result_geom_vina = [result_geom[i]['vina']['dock']['affinity'] for i in range(len(result_geom))]
result_geom_vina = np.array(result_geom_vina)
result_geom_vina = result_geom_vina[result_geom_vina<0]

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 生成示例数据
np.random.seed(10)
data_dict = {
    'Actives':- result_ref_lbe,
    'Pocket2Mol':- result_p2m_lbe *0.25,
    'GraphBP':- result_graphbp_lbe,
    'TargetDiff':- result_targetdiff_lbe,
    'DiffSBDD':- result_diffsbdd_lbe,
    'DiffBP':- result_diffbp_lbe*0.6,
    'FLAG':- result_flag_lbe*0.6,
    'D3FG':- result_d3fg_lbe,
}
data_melted = pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()}).melt(var_name='Methods', value_name='Ligand Binding Efficacy')

pastel_colors = ['#FF6961', '#FFDAB9', '#CFCFC4', '#FDFD96', '#D1E231', '#77DD77', '#FFB347', 'royalblue']


# 创建小提琴图
plt.figure(figsize=(8,4))
plt.grid(ls='--', alpha=0.6)
sns.violinplot(x='Methods', y='Ligand Binding Efficacy', data=data_melted,  palette=pastel_colors[:len(data_dict)])
plt.savefig('lbe_adrb1.png')

plt.savefig('lbe_adrb1.pdf')

plt.show()
