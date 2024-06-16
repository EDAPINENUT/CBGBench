# CBGBench: Fill in the Blank of Protein-Molecule Binding Graph

## Installation
#### Create environment with basic packages.
```
conda env create -f environment.yml
conda activate cbgbench
```

#### Install pytorch and torch_geometric

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg pytorch-scatter pytorch-cluster -c pyg
```

#### Install tools for chemistry
```
# install rdkit, efgs, obabel, etc.
pip install --use-pep517 EFGs
pip install biopython
conda install rdkit openbabel tensorboard tqdm pyyaml easydict python-lmdb -c conda-forge

# install plip
mkdir tools
cd tools
git clone https://github.com/pharmai/plip.git
cd plip
python setup.py install
alias plip='python plip/plip/plipcmd.py'
cd ..

### note that if there is error in setup.py, it can be ignored as long as openbabel is installed.

# install docking tools
conda install -c conda-forge numpy swig boost-cpp sphinx sphinx_rtd_theme
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
pip install meeko==0.1.dev3 scipy pdb2pqr vina

# If you encounter the following error:
# ImportError: libtiff.so.5: cannot open shared object file: No such file or directory
# Please try the following steps to resolve it:
pip uninstall pillow
pip install pillow
```

## Prepare Dataset

### CrossDocked2020
#### 1. Download processed datasets
(i) Download the processed dataset from [Google Drive](https://drive.google.com/drive/folders/1wm5_rMbemxqMiTxoBr_V-Vt5NyNtdZT7?usp=share_link) . For each datasets, it is corresponding to different methods and the tasks.  Please refer to the Table below to find the processed data that you need to use. Note that to train D3FG, it is a two stage model that firstly generate functional groups and then generate linker atoms, so that we named the two-stage model as D3FG_fg and D3FG_linker, and the training data is different for them.

Table: The data file name corresponding to different methods and tasks.
| model       | task       | file name                                                                 |
|-------------|------------|---------------------------------------------------------------------------|
| Pocket2Mol  | de novo    | data/pl/crossdocked_v1.1_rmsd1.0_pocket10_processed_fullatom.lmdb         |
| GraphBP     | de novo    | data/pl/crossdocked_v1.1_rmsd1.0_pocket10_processed_fullatom.lmdb         |
| DiffSBDD    | de novo    | data/pl/crossdocked_v1.1_rmsd1.0_pocket10_processed_fullatom.lmdb         |
| DiffBP      | de novo    | data/pl/crossdocked_v1.1_rmsd1.0_pocket10_processed_fullatom.lmdb         |
| TargetDiff  | de novo    | data/pl/crossdocked_v1.1_rmsd1.0_pocket10_processed_fullatom.lmdb         |
| FLAG        | de novo    | data/pl_arfg/crossdocked_v1.1_rmsd1.0_pocket10_processed_arfuncgroup.lmdb |
| D3FG_fg     | de novo    | data/pl_fg/crossdocked_v1.1_rmsd1.0_pocket10_processed_funcgroup.lmdb     |
| D3FG_linker | de novo    | data/pl/crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb           |
| Pocket2Mol  | fragment   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_frag.lmdb      |
| GraphBP     | fragment   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_frag.lmdb      |
| DiffSBDD    | fragment   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_frag.lmdb      |
| DiffBP      | fragment   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_frag.lmdb      |
| TargetDiff  | fragment   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_frag.lmdb      |
| Pocket2Mol  | linker     | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb    |
| GraphBP     | linker     | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb    |
| DiffSBDD    | linker     | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb    |
| DiffBP      | linker     | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb    |
| TargetDiff  | linker     | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb    |
| Pocket2Mol  | scaffold   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_scaffold.lmdb  |
| GraphBP     | scaffold   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_scaffold.lmdb  |
| DiffSBDD    | scaffold   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_scaffold.lmdb  |
| DiffBP      | scaffold   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_scaffold.lmdb  |
| TargetDiff  | scaffold   | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_scaffold.lmdb  |
| Pocket2Mol  | side chain | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_sidechain.lmdb |
| GraphBP     | side chain | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_sidechain.lmdb |
| DiffSBDD    | side chain | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_sidechain.lmdb |
| DiffBP      | side chain | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_sidechain.lmdb |
| TargetDiff  | side chain | data/pl_decomp/crossdocked_v1.1_rmsd1.0_pocket10_processed_sidechain.lmdb |

#### 2. Copy the datasets to `./data` dir, in which the complete data file directory will look like

```
- CGBBench
    - data
        - pl
            - crossdocked_name2id.pt
            - crossdocked_v1.1_rmsd1.0_pocket10_processed_fullatom.lmdb
            - crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb
        - pl_arfg
            - crossdocked_name2id_arfuncgroup.pt
            - crossdocked_v1.1_rmsd1.0_pocket10_processed_arfuncgroup.lmdb
        - pl_decomp
            - crossdocked_name2id_frag.pt
            - crossdocked_name2id_linker.pt
            - crossdocked_name2id_scaffold.pt
            - crossdocked_name2id_sidechain.pt
            - crossdocked_v1.1_rmsd1.0_pocket10_processed_frag.lmdb
            - crossdocked_v1.1_rmsd1.0_pocket10_processed_linker.lmdb
            - crossdocked_v1.1_rmsd1.0_pocket10_processed_scaffold.lmdb
            - crossdocked_v1.1_rmsd1.0_pocket10_processed_sidechain.lmdb
        - pl_fg
            - crossdocked_name2id_funcgroup.pt
            - crossdocked_v1.1_rmsd1.0_pocket10_processed_funcgroup.lmdb
```


#### 2. Download raw data and prepare from scratch

(i) Download `crossdocked_v1.1_rmsd1.0.tar.gz` from [TargetDiff Drive](https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK), and copy it to `./raw_data/` with
```
mkdir raw_data
tar -xzvf crossdocked_v1.1_rmsd1.0.tar.gz ./raw_data
```

(ii) Run the following:
```
python ./scripts/extract_pockets.py --source raw_data/crossdocked_v1.1_rmsd1.0 --dest raw_data/crossdocked_v1.1_rmsd1.0_pocket10
```

(iii) In training, the Dataset will prepare for each tasks. Pls refer to `Training`. 

### ADRB1 and DRD3 Targets
 ####  Download processed targets.
 Also, you can download the processes targets file `case_study` from [Google Drive](https://drive.google.com/drive/folders/1YulDlChalaIjqRJjCyOFSY1-KoZ-hYCP?usp=share_link), and copy it to the `./data` directory, which will lead the dir to
```
- CGBBench
    - data
        - case_study
            - processed
                - case_study_processed_fullatom.lmdb
                - case_study_processed_funcgroup.lmdb
                - case_study_name2id.pt
                ...
```

## Train from Scratch
python train xxx --tag selftrain

#### Also, you can use our pretrained ones from Google Drives [], the corresponding pretrained data file names to differet tasks and models are provided in the table below:

TODO: Table with pretrained models

Download the pretrained `.pt` checkpoints thatyou need, and copy them into `./logs/`, which will lead to the following directory structure:
```
# TODO
- CBGBench
    - logs
        - denovo
            - d3fg_fg
                - pretrained
                    - checkpoints
                        - 951000.pt
            - d3fg_linker
                - pretrained
                    - checkpoints
                        - 4840000.pt
            - diffbp
                - pretrained
                    - checkpoints
                        - 4848000.pt
            ... 
        - frag
            - diffbp
                - pretrained
                    - checkpoints
                        - 476000.pt
            ...
        ...
```

## Generation on test sets
#### Train from scratch

`bash generate.sh --config --ckpt `

note that `D3FG` uses two-step generation strategy, so that the command should be
`python generate -- -- `

#### Use pretrained models

`python generate.py -- --ckpt`

For `D3FG`, you should run
`python generate.py -- --`

## Evaluation
#### Train from scratch
If you have trained the model by yourself, run
```
cd evaluate_scripts
bash evaluate.sh --method xx --tasks xx --tag selftrain 

e.g.
# bash evaluate.sh --method targetdiff --tasks denovo --tag selftrain
# bash evaluate.sh --method targetdiff --tasks frag --tag selftrain
# bash evaluate.sh --method targetdiff --tasks linker --tag selftrain
# bash evaluate.sh --method targetdiff --tasks scaffold --tag selftrain
# bash evaluate.sh --method targetdiff --tasks sidechain --tag selftrain
```

#### Use the pretrained models
Or if you have downloaded pretrained models, please run

```
cd evaluate_scripts
bash bash evaluate.sh --method xx --tasks xx --tag pretrained 
```

## TODO List
1. Use the pretrained models to generate on real-world targets flexibly.

2. Include more models, such as 3DSBDD, DecompDiff, VoxMol and LiGAN.

If you find the repository is helpful to your research or projects, please refer to our benchmark paper:
```
TODO
```
