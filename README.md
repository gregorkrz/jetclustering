# SVJ clustering
The repo has evolved from [here](https://github.com/selvaggi/mlpf) - mainly, we use the dataloader and code for reading  the root files for the previous MLPF project

## Setup
**Important**: To make it easier and less time-consuming to move the commands across different machines, i.e. lxplus, T3 and Vega, we use relative paths. However, all commands can also be supplied absolute paths starting with `/`. **In case you use relative paths, make sure to modify the `env.sh` file with your paths!**
0. Environment setup
We use the environment defined in the following container: `gkrz/lgatr:v3`
1. Set the environment variables `source env.sh`


### Preprocess data

#### First version without dark quarks

`python -m src.preprocessing.preprocess_dataset --input scouting_PFNano_signals/SVJ_hadronic_std --output scouting_PFNano_signals/SVJ_hadronic_std`

Or, run it on slurm: `sbatch jobs/preprocess_v0.slurm` (make sure to update your local `env.sh` file!)

#### Evaluation of clustering

For AK8: `python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals/SVJ_hadronic_std --dataset-cap 1000`
For AK8 GenJets: `python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals/SVJ_hadronic_std --dataset-cap 1000 --jets-object genjets`
For any model: `python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/GATr_rinv_03_m_900  --eval-dir train/Test_betaPt_BC_all_datasets_2025_01_07_17_50_45  --dataset-cap 1000 --jets-object model_jets` (Add `--eval-dir` with the path to the eval run containing the clustering and the jets. Optionally, add --clustering-suffix in case there are multiple clusterings saved in the file.)


The script produces output in the `results` folder. The script goes over the events up to dataset-cap (optional).


### Training models

### Evaluation

### Datasets

`scouting_PFNano_signals1`: Contains special PFCands and PFCands in separate fields

`scouting_PFNano_signals2`: Contains both special PFCands and PFCands in the same field, under PFCands.
It was easier to just create this instead of always having special treatment for the special PFCands.
