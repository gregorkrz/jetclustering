# SVJ clustering
The repo has evolved from [here](https://github.com/selvaggi/mlpf) - mainly, we use the dataloader and code for reading  the root files for the previous MLPF project. The preprocessing part is not really needed but it does help with performance when we are doing a lot of experiments with the same dataset.

## Setup
**Important**: To make it easier and less time-consuming to move the commands across different machines, i.e. lxplus, T3 and Vega, we use relative paths. However, all commands can also be supplied absolute paths starting with `/`. **In case you use relative paths, make sure to modify the `env.sh` file with your paths!**

0. Environment setup: We use the Python with packages compiled in the following container: `gkrz/lgatr:v3`. The container can be built from scratch using the Dockerfile in this repo.


1. Set the environment variables `source env.sh`


### Preprocess data
See the script at `sbatch jobs/preprocess_v0.slurm` (make sure to update your local `env.sh` file!)

## Evaluation of clustering

For AK8: `python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals/SVJ_hadronic_std --dataset-cap 1000`


For AK8 GenJets: `python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals/SVJ_hadronic_std --dataset-cap 1000 --jets-object genjets`


For any model: `python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/GATr_rinv_03_m_900  --eval-dir train/Test_betaPt_BC_all_datasets_2025_01_07_17_50_45  --dataset-cap 1000 --jets-object model_jets` Add `--eval-dir` with the path to the eval run containing the coordinates and clustering labels. Optionally, add `--clustering-suffix` in case there are multiple clusterings saved in the same folder. (usually not unless you were fine-tuning the clustering)


The script produces output in the `results` folder. The script goes over the events up to dataset-cap (optional). 


## Training

See mainly `jobs/vega/lgatr_training.sh`, `jobs/vega/transformer_training.sh`, `jobs/vega/gatr_training_vega.sh` - you might need to modify the slurm file a bit to fit the system you are running on

## Evaluation
In order to move things faster, scripts to evaluate the trained models faster at a given ckpt are given. 

To evaluate at step 10k of the given training run: `python -m scripts.generate_test_jobs -template t3 -run Transformer_training_40k_5_64_4_2025_01_22_15_55_39 -step 10000 -tag params_study`
* Important: The step provided counts from the starting point of training the model: for example, if the run breaks in the middle and it's restarted from the latest ckpt, the command will identify that and load a checkpoint from the previous run if it contains one. You only need to provide the latest training with the `-run` argument.
* The `-tag` argument identifies the given study and can be later used to retrieve all the evals of all the models for a given run.


After the GPU eval, the CPU eval from above needs to be ran: `python -m scripts.test_plot_jobs --tag params_study`. The script will identify the runs that need to have evaluation figures produced. Uncommend the AK8 part in the file to also evaluate with AK8. Inside the produced folder, it also produces run_config.pkl that can be used later to make plots (of e.g. metrics vs number of params, model architecture, and amount of training).



Use the scripts in `scripts/` to produce the joint plots of F1 score, precision, recall etc.




### Datasets

`scouting_PFNano_signals1`: Contains special PFCands and PFCands in separate fields

`scouting_PFNano_signals2`: Contains both special PFCands and PFCands in the same field, under PFCands.

It was easier to just create this instead of always having special treatment for the special PFCands. As of January 2025, we are only using this version, accessible at `/pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/preprocessed_data/scouting_PFNano_signals2`.


