---
title: JetClustering
emoji: ⚛️
colorFrom: "red"
colorTo: "blue"
sdk: docker
app_file: app.py
pinned: false
---

# SVJ clustering
The repo has evolved from [here](https://github.com/selvaggi/mlpf) - mainly, we use the dataloader and code for reading  the root files for the previous MLPF project. The preprocessing part is not really needed but it does help with performance when we are doing a lot of experiments with the same dataset.

## Setup
**Important**: To make it easier and less time-consuming to move the commands across different machines, i.e. lxplus, T3 work, and T3 SE, we use relative paths. However, all commands can also be supplied absolute paths starting with `/`. **In case you use relative paths, make sure to modify the `env.sh` file with your paths!**

0. Environment setup: We use the Python with packages compiled in the following container: `gkrz/lgatr:v3`. The container can be also built from scratch using the Dockerfile in this repo (`Dockerfile_training`). You will need to install some additional libraries (`fastjet`, `HDBSCAN` etc.), so make sure you always set the same `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR`.
For example, you can use a script similar to this one to load the container and run commands in it interactively:
```bash
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
singularity shell  -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc   -B /t3home -H /t3home/gkrzmanc --nv docker://gkrz/lgatr:v3
```

1. Set the environment variables `source env.sh` or use the `.env` file (e.g. when running code in PyCharm).

### Generating Delphes data

See instructions in the other repo: http://github.com/gregorkrz/LJP

### Preprocessing Delphes data

Run the slurm job: `sbatch jobs/preprocess_v3_Delphes_QCDtrain.slurm` (make sure to update your local `env.sh` file!).  
Other jobs are located at `jobs/preprocess_v3_Delphes_QCDEval.slurm`, `jobs/preprocess_v3_Delphes_PU_PFfix_Train.slurm`, and `jobs/preprocess_v3_Delphes_PU_PFfix.slurm`.


### Download preprocessed datasets

Available at https://huggingface.co/datasets/gregorkrzmanc/jetclustering - put them in the preprocessed_data folder.

## Evaluation of clustering

For AK8: `python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals/SVJ_hadronic_std --dataset-cap 1000`


For AK8 GenJets: `python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals/SVJ_hadronic_std --dataset-cap 1000 --jets-object genjets`


For any model: `python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/GATr_rinv_03_m_900  --eval-dir train/Test_betaPt_BC_all_datasets_2025_01_07_17_50_45  --dataset-cap 1000 --jets-object model_jets` Add `--eval-dir` with the path to the eval run containing the coordinates and clustering labels. Optionally, add `--clustering-suffix` in case there are multiple clusterings saved in the same folder. (usually not unless you were fine-tuning the clustering)


The script produces output in the `results` folder. The script goes over the events up to dataset-cap (optional). 



### Automated evaluation
In order to move things faster, scripts to evaluate the trained models faster at a given ckpt are given. 

To evaluate at step 10k of the given training run: `python -m scripts.generate_test_jobs -template t3 -run Transformer_training_40k_5_64_4_2025_01_22_15_55_39 -step 10000 -tag params_study`
* Important: The step provided counts from the starting point of training the model: for example, if the run breaks in the middle and it's restarted from the latest ckpt, the command will identify that and load a checkpoint from the previous run if it contains one. You only need to provide the latest training with the `-run` argument.
* The `-tag` argument identifies the given study and can be later used to retrieve all the evals of all the models for a given run.
* The command pulls the config (e.g. model architecture and hyperparameters) automatically from the wandb run of the training.
* Add `-os` argument with a path to the objectness score checkpoint to use in the evaluation. (note that we are not really using this part of the model in the final paper, so it can be ignored)


After the GPU eval, the CPU eval from above needs to be ran: `python -m scripts.test_plot_jobs --tag params_study --input <input_dataset>`.
The script will identify the runs that need to have evaluation figures produced. The "analysis" part was separated from the GPU part, as it was changed many times over. E.g., you can decide to store some more info (e.g. pt of each jet) instead of just the precision, recall, and F1 scores for each dataset,
and then re-run this part. Once a job is finished, a txt file is placed in the resulting directory that marks it was done. If you run the above python command again, it will skip it (you can use the `-ow` flag.)
The command needs to be ran also to spawn the AK evaluation by adding `--submit-AKX` flag. For parton-level and gen-level  (PL and GL), the flags `-pl` and `-gl` need to be added too.
So basically for each evaluation dataset, you need to run the `python -m scripts.test_plot_jobs` command at least four times (3 times for AK and once for the completed GPU runs).

Inside the produced folder, it also produces run_config.pkl that can be used later to make plots (of e.g. metrics vs number of params, model architecture, and amount of training).

#### pT cutoff studies

In order to run the studies of the pT cutoff, run the `python -m scripts.test_plot_jobs` command with the following flag: `-pt 90`
This will create a new folder with suffix `_pt_90.0` - and similarly for other pT cutoffs. This is not a super efficient way to do it, but the jobs usually don't take much time, so we never really optimized it.

Afterwards, for the plots of the metrics vs. the pt cutoff, use the script in `scripts/metrics_plots_vs_pt_cutoff.py`.



Use the scripts in `scripts/` to produce the joint plots of F1 score, precision, recall etc.



### Download trained models

The final model weights can be accessed at https://huggingface.co/gregorkrzmanc/jetclustering/tree/main.

### Live demo

The live demo is available at https://huggingface.co/spaces/gregorkrzmanc/jetclustering.

### WandB runs

The WandB runs are at https://wandb.ai/fcc_ml/svj_clustering. You need to provide your own API key in env.sh.

### Training (deprecated)

See mainly `jobs/vega/lgatr_training.sh`, `jobs/vega/transformer_training.sh`, `jobs/vega/gatr_training_vega.sh` - you might need to modify the slurm file a bit to fit the system you are running on

### Datasets (deprecated)

`scouting_PFNano_signals1`: Contains special PFCands and PFCands in separate fields

`scouting_PFNano_signals2`: Contains both special PFCands and PFCands in the same field, under PFCands.

It was easier to just create this instead of always having special treatment for the special PFCands. As of January 2025, we are only using this version, accessible at `/pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/preprocessed_data/scouting_PFNano_signals2`.

