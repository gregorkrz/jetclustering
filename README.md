
# Jet clustering

## Setup
**Important**: To make it easier and less time-consuming to move the commands across different machines, we use relative paths. However, all commands can also be supplied absolute paths starting with `/`. **In case you use relative paths, make sure to modify the `env.sh` file with your paths!**

0. Environment setup: The container can be also built from scratch using the Dockerfile in this repo (`Dockerfile_training`). You will need to install some additional libraries (`fastjet`, `HDBSCAN` etc.), so make sure you always set the same `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR`.
For example, you can use a script similar to this one to load the container and run commands in it interactively:
```bash
export APPTAINER_TMPDIR=...
export APPTAINER_CACHEDIR=...
singularity shell  -B ... --nv docker://<CONTAINER_NAME>
```

1. Set the environment variables `source env.sh` or use the `.env` file (e.g. when running code in PyCharm).

### Training
`python -m src.train -train <DATASET(S)> -net src/models/LGATr/lgatr.py -bs 20 --gpus 0 --run-name <WANDB_RUN_NAME> --val-dataset-size 1000 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --num-blocks 10 -mv-ch 16 -s-ch 64 --spatial-part-only --validation-steps 2000 --no-pid`
Add the following flags: `--augment-soft-particles` for GP, `-irc` for the IRC safety loss (IRC_S or IRC_SN)

Important: you need to manually change from GP_IRC_SN to GP_IRC_S (line `if i % 2: # Every second one:` in `dataset/dataset.py`)!

### Automated evaluation
In order to move things faster, scripts to evaluate the trained models faster at a given ckpt are given. 

To evaluate at step 10k of the given training run: `python -m scripts.generate_test_jobs -template t3 -run Transformer_training_40k_5_64_4_2025_01_22_15_55_39 -step 10000 -tag params_study`
* Important: The step provided counts from the starting point of training the model: for example, if the run breaks in the middle and it's restarted from the latest ckpt, the command will identify that and load a checkpoint from the previous run if it contains one. You only need to provide the latest training with the `-run` argument.
Use `--steps-from-zero` if you want to avoid this behaviour.
* The `-tag` argument identifies the given study and can be later used to retrieve all the evals of all the models for a given run.
* The command pulls the config (e.g. model architecture and hyperparameters) automatically from the wandb run of the training.
* Add `-os` argument with a path to the objectness score checkpoint to use in the evaluation. (note that we are not really using this part of the model in the final paper, so it can be ignored)
* Add `-pl` and `-gl` flags to evaluate on the parton-level and gen-level particles, respectively. 

If it helps, you can try the `notebooks/gen_test_job_cmd_gen.py` script, which helps to generate the commands for evaluation.

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

### Producing plots

Run the script `python -m scripts.plot_eval_count_matched_quarks --input <input>` where input points to the directory produced by the `test_plot_jobs` (with the name the same as the tag). You need to modify the dictionary around line 320 that maps the training run IDs to 'standardized' names (e.g. LGATr_GP_IRC_SN).
The whole script has developed in a series of tiny additions of new plots, so itšs not the most efficient and it might benefit from restructuring.

### Live demo

A live interactive demo will be provided.


The script produces output in the `results` folder. The script goes over the events up to dataset-cap (optional). 

