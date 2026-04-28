---
title: JetClustering
emoji: ⚛️
colorFrom: "red"
colorTo: "blue"
sdk: docker
app_file: app.py
pinned: false
---

# Learning IRC-Safe Jet Clustering with Geometric Algebra Transformers

Authors: Gregor Kržmanc, Roberto Seidita, Annapaola de Cosa

Paper at ML4PS Workshop at NeurIPS: https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_59.pdf


A machine learning framework for jet clustering in CMS events using Geometric Algebra Transformers. This repository provides tools for preprocessing, training, and evaluating jet clustering models on Delphes simulation data.

## 🚀 Quick Start

**Live Demo**: Try the interactive demo at [https://huggingface.co/spaces/gregorkrzmanc/jetclustering](https://huggingface.co/spaces/gregorkrzmanc/jetclustering)

The demo allows you to:
- Upload particle-level data for an event (CSV of pt, eta, phi, mass, charge)
- Select different model variants and training datasets
- Visualize clustering results compared to Anti-kt jets
- View detailed jet information in JSON format

> **Note**: The live demo runs on the free HuggingFace tier, and it's extremely slow (1-5 minutes per event). For faster local execution, see the [Local Demo Setup](#local-demo-setup) section below.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Pre-trained Models](#pre-trained-models)
- [Project Structure](#project-structure)

## Overview

The repo has evolved from the [MLPF repository](https://github.com/selvaggi/mlpf) (we partially reuse the dataloader).

The framework supports:
- Multiple loss functions (GP, GP_IRC_S, GP_IRC_SN)
- Various training datasets (QCD, SVJ events with different parameters)
- Automated evaluation pipelines
- Comprehensive visualization tools

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (for training)
- Singularity/Apptainer (for containerized training)
- Access to CERN/PSI computing infrastructure (for full workflow)

## Installation

### Environment Setup

This project uses a Docker container with pre-compiled packages. The container image is `gkrz/lgatr:v3`.

#### Option 1: Use Pre-built Container

```bash
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
singularity shell -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc -B /t3home -H /t3home/gkrzmanc --nv docker://gkrz/lgatr:v3
```

#### Option 2: Build from Dockerfile

Build the training container from scratch:

```bash
docker build -f Dockerfile_training -t gkrz/lgatr:v3 .
```

**Important**: Ensure consistent `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` settings across sessions.

### Environment Variables

1. **Set up environment variables** by sourcing `env.sh`:

```bash
source env.sh
```

Or use the `.env` file for IDE integration (e.g., PyCharm).

2. **Configure paths**: Edit `env.sh` to set your local paths:

```bash
export SVJ_CODE_ROOT="/path/to/jetclustering/code"
export SVJ_DATA_ROOT="/path/to/jetclustering/data"
export SVJ_RESULTS_ROOT="/path/to/jetclustering/results"
export SVJ_PREPROCESSED_DATA_ROOT="/path/to/jetclustering/preprocessed_data"
export SVJ_RESULTS_ROOT_FALLBACK="/path/to/fallback/results"  # Optional: for SE storage
export SVJ_WANDB_ENTITY="your_wandb_entity"
```

**Path Configuration Notes**:
- Use relative paths for portability across machines (lxplus, T3 work, T3 SE)
- Absolute paths starting with `/` are also supported
- `SVJ_RESULTS_ROOT_FALLBACK` is used when files aren't available in the primary results directory

### Local Demo Setup

For faster local execution, use Docker Compose:

```yaml
version: '3.8'

services:
  jetclustering_demo:
    image: gkrz/jetclustering_demo_cpu:v0
    ports:
      - "7860:7860"
```

Save as `docker-compose.yml` and run:

```bash
docker-compose up
```

## Data Preparation

### Generating Delphes Data

See the [jetclustering_sim repository](https://github.com/gregorkrz/jetclustering_sim) for instructions on generating Delphes simulation data.

### Preprocessing Delphes Data

Preprocess your Delphes data using the provided SLURM jobs:

```bash
# For QCD training data
sbatch jobs/preprocess_v3_Delphes_QCDtrain.slurm

# For QCD evaluation data
sbatch jobs/preprocess_v3_Delphes_QCDEval.slurm

# For pile-up (PU) data
sbatch jobs/preprocess_v3_Delphes_PU_PFfix_Train.slurm
sbatch jobs/preprocess_v3_Delphes_PU_PFfix.slurm
```

**Important**: Update your local `env.sh` file before running preprocessing jobs!

### Download Preprocessed Datasets

Preprocessed datasets are available at:
- **Hugging Face Datasets**: [https://huggingface.co/datasets/gregorkrzmanc/jetclustering](https://huggingface.co/datasets/gregorkrzmanc/jetclustering)

Download and place them in the `preprocessed_data/` folder.

### Storage Management

To copy results to Storage Element (SE) and free up local storage:

```bash
rsync -avz -e "ssh" /work/gkrzmanc/jetclustering/results/ /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/results
```

The system automatically falls back to `SVJ_RESULTS_ROOT_FALLBACK` when files aren't found in the primary location.

## Training

All training is launched through `src.train`.  The wrapper SLURM scripts under
`jobs/` ultimately invoke the same CLI shown below; the commands here can be run
directly inside the training container or any environment with the
`requirements.txt` packages installed.

`-train` and `-val` take dataset directory names that resolve under
`$SVJ_PREPROCESSED_DATA_ROOT`; a typical run uses parts 0–8 for training and
part 9 for validation.

### Common arguments

| Flag | Meaning |
|---|---|
| `-net` / `--network-config` | Path to the model module (e.g. `src/models/LGATr/lgatr.py`) |
| `-bs` / `--batch-size` | Per-step batch size |
| `--num-steps` | Stop after this many optimizer steps (`-1` to use `--num-epochs`) |
| `--validation-steps` | How often to checkpoint and run validation |
| `--gt-radius` | Cone radius for matching dark quarks to PF candidates (label) |
| `--num-blocks` | Number of transformer / GATr blocks (encoder depth) |
| `--no-pid` | Drop the 9-dim PID one-hot from input scalars |
| `--load-model-weights` | Resume from / fine-tune a previous checkpoint |
| `-irc` / `--irc-safety-loss` | Add the IRC-safety auxiliary loss (uses an extra augmented loader) |
| `--irc-mode` | `IRC_S` (collinear splitting only) or `IRC_SN` (default — splitting + noise, alternated per event) |
| `--augment-soft-particles` | Add 500 ~0-pT particles to every event (separate from the IRC auxiliary stream) |

### Sample commands

#### L-GATr (base GP training, m=900, r_inv=0.3)

```bash
python -m src.train \
  -train Delphes_020425_train2_PU_PFfix_part{0..8}/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -val   Delphes_020425_train2_PU_PFfix_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -net   src/models/LGATr/lgatr.py \
  -bs 20 --gpus 0 \
  --run-name LGATr_GP \
  --num-steps 50000 --validation-steps 2000 \
  --num-blocks 10 -mv-ch 16 -s-ch 64 \
  --attr-loss-weight 0.1 --coord-loss-weight 0.1 \
  --beta-type pt+bc --gt-radius 0.8 \
  --spatial-part-only --no-pid \
  --val-dataset-size 1000
```

#### Transformer baseline

```bash
python -m src.train \
  -train Delphes_020425_train2_PU_PFfix_part{0..8}/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -val   Delphes_020425_train2_PU_PFfix_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -net   src/models/transformer/transformer.py \
  -bs 20 --gpus 0 \
  --run-name Transformer_GP \
  --num-steps 50000 --validation-steps 2000 \
  --num-blocks 10 --internal-dim 128 --n-heads 4 \
  --attr-loss-weight 0.1 --coord-loss-weight 0.1 \
  --beta-type pt+bc --gt-radius 0.8 \
  --no-pid --val-dataset-size 1000
```

#### GATr (Euclidean)

```bash
python -m src.train \
  -train Delphes_020425_train2_PU_PFfix_part{0..8}/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -val   Delphes_020425_train2_PU_PFfix_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -net   src/models/GATr/Gatr.py \
  -bs 20 --gpus 0 \
  --run-name GATr_GP \
  --num-steps 50000 --validation-steps 2000 \
  --num-blocks 10 -mv-ch 16 -s-ch 64 \
  --attr-loss-weight 0.1 --coord-loss-weight 0.1 \
  --beta-type pt+bc --gt-radius 0.8 \
  --no-pid --val-dataset-size 1000
```

#### IRC-safety fine-tuning (GP_IRC_S / GP_IRC_SN)

The IRC variants resume from a base GP checkpoint and add the IRC-safety
auxiliary loss.  Switch between IRC_S and IRC_SN with `--irc-mode` — no source
edits required.

```bash
# GP_IRC_SN: alternating per event -- odd-index events get collinear splitting,
# even-index events get soft-particle addition (i.e. both S and N are used)
python -m src.train \
  -train Delphes_020425_train2_PU_PFfix_part{0..8}/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -val   Delphes_020425_train2_PU_PFfix_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -net   src/models/LGATr/lgatr.py \
  -bs 8 --gpus 0 \
  --run-name LGATr_GP_IRC_SN \
  --num-steps 75000 --validation-steps 2000 \
  --num-blocks 10 -mv-ch 16 -s-ch 64 \
  --attr-loss-weight 0.1 --coord-loss-weight 0.1 \
  --beta-type pt+bc --gt-radius 0.8 \
  --spatial-part-only --no-pid \
  --augment-soft-particles \
  --irc-safety-loss --irc-mode IRC_SN \
  --load-model-weights train/LGATr_GP_<timestamp>/step_50000_epoch_12.ckpt

# GP_IRC_S: collinear splitting only (no soft-particle addition in the IRC stream)
python -m src.train ... --irc-safety-loss --irc-mode IRC_S ...
```

#### Mask2Former (this branch)

Mask2Former segments particles into jets directly via M learnable queries +
Hungarian matching, bypassing HDBSCAN at evaluation time.  It reuses the same
data pipeline as the other models.

```bash
python -m src.train \
  -train Delphes_020425_train2_PU_PFfix_part{0..8}/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -val   Delphes_020425_train2_PU_PFfix_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak \
  -net   src/models/Mask2Former/mask2former.py \
  -bs 20 --gpus 0 \
  --run-name Mask2Former \
  --num-steps 50000 --validation-steps 2000 \
  --num-blocks 6 --internal-dim 128 --n-heads 4 \
  --num-queries 16 --num-dec-layers 3 \
  --gt-radius 0.8 --no-pid \
  --val-dataset-size 1000
```

`--num-blocks` / `--internal-dim` / `--n-heads` configure the shared transformer
encoder; `--num-queries` and `--num-dec-layers` are Mask2Former-specific.  The
loss (`get_loss_func` inside the network module) replaces Object Condensation
with a Hungarian-matched mask-classification loss; the OC-only flags
(`--attr-loss-weight`, `--coord-loss-weight`, `--beta-type`, `--spatial-part-only`)
are ignored for Mask2Former runs.

To add the IRC-safety loss (with model-specific mask-consistency formulation):

```bash
python -m src.train ... \
  -net src/models/Mask2Former/mask2former.py \
  --irc-safety-loss --irc-mode IRC_SN  # or IRC_S
```

### Pre-built SLURM wrappers

The `jobs/` tree contains thin SLURM wrappers around the commands above:

- `jobs/base_training/` — base GP training (lgatr / transformer / gatr) on the
  m=900 / r_inv=0.3 dataset.
- `jobs/base_training_different_datasets/` — extended training on alternative
  datasets (m=700 / r_inv=0.7, QCD, mixtures), `aug/` for soft-particle
  augmentation, `aug_IRC_S/` and `aug_IRC_SN/` for the IRC-safety variants.
  These scripts pass `-irc` and `--augment-soft-particles` and load a base
  checkpoint via `--load-model-weights`; with the new `--irc-mode` flag the
  parity toggle no longer requires a source edit.

## Evaluation

### Automated Evaluation Pipeline

The evaluation process consists of two stages:

#### Stage 1: GPU Evaluation

Generate evaluation jobs for a specific checkpoint:

```bash
python -m scripts.generate_test_jobs \
    -template t3 \
    -run Transformer_training_40k_5_64_4_2025_01_22_15_55_39 \
    -step 10000 \
    -tag params_study
```

**Parameters**:
- `-template`: Job template (e.g., `t3`, `vega`)
- `-run`: Training run identifier
- `-step`: Checkpoint step (counts from training start)
- `-tag`: Study identifier for grouping evaluations
- `-os`: Path to objectness score checkpoint (optional, not used in final paper)
- `-pl`: Evaluate on parton-level particles
- `-gl`: Evaluate on gen-level particles
- `--steps-from-zero`: Disable automatic checkpoint detection from previous runs

**Checkpoint Resolution**:
- The script automatically detects if training was restarted from a checkpoint
- It loads the appropriate checkpoint from previous runs if needed
- Use `--steps-from-zero` to disable this behavior

**Helper Script**: Use `notebooks/gen_test_job_cmd_gen.py` to generate evaluation commands interactively.

#### Stage 2: CPU Evaluation and Analysis

After GPU evaluation completes, run analysis and plotting:

```bash
python -m scripts.test_plot_jobs \
    --tag params_study \
    --input <input_dataset>
```

**Additional Flags**:
- `--submit-AKX`: Spawn Anti-kt evaluation jobs
- `-pl`: For parton-level evaluation
- `-gl`: For gen-level evaluation
- `-ow`: Overwrite existing results
- `-pt <cutoff>`: Run pT cutoff studies (e.g., `-pt 90`)

**Evaluation Workflow**:
1. Run GPU evaluation for each dataset
2. Run CPU evaluation/analysis (4 times per dataset: 3 for AK variants + 1 for GPU results)
3. Results include `run_config.pkl` for later metric analysis

### pT Cutoff Studies

To study performance at different pT cutoffs:

```bash
python -m scripts.test_plot_jobs --tag params_study --input <dataset> -pt 90
```

This creates results with suffix `_pt_90.0`. Generate plots comparing metrics vs. pT cutoff:

```bash
python -m scripts/metrics_plots_vs_pt_cutoff.py
```

## Visualization

### Generating Evaluation Plots

Produce comprehensive evaluation plots:

```bash
python -m scripts.plot_eval_count_matched_quarks --input <input_directory>
```

**Input Directory**: Points to the directory produced by `test_plot_jobs` (named after the tag).

**Configuration**: Modify the dictionary around line 320 in the script to map training run IDs to standardized names (e.g., `LGATr_GP_IRC_SN`).

### Metric Analysis

Use scripts in `scripts/` to generate joint plots of:
- F1 score
- Precision
- Recall
- Other performance metrics

The `run_config.pkl` files generated during evaluation can be used to create plots comparing:
- Metrics vs. number of parameters
- Metrics vs. model architecture
- Metrics vs. training duration

## Pre-trained Models

### Model Weights

Pre-trained model weights are available at:
- **Hugging Face Model Hub**: [https://huggingface.co/gregorkrzmanc/jetclustering/tree/main](https://huggingface.co/gregorkrzmanc/jetclustering/tree/main)

### Weights & Biases Runs

Training runs and metrics are logged at:
- **WandB Project**: [https://wandb.ai/fcc_ml/svj_clustering](https://wandb.ai/fcc_ml/svj_clustering)

**Setup**: Add your WandB API key to `env.sh`:

```bash
export WANDB_API_KEY="your_api_key_here"
```

## Project Structure

```
jetclustering/
├── app.py                      # Gradio demo interface
├── Dockerfile                  # Demo container
├── Dockerfile_training         # Training container
├── env.sh                      # Environment variables
├── requirements.txt            # Python dependencies
├── config_files/               # Model and dataset configurations
├── jobs/                       # SLURM job scripts
│   ├── base_training/          # Base model training
│   ├── base_training_different_datasets/  # Extended training
│   └── preprocess_*.slurm      # Preprocessing jobs
├── notebooks/                  # Jupyter notebooks and helper scripts
├── scripts/                    # Evaluation and plotting scripts
│   ├── generate_test_jobs.py   # Generate evaluation jobs
│   ├── test_plot_jobs.py       # Run analysis and plotting
│   └── plot_eval_count_matched_quarks.py  # Main plotting script
└── src/                        # Source code
    ├── data/                   # Data loading utilities
    ├── dataset/                # Dataset classes
    ├── evaluation/             # Evaluation metrics
    ├── jetfinder/              # Jet finding algorithms
    ├── layers/                 # Neural network layers
    ├── models/                 # Model architectures
    ├── plotting/               # Visualization utilities
    ├── preprocessing/          # Data preprocessing
    ├── train.py                # Training script
    └── utils/                  # Utility functions
```
