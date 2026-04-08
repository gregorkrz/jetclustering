# AGENTS.md

## Cursor Cloud specific instructions

### Overview
This is a Python 3.10 ML framework for IRC-safe jet clustering using Lorentz Geometric Algebra Transformers (L-GATr). The main runnable service is a **Gradio demo web app** (`app.py`) on port 7860 that performs jet clustering inference using pre-trained models.

### Python version
The project requires **Python 3.10** (pinned by numba 0.58.1 and the Dockerfile). The VM needs `python3.10` installed via deadsnakes PPA. A virtualenv at `/workspace/.venv` is used.

### Key dependencies and install order
Dependencies must be installed in a specific order to avoid version conflicts:
1. `numba==0.58.1` (pins numpy < 1.27)
2. PyTorch 2.5.0 **CPU** from `https://download.pytorch.org/whl/cpu`
3. `torch_geometric` and PyG extensions (`pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`) from `https://data.pyg.org/whl/torch-2.5.0+cpu.html`
4. `xformers==0.0.29.post1` (must match torch 2.5.x; do NOT let lgatr upgrade torch)
5. L-GATr from `https://github.com/gregorkrz/lorentz-gatr` — install with `pip install --no-deps` to prevent it from upgrading torch
6. Remaining packages: `pytorch-lightning`, `fastjet`, `gradio`, `huggingface_hub`, `hdbscan`, `ruff`, etc.

### Gotcha: torch version conflicts
Installing `lgatr` from git will attempt to upgrade PyTorch to the latest CUDA version. Always install lgatr with `--no-deps` and manually install its dependencies first. After any dependency changes, verify `python -c "import torch; print(torch.__version__)"` shows `2.5.0+cpu`.

### Gotcha: xformers on CPU
xformers CUDA extensions won't load on CPU — the warning is expected. The demo uses `cpu_demo=True` which sets the attention mask to `None`, bypassing xformers attention. The import `from xformers.ops.fmha import BlockDiagonalMask` requires xformers >= 0.0.29.

### Gotcha: pytorch_cmspepr
`pytorch_cmspepr` requires CUDA to build and is NOT needed for the demo app. It's only used by `GravNetConv` layers for training.

### Running the demo app
```bash
source /workspace/.venv/bin/activate
cd /workspace
GRADIO_SERVER_NAME=0.0.0.0 python app.py
```
The app starts on port 7860. Pre-trained models must exist in `models/` and demo datasets in `demo_datasets/` (downloaded from HuggingFace Hub).

### Models and datasets
Downloaded from HuggingFace:
- Models: `huggingface_hub.snapshot_download(repo_id='gregorkrzmanc/jetclustering', local_dir='models/')`
- Demo datasets: `huggingface_hub.snapshot_download(repo_id='gregorkrzmanc/jetclustering_demo', local_dir='demo_datasets/', repo_type='dataset')`

### Linting
```bash
source /workspace/.venv/bin/activate
ruff check
```
The codebase has existing lint issues (research code); `ruff` is listed in `requirements.txt`.

### Testing
No automated test suite exists. The demo app can be tested via the Gradio API or browser UI. See README.md for project structure and usage.
