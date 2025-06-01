## gkrz/lgatr:v3
# docker build -t gkrz/lgatr:v4 .
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app



SHELL ["/bin/bash", "-c"]

USER root

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        git \
        python-is-python3 \
        python3-dev \
        python3-pip \
        && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 --version
RUN python3 --version
RUN python --version

RUN python3 -m pip install --no-cache-dir --upgrade pip
#python3 -m pip install --no-cache-dir --upgrade --requirement requirements.txt
RUN python3 -m pip install numba==0.58.1
# packages without conda
# RUN python3 -m pip install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN python3 -m pip install torch==2.5.0 torchvision torchaudio
RUN python3 -m pip install torch_geometric
#RUN python3 -m pip install  torch_scatter torch_sparse
#RUN python3 -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1.html
#RUN python3 -m pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1.html
#RUN python3 -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1.html
RUN python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv xformers==0.0.29.post1 -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

RUN python3 -m pip install pytorch-lightning yacs torchmetrics
RUN python3 -m pip install performer-pytorch
RUN python3 -m pip install tensorboardX
RUN python3 -m pip install ogb
RUN python3 -m pip install wandb
RUN python3 -m pip install seaborn
RUN python3 -m pip install  dgl
RUN python3 -m pip install numpy
RUN python3 -m pip install scipy
RUN python3 -m pip install pandas
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install matplotlib
RUN python3 -m pip install tqdm
RUN python3 -m pip install PyYAML
RUN python3 -m pip install awkward0
RUN python3 -m pip install uproot
RUN python3 -m pip install awkward
RUN python3 -m pip install vector
RUN python3 -m pip install lz4
RUN python3 -m pip install xxhash
RUN python3 -m pip install tables
RUN python3 -m pip install tensorboard
RUN python3 -m pip install plotly
RUN python3 -m pip install fastjet
RUN python3 -m pip install gradio
RUN python3 -m pip install huggingface_hub
RUN python3 -m pip install hdbscan
#RUN python3 -m pip install lgatr # This doesn't work
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='gregorkrzmanc/jetclustering', local_dir='models/'); \
snapshot_download(repo_id='gregorkrzmanc/jetclustering_demo', local_dir='demo_datasets/', repo_type='dataset')"
# remove pip cache
RUN python3 -m pip cache purge

# COPY docker/ext_packages /docker/ext_packages
# RUN python3 /docker/ext_packages/install_upstream_python_packages.py
RUN mkdir -p /opt/pepr

# Install GATr
#RUN cd /opt/pepr && git clone https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git geometric-algebra-transformer1
#RUN cd /opt/pepr/geometric-algebra-transformer1/ && python3 -m pip install .


# Install L-GATr - for some reason this only works if executed from the already-built container
RUN cd /opt/pepr && git clone https://github.com/gregorkrz/lorentz-gatr lgatr
RUN cd /opt/pepr/lgatr/ && python3 -m pip install .
RUN ls /usr/local/lib/python3.10/dist-packages/lgatr
RUN ls /usr/local/lib/python3.10/dist-packages/lgatr/layers
# Install torch_cmspepr

RUN cd /opt/pepr && git clone https://github.com/cms-pepr/pytorch_cmspepr
RUN cd /opt/pepr/pytorch_cmspepr/ && python3 -m pip install .

COPY . /app

# entrypoint run app.py with python
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]

