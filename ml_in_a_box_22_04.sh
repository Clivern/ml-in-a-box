#!/bin/bash

# ==================================================================
# Module list
# ------------------------------------------------------------------
# python                3.10.12          (apt)
# torch                 2.1.0            (pip)
# torchvision           0.16.0           (pip)
# torchaudio            2.1.0            (pip)
# tensorflow            2.14.0           (pip)
# jax                   0.3.17           (pip)
# transformers          4.34.1           (pip)
# datasets              2.14.5           (pip)
# jupyterlab            4.0.7            (pip)
# numpy                 1.26.1           (pip)
# scipy                 1.11.3           (pip)
# pandas                2.1.1            (pip)
# cloudpickle           3.0.0            (pip)
# scikit-image          0.22.0           (pip)
# scikit-learn          1.3.1            (pip)
# matplotlib            3.8.0            (pip)
# ipython               8.16.1           (pip)
# ipykernel             6.15.2           (pip)
# ipywidgets            8.0.2            (pip)
# cython                0.29.32          (pip)
# tqdm                  4.66.1           (pip)
# gdown                 4.7.1            (pip)
# xgboost               2.0.0            (pip)
# pillow                10.1.0           (pip)
# seaborn               0.13.0           (pip)
# sqlalchemy            1.4.40           (pip)
# spacy                 3.7.2            (pip)
# nltk                  3.8.1            (pip)
# boto3                 1.28.66          (pip)
# tabulate              0.9.0            (pip)
# future                0.18.3           (pip)
# gradient              2.0.6            (pip)
# jsonify               0.5              (pip)
# opencv-python         4.8.1.78         (pip)
# pyyaml                5.4.1            (pip)
# sentence-transformers 2.2.2            (pip)
# wandb                 0.15.12          (pip)
# nodejs                20.x latest      (apt)
# default-jre           latest           (apt)
# default-jdk           latest           (apt)


# ==================================================================
# Initial setup
# ------------------------------------------------------------------

    # Set ENV variables
    export APT_INSTALL="apt-get install -y --no-install-recommends"
    export PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"
    export GIT_CLONE="git clone --depth 10"

    # Update apt
    sudo apt update


# ==================================================================
# Tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive \
    sudo $APT_INSTALL \
        gcc \
        make \
        pkg-config \
        apt-transport-https \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        rsync \
        git \
        vim \
        mlocate \
        libssl-dev \
        curl \
        openssh-client \
        unzip \
        unrar \
        zip \
        awscli \
        csvkit \
        emacs \
        joe \
        jq \
        dialog \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        nano \
        iputils-ping \
        sudo \
        ffmpeg \
        libsm6 \
        libxext6 \
        libboost-all-dev


# ==================================================================
# Python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive \
    sudo $APT_INSTALL software-properties-common

    # Installing python3.10
    DEBIAN_FRONTEND=noninteractive sudo $APT_INSTALL \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-distutils-extra

    # Add symlink so python and python3 commands use same python3.10 executable
    sudo ln -s /usr/bin/python3.10 /usr/local/bin/python3
    sudo ln -s /usr/bin/python3.10 /usr/local/bin/python

    # Installing pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
    export PATH=$PATH:/home/paperspace/.local/bin


# ==================================================================
# Installing CUDA packages (CUDA Toolkit 12.2.2 & CUDNN 8.9.4)
# ------------------------------------------------------------------

    # Based on https://developer.nvidia.com/cuda-toolkit-archive
    # Based on https://developer.nvidia.com/rdp/cudnn-archive

    wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
    sudo sh cuda_12.2.2_535.104.05_linux.run --silent --toolkit
    export PATH=$PATH:/usr/local/cuda-12.2/bin
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64
    rm cuda_12.2.2_535.104.05_linux.run


    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
    sudo $APT_INSTALL libcudnn8=8.9.4.*-1+cuda12.2
    sudo $APT_INSTALL libcudnn8-dev=8.9.4.*-1+cuda12.2


# ==================================================================
# PyTorch
# ------------------------------------------------------------------

    # Based on https://pytorch.org/get-started/locally/

    $PIP_INSTALL torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
        

# ==================================================================
# JAX
# ------------------------------------------------------------------

    # Based on https://github.com/google/jax#pip-installation-gpu-cuda

    $PIP_INSTALL "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html



# ==================================================================
# TensorFlow
# ------------------------------------------------------------------

    # Based on https://www.tensorflow.org/install/pip

    # export LD_LIBRARY_PATH=${HOME}/anaconda3/lib
    $PIP_INSTALL tensorflow==2.14.0


# ==================================================================
# Hugging Face
# ------------------------------------------------------------------
    
    # Based on https://huggingface.co/docs/transformers/installation
    # Based on https://huggingface.co/docs/datasets/installation

    $PIP_INSTALL transformers==4.34.1 datasets==2.14.5


# ==================================================================
# JupyterLab
# ------------------------------------------------------------------

    # Based on https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html#pip

    $PIP_INSTALL jupyterlab==4.0.7


# ==================================================================
# Additional Python Packages
# ------------------------------------------------------------------

    $PIP_INSTALL \
        numpy==1.26.1 \
        scipy==1.11.3 \
        pandas==2.1.1 \
        cloudpickle==3.0.0 \
        scikit-image==0.22.0 \
        scikit-learn==1.3.1 \
        matplotlib==3.8.0 \
        ipython==8.16.1 \
        ipykernel==6.25.2 \
        ipywidgets==8.1.1 \
        cython==3.0.4 \
        tqdm==4.66.1 \
        gdown==4.7.1 \
        xgboost==2.0.0 \
        pillow==10.1.0 \
        seaborn==0.13.0 \
        sqlalchemy==1.4.40 \
        spacy==3.7.2 \
        nltk==3.8.1 \
        boto3==1.28.66 \
        tabulate==0.9.0 \
        future==0.18.3 \
        gradient==2.0.6 \
        jsonify==0.5 \
        opencv-python==4.8.1.78 \
        PyYAML==5.4.1 \
        sentence-transformers==2.2.2 \
        wandb==0.15.12


# ==================================================================
# Installing JRE and JDK
# ------------------------------------------------------------------

    sudo $APT_INSTALL default-jre
    sudo $APT_INSTALL default-jdk


# ==================================================================
# CMake
# ------------------------------------------------------------------

    sudo $GIT_CLONE https://github.com/Kitware/CMake ~/cmake
    cd ~/cmake
    sudo ./bootstrap
    sudo make -j"$(nproc)" install


# ==================================================================
# Node.js and Jupyter Notebook Extensions
# ------------------------------------------------------------------

    sudo $APT_INSTALL nodejs
    $PIP_INSTALL jupyter_contrib_nbextensions jupyterlab-git
    DEBIAN_FRONTEND=noninteractive jupyter contrib nbextension install --user


# ==================================================================
# Config & Cleanup
# ------------------------------------------------------------------

    echo "export PATH=${PATH}" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ~/.bashrc

