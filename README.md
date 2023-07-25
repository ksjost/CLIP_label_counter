# CLIP Label Counter

These files and scripts will allow you to use OpenAI's CLIP on your own set of images and labels.

## Instructions

### Create New Conda Environment
    conda create -n try_clip
    conda activate try_clip

### Install OpenAI CLIP Packages

This code is adapted from OpenAI's CLIP repository. Visit https://github.com/openai/CLIP to clone the repository or follow the installation instructions under "Usage" before continuing. These instructions are also included here:

    cd 
    conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
    # The previous line installs the repository as a python package as opposed to cloning it

#### Notice:
The CUDA Version on my machine is 12.0, but the GPU only connected when I set `cudatoolkit=11.0` while installing pytorch.

### More installation requirements
    pip install matplotlib
    pip install pyyaml

### Clone CLIP_label_counter repository
    git clone https://github.com/ksjost/CLIP_label_counter

### Enter desired labels into word_labels.yaml
    cd CLIP_label_counter
    vi word_labels.yaml

### Run program
    chmod +x label_counter.sh
    ./label_counter.sh <path to images>
    python probability_cutoff.py <probability>

#### Troubleshooting: 
Eventually I began running into a RuntimeError that was saying "CUDA out of memory." I was told that this could be because there were times that I was killing the program before it finished running. To correct this issue: 

    nvidia-smi
    # Find PID
    kill -9 <PID>
