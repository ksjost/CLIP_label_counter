# CLIP Label Counter

These files and scripts will allow you to use OpenAI's CLIP on your own set of images and labels.

label_counter.sh takes a path to a directory of images as a command line argument. This directory is an argument for label_counter.py. 

label_counter.py uses clip to encode the labels in word_labels.yaml and the images in the directory you provide to compute the similarity percentage between each text-image pair. It then uses the labels in word_labels.yaml as keys for a Counter. The top 5 similarity percentages for each image are updated in the counter (so each word in the counter is followed by the sum of its corresponding percentage when it scored in the top 5 similarity labels for each image). This counter is then dumped and saved to a pickle file called label_counter.pkl

I have a directory of approximately 154,000 images, and I put 8 labels into word_labels.yaml, and this program (connected to CUDA) ran in about an hour.

Pickling the data allows the data in the counter to be saved/accessed/analyzed without having to rerun label_counter.py (which takes about an hour to run).

predictions_graph.py loads the counter from label_counter.pkl so that it can be used to create and save a bar graph. The horizontal bar chart created in this script shows the summed percentages corresponding to each label in word_labels.yaml.

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
    vi word_labels.yaml

### Run program
    chmod +x label_counter.sh
    ./label_counter.sh <path to images>

#### Troubleshooting: 
Eventually I began running into a RuntimeError that was saying "CUDA out of memory." I was told that this could be because there were times that I was killing the program before it finished running. To correct this issue: 

    nvidia-smi
    # Find PID
    kill -9 <PID>
