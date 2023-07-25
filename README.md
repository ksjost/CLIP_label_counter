# CLIP Label Counter

These files and scripts will allow you to use OpenAI's CLIP on your own set of images and labels.

label_counter.sh takes a path to a directory of images as a command line argument. This directory is an argument for label_counter.py. label_counter.sh is responsible for running label_counter.py, predictions_bar_graph.py, and predictions_hist.py.

label_counter.py uses clip to encode the labels in word_labels.yaml and the images in the directory you provide to compute the similarity percentage between each text-image pair. It then uses the labels in word_labels.yaml as keys for a Counter. The top 5 similarity percentages for each image are updated in the counter (so each word in the counter is followed by the sum of its corresponding percentage when it scored in the top 5 similarity labels for each image). This counter is then dumped and saved to a pickle file called label_counter.pkl. This program also creates a dictionary containing each word in word_labels.yaml as keys and a list of the corresponding similarity percentages as values. For example, the list that corresponds to the key "boat" contains all the similarity percentages when the label "boat" is one of the top 5 matches for an image. This dictionary is dumped and saved to a pickle file called similarity_dict.pkl.

I have a directory of approximately 154,000 images, and I put 8 labels into word_labels.yaml, and this program (connected to CUDA) ran in about 1.5 hours. Pickling the data in the counter and the dictionary allows me to save/access/analyze them without having to rerun label_counter.py.

predictions_bar_graph.py loads the counter from label_counter.pkl so that it can be used to create and save a bar graph. The horizontal bar chart created in this script shows the summed percentages corresponding to each label in word_labels.yaml.

predictions_hist.py loads the dictionary from similarity_dict.pkl so that it can be used to create and save multiple histograms. One histogram is made for each word in word_labels.yaml. The histogram plots the distribution of similarity percentages that are greater than or equal to 20%. The plot title contains the count of files represented by the plot. This program makes a directory called "histograms" where each plot is saved.

probability_cutoff.py loads the dictionary from similarity_dict.pkl. It takes one argument **a float representing the probability at which to cut off or a integer/float representing the percent similarity at which to cutoff**. Then it prints each label in word_labels.yaml and the corresponding number of files that meet this **percent/probability** requirement.

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
