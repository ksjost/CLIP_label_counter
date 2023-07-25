import numpy as np
import os
import torch
import pickle
import faiss.contrib.torch_utils
import sys
import yaml
import clip
import shutil

def main():
    """Main program"""

    # Get info from .yaml
    with open("faiss/keyword_search.yaml", "r") as file: 
        yaml_doc = yaml.safe_load(file)
    
    keywords = yaml_doc["words"]
    image_path = yaml_doc["path to images"]
    index_path = yaml_doc["path to index"]
    k = yaml_doc["k nearest neighbors"]

    # Get images
    image_files = os.listdir(image_path)
    image_files.sort()

    # Get index
    index = faiss.read_index(index_path)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model 
    model, preprocess = clip.load("ViT-B/32", device)

    os.makedirs("knn", exist_ok=True)

    # For each word, get knn
    for word in keywords: 
        # Tokenize word
        text_input = clip.tokenize(f"a photo of a {word}").to(device)
        
        with torch.no_grad():
            text_feature = model.encode_text(text_input)

        D, I = index.search(text_feature, k)
        D.to(device)

        knn_list = I.tolist()

        print(f"The {k} Nearest Neighbors for '{word}' are:")
        
        os.makedirs(f"knn/{word}", exist_ok=True)
        for i in knn_list[0]:
            print(image_files[i])
            shutil.copy(image_path + image_files[i], f"knn/{word}/{image_files[i]}")

if __name__ == "__main__":
    main()