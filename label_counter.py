import torch
import clip
import os
from PIL import Image
from collections import Counter
import yaml
import sys
import pickle

def main():
    """Main Program"""

    # Get the directory path from the command-line argument
    images_dir = sys.argv[1]
    images_list = os.listdir(images_dir)

    # Load the YAML data from the file 
    with open("word_labels.yaml", "r") as file: 
        words = yaml.safe_load(file)
    
    # Access the list of labels
    labels_list = words["labels"]

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    label_counter = Counter()

    for image in images_list:
        image_input = preprocess(Image.open(images_dir+image)).unsqueeze(0).to(device)
        text_input = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels_list]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        values = values.tolist()
        indices = indices.tolist()

        listed_zip = list(zip(values, indices))

        for value, index in listed_zip: 
            label_counter.update({labels_list[index]:round(100 * value, 2)})

    # Save the label_counter dictionary to a pickle file
    with open("label_counter.pkl", "wb") as file: 
        pickle.dump(label_counter, file)
        print("Data successfully pickled and saved")

if __name__ == "__main__":
    main()