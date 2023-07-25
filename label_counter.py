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
    images_list.sort()

    # Load the YAML data from the file 
    with open("word_labels.yaml", "r") as file: 
        yaml_doc = yaml.safe_load(file)
    
    # Access the list of labels
    labels_list = yaml_doc["labels"]
    pickle_vectors = yaml_doc["pickle vectors"]

    if pickle_vectors: 
        os.makedirs("pickled_vectors", exist_ok=True)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    label_counter = Counter()

    # Initialize dictionary for similarity probabilities
    similarity_dict = {}

    for word in labels_list:
        similarity_dict[word] = []

    # Encode text
    text_input = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels_list]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    
    # Initialize list for encoded image features
    vector_list = []
    a = 0

    for i, image in enumerate(images_list):
        # Encode image
        image_input = preprocess(Image.open(images_dir+image)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        # Pickle image_features if pickle_vectors == True
        if pickle_vectors: 
            vector_list.append(image_features)

            if (i + 1) % 100000 == 0:
                # File name: "startIndex_endIndex.pkl"
                with open(f"pickled_vectors/{a}_{i}.pkl", "wb") as file:
                    pickle.dump(vector_list, file)
                vector_list = []
                a = i + 1

        # Compute similarity 
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        values = values.tolist()
        indices = indices.tolist()

        listed_zip = list(zip(values, indices))

        # Update the counter and dictionary
        for value, index in listed_zip: 
            label_counter.update({labels_list[index]:round(value, 4)})
            similarity_dict[labels_list[index]].append(round(value, 4))
    
    # Pickle remaining image_features (if applicable)
    if len(vector_list) != 0:
        if pickle_vectors: 
            with open(f"pickled_vectors/{a}_{i}.pkl", "wb") as file:
                pickle.dump(vector_list, file)

    # Save the label_counter dictionary to a pickle file
    with open("label_counter.pkl", "wb") as file: 
        pickle.dump(label_counter, file)
        print("Label Counter pickled and saved")

    # Save the similarity_dict dictionary to a pickle file
    with open("similarity_dict.pkl", "wb") as file:
        pickle.dump(similarity_dict, file)
        print("Similarity scores successfully pickled and saved") 

if __name__ == "__main__":
    main()
    
