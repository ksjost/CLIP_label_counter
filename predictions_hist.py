import matplotlib.pyplot as plt
import pickle
import os
import yaml

def main():
    """Main Program"""

    # Load the YAML data from the file 
    with open("word_labels.yaml", "r") as file: 
        yaml_doc = yaml.safe_load(file)

    probability = yaml_doc["probability cutoff"][0]

    # Load the pickled data from similarity_dict.pkl
    with open("similarity_dict.pkl", "rb") as file:
        similarity_dict = pickle.load(file)

    os.makedirs("histograms", exist_ok=True)
    
    for word in similarity_dict.keys():
        probs = [p for p in similarity_dict[word] if p >= probability]
        plt.figure()
        plt.hist(similarity_dict[word], bins=50)
        plt.axvline(probability, color="r")
        plt.xlabel("Similarity Probability")
        plt.ylabel("Frequency")
        plt.title(f"probability({word} is similar) >= {probability} for {len(probs)} files")
        plt.savefig(f"histograms/{word}_hist.png")
    
if __name__ == "__main__":
    main()