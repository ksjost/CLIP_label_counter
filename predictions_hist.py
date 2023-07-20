import matplotlib.pyplot as plt
import pickle
import os

def main():
    """Main Program"""

    # Load the pickled data fron similarity_dict.pkl
    with open("similarity_dict.pkl", "rb") as file:
        similarity_dict = pickle.load(file)

    os.makedirs("histograms", exist_ok=True)
    
    for word in similarity_dict.keys():
        plt.figure()
        plt.hist(similarity_dict[word], bins=50)
        plt.xlabel("Rounded Probability Distribution")
        plt.ylabel("Number of Occurences")
        plt.title(f"Distribution for label: {word}")
        plt.savefig(f"histograms/{word}_hist.png")
    
if __name__ == "__main__":
    main()