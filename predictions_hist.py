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
        percents = [p for p in similarity_dict[word] if p >= 20]
        plt.figure()
        plt.hist(percents, bins=50)
        plt.xlabel("Percent Similarity")
        plt.ylabel("Frequency")
        plt.title(f"{word} is a match for {len(percents)} files")
        plt.savefig(f"histograms/{word}_hist.png")
    
if __name__ == "__main__":
    main()