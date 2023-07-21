import pickle
import sys

def main():
    """Main Program"""

    # Load the pickled data from similarity_dict.pkl
    with open("similarity_dict.pkl", "rb") as file: 
        similarity_dict = pickle.load(file)

    # Get the probability from the command-line argument
    probability = float(sys.argv[1])
    percent = float(sys.argv[1]) * 100

    print(f"\nNumber of files where probability >= {probability}\n")

    for word in similarity_dict.keys():
        percents = [p for p in similarity_dict[word] if p >= percent]
        print(f"\t{word}: {len(percents)}")

if __name__ == "__main__":
    main()