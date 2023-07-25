import matplotlib.pyplot as plt
import pickle

def main():
    """Main Program"""

    # Load the pickled data fron label_counter.pkl
    with open("label_counter.pkl", "rb") as file:
        label_counter = pickle.load(file)
    
    labels = list(label_counter.keys())
    scores = list(label_counter.values())

    plt.barh(labels, scores)
    plt.xlabel("Rounded Probability Sum")
    plt.ylabel("Predicted Label")
    plt.title("Bar Graph of Summed Predicted Label Probabilities")
    plt.savefig("predictions_graph.png")

if __name__ == "__main__":
    main()
    
