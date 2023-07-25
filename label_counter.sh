#!/bin/bash

# Check if the correct number of arguments is provided 
if [ $# -ne 1 ]; then 
    echo "Usage: $0 <path_to_directory>"
    exit 1
fi

# Get the directory path from the first argument 
directory="$1"

# Check if the provided directory exists
if [ ! -d "$directory" ]; then 
    echo "Error: Directory '$directory' does not exist."
    exit 1
fi

# Run the python script with the directory as an argument to get label predictions
echo "Running label_counter.py..."
python label_counter.py "$directory"

# Run the python script to plot predictions
echo "Running predictions_bar_graph.py..."
python predictions_bar_graph.py

# Run the python script to plot the distribution of label probabilities
echo "Running predictions_hist.py..."
python predictions_hist.py

# Check if the python script successfully generated the bar graph file 
if [ ! -f "predictions_graph.png" ]; then 
    echo "Error: The python script did not generate the bar graph file."
    exit 1
fi 

echo "Plot files 'predictions_bar_graph.png' created and saved in the current directory."
echo "Done!"
