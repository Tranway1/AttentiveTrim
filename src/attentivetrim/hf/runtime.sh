#!/bin/bash

# Define the list of ratios
ratios=(0.005 0.01 0.05 0.1 0.2 0.3)

# Loop from 0 to 2 inclusive for 'q'
for q in {0..2}; do
    # Loop through each value in the 'ratios' array
    for ratio in "${ratios[@]}"; do
        echo "Running experiment with Ratio: $ratio, Question: $q"
        # Run the Python script with the specified parameters, substituting in the current 'ratio' and 'q' values
        python extraction_sample_hf.py --question $q --ratio $ratio
    done
done