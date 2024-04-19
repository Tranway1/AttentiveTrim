#!/bin/bash

# Define the list of ratios
ratios=(0.005 0.01 0.05 0.1 0.2 0.3)

# Loop from 0 to 2 inclusive for 'q'
for q in {0..2}; do
    # Loop through each value in the 'ratios' array
    for ratio in "${ratios[@]}"; do
        echo "Running experiment with Ratio: $ratio, Question: $q"
        # Run the Python script with the specified parameters, substituting in the current 'ratio' and 'q' values
        python extraction_sample_vllm.py --batch_size 1 --input_size 7500 --max_new_tokens 128 --model /home/gridsan/cliu/hf/Mistral-7B-Instruct-v0.2 --tensor_para_size 1 --iters 1 --text_ratio $ratio --question $q --print_output --test_perf
    done
done