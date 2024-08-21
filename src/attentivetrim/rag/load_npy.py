import numpy as np


def load_encoded_prompts(filename):
    # Load the encoded prompts from the .npy file
    encoded_prompts = np.load(filename, allow_pickle=True)

    # Handle the loaded data
    print("Loaded encoded prompts:")
    # print(encoded_prompts)

    return encoded_prompts


# Example usage
filename = '/Users/chunwei/research/llm-scheduling/profiling-Meta-Llama-3-8B-Instruct-length-sfr-embed.npy'
# filename = '/Users/chunwei/research/llm-scheduling/alpaca10k-uae-embed.npy'

encoded_prompts = load_encoded_prompts(filename)
print("shape of encoded prompts: ", encoded_prompts.shape)

# convert the encoded prompts to a data frame
