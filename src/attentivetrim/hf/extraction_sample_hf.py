import argparse
import json
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.attentivetrim.tool.histogram_range import get_range_from_hist

QUESTIONS = ["What is the paper title?",
             "What is the authors of the paper?",
             "What is the main contribution of the paper?"]

HISTS = ["../data/frequency-test-title.csv",
            "../data/frequency-test-authors.csv",
            "../data/frequency-test-contribution.csv"]

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--question', type=int, default=0, help='Index of the question to ask')
parser.add_argument('--ratio', type=float, default=0.1, help='Ratio of the text to use for the question')

args = parser.parse_args()

# Assuming QUESTIONS is defined somewhere in your code
# Make sure QUESTIONS is a list of questions and args.question is a valid index
question = QUESTIONS[args.question]
hist_file = HISTS[args.question]
base_dir = "/home/gridsan/cliu/"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Ensure this matches your desired device
device = "cuda"  # the device to load the model onto
hf_token = "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"
local_model_path = '/home/gridsan/cliu/hf/Mistral-7B-Instruct-v0.2'
# local_model_path = '/home/gridsan/cliu/hf/dbrx-instruct'
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.to(device)

# Assuming you have a list of file paths
list_file = "../data/test_v16_inputfile100.txt"
with open(list_file) as f:
    list_of_files = f.readlines()
list_of_files = [x.strip() for x in list_of_files]
ratio = args.ratio

sr, er = get_range_from_hist(hist_file, ratio, resolution=0.001, trim_zeros=False)
results = {"question": question, "ratio": ratio, "hist_file": hist_file, "start_ratio": sr, "end_ratio": er, "files": []}
print("question:", question, "ratio:", ratio, "hist_file:", hist_file, "start ratio:", sr, "end ratio:", er)

for file in list_of_files:
    with open(base_dir + 'pvldb_1-16/16/' + file) as f_in:
        doc_dict = json.load(f_in)

    context = doc_dict["symbols"]
    test_len = len(context)
    start = int(sr * test_len)
    end = int((er + 0.001) * test_len)

    print("character start:", start, "end:", end)
    sample = context[start:end]
    sample_length = len(sample)
    total_length = len(context)
    print(f"Total length: {total_length}, Sample length: {sample_length}, Question: {question}")

    prompt = f"Here is a paper snippet:'''{sample}'''\n\nNow please answer the question: {question} \n Please only answer the question I asked above and print the answer in a single line."

    start_time = time.time()

    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    input_length = len(encodeds[0])
    print(f"Encodeds shape: {encodeds.shape}")
    # Sending data to device...
    input_ids = encodeds.to(device)

    sending_data = time.time() - start_time
    # print(f"Data sent to device in: {sending_data:.4f} seconds")

    # print("Generating response...")
    try:
        # Generate the response using the model
        generated_ids = model.generate(input_ids, max_new_tokens=1000)

        print(f"Generated shape: {generated_ids.shape}")
        output_length = len(generated_ids[0])
        output_ids = generated_ids[:, input_length:output_length+1]

        generate_res = time.time() - start_time
        # print(f"Response generated in: {generate_res:.4f} seconds")

        # Decode the generated ids to get the textual response
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    except Exception as e:
        duration = time.time() - start_time
        results["files"].append({"file": file, "result": '', "duration": duration})
        print(f"File: {file}, Result: '', Duration: {duration}")
        continue

    duration = time.time() - start_time
    output_length = len(decoded[0])
    print(f"decoded length: {output_length}, input_length: {total_length},  duration: {duration:.4f} seconds")

    results["files"].append({"file": file, "result": decoded, "duration": duration})
    print(f"File: {file}, Result: {decoded}, Duration: {duration}")

json_string = json.dumps(results, indent=4)
with open(f'../data/hf/hf-results-{question[:15]}-{ratio}.json', 'w') as f:
    f.write(json_string)