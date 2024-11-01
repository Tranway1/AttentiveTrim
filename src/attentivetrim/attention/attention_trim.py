import json
import time
import numpy as np
from transformers import AutoTokenizer
import dsp
from src.attentivetrim import dspyCOT
from src.attentivetrim.hf.attention_llama_hf_reverse_eng import clean_token
from src.attentivetrim.tool.extraction_sample_fallback import SingleQuestionOverSampleFallback
import os
import dspy
from src.attentivetrim.tool.diverse.eval_with_grd_local_diverse import evaluate_rouge_results
from src.attentivetrim.tool.diverse.eval_with_grd_diverse import evaluate_results


from pathlib import Path
HOME_DIR = Path.home()
model_name = f"{HOME_DIR}/hf/Llama-3.2-1B-Instruct"


def calculate_average_attention_by_sentence(tokens, attention_scores, offset_pairs):
    # Initialize a list to store the average scores for each offset pair
    average_scores = []

    # Calculate the starting character index of each token
    token_char_indices = []
    current_index = 0
    for token in tokens:
        token_char_indices.append((current_index, current_index + len(token) - 1))
        current_index += len(token) + 1  # Assuming one space between tokens

    # Iterate over each offset pair
    for idx, pair in enumerate(offset_pairs):
        start, end = pair[0], pair[1]
        sum_attention = 0.0
        count = 0

        # Check each token to see if it falls within the current offset pair range
        for idx, (token_start, token_end) in enumerate(token_char_indices):
            # Check if the token overlaps with the offset range
            if token_start <= end and token_end >= start:
                sum_attention += attention_scores[idx]
                count += 1

        # Compute the average and add it to the list
        if count > 0:
            average_score = sum_attention / count
        else:
            average_score = 0  # To handle any unexpected case of zero count

        average_scores.append(average_score)

    return average_scores

def fetch_best_sentences(context, average_scores, offset_pairs, top_k=3):
    # Sort the average scores in descending order and keep track of the original indices
    sorted_scores = sorted(enumerate(average_scores), key=lambda x: x[1], reverse=True)
    top_sentences = []
    top_indices = []
    for idx, score in sorted_scores[:top_k]:
        top_indices.append(idx)
    # Sort the top indices to maintain the original order
    top_indices.sort()
    for idx in top_indices:
        start, end = offset_pairs[idx][0], offset_pairs[idx][1]
        top_sentences.append(context[start:end+1])
    return top_sentences

def get_sample_result(sample, question):
    cot = dspyCOT(SingleQuestionOverSampleFallback)
    pred = cot(question, sample)
    return pred.answer, pred.rationale


def get_test_result(file_path, question, attention_scores, tokens, topk=3):
    # Load document
    with open(file_path) as f_in:
        doc_dict = json.load(f_in)

    context = doc_dict["symbols"]
    sentences = doc_dict["entities"]["sentences"]
    offset_pairs = []
    for sentence in sentences:
        span = sentence["spans"][0]
        offset_pairs.append(span)


    average_scores = calculate_average_attention_by_sentence(tokens, attention_scores, offset_pairs)

    # Fetch the top-k sentences based on the average attention scores
    top_sentences = fetch_best_sentences(context, average_scores, offset_pairs, top_k=topk)

    # put top_sentences into string connected by ...
    top_sentences = "...".join(top_sentences)
    char_used = len(top_sentences)-3*(topk-1)

    t_start = time.time()
    res, rationale = get_sample_result(top_sentences, question)
    t_end = time.time()

    duration = t_end - t_start

    ratio = char_used/len(context)

    return res, rationale, ratio, duration

def run_sample_pred_batch(list_of_files, base_list, dataset, question, query_idx, topk=3):
    token_dir = "../data/tokens"
    attention_summary_dir = "../data/attention_summary"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    results = {"question": question, "files": []}
    sum_ratio = 0.0
    for file in list_of_files:
        # find the index if the file from base_list
        file_idx = base_list.index(file)
        print(f"file_idx: {file_idx}")
        # get the tokens and attention_scores
        token_file = f"{token_dir}/{dataset}_{file_idx}.json"
        attention_summary_file = f"{attention_summary_dir}/{dataset}_{file_idx}_{query_idx}.npy"
        print(f"token_file: {token_file}, attention_summary_file: {attention_summary_file}")

        with open(token_file) as f:
            tokens = json.load(f)
        attention_scores = np.load(attention_summary_file)
        token_strs = clean_token(tokens, tokenizer)

        res, rationale, ratio, duration = get_test_result(file, question, attention_scores[3:], token_strs[3:], topk=topk)
        sum_ratio += ratio
        results["files"].append({"file": file, "result": res, "rationale": rationale, "ratio": ratio, "duration": duration})
        print(f"file: {file}, result: {res}, ratio: {ratio}, duration: {duration}")
    return results, sum_ratio/len(list_of_files)

def run_prediction_process(dataset='paper'):
    # Ensure the OpenAI API key is set
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-1106-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)

    # Load configuration
    config_path = '../../../questions/question.json'
    with open(config_path) as f:
        data = json.load(f)

    base_list = data["datasets"][dataset]["list"]
    questions = data["query"][f'{dataset.upper()}_QUESTIONS']
    loc_dir = "../../../grd_loc"
    heatmap_dir = "../../../heatmap"
    pred_dir = "../../../pred_att"
    budgets = [20, 30]
    dataset_pred_dir = os.path.join(pred_dir, dataset)

    if not os.path.exists(dataset_pred_dir):
        os.makedirs(dataset_pred_dir)

    for q_idx, question in enumerate(questions):
        task_file = f"{dataset}_{question}-location.json"

        # Read the whole file list
        with open(f"{loc_dir}/{task_file}", 'r') as f:
            json_obj = json.loads(f.read())
        whole_file_list = [entry["file"] for entry in json_obj["files"]]

        # Read the heatmap sample file list
        with open(f"{heatmap_dir}/heatmap-{task_file}", 'r') as f:
            heatmap_json_obj = json.loads(f.read())
        sample_file_list = [entry["file"] for entry in heatmap_json_obj["chosen_files"]]

        # Get the difference and keep the order
        diff_list = [file for file in whole_file_list if file not in sample_file_list]
        print("whole_file_list:", len(whole_file_list), "\nsample_file_list:", len(sample_file_list), "\ndiff_list:",
              len(diff_list))
        list_of_files = diff_list


        for topk in budgets:
            print("question:", question,  "topk:", topk)
            results, avg_ratio = run_sample_pred_batch(list_of_files, base_list, dataset, question, q_idx, topk=topk)
            json_string = json.dumps(results, indent=4)
            pred_file = os.path.join(dataset_pred_dir, f"results-{avg_ratio:.2f}-{task_file}")
            with open(pred_file, 'w') as f:
                f.write(json_string)

def eval_acc(dataset = 'paper'):
    dsp.modules.cache_utils.cache_turn_on = False
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-1106-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)

    budget_map = {
        "What are the state abbreviation and ZIP code of the company?": [0.01, 0.02, 0.05, 0.10, 0.23, 0.35],
        "What are the violation items?": [0.01, 0.04, 0.09, 0.21, 0.34],
        "What is the civil penalty amount suggested for the probable violation?": [0.01, 0.04, 0.10, 0.21, 0.34],
        "What is the date of the notice?": [0.01, 0.02, 0.05, 0.10, 0.22, 0.35],
        "What is the full title of the sender in the notice?": [0.01, 0.02, 0.05, 0.10, 0.22, 0.35],
        "What is the name of the company?": [0.01, 0.02, 0.05, 0.10, 0.22, 0.34],
        "What is the type of violation item?": [0.01, 0.04, 0.09, 0.21, 0.34],
        "What statutory authority is used to propose the Compliance Order?": [0.01, 0.02, 0.05, 0.11, 0.22, 0.35]
    }

    # Run the model with given budget
    config_path = '../../../questions/question.json'

    with open(config_path) as f:
        data = json.load(f)

    file_list = data["datasets"][dataset]["list"]
    questions = data["query"][f'{dataset.upper()}_QUESTIONS']
    loc_dir = "../../../grd_loc"
    heatmap_dir = "../../../heatmap"
    pred_dir = "../../../pred_att"
    acc_gpt_dir = "../../../acc/gpt"
    acc_rouge_dir = "../../../acc/rouge"


    dataset_pred_dir = os.path.join(pred_dir, dataset)
    if not os.path.exists(dataset_pred_dir):
        os.makedirs(dataset_pred_dir)

    for question in questions:
        task_file = f"{dataset}_{question}-location.json"
        budgets = budget_map[question]
        for budget in budgets:
            print("question:", question, "budget:", budget)
            results_file = os.path.join(dataset_pred_dir, f"results-{budget:.2f}-{task_file}")
            groundtruth_file = os.path.join(loc_dir, task_file)
            acc_gpt_file = os.path.join(acc_gpt_dir, f"accAtt-{budget:.2f}-{task_file}".replace(".json", "-acc.json"))
            acc_rouge_file = os.path.join(acc_rouge_dir, f"accAtt-{budget:.2f}-{task_file}".replace(".json", "-acc.json"))
            evaluate_results(results_file, groundtruth_file, acc_gpt_file)
            evaluate_rouge_results(results_file, groundtruth_file, acc_rouge_file)


if __name__ == "__main__":

    # run_prediction_process("notice")
    eval_acc("notice")