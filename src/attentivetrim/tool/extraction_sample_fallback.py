import os
import json
import time

import dsp
import dspy
from src.attentivetrim.tool import histogram_range
from src.attentivetrim.tool.dspy_interface import dspyCOT



QUESTIONS = ["What is the paper title?",
             "What is the authors of the paper?",
             "What is the main contribution of the paper?"]

HISTS = ["../data/frequency-test-title.csv",
            "../data/frequency-test-authors.csv",
            "../data/frequency-test-contribution.csv"]


class SingleQuestionOverSampleFallback(dspy.Signature):
    """Answer question(s) about a scientific paper."""

    context = dspy.InputField(desc="contains a snippet of the paper, including the most possible answer to the question.")
    question = dspy.InputField(desc="one question about the paper")
    answer = dspy.OutputField(desc="print the answer close to the original text as you can, and print 'None' if an answer cannot be found or the answer is truncated at the beginning or the end of the given context. Please do not helucinate the answer")



def get_test_result(file_path, question, sr, er):
    # Load document
    with open('/Users/chunwei/pvldb_1-16/16/' + file_path) as f_in:
        doc_dict = json.load(f_in)

    context = doc_dict["symbols"]
    test_len = len(context)


    start = int(sr * test_len)
    end = int(er * test_len)

    print("character start:", start, "end:", end)
    sample = context[start:end]
    print("sample size:", len(sample))

    # Generate prediction
    cot = dspyCOT(SingleQuestionOverSampleFallback)
    pred = cot(question, sample)
    return pred.answer

def run_file_batch(list_of_files, question, hist_file, budget=0.05):
    sr, er = histogram_range.get_range_from_hist(hist_file, budget, resolution=0.001, trim_zeros=False)
    print("start ratio:", sr, "end ratio:", er)

    results = {"question": question, "files": []}
    for file in list_of_files:
        print(f"using budget: {budget}")
        cur_budget = budget
        print("file:", file)
        start_time = time.time()
        test_result = get_test_result(file, question, sr, er)
        print("test_result:", test_result)
        while test_result == "None":
            print(f"fallback to higher budget {cur_budget} -> {cur_budget*2}")
            if cur_budget <= 0.4:
                cur_budget = cur_budget*2
            elif cur_budget < 1.0:
                cur_budget = 1.0
            else:
                break
            cur_sr, cur_er = histogram_range.get_range_from_hist(hist_file, cur_budget, resolution=0.001, trim_zeros=False)
            print("fallback start ratio:", cur_sr, "end ratio:", cur_er)
            test_result = get_test_result(file, question, cur_sr, cur_er)
        duration = time.time() - start_time

        results["files"].append({"file": file, "result": test_result, "duration": duration, "budget": cur_budget})
        print("file:", file, " result:", test_result)

    return results

if __name__ == "__main__":
    dsp.modules.cache_utils.cache_turn_on = False
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-1106-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)

    list_file = "../data/test_v16_inputfile100.txt"
    with open(list_file) as f:
        list_of_files = f.readlines()
    list_of_files = [x.strip() for x in list_of_files]
    q_idx= 1
    question = QUESTIONS[q_idx]
    hist_file = HISTS[q_idx]
    budget = 0.005
    print("question:", question, "hist_file:", hist_file, "budget:", budget)
    results = run_file_batch(list_of_files, question, hist_file, budget=budget)
    json_string = json.dumps(results, indent=4)
    with open(f'../data/results-fallback-{question[:15]}-{budget}.json', 'w') as f:
        f.write(json_string)