import os
import json
import time

import dsp
import dspy
import histogram_range
from dspy_interface import dspyCOT



QUESTIONS = ["What is the main contribution of the paper?",
             "What is the authors of the paper?",
             "What is the paper title?"]


class SingleQuestionOverSample(dspy.Signature):
    """Answer question(s) about a scientific paper."""

    context = dspy.InputField(desc="contains a snippet of the paper, including the most possible answer to the question.")
    question = dspy.InputField(desc="one question about the paper")
    answer = dspy.OutputField(desc="print the answer close to the original text as you can, and print 'None' if an answer cannot be found. Please do not helucinate the answer")



def get_test_result(file_path, question, sr, er):
    # Load document
    with open('/Users/chunwei/pvldb_1-16/16/' + file_path) as f_in:
        doc_dict = json.load(f_in)

    context = doc_dict["symbols"]
    test_len = len(context)


    start = int(sr * test_len)
    end = int(er * test_len)

    print("character start:", start, "end:", end)
    sample = context[start:end+1]
    if sr == er:
        end = int((er+0.001) * test_len)-1
        sample = context[start:end+1]
    print("sample size:", len(sample))

    # Generate prediction
    cot = dspyCOT(SingleQuestionOverSample)
    pred = cot(question, sample)
    return pred.answer

def run_file_batch(list_of_files, question, hist_file, budget=0.05):
    sr, er = histogram_range.get_range_from_hist(hist_file, budget, resolution=0.001, trim_zeros=False)
    print("start ratio:", sr, "end ratio:", er)

    results = {"question": question, "files": []}
    for file in list_of_files:
        start_time = time.time()
        test_result = get_test_result(file, question, sr, er)
        duration = time.time() - start_time

        results["files"].append({"file": file, "result": test_result, "duration": duration})
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
    question = QUESTIONS[0]
    hist_file = "../data/frequency-test-contribution.csv"
    budget = 0.2
    print("question:", question, "hist_file:", hist_file, "budget:", budget)
    results = run_file_batch(list_of_files, question, hist_file, budget=budget)
    json_string = json.dumps(results, indent=4)
    with open(f'../data/results-{question[:15]}-{budget}.json', 'w') as f:
        f.write(json_string)