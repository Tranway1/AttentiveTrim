import os
import json
import time
from textwrap import indent

import dsp
import dspy

from dspy_interface import dspyCOT, VeriCorrectness


class ValidationWithTestAndGroundTruth(dspy.Signature):
    """Compare the test result with the ground truth"""

    context = dspy.InputField(desc="a statement of a scientific paper as ground truth")
    question = dspy.InputField(desc="another statement of a scientific paper as a test")
    answer = dspy.OutputField(desc="Please print the result of the comparison. If the test is the sementically similar to the ground truth, print 'True'. Otherwise, print 'False'.")

class ValidationCorrectnessSignature(dspy.Signature):
    """Verify if the predicted answer semantically matches the gold answer. And return boolean value."""

    question = dspy.InputField(desc="a question of a scientific paper")
    gold_answer = dspy.InputField(desc="a statement of a scientific paper as ground truth")
    predicted_answer = dspy.InputField(desc="another statement of a scientific paper as a test")
    is_correct = dspy.OutputField(desc="Please print the result of the comparison in 'True' or 'False'. If the test is the sementically similar to the ground truth, print 'True'. Otherwise, print 'False'.")

def evaluate_results(results_file, groundtruth_file, acc_file):


    with open(results_file) as fr:
        results = json.load(fr)

    with open(groundtruth_file) as fgrd:
        grd_json = json.load(fgrd)
        groundtruth_data = grd_json["files"]

    acc_res = {"question": grd_json["question"], "files": []}
    question = grd_json["question"]
    print("Question: ", question)

    verification_cot = VeriCorrectness(ValidationCorrectnessSignature)
    cnt = 0
    for result in results["files"]:
        file = result["file"]
        test = result["result"]
        ground_truth = next((item["groundtruth"] for item in groundtruth_data if item["file"] == file), None)

        if ground_truth:
            pred = verification_cot(question, ground_truth, test)
            # print("prompt: ", pred.rationale)
            match = pred.is_correct
            if match == "True":
                cnt += 1
            print("file:", file, " groundtruth: ", ground_truth, " result:", test, " match:", match)
            acc_res["files"].append({"file": file, "groundtruth": ground_truth, "result": test, "match": match, "rationale": pred.rationale})
    acc_res["total_matches"] = cnt
    acc_res["total_files"] = len(results["files"])
    with open(acc_file, "w") as f:
        f.write(json.dumps(acc_res, indent=4))
    print(f"Total matches: {cnt} out of {len(results['files'])}")

if __name__ == "__main__":
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-1106-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)
    results_file = '../data/gpt4/results-What is the mai-0.4.json'
    groundtruth_file = '../data/test_v16_inputfile100-result-What is the mai.json'
    acc_file = results_file.replace(".json", "-acc-full.json")
    evaluate_results(results_file, groundtruth_file, acc_file)