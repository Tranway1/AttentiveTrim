import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


import dsp
import dspy

from src.attentivetrim.tool.dspy_interface import dspyCOT, VeriCorrectness


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

    model = SentenceTransformer('all-MiniLM-L6-v2')

    verification_cot = VeriCorrectness(ValidationCorrectnessSignature)
    cnt = 0
    for result in results["files"]:
        file = result["file"]
        test = result["result"]
        ground_truth = next((item["groundtruth"] for item in groundtruth_data if item["file"] == file), None)
        if ground_truth:
            pair = [ground_truth, test]
            pair_embeddings = model.encode(pair)
            cos_sim = cosine_similarity(
                [pair_embeddings[0]],
                [pair_embeddings[1]]
            )
            print("file:", file, " groundtruth: ", ground_truth, " result:", test, " match:", float(cos_sim[0][0]))
            acc_res["files"].append({"file": file, "groundtruth": ground_truth, "result": test, "match": float(cos_sim[0][0]), "rationale": "cosine similarity"})
    acc_res["total_matches"] = 0
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
    results_file = '../data/results-What is the pap-0.001.json'
    groundtruth_file = '../data/test_v16_inputfile100-result-What is the pap-0.3.json'
    acc_file = results_file.replace(".json", "-acc-local-0.1.json")
    evaluate_results(results_file, groundtruth_file, acc_file)