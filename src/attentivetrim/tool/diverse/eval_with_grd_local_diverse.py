import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


import dsp
import dspy

from src.attentivetrim.tool.dspy_interface import dspyCOT, VeriCorrectness
from rouge_score import rouge_scorer

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


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

def evaluate_rouge_results(results_file, groundtruth_file, acc_file):


    with open(results_file) as fr:
        results = json.load(fr)

    with open(groundtruth_file) as fgrd:
        grd_json = json.load(fgrd)
        groundtruth_data = grd_json["files"]

    acc_res = {"question": grd_json["question"], "files": []}
    question = grd_json["question"]
    # print("Question: ", question)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    verification_cot = VeriCorrectness(ValidationCorrectnessSignature)
    cnt = 0
    for result in results["files"]:
        file = result["file"]
        test = result["result"]
        ground_truth = next((item["groundtruth"] for item in groundtruth_data if item["file"].endswith(file)), None)
        if ground_truth:
            # Calculate ROUGE scores
            rouge_scores = scorer.score(ground_truth, test)

            # Calculate cosine similarity
            pair = [ground_truth, test]
            pair_embeddings = model.encode(pair)
            cos_sim = cosine_similarity(
                [pair_embeddings[0]],
                [pair_embeddings[1]]
            )

            # Log results
            # print("file:", file,
            #       "groundtruth:", ground_truth,
            #       "result:", test,
            #       "match:", min(float(cos_sim[0][0]),1.0),
            #       "ROUGE-1:", rouge_scores['rouge1'],
            #       "ROUGE-2:", rouge_scores['rouge2'],
            #       "ROUGE-L:", rouge_scores['rougeL'])

            # Append results to acc_res
            acc_res["files"].append({
                "file": file,
                "groundtruth": ground_truth,
                "result": test,
                "match": min(float(cos_sim[0][0]),1.0),
                "ROUGE-1": rouge_scores['rouge1'].fmeasure,
                "ROUGE-2": rouge_scores['rouge2'].fmeasure,
                "ROUGE-L": rouge_scores['rougeL'].fmeasure,
                "rationale": "cosine similarity and ROUGE scores"
            })
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
    tests = [
        ["../../data/diverse/results-What is the pap-0.001.json",
         "../../data/diverse/results-What is the pap-0.005.json",
         "../../data/diverse/results-What is the pap-0.05.json",
         "../../data/diverse/results-What is the pap-0.3.json",],
        ["../../data/diverse/results-Who are the aut-0.005.json",
         "../../data/diverse/results-Who are the aut-0.01.json",
         "../../data/diverse/results-Who are the aut-0.05.json",
         "../../data/diverse/results-Who are the aut-0.1.json"],
        ["../../data/diverse/results-What is the mai-0.05.json",
         "../../data/diverse/results-What is the mai-0.1.json",
         "../../data/diverse/results-What is the mai-0.15.json",
         "../../data/diverse/results-What is the mai-0.2.json",
         "../../data/diverse/results-What is the mai-0.4.json",
         "../../data/diverse/results-What is the mai-0.9.json"],
        ["../../data/diverse/results-What are the ba-0.05.json",
        "../../data/diverse/results-What are the ba-0.1.json",
        "../../data/diverse/results-What are the ba-0.15.json",
        "../../data/diverse/results-What are the ba-0.2.json",
        "../../data/diverse/results-What are the ba-0.4.json",
        "../../data/diverse/results-What are the ba-0.9.json"]
    ]

    grds = ["../../data/test_diverse_inputfile100-result-What is the pap-0.3.json",
           "../../data/test_diverse_inputfile100-result-Who are the aut-0.3.json",
           "../../data/test_diverse_inputfile100-result-What is the mai.json",
           "../../data/test_diverse_inputfile100-result-What are the ba.json"]

    idx = 0
    for results_file in tests[idx]:
        groundtruth_file = grds[idx]
        acc_file = results_file.replace(".json", "-acc-local-full.json")
        evaluate_rouge_results(results_file, groundtruth_file, acc_file)
