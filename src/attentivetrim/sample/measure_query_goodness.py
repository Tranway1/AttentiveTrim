import os
import json
import time
import random
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.attentivetrim.sample.get_grd_batch import SingleQuestionOverPaper
import dspy
from papermage import Document

from src.attentivetrim.tool.dspy_interface import dspyCOT, QuestionOverPaper
from rouge_score import rouge_scorer

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)



def get_groundtruth(file_path, question, model='gpt-4-1106-preview', temperature=0.0, enable_cache=True):
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.LM(f'openai/{model}', api_key=openai_key, temperature=temperature, cache=enable_cache)
    dspy.settings.configure(lm=turbo)

    # Load document
    with open(file_path) as f_in:
        doc_dict = json.load(f_in)
        doc = Document.from_json(doc_dict)


    context = doc_dict["symbols"]

    # Generate prediction
    cot = dspyCOT(SingleQuestionOverPaper)
    pred = cot(question, context)
    return pred.answer

def compare_groundtruth_with_test(ground_truth, test, file):
    rouge_scores = scorer.score(ground_truth, test)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Calculate cosine similarity
    pair = [ground_truth, test]
    pair_embeddings = model.encode(pair)
    cos_sim = cosine_similarity(
        [pair_embeddings[0]],
        [pair_embeddings[1]]
    )

    # Log results
    print("file:", file,
          "groundtruth:", ground_truth,
          "result:", test,
          "match:", min(float(cos_sim[0][0]), 1.0),
          "ROUGE-1:", rouge_scores['rouge1'],
          "ROUGE-2:", rouge_scores['rouge2'],
          "ROUGE-L:", rouge_scores['rougeL'])

    # Append results to acc_res
    return ({
        "file": file,
        "groundtruth": ground_truth,
        "result": test,
        "match": min(float(cos_sim[0][0]), 1.0),
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-2": rouge_scores['rouge2'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        "rationale": "cosine similarity and ROUGE scores"
    })

def measure_goodness(dataset, model='gpt-4-1106-preview', sample_ratio=0.3, init_seed=0, iteration=3):
    config_path = '../../../questions/question.json'
    with open(config_path) as f:
        data = json.load(f)
    files = data["datasets"][dataset]["list"]
    questions = data["query"][f'{dataset.upper()}_QUESTIONS']

    random.seed(init_seed)
    sample_size = int(len(files) * sample_ratio)
    chosen_files = random.sample(files, sample_size)
    print(f"Chosen files size: {len(chosen_files)}")

    for question in questions:
        print(f"Question: {question}")
        result = {}
        result["question"] = question
        result["files"] = []
        for file in chosen_files:
            # run the grd for three times
            for _ in range(iteration):
                try:
                    start_time = time.time()
                    groundtruth = get_groundtruth(file, question, model=model, temperature=1.0, enable_cache=False)
                    print(f"temperature=1.0, enable_cache=False")
                    duration = time.time() - start_time
                    print(f"Duration: {duration}")
                    result["files"].append(
                        {"file": file, "groundtruth": groundtruth, "rationale": None, "duration": duration})
                    print("file:", file, " groundtruth: ", groundtruth, " rationale: ", None)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        json_string = json.dumps(result, indent=4)
        grd_output_file = f"../../../grd/goodness_{dataset}_{question}.json"
        with open(grd_output_file, 'w') as f:
            f.write(json_string)
    print(f"{dataset} Done")

def goodness_score(dataset, iteration = 3):
    config_path = '../../../questions/question.json'
    with open(config_path) as f:
        data = json.load(f)
    questions = data["query"][f'{dataset.upper()}_QUESTIONS']

    for question in questions:
        goodness_file = f"../../../grd/goodness_{dataset}_{question}.json"
        with open(goodness_file) as f:
            data = json.load(f)
        total_files = len(data["files"])
        data["comparison"] = []
        cosine_score = 0.0
        rouge1_score = 0.0
        rouge2_score = 0.0
        rougeL_score = 0.0

        grds = []
        for file in data["files"]:
            grds.append(file)
            if len(grds) == iteration:
                # get all combination of grds
                for i in range(iteration):
                    for j in range(i+1, iteration):
                        assert grds[i]["file"] == grds[j]["file"]
                        ground_truth = grds[i]["groundtruth"]
                        test = grds[j]["groundtruth"]
                        result = compare_groundtruth_with_test(ground_truth, test, grds[i]["file"])
                        cosine_score += result["match"]
                        rouge1_score += result["ROUGE-1"]
                        rouge2_score += result["ROUGE-2"]
                        rougeL_score += result["ROUGE-L"]
                        data["comparison"].append(result)
                        # print(f"accumulated cosine_score: {cosine_score}, rouge1_score: {rouge1_score}, rouge2_score: {rouge2_score}, rougeL_score: {rougeL_score}")
                grds = []
        data["goodness_score"] = {
            "cosine_score": cosine_score / total_files,
            "rouge1_score": rouge1_score / total_files,
            "rouge2_score": rouge2_score / total_files,
            "rougeL_score": rougeL_score / total_files
        }

        json_string = json.dumps(data, indent=4)
        with open(goodness_file, 'w') as f:
            f.write(json_string)
        print(f"{dataset} {question} Done")
    print(f"{dataset} Done")

# Read the goodness score and output into a csv file for analysis
# The output will file name as each row, and the columns will be the goodness score
def format_output(dataset, iteration = 3):
    config_path = '../../../questions/question.json'
    with open(config_path) as f:
        data = json.load(f)
    questions = data["query"][f'{dataset.upper()}_QUESTIONS']
    csv_file_path = f"../../../questions/goodness_score_{dataset}.csv"
    with open(csv_file_path, 'w') as csv_file:
        csv_file.write("dataset,question,cosine_score,rouge1_score,rouge2_score,rougeL_score\n")

    for question in questions:
        goodness_file = f"../../../grd/goodness_{dataset}_{question}.json"
        with open(goodness_file) as f:
            data = json.load(f)
        cosine_score = data["goodness_score"]["cosine_score"]
        rouge1_score = data["goodness_score"]["rouge1_score"]
        rouge2_score = data["goodness_score"]["rouge2_score"]
        rougeL_score = data["goodness_score"]["rougeL_score"]
        with open(csv_file_path, 'a') as csv_file:
            csv_file.write(f"{dataset},{question},{cosine_score},{rouge1_score},{rouge2_score},{rougeL_score}\n")
    print(f"{dataset} Done")






if __name__ == "__main__":
    # measure_goodness("paper", model='gpt-4-1106-preview', sample_ratio=0.3, init_seed=0, iteration=3)
    # goodness_score("notice", iteration=3)
    format_output("paper", iteration=3)