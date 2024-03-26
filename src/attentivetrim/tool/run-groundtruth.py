
import os
import json
import time

import dsp
import dspy
from papermage import Document

from src.attentivetrim.tool.dspy_interface import dspyCOT, QuestionOverPaper

class SingleQuestionOverPaper(dspy.Signature):
    """Answer question(s) about a scientific paper."""

    context = dspy.InputField(desc="contains full text of the paper, including author, institution, title, and body")
    question = dspy.InputField(desc="one question about the paper")
    answer = dspy.OutputField(desc="print the answer only and print None if you cannot find a answer. Please do not helucinate the answer")


def get_groundtruth (file_path, question):

    # Turn off caching
    dsp.modules.cache_utils.cache_turn_on = False
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-1106-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)

    # Load document
    with open('/Users/chunwei/pvldb_1-16/17/'+file_path) as f_in:
        doc_dict = json.load(f_in)
        doc = Document.from_json(doc_dict)


    context = doc_dict["symbols"]

    # Generate prediction
    cot = dspyCOT(QuestionOverPaper)
    pred = cot(question, context)
    return pred.answer

def run_file_batch (list_of_files, question):
    # Define question and context

    result = {}
    result["question"] = question
    result["files"] = []
    for file in list_of_files:

        start_time = time.time()
        groundtruth = get_groundtruth(file, question)
        duration = time.time() - start_time
        print(f"Duration: {duration}")
        result["files"].append({"file": file, "groundtruth": groundtruth, "duration": duration})
        print("file:", file, " groundtruth: ", groundtruth)
    return result


if __name__ == "__main__":
    list_file = "../data/inputfile.txt"
    with open(list_file) as f:
        list_of_files = f.readlines()
    list_of_files = [x.strip() for x in list_of_files]
    question = "What is the main contribution of the paper?"
    json_obj = run_file_batch(list_of_files, question)
    json_string = json.dumps(json_obj, indent=4)
    with open(f'../data/result-{question[:10]}.json', 'w') as f:
        f.write(json_string)