
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
    answer = dspy.OutputField(desc="print the answer close to the original text as you can, and print 'None' if an answer cannot be found. Please do not helucinate the answer")


class ValidationWithTestAndGroundtruth(dspy.Signature):
    """Compare the test result with the groundtruth"""

    groundturth = dspy.InputField(desc="statement of a scientific paper as groundtruth")
    test = dspy.InputField(desc="another statement of a scientific paper as a test")
    answer = dspy.OutputField(desc="Please print the result of the comparison. If the test is the sementically similar to the groundtruth, print 'True'. Otherwise, print 'False'.")



def get_groundtruth (file_path, question, model='gpt-4-1106-preview'):


    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model=model, api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)

    # Load document
    with open(file_path) as f_in:
        doc_dict = json.load(f_in)
        doc = Document.from_json(doc_dict)


    context = doc_dict["symbols"]

    # Generate prediction
    cot = dspyCOT(SingleQuestionOverPaper)
    pred = cot(question, context)
    return pred.answer, pred.rationale

def run_grd_batch (list_of_files, question, model='gpt-4-1106-preview'):
    # Define question and context

    result = {}
    result["question"] = question
    result["files"] = []
    for file in list_of_files:
        try:
            start_time = time.time()
            groundtruth, rationale = get_groundtruth(file, question, model)
            duration = time.time() - start_time
            print(f"Duration: {duration}")
            result["files"].append({"file": file, "groundtruth": groundtruth, "rationale": rationale, "duration": duration})
            print("file:", file, " groundtruth: ", groundtruth, " rationale: ", rationale)
        except Exception as e:
            print(f"Error: {e}")
            continue
    return result

if __name__ == "__main__":
    dsp.modules.cache_utils.cache_turn_on = False
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-1106-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)

