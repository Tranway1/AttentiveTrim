import os
import json
import time
from src.attentivetrim.tool import histogram_range
from src.attentivetrim.tool.diverse.extraction_sample_diverse import SingleQuestionOverSample
from src.attentivetrim.tool.dspy_interface import dspyCOT



def get_test_result(file_path, question, sr, er):
    # Load document
    with open(file_path) as f_in:
        doc_dict = json.load(f_in)

    context = doc_dict["symbols"]
    test_len = len(context)


    start = int(sr * test_len)
    end = int(er * test_len)

    print("character start:", start, "end:", end)
    sample = context[start:end]
    print("sample size:", len(sample))

    # Generate prediction
    cot = dspyCOT(SingleQuestionOverSample)
    pred = cot(question, sample)
    return pred.answer, pred.rationale

def run_sample_pred_batch(list_of_files, question, hist_file, budget=0.05):
    sr, er = histogram_range.get_range_from_hist_json(hist_file, budget, resolution=0.001, trim_zeros=False)
    print("start ratio:", sr, "end ratio:", er)

    results = {"question": question, "files": []}
    for file in list_of_files:
        start_time = time.time()
        test_result, rationale = get_test_result(file, question, sr, er)
        duration = time.time() - start_time

        results["files"].append({"file": file, "result": test_result, "rationale": rationale, "duration": duration, "budget": budget})
        print("file:", file, " result:", test_result)

    return results

