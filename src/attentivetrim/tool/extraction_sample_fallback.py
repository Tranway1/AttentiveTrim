import os
import json
import time

import dsp
import dspy
from src.attentivetrim.tool import histogram_range
from src.attentivetrim.tool.dspy_interface import dspyCOT



QUESTIONS = ["What is the paper title?",
             "What is the authors of the paper?",
             "What is the main contribution of the paper?",
             "What are the baselines used in the evaluation?"]

PAPER_QUESTIONS = [
    "What is the publication year?",
    "What is the type of contribution?",
    "What is the domain?",
    "What is the type of study?",
    "What is the venue?",
    "What is the artifact?",
    "What is the theory?",
    "What is the type of population being studied or designed?",
    "What is the number of authors?"
]


NOTICE_QUESTIONS = [
    "What is the name of the company?",
    "What is the region of the company?",
    "What is the state of the company?",
    "What is the date of the notice?",
    "What is the proposed civil penalty?",
    "What is the type of violation item?",
    "What is the number of violation items?",
    "What is the compliance order?"
]

CIVIC_QUESTIONS = [
    "What is the name of the project?",
    "What is the begin construction time?",
    "What is the complete design time or completion?",
    "What is the advertise time?",
    "What is the topic?",
    "What is the type?",
    "What is the status?"
]


HISTS = ["../data/frequency-test-title.csv",
            "../data/frequency-test-authors.csv",
            "../data/frequency-test-contribution.csv",
            "../data/frequency-test-baselines.csv"]

BUDGETS = [
    [0.001, 0.005, 0.05, 0.3],
    [0.005, 0.01, 0.05, 0.1],
    [0.05, 0.1, 0.15, 0.2, 0.4, 0.9],
    [0.05, 0.1, 0.15, 0.2, 0.4, 0.9]
]





class SingleQuestionOverSampleFallback(dspy.Signature):
    """Answer question(s) about a scientific paper."""

    context = dspy.InputField(desc="contains a snippet of the paper, including the most possible answer to the question.")
    question = dspy.InputField(desc="one question about the paper")
    answer = dspy.OutputField(desc="print the answer close to the original text as you can, and print 'None' if an answer cannot be found or the answer is truncated at the beginning or the end of the given context. Please do not helucinate the answer")


# automatically fill up the relevant range by character
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

    real_ratio = (end - start)/test_len

    # Generate prediction
    cot = dspyCOT(SingleQuestionOverSampleFallback)
    pred = cot(question, sample)
    return pred.answer, real_ratio

# automatically fill up the sentence at the beginning and the end
def get_test_result_sentence(file_path, question, sr, er):

    with open('/Users/chunwei/pvldb_1-16/16/' + file_path) as f_in:
        doc_dict = json.load(f_in)
    context = doc_dict["symbols"]
    sentences = doc_dict["entities"]["sentences"]
    # sentences look like this: [{"spans": [[0, 61]]}, {"spans": [[62, 64]]}, {"spans": [[65, 236]]}, {"spans": [[237, 450]]},...]

    test_len = len(context)
    start = int(sr * test_len)
    end = int(er * test_len)


    print("proposed start:", start, "end:", end)
    # return all the sentences that overlaps the range

    real_start = start
    real_end = end
    # find the read start and end with regard to the sentence
    for sentence in sentences:
        span = sentence["spans"][0]
        if (span[0] >= start and span[0] <= end) and span[1] >= end:
            real_end = span[1]
            break
        elif (span[1] >= start and span[1] <= end) and span[0] <= start:
            real_start = span[0]

    print("character real start:", real_start, "real end:", real_end)
    real_ratio = (real_end - real_start)/test_len

    sample = context[real_start:real_end]
    print("sample size:", len(sample))
    cot = dspyCOT(SingleQuestionOverSampleFallback)
    pred = cot(question, sample)
    return pred.answer, real_ratio


# automatically fill up the relevant range by paragraph
# pagemage paragraph is not built well
def get_test_result_paragraph(file_path, question, sr, er):
    with open('/Users/chunwei/pvldb_1-16/16/' + file_path) as f_in:
        doc_dict = json.load(f_in)
    context = doc_dict["symbols"]
    paragraphs = doc_dict["entities"]["paragraphs"]
    # paragraphs look like this: [{"spans": [[0, 61]]}, {"spans": [[62, 64]]}, {"spans": [[65, 236]]}, {"spans": [[237, 450]]},...]
    test_len = len(context)
    start = int(sr * test_len)
    end = int(er * test_len)

    print("proposed start:", start, "end:", end)
    real_start = start
    real_end = end

    for paragraph in paragraphs:
        span = paragraph["spans"][0]
        if (span[0] >= start and span[0] <= end) and span[1] >= end:
            real_end = span[1]
            break
        elif (span[1] >= start and span[1] <= end) and span[0] <= start:
            real_start = span[0]

    print("character real start:", real_start, "real end:", real_end)
    real_ratio = (real_end - real_start)/test_len

    sample = context[real_start:real_end]
    print("sample size:", len(sample))
    cot = dspyCOT(SingleQuestionOverSampleFallback)
    pred = cot(question, sample)
    return pred.answer, real_ratio




# automatically fill up the relevant range by section
def get_test_result_section(file_path, question, sr, er):
    with open('/Users/chunwei/pvldb_1-16/16/' + file_path) as f_in:
        doc_dict = json.load(f_in)
    context = doc_dict["symbols"]
    sections = doc_dict["entities"]["sections"]
    # sections look like this: [{"spans": [[0, 61]]}, {"spans": [[62, 64]]}, {"spans": [[65, 236]]}, {"spans": [[237, 450]]},...]
    test_len = len(context)
    sectionbasedranges = []
    pre_offset = 0
    if len(doc_dict["entities"]["abstracts"])>0:
        abstract = doc_dict["entities"]["abstracts"][0]["spans"][0]
        sectionbasedranges.append((0, abstract[0]-1))
        pre_offset=abstract[1]
    for section in sections:
        span0 = section["spans"][0][0]
        sectionbasedranges.append((pre_offset, span0-1))
        pre_offset = span0
    sectionbasedranges.append((pre_offset, test_len-1))
    assert len(sectionbasedranges) == len(sections)+2

    start = int(sr * test_len)
    end = int(er * test_len)
    print("proposed start:", start, "end:", end)

    real_start = start
    real_end = end
    cnt = 0
    for section in sectionbasedranges:
        (s, e) = section
        if (s >= start and s <= end) and e >= end:
            real_end = e
            print("End section idx:", cnt, "start:", s, "end:", e)
            break
        elif (e >= start and e <= end) and s <= start:
            print("Start section idx:", cnt, "start:", s, "end:", e)
            real_start = s
        cnt += 1
    print("character real start:", real_start, "real end:", real_end)
    real_ratio = (real_end - real_start)/test_len

    sample = context[real_start:real_end]
    print("sample size:", len(sample))
    cot = dspyCOT(SingleQuestionOverSampleFallback)
    pred = cot(question, sample)
    return pred.answer, real_ratio



def run_file_batch(list_of_files, question, hist_file, budget=0.05, mode="fallback", enable_general_fallback=True):
    sr, er = histogram_range.get_range_from_hist(hist_file, budget, resolution=0.001, trim_zeros=False)
    print("start ratio:", sr, "end ratio:", er)

    results = {"question": question, "files": []}
    if mode == ("fallback"):
        for file in list_of_files:
            print(f"using budget: {budget}")
            cur_budget = budget
            print("file:", file)
            start_time = time.time()
            test_result, _ = get_test_result(file, question, sr, er)
            print("test_result:", test_result)
            while test_result == "None":
                print(f"fallback to higher budget {cur_budget} -> {cur_budget*2}")
                if cur_budget <0.4:
                    cur_budget = cur_budget*2
                elif cur_budget < 1.0:
                    cur_budget = 1.0
                else:
                    break
                print(f"using budget: {cur_budget}")
                cur_sr, cur_er = histogram_range.get_range_from_hist(hist_file, cur_budget, resolution=0.001, trim_zeros=False)
                print("fallback start ratio:", cur_sr, "end ratio:", cur_er)
                test_result, _ = get_test_result(file, question, cur_sr, cur_er)
            duration = time.time() - start_time

            results["files"].append({"file": file, "result": test_result, "duration": duration, "budget": cur_budget})
            print("file:", file, " result:", test_result)
    else:
        for file in list_of_files:
            print(f"using budget: {budget}")
            cur_budget = budget
            print("file:", file)
            start_time = time.time()
            if mode == "sentence":
                test_result, real_ratio = get_test_result_sentence(file, question, sr, er)
            elif mode == "paragraph":
                test_result, real_ratio = get_test_result_paragraph(file, question, sr, er)
            elif mode == "section":
                test_result, real_ratio = get_test_result_section(file, question, sr, er)
            else:
                raise ValueError("mode not supported")
            print("test_result:", test_result)
            if enable_general_fallback:
                while test_result == "None":
                    print(f"fallback to higher budget {cur_budget} -> {cur_budget*2}")
                    if cur_budget <0.4:
                        cur_budget = cur_budget*2
                    elif cur_budget < 1.0:
                        cur_budget = 1.0
                    else:
                        break
                    print(f"using budget: {cur_budget}")
                    cur_sr, cur_er = histogram_range.get_range_from_hist(hist_file, cur_budget, resolution=0.001, trim_zeros=False)
                    print("fallback start ratio:", cur_sr, "end ratio:", cur_er)
                    if mode == "sentence":
                        test_result, real_ratio = get_test_result_sentence(file, question, cur_sr, cur_er)
                    elif mode == "paragraph":
                        test_result, real_ratio = get_test_result_paragraph(file, question, cur_sr, cur_er)
                    elif mode == "section":
                        test_result, real_ratio = get_test_result_section(file, question, cur_sr, cur_er)
                    else:
                        raise ValueError("mode not supported")
            duration = time.time() - start_time

            results["files"].append({"file": file, "result": test_result, "duration": duration, "budget": real_ratio})
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
    budgets = BUDGETS[q_idx]
    mode = "section"
    enable_general_fallback = True
    for budget in [budgets[0]]:
        print("question:", question, "hist_file:", hist_file, "budget:", budget, "mode:", mode, "enable_general_fallback:", enable_general_fallback)
        results = run_file_batch(list_of_files, question, hist_file, budget=budget, mode=mode, enable_general_fallback=enable_general_fallback)
        json_string = json.dumps(results, indent=4)
        with open(f'../data/{mode}/results-{mode}-{enable_general_fallback}-{question[:15]}-{budget}.json', 'w') as f:
            f.write(json_string)