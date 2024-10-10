import json
import os
import time
from scipy import spatial
from angle_emb import AnglE, Prompts

from src.attentivetrim import dspyCOT
from src.attentivetrim.rag.char_chunker import get_chunks_char
from src.attentivetrim.tool import histogram_range
from src.attentivetrim.tool.diverse.extraction_sample_diverse import SingleQuestionOverSample
import dsp
import dspy

def process_document_with_question(list_of_files, question, task, hist_file, chunk_size=500, budget=1.0, k=30,
                                   output_dir="../../../rag/", model_name='WhereIsAI/UAE-Large-V1'):
    """
    Process a list of documents to find answers to a specified question using a pretrained model and guided by histogram data.

    Args:
    - list_file (str): list of document paths.
    - model_name (str): Model identifier for loading the pretrained model.
    - question (str): The question to process.
    - hist_file (str): Path to the histogram file for additional data guidance.
    - chunk_size (int): Size of text chunks to process.
    - ratio (float): Token reduction ratio for processing.
    - k (int): Number of top results to return.
    - output_dir (str): Directory to save the output JSON file.
    """
    # Load the model
    angle = AnglE.from_pretrained(model_name, pooling_strategy='cls').cuda()
    if budget >= 1.0:
        sr = 0.0
        er = 1.0
    else:
        sr, er = histogram_range.get_range_from_hist_json(hist_file, budget, resolution=0.001, trim_zeros=False)
        print("enable token reduction")

    print("start ratio:", sr, "end ratio:", er)

    # Prepare the output structure
    output = {
        "question": question,
        "task": task,
        "model": model_name,
        "chunk_size": chunk_size,
        "ratio": budget,
        "range": [sr, er],
        "files": []
    }

    # Process each file
    for file_path in list_of_files:
        with open(file_path) as f_in:
            doc_dict = json.load(f_in)

        context = doc_dict["symbols"]
        total_chars = len(context)

        chunks = get_chunks_char(file_path, chunk_char_size=chunk_size, start_ratio=sr, end_ratio=er)
        doc_vecs = angle.encode(chunks)
        agg_time = 0
        file_info = {
            "file": file_path.split('/')[-1],
            "total_chars_used": sum(len(chunk) for chunk in chunks),
            "total_chars": total_chars,
            "total_chunks": len(chunks),
            "runtime": agg_time,
            "results": []
        }

        start_time = time.time()
        qv = angle.encode(Prompts.C.format(text=question))
        res = []
        scores = []

        for idx, dv in enumerate(doc_vecs):
            similarity = 1 - spatial.distance.cosine(qv[0], dv)
            res.append((similarity, idx, chunks[idx]))
            scores.append(similarity)

        # Sort by weighted similarity
        res.sort(key=lambda x: x[0], reverse=True)
        runtime = time.time() - start_time
        agg_time += runtime

        question_result = {
            "top_30": [(sim, idx, text) for sim, idx, text in res[:min(k, len(res))]],
            "runtime": runtime,
            "scores": scores,
        }
        file_info["results"].append(question_result)
        file_info["runtime"] = agg_time
        output["files"].append(file_info)

    # Write to JSON file
    output_file = f'{output_dir}/rag-top30-{task}-{model_name.split("/")[-1]}-{chunk_size}-{budget}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)


def get_answer(question, top_n_indices):
    """
    Extracts answers for a given question from specified chunks of a document.
    """


    selected_chunks = [chunks[2] for chunks in top_n_indices]
    sample = "\n ... ".join(selected_chunks)

    # Initialize the cot model
    cot = dspyCOT(SingleQuestionOverSample)

    # Measure the runtime of the prediction
    start_time = time.time()
    pred = cot(question, sample)
    end_time = time.time()

    # Calculate duration
    duration = end_time - start_time
    return pred.answer, duration, pred.rationale


def process_question(rag_file, question, task, n_values=[1, 5, 10, 20, 30], output_dir="../../../pred_rag/"):
    """
    Processes a single question across multiple documents and saves the results.
    """
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Load JSON file
    with open(rag_file, 'r') as file:
        data = json.load(file)

    text_ratio = float(data['ratio'])
    if text_ratio < 1.0:
        print("using token reduction")
        task = f"{task}-trrag"
    else:
        task = f"{task}-rag"
    results = {}
    total_chars = 0

    # Process each file
    for file_obj in data['files']:
        total_chars += (int(file_obj['total_chars']))
        print(f'File: {file_obj["file"]}')

        top = file_obj['results'][0]['top_30']  # Assuming only one question result per file

        for n in n_values:
            cur_n = min(n, len(top))
            top_n_indices = top[:cur_n]
            # sort the top_n_indices by ascending order on the second element of the tuple
            top_n_indices.sort(key=lambda x: x[1])
            # print the sorted top_n_indices ie the second element of the tuple
            print(f"Top {cur_n} indices: {[x[1] for x in top_n_indices]}")


            try:
                answer, duration, rationale = get_answer(question, top_n_indices)
            except Exception as e:
                print(f"Error processing file {file_obj['file']}: {e}")
                answer = "None"
                duration = 0.0
                rationale = "Processing error"

            if n not in results:
                results[n] = []
            results[n].append({
                "file": file_obj["file"],
                "result": answer,
                "duration": duration,
                "rationale": rationale,
                "cur_n": cur_n
            })
            print(f'Question: {question}, N: {cur_n}, Answer: {answer}, Duration: {duration:.2f}s')



    # Save results to JSON files
    for n in results:
        sum_cur_n = sum(res['cur_n'] for res in results[n])
        ratio = f'{sum_cur_n * 500.0 / total_chars:.2f}'
        json_results = {
            "question": question,
            "files": results[n]
        }
        output_path = f'{output_dir}/ragresults-{ratio}-{task}.json'
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f'File saved: {output_path}')

if __name__ == "__main__":
    dsp.modules.cache_utils.cache_turn_on = False
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-1106-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)
    # Example usage
    rag_file = "../../../rag/rag-top30-paper_What is the venue that the paper published in?-UAE-Large-V1-500-0.4.json"
    question = "What is the domain of this paper?"
    process_question(rag_file, question, f"paper_{question}")


