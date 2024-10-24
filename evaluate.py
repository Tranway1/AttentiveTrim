"""
This script is used to evaluate the performance of the attentive trim method.
The main method that is exposed is answer_question, which takes in a context and a question, and an optional list of indices of token to retain in the context, and returns the answer.
This answer is then evaluated by comparing it to the ground truth answer.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import pipeline, set_seed
import torch
import pickle
import argparse

model_name = "openai-community/gpt2"
model_name = "/home/gridsan/cliu/hf/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, strip_accents=True)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, device_map="auto")


contexts =  [
            """A DECLARATIVE SYSTEM FOR OPTIMIZING AI WORKLOADS
            A PREPRINT
            Chunwei Liu*
            , Matthew Russo*
            , Michael Cafarella,
            Lei Cao†
            , Peter Baille Chen, Zui Chen, Michael Franklin‡
            ,
            Tim Kraska, Samuel Madden, Gerardo Vitagliano
            MIT, †University of Arizona, ‡University of Chicago
            chunwei@mit.edu, mdrusso@mit.edu, michjc@csail.mit.edu
            caolei@arizona.edu, peterbc@mit.edu, chenz429@mit.edu, mjfranklin@uchicago.edu,
            kraska@mit.edu, madden@csail.mit.edu, gerarvit@csail.mit.edu
            ABSTRACT
            A long-standing goal of data management systems has been to build systems which can compute
            quantitative insights over large corpora of unstructured data in a cost-effective manner. Until recently,
            it was difficult and expensive to extract facts from company documents, data from scientific papers, or
            metrics from image and video corpora. Today’s models can accomplish these tasks with high accuracy.
            """]

questions = ["What is the paper title?",
            "What is the authors of the paper?"]

answers = [["A Deep Dive into Common Open Formats for Analytical DBMSs",
            "A DECLARATIVE SYSTEM FOR OPTIMIZING AI WORKLOADS",
            "Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods",
            "A Lived Informatics Model of Personal Informatics",
            "A Stage-Based Model of Personal Informatics Systems",
            "A Stage-Based Model of Personal Informatics Systems"],
            ["Chunwei Liu Anna Pavlenko Matteo Interlandi Brandon Haynes",
            "Chunwei Liu Matthew Russo Michael Cafarella Lei Cao Peter Baille Chen Zui Chen Michael Franklin Tim Kraska Samuel Madden Gerardo Vitagliano",
            "Kathleen C. Fraser Hillary Dawkins Svetlana Kiritchenko",
            "Daniel A. Epstein An Ping James Fogarty Sean A. Munson",
            "Ian Li Anind Dey Jodi Forlizzi",
            ""]]

def evaluate_answer(predicted, ground_truth):
    tok_answer = predicted.lower().split(" ")
    tok_gt = ground_truth.lower().split(" ")
    try:
        recall = len([x for x in tok_gt if x in tok_answer]) / len(tok_gt)
    except ZeroDivisionError:
        return breakpoint()
    return recall

def answer_question(context, question, token_indices=None):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
   
    if token_indices:
        inputs = inputs[:, token_indices]

    outputs = model.generate(inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    output_tokens = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = output_tokens.split("Answer:")[1].strip()
    return answer


top50_tophead = [0,  17,  22,  27,  18,  32,   2,  42,   4, 112,  21,   1,  28,  29, 24, 171,  88, 170, 189,  38, 250,  20,  26,  19,  31,  23,   3,  58, 34, 165,  43,   8, 172,  74, 209, 169,  30,  37, 260, 167,  36, 173, 25,  78,  60,   9,  44,  82,  77,  35]

top50_toplayer = [0, 250, 252, 256, 253, 251, 260,  17, 255, 257, 254, 261,  27, 259, 189, 112, 171, 165,  21,  22, 170,   2,  58, 258, 262, 249,  74,  18, 1,  88,  43, 167,  29,   8, 206,  24,  38, 173, 208,  32,  42,  28, 33,  34,  26, 169, 238,  95, 229, 209]

top50_all = [0, 256, 253, 252, 251, 260, 250, 257, 254, 255, 261,  17, 258, 259, 262, 249,  21,   2, 170,  27,   1,  22, 165, 189, 171,  18, 112,  24, 238, 206, 176,  26,   8,   4, 175, 156, 173, 240, 167,   6, 241, 239, 19,  88,  85, 172, 169,   7, 209, 229]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the performance of the attentive trim method.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output")
    parser.add_argument("--token_indices", choices=["none", "tophead", "toplayer", "all"], default="none", help="Token indices to use for the second question (default: none)")
    args = parser.parse_args()
    print(f"using the variables {args}")
    scores = []
    for i, c in enumerate(contexts):
        for j, q in enumerate(questions):
            if j == 0:
                token_indices = None
            if j == 1:
                if args.token_indices == "none":
                    token_indices = None
                elif args.token_indices == "tophead":
                    token_indices = list(sorted(top50_tophead))
                elif args.token_indices == "toplayer":
                    token_indices = list(sorted(top50_toplayer))
                elif args.token_indices == "all":
                    token_indices = list(sorted(top50_all))

            answer = answers[j][i]
            predicted = answer_question(c, q, token_indices)
            score = evaluate_answer(predicted, answer)
            scores.append(score)
            if args.verbose:
                print("Input:", q[:, token_indices])
                print("\t", predicted)
                print("\t (Ground Truth:", answer, ")")
                print("score", score)

    print(f"Done with {len(contexts)} contexts and {len(questions)} questions")
    print("Average score:", sum(scores) / len(scores))