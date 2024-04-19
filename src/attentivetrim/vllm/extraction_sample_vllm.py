import json
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
import time
import os

from src.attentivetrim.tool.histogram_range import get_range_from_hist

QUESTIONS = ["What is the paper title?",
             "What is the authors of the paper?",
             "What is the main contribution of the paper?"]

HISTS = ["../data/frequency-test-title.csv",
            "../data/frequency-test-authors.csv",
            "../data/frequency-test-contribution.csv"]
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
parser = ArgumentParser()
parser.add_argument("--model", default='bigscience/bloom', type=str, help="model_name")
parser.add_argument("--input_size", type=int, default=128, help="input prompt token size")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--tensor_para_size", default=8, type=int, help="tensor parallelism")
parser.add_argument("--iters", default=5, type=int, help="number of iterations")
parser.add_argument("--greedy", action='store_true', help="greedy generation mode - temperature=0")
parser.add_argument("--print_output", action='store_true', help="print generated output text")
parser.add_argument("--test_perf", action='store_true', help="test performance to include warmup runs")
parser.add_argument("--text_ratio", default=0.1, type=float, help="ratio of the text to use for the prompt")
parser.add_argument("--question", default=0, type=int, help="question index to use")
args = parser.parse_args()
 
max_model_len = (args.max_new_tokens + args.input_size + 5) * args.batch_size
llm = LLM(args.model,
        tensor_parallel_size=args.tensor_para_size,
        max_model_len=max_model_len,
        #download_dir='/dev/shm/',
        disable_log_stats=False,
        dtype="float16")
 
tokenizer = llm.get_tokenizer()

# with open('./nate_the_snake.txt', 'r') as f:
#     text = f.read()
#     encoded_text = tokenizer(text).input_ids
#     prompt = tokenizer.decode(encoded_text[:args.input_size])
#
# prompts = [prompt]
# if args.batch_size > len(prompts):
#     prompts *= math.ceil(args.batch_size / len(prompts))
#  
# prompts = prompts[:args.batch_size]
base_dir = "/home/gridsan/cliu/"
list_file = "../data/test_v16_inputfile100.txt"
with open(list_file) as f:
    list_of_files = f.readlines()
list_of_files = [x.strip() for x in list_of_files]


ratio = args.text_ratio
q_idx = args.question
hist_file = HISTS[q_idx]

prompt = "Tell me about Boston"
prompts = [prompt]
if args.greedy:
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
else:
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens)
 
# warmup
if args.test_perf:
    outputs = llm.generate(prompts, sampling_params)

question = QUESTIONS[q_idx]
sr, er = get_range_from_hist(hist_file, ratio, resolution=0.001, trim_zeros=False)
results = {"question": question, "ratio": ratio, "hist_file": hist_file, "start_ratio": sr, "end_ratio": er, "files": []}
print("question:", question, "ratio:", ratio, "hist_file:", hist_file, "start ratio:", sr, "end ratio:", er)

for file in list_of_files:
    with open(base_dir+'pvldb_1-16/16/' + file) as f_in:
        doc_dict = json.load(f_in)

    context = doc_dict["symbols"]
    test_len = len(context)
    start = int(sr * test_len)
    end = int((er+0.001) * test_len)

    print("character start:", start, "end:", end)
    sample = context[start:end]
    print("given ratio", ratio, "sample size:", len(sample), "total size:", len(context), "ratio:", len(sample)/len(context))

    prompt = f"Here is a paper snippet:'''{sample}'''\n\nNow please answer the question: {question} \n Please only answer the question I asked above and print the answer in a single line."
    prompts = [prompt]
    test_result = ""
    start_time = time.time()
    for i in range(args.iters):
        start = time.time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        print("elapsed time:", time.time() - start)

        if args.print_output:
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                # print(f"output: {output}")
                test_result = generated_text
                print(f"Generated text: {test_result!r}")

    duration = (time.time() - start_time) / (1.0*args.iters)

    results["files"].append({"file": file, "result": f"{test_result!r}", "duration": duration})
    print("file:", file, " result:", f"{test_result!r}")

json_string = json.dumps(results, indent=4)
with open(f'../data/vllm/vllm-results-{question[:15]}-{ratio}.json', 'w') as f:
    f.write(json_string)


