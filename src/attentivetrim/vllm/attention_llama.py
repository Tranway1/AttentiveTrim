from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Load the model and tokenizer
model = LLM("/home/gridsan/cliu/hf/Meta-Llama-3-8B-Instruct", dtype="float16")
tokenizer = AutoTokenizer.from_pretrained("/home/gridsan/cliu/hf/Meta-Llama-3-8B-Instruct")

def profile_token_ids(tokenizer, token_ids):
    # Print index, token ID, and word
    print(f"Number of tokens: {len(token_ids)}")
    for idx, token_id in enumerate(token_ids):
        word = tokenizer.decode([token_id])
        print(f"{idx}, {token_id}, {word}")

def generate_response(context, question):
    # Format the prompt based on the provided context and question
    if context.strip():
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"

    # Tokenize the prompt
    messages = [{"role": "user", "content": {prompt}}]
    encoded_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Formatted prompt: {encoded_input}")

    token_input = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print(f"Tokenized prompt: {token_input}")
    token_ids = token_input

    # Profile token IDs
    profile_token_ids(tokenizer, token_ids)

    # Generate the response using the model
    output = model.generate(encoded_input, SamplingParams(max_tokens=32, temperature=0))
    return output

# Example usage
context =  """A Deep Dive into Common Open Formats for Analytical DBMSs
Chunwei Liu MIT CSAIL chunwei@csail.mit.edu
ABSTRACT
Anna Pavlenko Microsoft annapa@microsoft.com
Matteo Interlandi Microsoft mainterl@microsoft.com
Brandon Haynes Microsoft brhaynes@microsoft.com
This paper evaluates the suitability of Apache Arrow, Parquet, and ORC as formats for subsumption in an analytical DBMS. We sys- tematically identify and explore the high-level features that are important to support efficient querying in modern OLAP DBMSs and evaluate the ability of each format to support these features. We find that each format has trade-offs that make it more or less suitable for use as a format in a DBMS and identify opportunities to more holistically co-design a unified in-memory and on-disk data representation. Our hope is that this study can be used as a guide for system developers designing and using these formats, as well as provide the community with directions to pursue for improving these common open formats.
PVLDB Reference Format:
Chunwei Liu, Anna Pavlenko, Matteo Interlandi, and Brandon Haynes. A Deep Dive into Common Open Formats for Analytical DBMSs . PVLDB, 16(11): 3044 - 3056, 2023.
doi:10.14778/3611479.3611507
PVLDB Artifact Availability:
The source code, data, and/or other artifacts have been made available at https://github.com/Tranway1/ColumnarFormatsEval.
1 INTRODUCTION
"""

question = "What is the paper title?"
response = generate_response(context, question)
print(f"Response~{datetime.now()}: {response}")