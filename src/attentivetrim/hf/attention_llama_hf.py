from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch


# local_model_path = '/home/gridsan/cliu/hf/Mistral-7B-Instruct-v0.2'
# local_model_path = '/home/gridsan/cliu/hf/dbrx-instruct/'
model_name = "/home/gridsan/cliu/hf/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True, load_in_4bit=True, device_map="auto")

context =  """A Deep Dive into Common Open Formats for Analytical DBMSs
Chunwei Liu MIT CSAIL chunwei@csail.mit.edu
ABSTRACT
Anna Pavlenko Microsoft annapa@microsoft.com
Matteo Interlandi Microsoft mainterl@microsoft.com
Brandon Haynes Microsoft brhaynes@microsoft.com
This paper evaluates the suitability of Apache Arrow, Parquet, and ORC as formats for subsumption in an analytical DBMS. 
"""

question = "What is the abstract of the paper?"

if context.strip():
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
else:
    prompt = f"Question: {question}\nAnswer:"

inputs = tokenizer.encode(prompt, return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0])
# save the attention weights into a file
import pickle

with open('../data/tensor/abstract_attention.pkl', 'wb') as f:
    pickle.dump(attention, f)

# save the tokens into a file
with open('../data/tensor/abstract_tokens.pkl', 'wb') as f:
    pickle.dump(tokens, f)