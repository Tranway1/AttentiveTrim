from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
hf_token = "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"
local_model_path = '/home/gridsan/cliu/hf/dbrx-instruct/'

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, token=hf_token)

input_text = "Databricks was founded in "
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))