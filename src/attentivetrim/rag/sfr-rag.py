import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from char_chunker import get_chunks_char


QUESTIONS = ["What is the main contribution of the paper?",
             "What is the authors of the paper?",
             "What is the paper title?"]

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a paper question, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task,  QUESTIONS[0],),
    get_detailed_instruct(task, QUESTIONS[2])
]


chunks = get_chunks_char('/Users/chunwei/pvldb_1-16/17/p148-zeng_pm.json', truncate=12000)
print("number of chunks:")
print(len(chunks))




# No need to add instruction for retrieval documents
passages = chunks

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')

# get the embeddings
max_length = 1024
input_texts = queries + passages
batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print(scores.tolist())
# given me the index of the top 5 most relevant chunks
top_k = scores.topk(5, dim=1)
for i, (query, top_k_idx) in enumerate(zip(queries, top_k.indices)):
    print(f"Top 5 most relevant chunks for query {queries[i]}:")
    for j, idx in enumerate(top_k_idx):
        print(f"score: {top_k.values[i][j]:.2f}, index: {idx}, text: {passages[idx]}")
