import json

from angle_emb import AnglE, Prompts
from scipy import spatial

from char_chunker import get_chunks_char




question = 'What is the main contribution of the paper?'
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
qv = angle.encode(Prompts.C.format(text=question))
chunks = get_chunks_char('/Users/chunwei/pvldb_1-16/17/p148-zeng_pm.json', chunk_char_size=500)
doc_vecs = angle.encode(chunks)
print("number of chunks:")
print(len(chunks))

res = []
idx = 0
for dv in doc_vecs:
    res.append((1 - spatial.distance.cosine(qv[0], dv), idx, chunks[idx]))
    print(idx, 1-spatial.distance.cosine(qv[0], dv) )
    idx+=1


# sort by similarity
res.sort(key=lambda x: x[0], reverse=True)
print(f'Top 20 most relevant chunks for {question}:')
for i in range(20):
    print(res[i])