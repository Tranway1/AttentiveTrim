from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences005 = ["The main contribution of the paper is the design of C5, the first cloned concurrency control protocol to provide bounded replication lag.",
                "The main contribution of the paper is the introduction of the C5 protocol, which is a commensurate-granularity"]
sentences02 =["In sum, this paper\u2019s contributions stem from our key insight that cloned concurrency control must have commensurate constraints with the primary: first, we prove neither a transaction-granularity nor a page-granularity protocol can always keep up with a two-phase locking primary; next, we prove a commensurate-granularity protocol has the potential to keep",
              "In sum, this paper\u2019s contributions stem from our key insight that cloned concurrency control must have commensurate constraints with the primary: first, we prove neither a transaction-granularity nor a page-granularity protocol can always keep up with"]
sentences01 = ["The main contribution of the paper is the design of C5, the first cloned concurrency control protocol to provide bounded replication lag.",
               "The main contribution of the paper is the key insight that cloned concurrency control must have commensurate constraints with the primary, the proof that neither a transaction-granularity nor a page-granularity protocol can always keep up with a two-phase locking primary, the proof that a commensurate-granularity protocol has the potential to keep up with an unrestricted primary, and the description and implementation of such a protocol, C5, which is demonstrated to always keep up in practice."]
sentence_embeddings = model.encode(sentences02)

cos_sim = cosine_similarity(
    [sentence_embeddings[0]],
    [sentence_embeddings[1]]
)
print(cos_sim)