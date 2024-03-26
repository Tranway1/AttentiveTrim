import json
import os
import getpass

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import LanceDB

import lancedb

def lance_db_rag():
    db = lancedb.connect("/tmp/lancedb")
    table = db.create_table(
        "rag_table",
        data=[
            {
                "vector": OpenAIEmbeddings().embed_query("Hello World"),
                "text": "Hello World",
                "id": "1",
            }
        ],
        mode="overwrite",
    )

    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    file_json = "/Users/chunwei/pvldb_1-16/17/p3044-liu_pm.json"
    with open(file_json) as f_in:
        doc_dict = json.load(f_in)
        # print(doc_dict["symbols"])


    text = doc_dict["symbols"]
    from langchain_text_splitters import TokenTextSplitter

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)


    texts = text_splitter.split_text(text)
    # convert the list of texts to a list of
    print(len(texts))
    # print(texts[0])

    db = LanceDB.from_texts(texts, OpenAIEmbeddings())

    query = "Chunwei Liu, Anna Pavlenko, Matteo Interlandi, Brandon Haynes"
    docs = db.similarity_search(query)
    print(len(docs))
    print(docs[0].page_content)

    # embedding_vector = OpenAIEmbeddings().embed_query(query)
    # docs = db.similarity_search_by_vector(embedding_vector)
    # print(docs[0].page_content)

if __name__ == "__main__":
    lance_db_rag()