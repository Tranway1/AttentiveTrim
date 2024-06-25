import json
from typing import List
from langchain_text_splitters import TokenTextSplitter


def split_text_on_characters(*, text: str, chars_per_chunk: int, chunk_overlap: int = 0) -> List[str]:
    """Split incoming text and return chunks based on character count."""
    splits: List[str] = []
    start_idx = 0
    cur_idx = min(start_idx + chars_per_chunk, len(text))

    while start_idx < len(text):
        splits.append(text[start_idx:cur_idx])
        if cur_idx == len(text):
            break
        start_idx += chars_per_chunk - chunk_overlap
        cur_idx = min(start_idx + chars_per_chunk, len(text))

    return splits

def get_chunks_token(file_json,truncate=0):
    # file_json = "/Users/chunwei/pvldb_1-16/17/p3044-liu_pm.json"
    with open(file_json) as f_in:
        doc_dict = json.load(f_in)
        # print(doc_dict["symbols"])
    if truncate!=0:
        text = doc_dict["symbols"][:truncate]
    else:
        text = doc_dict["symbols"]

    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)

    texts = text_splitter.split_text(text)
    # convert the list of texts to a list of
    print(len(texts))
    return texts

def get_chunks_char(file_json,chunk_char_size=500, truncate=0, start_ratio=0.0, end_ratio=1.0):
    # file_json = "/Users/chunwei/pvldb_1-16/17/p3044-liu_pm.json"
    print(f'chunking file: {file_json}')
    with open(file_json) as f_in:
        doc_dict = json.load(f_in)
        # print(doc_dict["symbols"])
    if truncate!=0:
        text = doc_dict["symbols"][:truncate]
    else:
        text = doc_dict["symbols"]
    if start_ratio != 0.0 or end_ratio != 1.0:
        print(f"truncate text from {start_ratio} to {end_ratio}")
        text = text[int(start_ratio * len(text)):int(end_ratio * len(text))]


    texts = split_text_on_characters(text=text, chars_per_chunk=chunk_char_size, chunk_overlap=0)
    return texts


if __name__ == "__main__":
    text = "This is a sample text to demonstrate character-based splitting."
    chars_per_chunk = 10
    chunk_overlap = 3
    chunks = split_text_on_characters(text=text, chars_per_chunk=chars_per_chunk, chunk_overlap=chunk_overlap)
    print(chunks)