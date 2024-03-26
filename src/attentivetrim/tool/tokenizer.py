from typing import List

import tiktoken

def get_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    return encoding.encode(string)


def decode_tokens_to_string(tokens: list[int], encoding_name: str) -> list[bytes]:
    encoding = tiktoken.encoding_for_model(encoding_name)
    return encoding.decode_tokens_bytes(tokens)

if __name__ == "__main__":
    tokens = get_tokens_from_string("Hello world, let's test tiktoken.", "gpt-4-turbo")
    # decode tokens to strings
    tokens = [int(i) for i in tokens]
    byte_tokens = decode_tokens_to_string(tokens, "gpt-4-turbo")
    str_tokens = [token.decode("utf-8") for token in byte_tokens]
    print("tokens:", str_tokens )
    num_tokens = len(tokens)
    print("tokens size:", num_tokens)