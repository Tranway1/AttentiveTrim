import json
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

from pathlib import Path
HOME_DIR = Path.home()


# local_model_path = f'{HOME_DIR}/hf/Mistral-7B-Instruct-v0.2'
# local_model_path = f'{HOME_DIR}/hf/dbrx-instruct/'
model_name = f"{HOME_DIR}/hf/Llama-3.2-1B-Instruct"



def clean_token(tokens, cur_tokenizer):
    clearned_tokens = []
    for token in tokens:
        t_list = [token]
        clean_token = cur_tokenizer.convert_tokens_to_string(t_list)
        clearned_tokens.append(clean_token)
    return clearned_tokens

def extend_text_range(start_char, end_char, text, extended_char_size=8000):
    # Check if the entire text is shorter than the extended size
    if len(text) < extended_char_size:
        return text

    # Calculate the middle point of the given range
    middle_point = (start_char + end_char) // 2

    # Calculate half of the extended size
    half_extended_size = extended_char_size // 2

    # Calculate new start and end points
    new_start = max(0, middle_point - half_extended_size)
    new_end = min(len(text), middle_point + half_extended_size)

    # Adjust if the extended range is out of the text boundaries
    if new_start == 0:
        new_end = min(extended_char_size, len(text))
    if new_end == len(text):
        new_start = max(0, len(text) - extended_char_size)
    print(f"new_start: {new_start}, new_end: {new_end}")
    # Return the extended text range
    return new_start, new_end

def get_token_range_from_char_range(tokens, start_char, end_char):
    assert start_char < end_char, "start_char should be less than end_char"
    print(f"original start_char: {start_char}, original end_char: {end_char}")

    total_chars = sum(len(token) for token in tokens)
    print(f"total_chars: {total_chars}")
    if start_char < 0:
        start_char = total_chars + start_char
    if end_char <= 0:
        end_char = total_chars + end_char

    # Ensure the adjusted indices are within bounds
    start_char = max(0, min(start_char, total_chars))
    end_char = max(0, min(end_char, total_chars))

    print(f"adjusted start_char: {start_char}, adjusted end_char: {end_char}")

    start_token = end_token = None
    char_count = 0
    for idx, token in enumerate(tokens):
        token_length = len(token)
        if start_token is None and char_count + token_length > start_char:
            print(f"start_token: {start_token}, char_count: {char_count}, token_length: {token_length}, index: {idx}， token: {token}")
            start_token = idx
        if end_token is None and char_count + token_length >= end_char:
            print(f"end_token: {end_token}, char_count: {char_count}, token_length: {token_length}, index: {idx}， token: {token}")
            end_token = idx+1
            break
        char_count += token_length

    return start_token, end_token

def analyze_attention_layers(attention, grd_s = 3, grd_e = 16, question_length = 10, k = 30):
    """ Analyze attention across layers focusing on specific token indices. """
    att_layers = []
    for layer in range(len(attention)):
        sum_attention = torch.sum(attention[layer], dim=(0, 1))
        topk = torch.topk(sum_attention[-question_length:], k)
        count = sum(grd_s <= index[i] <= grd_e for index in topk.indices for i in range(k))
        att_layers.append([f"Layer {layer}", count])
    sorted_att_layers = sorted(att_layers, key=lambda x: x[1], reverse=True)
    return sorted_att_layers[:10]


def analyze_attention_heads(attention, grd_s = 3, grd_e = 16, question_length = 10, k = 30):
    """ Analyze attention across heads in each layer focusing on specific token indices. """
    print("Analyzing attention heads...")
    att_heads = []
    print("Grd_s: ", grd_s,     "Grd_e: ", grd_e)
    for layer in range(len(attention)):
        multi_head = torch.sum(attention[layer], dim=0)
        for head in range(len(multi_head)):
            sum_attention = multi_head[head]
            topk = torch.topk(sum_attention[-question_length:], k)
            count = sum(grd_s <= index[i] <= grd_e for index in topk.indices for i in range(k))
            att_heads.append([f"Layer: {layer}, Head: {head}", count])
    sorted_att_heads = sorted(att_heads, key=lambda x: x[1], reverse=True)
    return sorted_att_heads[:10]

def run_reverse_engineer(dataset):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True, load_in_4bit=True, device_map="auto")
    print(f"Model loaded: {model_name}")

    config_path = '../../../questions/question.json'
    with open(config_path) as f:
        data = json.load(f)

    questions = data["query"][f'{dataset.upper()}_QUESTIONS']
    data["head_stats"] = data.get("head_stats", {})
    data["layer_stats"] = data.get("layer_stats", {})
    data["head_stats"][dataset] = {}
    data["layer_stats"][dataset] = {}


    for question in questions:
        layer_count = {}
        head_count = {}
        print(f"Processing question: {question} on dataset: {dataset}")
        cnt = 0
        with open(f"../../../grd_loc/{dataset}_{question}-location.json", 'r') as f:
            json_obj = json.loads(f.read())

        for entry in json_obj["files"][:30]:
            print(f"Processing file: {entry['file']}")
            file_path = entry["file"].replace("/Users/chunwei/research/", f"{HOME_DIR}/")
            if HOME_DIR == "/home/chunwei":
                file_path = file_path.replace("/Users/chunwei/research/", f"{HOME_DIR}/").replace("(", "\\( ").replace(")", "\\)")

            grdt = entry["groundtruth"]
            print(f"Ground truth: {grdt}")
            if grdt.lower() == "none":
                print("No ground truth. Skipping this entry.")
                continue
            with open(file_path) as f_in:
                doc_dict = json.load(f_in)
            context = doc_dict["symbols"]
            grd_start = entry["start"]
            grd_end = entry["end"]

            enable_trim = False

            if context.strip():
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            else:
                prompt = f"Question: {question}\nAnswer:"
            try:
                inputs = tokenizer.encode(prompt, return_tensors='pt')
                outputs = model(inputs)
                cnt += 1
            except RuntimeError as e:
                enable_trim = True
                print(f"RuntimeError: {e}")
                torch.cuda.empty_cache()

            if enable_trim:
                new_start, new_end = extend_text_range(grd_start, grd_end, context, extended_char_size=8000)
                context = context[new_start:new_end]
                print(f"old grd_start: {grd_start}, old grd_end: {grd_end}")
                grd_start -= new_start
                grd_end -= new_start
                print(f"new grd_start: {grd_start}, new grd_end: {grd_end} with new start base offset: {new_start}")
                print(f"Trimming the context with size {len(context)} and retrying")
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                try:
                    inputs = tokenizer.encode(prompt, return_tensors='pt')
                    outputs = model(inputs)
                    cnt += 1
                except RuntimeError as e:
                    print(f"The trimmed context is still too long. Skipping this entry.")
                    print(f"RuntimeError: {e}")
                    torch.cuda.empty_cache()
                    continue
            print(f"Original context length: {len(context)}")
            print(len(outputs))
            attention = outputs[-1]
            tokens = tokenizer.convert_ids_to_tokens(inputs[0])

            str_token = clean_token(tokens, tokenizer)

            # save the attention weights into a file

            print(f"tokens length: {len(tokens)}")
            print(f"str_token length: {len(str_token)}")

            # get question range in the tokens
            question_section = f"Question: {question}\nAnswer:"
            print(f"{question_section}")
            print(f"Question section length: {len(question_section)}")
            startq_token_idx, endq_token_idx = get_token_range_from_char_range(str_token, -len(question_section), 0)
            print(f"Question token range: {startq_token_idx}, {endq_token_idx}")
            print(f"Question: {tokenizer.convert_tokens_to_string(tokens[startq_token_idx:endq_token_idx])}")


            # get grd range in the tokens
            print(f"Ground truth char range: {grd_start}, {grd_end}")
            # add offset because of the prompt template total selected token length: 51
            # 0: <|begin_of_text|>, 17
            # 1: Context, 7
            # 2: :, 1
            offset = 25
            start_grd_idx, end_grd_idx = get_token_range_from_char_range(str_token, grd_start+offset, grd_end+offset)
            print(f"Ground truth token range: {start_grd_idx}, {end_grd_idx}")
            print(f"ground truth string: {tokenizer.convert_tokens_to_string(tokens[start_grd_idx:end_grd_idx])}")

            print(f"Attention shape: {len(attention)}")
            print(f"Tokens length: {len(tokens)}")

            top_10_layers = analyze_attention_layers(attention, grd_s=start_grd_idx, grd_e=end_grd_idx, question_length=endq_token_idx-startq_token_idx, k=50)
            print("Top 10 attention layers:")
            for item in top_10_layers:
                layer_count[item[0]] = layer_count.get(item[0], 0) + 1
                print(item)

            top_10_heads = analyze_attention_heads(attention, grd_s=start_grd_idx, grd_e=end_grd_idx, question_length=endq_token_idx-startq_token_idx, k=50)
            print("Top 10 attention heads:")
            for item in top_10_heads:
                head_count[item[0]] = head_count.get(item[0], 0) + 1
                print(item)
            # Explicitly delete attention and tokens to free up memory
            del attention
            del tokens
            gc.collect()  # Explicitly invoke garbage collector
            torch.cuda.empty_cache()
            print("Memory freed")

        # sort descending for each question
        layer_count = dict(sorted(layer_count.items(), key=lambda item: item[1], reverse=True))
        head_count = dict(sorted(head_count.items(), key=lambda item: item[1], reverse=True))
        data["head_stats"][dataset][question] = head_count
        data["layer_stats"][dataset][question] = layer_count
        print(f"==========Popular layers and heads for {dataset} set with question: {question} with {cnt} success entries==========")
        print("Layer count:")
        for key, value in layer_count.items():
            print(key, value)
        print("Head count:")
        for key, value in head_count.items():
            print(key, value)
    print(f"Done with {dataset} set and {len(questions)} questions")
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":

    dataset = "notice"
    # dataset = "paper"
    run_reverse_engineer(dataset)


