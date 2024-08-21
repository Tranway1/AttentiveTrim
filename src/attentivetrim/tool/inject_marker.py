import json
import re


def add_sentence_counter(text):
    # Split text on potential sentence-ending punctuation followed by a space and a capital letter
    potential_sentences = re.split(r'([.!?])\s+(?=[A-Z])', text)

    # Reassemble sentences considering abbreviations and initials
    sentences = []
    sentence = ''
    for part in potential_sentences:
        if sentence and part in '.!?':
            sentence += part
        elif sentence:
            # Check if the last part ends with a known abbreviation or initial
            if re.search(r'\b(?:Mr|Mrs|Dr|Ms|Prof|St|Lt|Col|Sr|Jr| [A-Z])\.$', sentence.strip()):
                sentence += ' ' + part
            else:
                sentences.append(sentence.strip())
                sentence = part
        else:
            sentence = part

    if sentence:
        sentences.append(sentence.strip())

    # Initialize an empty list to store sentences with counters
    numbered_sentences = []

    # Loop through each sentence and add a counter
    for i, sentence in enumerate(sentences, 1):
        # Format the sentence with a counter prefix
        numbered_sentence = f"S{i}: {sentence}"
        numbered_sentences.append(numbered_sentence)

    # Join all the numbered sentences into a single string with new lines
    result_text = "\n".join(numbered_sentences)
    return result_text

list_file = "../data/test_v16_inputfile100.txt"
with open(list_file) as f:
    list_of_files = f.readlines()
list_of_files = [x.strip() for x in list_of_files]

for file in list_of_files:
    with open('/Users/chunwei/pvldb_1-16/16/' + file) as f_in:
        doc_dict = json.load(f_in)
    context = doc_dict["symbols"]
    result = add_sentence_counter(context)
    output_file = file.replace('.json', '_numbered.json')
    with open('/Users/chunwei/pvldb_1-16/16/' + output_file, 'w') as f_out:
        json.dump({"symbols": result}, f_out, indent=4)


