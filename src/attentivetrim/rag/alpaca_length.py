import json

# Path to the JSON file
json_file_path = '/Users/chunwei/research/llm-scheduling/profiling-Meta-Llama-3-8B-Instruct.json'


# Function to extract the required information
def extract_info(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # List to hold the extracted information
    extracted_info = []

    # Iterate through each record in the JSON data
    for record in data['records']:
        # Extract record_id
        record_id = record.get('record_id', None)

        # Extract prompt from the output field
        output = record.get('output', '')
        prompt_start = output.find('prompt=') + 10
        prompt_end = output.find('### Response:', prompt_start)
        prompt = output[prompt_start:prompt_end] if prompt_start != 1 and prompt_end != -1 else None

        # Extract iteration_count if available
        iteration_count = record.get('iteration_count', None)

        # Append the extracted information to the list
        extracted_info.append({
            'record_id': record_id,
            'prompt': prompt,
            'iteration_count': iteration_count
        })

    return extracted_info


# Call the function and print the results
info = extract_info(json_file_path)
#  save the extracted information to a new JSON file
output_file_path = '/Users/chunwei/research/llm-scheduling/profiling-Meta-Llama-3-8B-Instruct-length.json'
with open(output_file_path, 'w') as json_file:
    json.dump({'records': info, 'total': len(info)}, json_file, indent=4)

