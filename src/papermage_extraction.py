import os
import json

from papermage.recipes import CoreRecipe


def process_pdf_directory(directory_path):
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # Iterate through all files in the directory with sorted order
    for filename in sorted(os.listdir(directory_path)):
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Processing {pdf_path}...")


            json_filename = filename.replace('.pdf', '_pm.json')
            json_path = os.path.join(directory_path, json_filename)
            if os.path.exists(json_path):
                print(f"Skipping {pdf_path} as {json_path} already exists.")
                continue

            # Initialize your CoreRecipe or similar class
            recipe = CoreRecipe()

            # Process the PDF file
            try:
                doc = recipe.from_pdf(pdf=pdf_path)
            except Exception as e:
                print(f"Failed to process {pdf_path}: {e}")
                continue

            # Convert the extracted information to JSON
            doc_json = json.dumps(doc.to_json())

            # Save the JSON to a file

            with open(json_path, 'w') as json_file:
                json_file.write(doc_json)

            print(f"Saved extracted data to {json_path}")

# Example usage
directory_path = '/Users/chunwei/sigmod/sigmod2024'
process_pdf_directory(directory_path)