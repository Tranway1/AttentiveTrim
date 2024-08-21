import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

# Define the CSV file path
csv_file_path = '/Users/chunwei/Downloads/cases_initiated_NOPV_7_15_2024.csv'

# Read the CSV file using pandas
df = pd.read_csv(csv_file_path)

# Directory to save PDF files
pdf_directory = 'downloaded_pdfs'
os.makedirs(pdf_directory, exist_ok=True)


# Function to download PDF
def download_pdf(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")


# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    case_number = row['Case Number']
    url = f"https://primis.phmsa.dot.gov/enforcement-data/case/{case_number}"

    # Fetch the webpage
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the first PDF link
        pdf_link = soup.find('a', href=lambda href: href and ".pdf" in href)
        if pdf_link:
            pdf_url = pdf_link['href']
            if not pdf_url.startswith('http'):
                pdf_url = f"https://primis.phmsa.dot.gov{pdf_url}"

            # Download the PDF
            pdf_filename = os.path.join(pdf_directory, pdf_url.split('/')[-1])
            download_pdf(pdf_url, pdf_filename)
        else:
            print(f"No PDF found for case {case_number}")
    else:
        print(f"Failed to fetch webpage for case {case_number}")
