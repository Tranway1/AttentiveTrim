import os
import requests
import random
import time
from bs4 import BeautifulSoup

DIR = '/Users/chunwei/arxiv/0625_2000'

html_file = "/Users/chunwei/Downloads/ComputerScience.html"

# Sample HTML content
html_content = open(html_file).read()

# Directory to save the downloaded PDFs
save_directory = DIR
os.makedirs(save_directory, exist_ok=True)

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Find all PDF links
pdf_links = soup.find_all('a', title="Download PDF")
cnt = 0
# Download each PDF found
for link in pdf_links:
    pdf_url = link['href']
    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            pdf_filename = os.path.join(save_directory, pdf_url.split('/')[-1] + ".pdf")
            with open(pdf_filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {cnt}th paper with {pdf_filename}")
        else:
            print(f"Failed to download PDF from {pdf_url}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF from {pdf_url}. Error: {e}")
    cnt += 1
    if cnt >=600:
        break
    # Random delay between 10 to 40 seconds
    delay = random.randint(5, 20)
    print(f"Waiting for {delay} seconds before next download...")
    time.sleep(delay)
