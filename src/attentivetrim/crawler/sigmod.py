import os
import requests
import random
import time
from bs4 import BeautifulSoup



# iter through all the html files: PACMMOD_Vol1No1.html, PACMMOD_Vol1No2.html,
# PACMMOD_Vol1No3.html, PACMMOD_Vol1No4.html, PACMMOD_Vol2No1.html, PACMMOD_Vol2No2.html
volumn_files = ["PACMMOD_Vol1No1.html", "PACMMOD_Vol1No2.html", "PACMMOD_Vol1No3.html", "PACMMOD_Vol1No4.html", "PACMMOD_Vol2No1.html", "PACMMOD_Vol2No2.html"]
volume_year = [2023, 2023, 2023, 2023, 2024, 2024]

for idx in range(2,len(volumn_files)):
    i = len(volumn_files) - 1 - idx
    # Define the directory where PDFs will be saved
    DIR = '/Users/chunwei/sigmod/sigmod' + str(volume_year[i])

    # Ensure the directory exists
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    html_file = "/Users/chunwei/Downloads/" + volumn_files[i]

    # Sample HTML content
    html_content = open(html_file).read()

    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all divs that potentially contain a research article
    issue_items = soup.find_all('div', class_='issue-item')

    for item in issue_items:
        # Check if the text 'research-article' is in the issue heading
        issue_heading = item.find('div', class_='issue-heading')
        if issue_heading and 'research-article' in issue_heading.get_text().lower():
            # Find the PDF link
            pdf_link = item.find('a', attrs={'aria-label': 'PDF'})
            if pdf_link:
                pdf_url = pdf_link['href']
                print(f"Downloading PDF from: {pdf_url}")

                pdf_filename = os.path.join(DIR, str(volume_year[i]) + "_" + pdf_url.split('/')[-1] + '.pdf')
                if os.path.exists(pdf_filename):
                    print(f"PDF already exists: {pdf_filename}")
                    continue

                try:
                    # Download the PDF file
                    response = requests.get(pdf_url)
                    if response.status_code == 200:
                        with open(pdf_filename, 'wb') as f:
                            f.write(response.content)
                        print(f"PDF saved as: {pdf_filename}")
                    else:
                        print(f"Failed to download PDF from: {pdf_url}")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download PDF from: {pdf_url}. Error: {e}")


                # Random delay between 10 to 40 seconds
                delay = random.randint(10, 40)
                print(f"Waiting for {delay} seconds before next download...")
                time.sleep(delay)
            else:
                print("No PDF link found for this article.")

print("All PDFs downloaded successfully!")