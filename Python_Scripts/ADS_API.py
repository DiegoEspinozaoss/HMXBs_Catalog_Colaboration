import os
import re

import ads
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
ADS_API_TOKEN = os.getenv("ADS_API_TOKEN")
ads.config.token = ADS_API_TOKEN
PAPERS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Papers'))
os.makedirs(PAPERS_DIR, exist_ok=True)

authors = {
    "Fortin, F.",
    "Kim, V.",
    "Malacaria, C.",
    "Neumann, M."
}

author_query = " OR ".join(f"\"{a}\"" for a in authors)

N_documents = 10

query_str = (
    f"author:({author_query}) "
    "AND full:(HMXB AND catalog AND binaries)"
)

results = list(ads.SearchQuery(q=query_str, fl=['title', 'bibcode', 'author'], rows=N_documents))

for r in results:
    print(f"{r.title[0]} ({r.author[0]}) - {r.bibcode}")


[os.remove(os.path.join('..', 'Papers', f)) for f in os.listdir(os.path.join('..', 'Papers')) if os.path.isfile(os.path.join('..', 'Papers', f))]


import re

def clean_title_for_filename(raw_title):
    pattern = r"(.*?)\d{4}[A-Z&\.]+[\.]{3}[\dA-Z\.]+$"
    match = re.match(pattern, raw_title)
    if match:
        return match.group(1).rstrip(" _-")
    else:
        return raw_title

with tqdm(total=len(results), desc="Descargando artículos", unit="pdf") as pbar:
    for article in results:
        bibcode = article.bibcode
        raw_title = article.title[0]
        clean_title = clean_title_for_filename(raw_title)
        filename_title = clean_title.replace(" ", "_").replace("/", "-")[:100]
        filename = f"{filename_title}_{bibcode}.pdf"
        output_path = os.path.join(PAPERS_DIR, filename)

        try:
            url = f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/EPRINT_PDF"
            response = requests.get(url)

            if response.status_code == 200 and response.headers.get('Content-Type', '').lower() == 'application/pdf':
                with open(output_path, 'wb') as f:
                    f.write(response.content)
            else:
                url_pub = f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/PUB_PDF"
                response_pub = requests.get(url_pub)

                if response_pub.status_code == 200 and response_pub.headers.get('Content-Type', '').lower() == 'application/pdf':
                    with open(output_path, 'wb') as f:
                        f.write(response_pub.content)
                    tqdm.write(f"✔ Descargado desde publisher: {filename}")

        except Exception as e:
            tqdm.write(f"⚠ Error descargando {bibcode}: {e}")

        pbar.update(1)

#%%
folder = "../Papers"

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        old_path = os.path.join(folder, filename)
        basename, ext = os.path.splitext(filename)

        new_basename = re.sub(r'_(20\d{2}).*$', '', basename)

        if new_basename != basename:
            new_filename = new_basename + ext
            new_path = os.path.join(folder, new_filename)
            os.rename(old_path, new_path)
