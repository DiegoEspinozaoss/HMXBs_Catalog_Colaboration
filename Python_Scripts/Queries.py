#%%
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
#%%
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
            tqdm.write(f"Error descargando {bibcode}: {e}")

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

#%%
from Load_Data import load_catalogs
data=load_catalogs()
cat_fortin=data['cat_fortin']
names=cat_fortin['Main_ID']
#%%

for index in range(len(names)):
    N_documents = 1
    name=names[index]
    query_str = (
        f'full:("{name}") AND full:catalog'#("Magellanic Cloud" OR "Milky Way")'
    )

    results = list(ads.SearchQuery(
        q=query_str,
        fl=['title', 'bibcode', 'author'],
        rows=N_documents
    ))

    for r in results:
        print(f"{r.title[0]} ({r.author[0]}) - {r.bibcode}")
        
        
        
#%%
from ads import SearchQuery

N_documents = 100  
author_name = "El Mellah"#, Ileyk"
query_str = f'author:"{author_name}" AND full catalog'

results = list(SearchQuery(
    q=query_str,
    fl=['title', 'bibcode', 'author'],
    rows=N_documents
))

for r in results:
    title = r.title[0] if r.title else "Sin título"
    first_author = r.author[0] if r.author else "Autor desconocido"
    print(f"{title} ({first_author}) - {r.bibcode}")










































































































































































########################################################################################
from Load_Data import load_catalogs
from lightkurve import search_lightcurve
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import matplotlib.dates as mdates
import datetime

import numpy as np
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve

def plot_all_lightcurves_and_periodogram(cat_fortin, system_index=0):
    title_size = 16
    label_size = 14
    line_width = 1.2
    color = '#003366'

    FIRST = cat_fortin['Main_ID'][system_index]

    search_result = search_lightcurve(FIRST)
    if len(search_result) == 0:
        print(f"No se encontraron curvas de luz para {FIRST}")
        return

    curves = []
    for i, lc_search in enumerate(search_result):
        lc = lc_search.download()
        if lc is not None:
            curves.append((i, lc_search, lc))
        else:
            print(f"No se pudo descargar la curva de luz #{i+1} para {FIRST}")

    if len(curves) == 0:
        print(f"No se pudo descargar ninguna curva de luz para {FIRST}.")
        return

    last_index, last_lc_search, last_lc = max(curves, key=lambda x: np.max(x[2].time.value))
    time_days = last_lc.time.value
    flux = last_lc.flux.value
    mission_name = last_lc_search.mission[0] if isinstance(last_lc_search.mission, (list, tuple)) else last_lc_search.mission

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(time_days, flux, lw=line_width, alpha=0.7, label=f"Misión: {mission_name}")
    ax1.set_title(f"Última curva de luz de {FIRST}", fontsize=title_size)
    ax1.set_xlabel("Tiempo (días, BJD - 2450000)", fontsize=label_size)
    ax1.set_ylabel("Flujo", fontsize=label_size)
    ax1.legend()
    plt.tight_layout()
    plt.show()


    
catalogs = load_catalogs()
fortin = catalogs['cat_fortin']
for i in range(len(fortin)):
    plot_all_lightcurves_and_periodogram(fortin, system_index=i)


#%%
# first_catalog_key = 'V/106/hmxbcat'
from astroquery.vizier import Vizier
from tqdm import tqdm
import pandas as pd

def get_catalog_base(full_key: str) -> str:
    parts = full_key.split('/')
    if len(parts) >= 4:
        #J/ApJS/154/585/table4  -->  J/ApJS/154/585
        return "/".join(parts[:3])
    elif len(parts) == 3:
        #IX/43/sources  -->  IX/43
        return "/".join(parts[:2])
    elif len(parts) == 2:
        #IX/58
        return full_key
    else:
        return full_key 

NAME = "4U 0115+63"

results = Vizier.query_object(NAME)
print("Catálogos encontrados:")
print(results)

catalogs_detected = []
abstracts_detected = []
rows_detected = []

for index in tqdm(range(len(results)), desc="Revisando catálogos"):
    full_catalog_key = str(list(results.keys())[index])
    
    if full_catalog_key.startswith(('METAobj', 'ReadMeObj')):
        continue
    
    catalog_base = get_catalog_base(full_catalog_key) 
    
    try:
        viz = Vizier(catalog=catalog_base)
        metadata = viz.get_catalog_metadata()
        description = metadata['abstract'][0]
        
        if ('X-ray' in description) or ('binaries' in description):
            Vizier.ROW_LIMIT = -1
            query = Vizier(columns=['*'])
            object_results = query.query_object(NAME, catalog=full_catalog_key)
            
            if len(object_results) > 0:
                table = object_results[0]
                df = table.to_pandas()
                
                if not df.empty:
                    print(f"\nCatálogo: {full_catalog_key}")
                    print(f"Abstracto: {description}")
                    print(f"Filas encontradas para {NAME}:")
                    print(df)
                    
                    catalogs_detected.append(full_catalog_key)
                    abstracts_detected.append(description)
                    rows_detected.append(df)
                    
    except Exception as e:
        print(f"Error al obtener metadatos o datos para {full_catalog_key}: {e}")

print("\nResumen:")
print(f"Catálogos detectados: {catalogs_detected}")
print(f"Abstracts detectados: {abstracts_detected}")
print(f"Filas detectadas por catálogo (como DataFrames):")
for i, df in enumerate(rows_detected):
    print(f"\nCatalogo: {catalogs_detected[i]}")
    print(df.head())

#%%
# number = 0 

# if catalogs_detected:
#     print(f"\nCatálogo seleccionado: {catalogs_detected[number]}")
#     print(f"Abstracto: {abstracts_detected[number]}")
    
#     catalog_name = catalogs_detected[number]
#     catalog_tables = rows_detected[catalog_name]
    
#     print("\nSubtablas disponibles en el catálogo:")
#     print(catalog_tables.keys())
    
#     first_table_key = list(catalog_tables.keys())[0]
#     print(f"\nNombre de la primera tabla: {first_table_key}")
    
#     table = catalog_tables[first_table_key]
#     df = table.to_pandas()
    
#     print("\nPrimeras filas del DataFrame:")
#     print(df.head())
    
#     NAME = '4U 0115+63'
#     mask = df.apply(lambda row: row.astype(str).str.contains(NAME.replace('+', '\\+'), case=False, regex=True).any(), axis=1)
#     df_filtered = df[mask]
    
#     if not df_filtered.empty:
#         print(f"\nFilas filtradas que contienen '{NAME}':")
#         print(df_filtered)
#     else:
#         print(f"\nNo se encontraron filas con '{NAME}' en la tabla.")
# else:
#     print("\nNo hay catálogos con 'X-ray' ni 'binaries' encontrados.")


