import streamlit as st
import os
import sys
sys.path.append('./Python_Scripts')  # añade esta línea al inicio

from Load_Data import load_catalogs

# Título principal
st.title("HMXB Catalog Explorer")

# Subtítulo
st.subheader("Explore images and data from the Milky Way catalogs")

# Mostrar imágenes desde la carpeta "Images"
st.header("📸 Galactic Images")

# Listar y mostrar imágenes
image_folder = "Images"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for img in image_files:
    st.image(os.path.join(image_folder, img), caption=img, use_container_width=True)

# Explorar archivos de datasets (opcional)
st.header("📂 Available Catalogs")
dataset_folder = "Datasets"
dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith('.xlsx')]

selected_dataset = st.selectbox("Choose a dataset to preview", dataset_files)

if selected_dataset:
    import pandas as pd
    df = pd.read_excel(os.path.join(dataset_folder, selected_dataset))
    st.write(f"Preview of `{selected_dataset}`:")
    st.dataframe(df.head(10))
