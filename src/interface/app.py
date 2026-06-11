import streamlit as st
import cv2
import numpy as np

from kl import KLT

st.set_page_config(layout="wide")

st.title("Transformée de Karhunen-Loève")

uploaded = st.file_uploader(
    "Choisir une image",
    type=["pgm", "png", "jpg", "jpeg"]
)

if uploaded is None:
    st.info("Importer une image pour commencer")
    st.stop()

# Lecture image uploadée
file_bytes = np.asarray(
    bytearray(uploaded.read()),
    dtype=np.uint8
)

img = cv2.imdecode(
    file_bytes,
    cv2.IMREAD_GRAYSCALE
)

# Redimensionnement
img = cv2.resize(
    img,
    (512, 512)
)

# Fenêtre de sélection de la taille des blocs
size = st.selectbox(
    "Taille des blocs",
    [8, 16, 32]
)

@st.cache_resource
def init_klt(image, size):
    return KLT(image, size)

algo = init_klt(img, size)

# Curseur pour chosir le nombre de coefficients
k = st.slider(
    "Nombre de coefficients gardés",
    1,
    size * size,
    size * size
)

# Application de la transformée inverse pour reconstruire l'image
image_rec = algo.reconstruire(k)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image originale")
    st.image(
        img,
        clamp=True
    )

with col2:
    st.subheader(f"Image reconstruite - k={k}")
    st.image(
        image_rec,
        clamp=True
    )