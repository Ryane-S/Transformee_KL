import streamlit as st
import cv2

from kl import KLT

# Initialisation de l'interface web
st.title("Transformée de Karhunen-Loève")

img = cv2.imread(
    "img/Baboon512.pgm",
    cv2.IMREAD_GRAYSCALE
)

size = 8

@st.cache_resource
def init():

    return KLT(
        img,
        size=size
    )

algo = init()

# Ajout d'un curseur pour régler le nombre de coefficients à conserver
k = st.slider(
    "Nombre de coefficients gardés",
    1,
    size**2,
    size**2
)

# Reconstruction de l'image par transformée inverse
image_rec = (
    algo
    .reconstruire(k)
)

c1, c2 = st.columns(2)

with c1:
    st.image(
        img,
        caption="Originale"
    )

with c2:
    st.image(
        image_rec,
        caption=f"k = {k}"
    )