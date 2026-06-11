import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import entropy

# Paramètres
img_path = "img/Baboon512.pgm"
save_path = "results/" + img_path.split("/")[-1]
size = 8
k = 20  # nombre de coefficients gardés

# Chargement image et Visualisation
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))

plt.figure()
plt.imshow(img, cmap="gray")
plt.colorbar()
plt.title("Image originale")


def calculer_entropie(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return entropy(p, base=2)


print("Entropie originale :", calculer_entropie(img))

# Découpage en blocs
h, w = img.shape

blocks = (
    img
    .reshape(h // size, size, w // size, size)
    .transpose(0, 2, 1, 3)
    .reshape(-1, size, size)
)

# Vectorisation
vecteurs = blocks.reshape(-1, size * size).astype(np.float64)

print("Vecteurs :", vecteurs.shape)

# Centrage
moyenne = np.mean(vecteurs, axis=0)
vecteurs_centres = vecteurs - moyenne

# Covariance
cov = np.cov(vecteurs_centres, rowvar=False)

# Décomposition spectrale
valeurs_propres, vecteurs_propres = np.linalg.eigh(cov)

# Tri croissant (énoncé)
idx = np.argsort(valeurs_propres)

valeurs_propres = valeurs_propres[idx]
FI = vecteurs_propres[:, idx]

print("FI :", FI.shape)

# Transformée KL
transformee_KL = vecteurs_centres @ FI

# Compression
nc = size * size - k

Y_mod = transformee_KL.copy()
Y_mod[:, :nc] = 0

print(f"Compression : {size*size} → {k}")
print(f"Coefficients annulés : {nc}")

# Transformée inverse
vecteurs_rec = Y_mod @ FI.T + moyenne

# Reconstruction image
image_rec = (
    vecteurs_rec
    .reshape(h // size, w // size, size, size)
    .transpose(0, 2, 1, 3)
    .reshape(h, w)
)

image_rec = np.clip(image_rec, 0, 255).astype(np.uint8)

print("Entropie reconstruite :", calculer_entropie(image_rec))

# Sauvegarde
cv2.imwrite(save_path, image_rec)

# Affichage
plt.figure()
plt.imshow(image_rec, cmap="gray")
plt.colorbar()
plt.title(f"Image reconstruite (k={k})")

plt.show()