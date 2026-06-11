# Transformée de Karhunen-Loève pour la compression d’images

Implémentation de la transformée de Karhunen-Loève (KLT/KL) pour la compression d’images en niveaux de gris.

## Objectifs

* Partition de l’image en blocs 8×8 et 16×16
* Calcul des matrices de covariance
* Diagonalisation et calcul des vecteurs propres
* Transformée KL directe et inverse
* Compression par annulation des composantes de faible énergie
* Reconstruction et évaluation visuelle de l’image
* Interface interactive pour visualiser l’impact du nombre de coefficients conservés

---

## Pipeline

1. Découpage de l’image en blocs
2. Vectorisation des blocs
3. Calcul de la matrice de covariance
4. Calcul des valeurs et vecteurs propres
5. Projection dans la base KL
6. Suppression des coefficients associés aux plus petites valeurs propres
7. Reconstruction de l’image

---

## Structure

```text
.
├── src/
│   ├── transforme.py      # Implémentation de la transformée KL
│   └── interface/
│       ├── app.py         # Interface Streamlit
│       └── klt.py         # Pipeline KLT utilisé par l'interface
│
├── img/                   # Images de test
├── results/               # Images reconstruites
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Guide d'installation

## Prérequis

* Python 3.12+
* uv

Installer `uv` :

```bash
pip install uv
```

## Installer le projet

Depuis la racine du dépôt :

```bash
uv sync
```

Cette commande crée automatiquement l’environnement virtuel et installe toutes les dépendances définies dans `pyproject.toml`.

---

# Utilisation

## Exécution classique

1. Placer une image dans le dossier `img`
2. Lancer le script :

```bash
uv run python src/kl_transform.py
```

3. Le résultat sera sauvegardé dans `results/`

---

## Interface interactive

L’interface permet de modifier dynamiquement le nombre de coefficients conservés (`k`) ainsi que la taille des blocs (8x8 ou 16x16) et d’observer l’impact sur l’image reconstruite.

Lancer :

```bash
uv run streamlit run src/interface/app.py
```
