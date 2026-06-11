# Transformée de Karhunen-Loève pour la compression d’images

Implémentation de la transformée de Karhunen-Loève (KLT/KL) pour la compression d’images en niveaux de gris.

## Objectifs
- Partition de l’image en blocs 8×8 et 16×16
- Calcul des matrices de covariance
- Diagonalisation et calcul des vecteurs propres
- Transformée KL directe et inverse
- Compression par annulation des composantes de faible énergie
- Reconstruction et évaluation visuelle de l’image

## Pipeline
1. Découpage de l’image en blocs
2. Vectorisation des blocs
3. Calcul de la matrice de covariance
4. Calcul des valeurs/vecteurs propres
5. Projection dans la base KL
6. Suppression des coefficients associés aux plus petites valeurs propres
7. Reconstruction de l’image

## Structure
* src/        # Implémentation
* images/     # Images de test
* results/    # Images reconstruites et résultats

# Guide d'installation

## Prérequis

* Python 3.10+
* uv

Installation de `uv` :
```bash
pip install uv
```

## Créer un environnement virtuel
```bash
uv venv
```
* Installer les dépendances du projet :
```bash
uv sync
```

## Lancer le projet
* Placer votre image dans le dossier `img`
* Appliquer la transformée de KL sur l'image :
```bash
uv run python src/transforme.py
```
* Visualiser le résultat dans le dossier `results`