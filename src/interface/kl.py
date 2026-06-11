import numpy as np

class KLT:
    def __init__(self, img, size=8):
        '''Constructeur'''

        self.size = size
        self.img = img

        self.partition()
        self.vectorisation()
        self.covariance()
        self.diagonalisation()
        self.transformee()

    def partition(self):
        '''Partition de l'image en blocs de taille size*size'''

        h, w = self.img.shape

        self.blocks = (
            self.img
            .reshape(h // self.size, self.size,
                     w // self.size, self.size)
            .transpose(0, 2, 1, 3)
            .reshape(-1, self.size, self.size)
        )

    def vectorisation(self):
        '''Transformation des blocs en vecteurs'''

        self.vecteurs = (
            self.blocks
            .reshape(-1, self.size * self.size)
            .astype(np.float64)
        )

    def covariance(self):
        '''Calcul de la matrice de covariance'''

        self.moyenne = np.mean(
            self.vecteurs,
            axis=0
        )

        self.vecteurs_centres = (
            self.vecteurs
            - self.moyenne
        )

        self.cov = np.cov(
            self.vecteurs_centres,
            rowvar=False
        )

    def diagonalisation(self):
        '''Diagonalisation de la matrice de covariance'''

        valeurs, vecteurs = (
            np.linalg.eigh(self.cov)
        )

        idx = np.argsort(valeurs)

        self.FI = vecteurs[:, idx]

    def transformee(self):
        '''Application de la transformée directe'''

        self.Y = (
            self.vecteurs_centres
            @ self.FI
        )

    def reconstruire(self, k):
        '''Application de la transformée inverse et reconstruction de l'image'''

        nc = self.size**2 - k

        Y_mod = self.Y.copy()

        Y_mod[:, :nc] = 0

        vecteurs_rec = (
            Y_mod
            @ self.FI.T
            + self.moyenne
        )

        h, w = self.img.shape

        image = (
            vecteurs_rec
            .reshape(
                h // self.size,
                w // self.size,
                self.size,
                self.size
            )
            .transpose(0, 2, 1, 3)
            .reshape(h, w)
        )

        return (
            np.clip(image, 0, 255)
            .astype(np.uint8)
        )