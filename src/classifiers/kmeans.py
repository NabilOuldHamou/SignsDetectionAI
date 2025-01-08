import numpy as np
import os

class KMeansClassifier:
    def __init__(self, num_clusters=6, max_iter=100, tol=1e-4):
        """
        Initialiser le classifieur KMeans.

        Paramètres :
        - num_clusters : Nombre de clusters (classes).
        - max_iter : Nombre maximal d'itérations pour l'algorithme k-means.
        - tol : Tolérance pour la convergence.
        """
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, features):
        """
        Entraîner le modèle k-means sur les données fournies.

        Paramètres :
        - features : Un tableau numpy de forme (n_samples, n_features).
        """
        if len(features) < self.num_clusters:
            raise ValueError("Le nombre d'échantillons est inférieur au nombre de clusters.")

        np.random.seed(42)
        random_indices = np.random.choice(len(features), self.num_clusters, replace=False)
        self.cluster_centers_ = features[random_indices]

        for iteration in range(self.max_iter):
            # Assigner des étiquettes en fonction du centre le plus proche
            distances = self._compute_distances(features)
            self.labels_ = np.argmin(distances, axis=1)

            # Mettre à jour les centres des clusters
            new_centers = np.array([features[self.labels_ == k].mean(axis=0) for k in range(self.num_clusters)])

            # Vérifier la convergence
            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tol):
                print(f"Convergence atteinte en {iteration + 1} itérations.")
                break

            self.cluster_centers_ = new_centers

    def predict(self, features):
        """
        Prédire le cluster le plus proche pour les données fournies.

        Paramètres :
        - features : Un tableau numpy de forme (n_samples, n_features).

        Retourne :
        - Les étiquettes des clusters pour chaque échantillon.
        """
        distances = self._compute_distances(features)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, features):
        """
        Calculer les distances entre les données et les centres des clusters.

        Paramètres :
        - features : Un tableau numpy de forme (n_samples, n_features).

        Retourne :
        - Un tableau numpy des distances de forme (n_samples, num_clusters).
        """
        return np.linalg.norm(features[:, np.newaxis] - self.cluster_centers_, axis=2)

    def save_model(self, path):
        """
        Sauvegarder le modèle KMeans dans un fichier .npy.

        Paramètres :
        - path : Chemin pour sauvegarder le modèle.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Rien à sauvegarder.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.cluster_centers_)
        print(f"Modèle KMeans sauvegardé à l'emplacement {path}")

    def load_model(self, path):
        """
        Charger le modèle KMeans depuis un fichier .npy.

        Paramètres :
        - path : Chemin vers le fichier sauvegardé.
        """
        if os.path.exists(path):
            self.cluster_centers_ = np.load(path)
            print(f"Modèle KMeans chargé depuis {path}")
        else:
            raise FileNotFoundError(f"Fichier du modèle KMeans introuvable à l'emplacement {path}.")
