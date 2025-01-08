import os
import cv2
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

class BayesianClassifier:
    def __init__(self, mode="page"):
        self.feature_means = {}
        self.feature_variances = {}
        self.class_priors = {}
        self.classes = []

        # Définir les classes autorisées selon le mode choisi
        self.allowed_classes = (
            ['Figure1', 'Figure2', 'Figure3', 'Figure4', 'Figure5', 'Figure6']
            if mode == "plan"
            else ['2', 'd', 'I', 'n', 'o', 'u']
        )

        # Initialisation du descripteur HOG avec des paramètres standards
        self.hog = cv2.HOGDescriptor(
            _winSize=(28, 28),
            _blockSize=(8, 8),
            _blockStride=(4, 4),
            _cellSize=(8, 8),
            _nbins=9,
        )

    def extract_features(self, image):
        """Extraire les caractéristiques d'une image donnée."""
        try:
            # Convertir en niveaux de gris si l'image est en couleurs
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Appliquer un seuillage adaptatif pour la segmentation
            binary_image = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Trouver les contours
            contours, _ = cv2.findContours(
                binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                print("Aucun contour trouvé.")
                return np.array([])

            features = []
            for contour in contours:
                if cv2.contourArea(contour) < 20:  # Filtrer les petites zones
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                letter_image = gray_image[y:y + h, x:x + w]
                letter_image = cv2.resize(letter_image, (28, 28))

                # Calculer les descripteurs HOG
                hog_features = self.hog.compute(letter_image)
                features.append(hog_features.flatten())

            features = np.array(features)
            if features.size == 0:
                return np.array([])

            # Normaliser les caractéristiques
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.where(norms > 1e-6, norms, 1)

            return features
        except Exception as e:
            print(f"Erreur dans l'extraction des caractéristiques : {e}")
            return np.array([])

    def train(self, dataset_path):
        """Entraîner le modèle Bayésien à partir d'un ensemble de données."""
        class_features = defaultdict(list)
        total_samples = 0

        for class_name in os.listdir(dataset_path):
            if class_name not in self.allowed_classes:
                continue

            class_folder_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_folder_path):
                if class_name not in self.classes:
                    self.classes.append(class_name)

                for img_name in os.listdir(class_folder_path):
                    img_path = os.path.join(class_folder_path, img_name)
                    if os.path.isfile(img_path):
                        image = cv2.imread(img_path)
                        if image is not None:
                            features = self.extract_features(image)
                            if features.size > 0:
                                for feature in features:
                                    class_features[class_name].append(feature)
                                total_samples += len(features)
                            else:
                                print(f"Aucune caractéristique extraite pour {img_path}")
                        else:
                            print(f"Échec du chargement de l'image : {img_path}")

        # Calculer les moyennes, variances et probabilités a priori
        for class_name in self.classes:
            if class_name in class_features:
                features = np.array(class_features[class_name])
                self.feature_means[class_name] = np.mean(features, axis=0)
                self.feature_variances[class_name] = np.var(features, axis=0) + 1e-6  # Éviter une variance nulle
                self.class_priors[class_name] = len(features) / total_samples

        print("Entraînement terminé pour les classes :", self.classes)

    def save_model(self, model_path):
        """Sauvegarder le modèle Bayésien sur le disque."""
        model_data = {
            "feature_means": self.feature_means,
            "feature_variances": self.feature_variances,
            "class_priors": self.class_priors,
            "classes": self.classes,
        }
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model_data, model_path)
        print(f"Modèle sauvegardé à l'emplacement {model_path}")

    def load_model(self, model_path):
        """Charger un modèle Bayésien sauvegardé."""
        if os.path.exists(model_path):
            model_data = torch.load(model_path)
            self.feature_means = model_data["feature_means"]
            self.feature_variances = model_data["feature_variances"]
            self.class_priors = model_data["class_priors"]
            self.classes = model_data["classes"]
            print(f"Modèle chargé depuis {model_path}")
        else:
            print(f"Modèle introuvable à l'emplacement {model_path}.")

    def predict(self, image, threshold=0.3):
        """Prédire la classe d'une image donnée."""
        try:
            features = self.extract_features(image)
            if features.size == 0:
                return None

            posteriors = {}
            for class_name in self.classes:
                mean = self.feature_means[class_name]
                variance = self.feature_variances[class_name]
                prior = self.class_priors[class_name]

                # Calculer la log-vraisemblance
                log_likelihood = -0.5 * np.sum(
                    ((features - mean) ** 2) / variance + np.log(2 * np.pi * variance),
                    axis=1,
                )
                posterior = log_likelihood + np.log(prior)
                posteriors[class_name] = np.sum(posterior)

            max_class = max(posteriors, key=posteriors.get)
            max_posterior = posteriors[max_class]

            if max_posterior < threshold:
                return None
            return max_class
        except Exception as e:
            print(f"Erreur dans la prédiction : {e}")
            return None

    def visualize(self):
        """Visualiser les moyennes des caractéristiques pour chaque classe."""
        if not self.classes:
            print("Aucune classe à visualiser.")
            return

        for class_name in self.classes:
            mean_features = self.feature_means[class_name]

            plt.figure(figsize=(10, 4))
            plt.title(f"Moyennes des caractéristiques pour la classe : {class_name}")
            plt.plot(mean_features)
            plt.xlabel("Indice des caractéristiques")
            plt.ylabel("Valeur moyenne")
            plt.grid(True)
            plt.show()
