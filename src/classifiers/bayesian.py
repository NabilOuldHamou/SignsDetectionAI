import os
import cv2
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

class BayesianClassifier:
    def __init__(self):
        """
        Initialisation du classificateur Bayésien avec les paramètres nécessaires.
        """
        self.feature_means = {}  # Moyennes des caractéristiques pour chaque classe
        self.feature_variances = {}  # Variances des caractéristiques pour chaque classe
        self.class_priors = {}  # Probabilités a priori pour chaque classe
        self.classes = []  # Liste des classes disponibles

    def extract_features(self, image):
        """
        Extraire les caractéristiques d'une image donnée (Histogramme des Gradients Orientés - HOG).

        :param image: Image en entrée
        :return: Tableau des caractéristiques normalisées
        """
        # Conversion de l'image en niveaux de gris si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Binarisation de l'image
        binary_image = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Extraction des contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = []
        for contour in contours:
            if cv2.contourArea(contour) < 22:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            letter_image = gray_image[y:y + height, x:x + width]
            letter_image = cv2.resize(letter_image, (28, 28))

            # Calcul des caractéristiques HOG
            hog = cv2.HOGDescriptor()
            hog_features = hog.compute(letter_image)

            features.append(hog_features.flatten())

        features = np.array(features)

        # Normalisation des caractéristiques
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.where(norms > 1e-6, norms, 1)

        return features

    def train(self, dataset_path):
        """
        Entraîner le classificateur avec un catalogue d'images organisées par classe.

        :param dataset_path: Chemin vers le dossier contenant les images classées
        """
        class_features = defaultdict(list)
        total_images = 0

        for class_name in os.listdir(dataset_path):
            class_folder_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_folder_path):
                if class_name not in self.classes:
                    self.classes.append(class_name)

                for image_name in os.listdir(class_folder_path):
                    image_path = os.path.join(class_folder_path, image_name)
                    if os.path.isfile(image_path):
                        image = cv2.imread(image_path)
                        if image is not None:
                            features = self.extract_features(image)
                            for feature in features:
                                class_features[class_name].append(feature)
                            total_images += 1

        # Calcul des moyennes, variances et probabilités a priori
        for class_name in self.classes:
            if class_name in class_features:
                features = np.array(class_features[class_name])
                self.feature_means[class_name] = np.mean(features, axis=0)
                self.feature_variances[class_name] = np.var(features, axis=0) + 1e-6
                self.class_priors[class_name] = len(features) / total_images

        print("Training completed for classes:", self.classes)

    def predict(self, image):
        """
        Prédire la classe d'une image donnée.

        :param image: Image à classer
        :return: Classe prédite
        """
        rotation_weights = {
            0: 1.0,
            90: 0.5,
            180: 0.5,
            270: 0.5
        }

        posterior_probabilities = {}

        for rotation, weight in rotation_weights.items():
            k = rotation // 90
            rotated_image = np.rot90(image, k)
            features = self.extract_features(rotated_image)

            for class_name in self.classes:
                mean = self.feature_means[class_name]
                variance = self.feature_variances[class_name]
                prior = self.class_priors[class_name]

                likelihood = -0.5 * np.sum((features - mean) ** 2 / variance) + np.log(2 * np.pi * variance)
                posterior = likelihood + np.log(prior)

                weighted_posterior = posterior * (1 - weight * 0.5)

                if class_name not in posterior_probabilities:
                    posterior_probabilities[class_name] = weighted_posterior
                else:
                    posterior_probabilities[class_name] = max(posterior_probabilities[class_name], weighted_posterior)

        return max(posterior_probabilities, key=posterior_probabilities.get)

    def save_model(self, model_path):
        """
        Sauvegarder le modèle Bayésien dans un fichier.

        :param model_path: Chemin du fichier de sauvegarde
        """
        model_data = {
            "feature_means": self.feature_means,
            "feature_variances": self.feature_variances,
            "class_priors": self.class_priors,
            "classes": self.classes
        }
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(model_data, model_path)
        print("Model saved in {}".format(model_path))

    def load_model(self, model_path):
        """
        Charger un modèle Bayésien sauvegardé.

        :param model_path: Chemin du fichier de modèle
        """
        if os.path.exists(model_path):
            model_data = torch.load(model_path)
            self.feature_means = model_data["feature_means"]
            self.feature_variances = model_data["feature_variances"]
            self.class_priors = model_data["class_priors"]
            self.classes = model_data["classes"]
            print("Model loaded from {}".format(model_path))
        else:
            print("Model does not exist: {}".format(model_path))

    def visualize(self):
        """
        Visualiser les moyennes des caractéristiques pour chaque classe.
        """
        if not self.classes:
            print("No classes to visualize")
            return

        for class_name in self.classes:
            mean_features = self.feature_means[class_name]

            plt.figure(figsize=(10, 4))
            plt.title("Mean features for class: {}".format(class_name))
            plt.plot(mean_features)
            plt.xlabel("Feature Index")
            plt.ylabel("Mean Value")
            plt.grid(True)
            plt.show()
