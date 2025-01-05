import os
from collections import defaultdict
import numpy as np
import cv2

from src.classifiers.bayesian import BayesianClassifier

if __name__ == "__main__":
    # Chemin vers le dataset d'entraînement
    dataset_path = "data/catalogue"

    # Initialisation du classifieur Bayésien
    bayesian_model = BayesianClassifier()

    print("Début de l'entraînement...")

    # Dictionnaire pour stocker les caractéristiques par classe
    class_features = defaultdict(list)
    total_images = 0

    # Parcours des classes dans le dataset
    for class_name in os.listdir(dataset_path):
        class_folder_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder_path):
            continue  # Ignorer les fichiers qui ne sont pas des dossiers

        # Ajouter la classe au modèle si elle n'existe pas déjà
        if class_name not in bayesian_model.classes:
            bayesian_model.classes.append(class_name)

        # Parcours des images dans le dossier de la classe
        for image_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                # Extraire les caractéristiques de l'image
                features = bayesian_model.extract_features(image)
                for feature in features:
                    class_features[class_name].append(feature)
                total_images += 1

    # Calcul des statistiques pour chaque classe
    for class_name in bayesian_model.classes:
        if class_name in class_features:
            features = np.array(class_features[class_name])
            bayesian_model.feature_means[class_name] = np.mean(features, axis=0)
            bayesian_model.feature_variances[class_name] = np.var(features, axis=0) + 1e-6  # Éviter la division par zéro
            bayesian_model.class_priors[class_name] = len(features) / total_images

    print("Entraînement terminé.")

    # Sauvegarde du modèle entraîné
    model_path = "models/bayesian_model.pth"
    bayesian_model.save_model(model_path)
    print(f"Modèle sauvegardé dans : {model_path}")
