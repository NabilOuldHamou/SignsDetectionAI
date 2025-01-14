import os
import argparse
from collections import defaultdict
import numpy as np
import cv2
from src.classifiers.bayesian import BayesianClassifier

if __name__ == "__main__":
    # Analyse des arguments pour configurer le mode
    parser = argparse.ArgumentParser(description="Entraîner le modèle Bayésien.")
    parser.add_argument("--mode", type=str, choices=["page", "plan"], default="page",
                        help="Mode de fonctionnement : 'page' pour les pages ou 'plan' pour les plans.")
    args = parser.parse_args()

    # Définir les chemins en fonction du mode
    mode = args.mode
    dataset_path = f"data/catalogue{'' if mode == 'page' else 'Symbol'}"
    allowed_classes = ['Figure1', 'Figure2', 'Figure3', 'Figure4', 'Figure5', 'Figure6'] \
        if mode == "plan" else ['2', 'd', 'I', 'n', 'o', 'u']
    model_path = f"models/bayesian_model{mode.upper()}.pth"

    # Initialiser le classifieur Bayésien
    bayesian_model = BayesianClassifier(mode=mode)

    print("Lancement de l'entraînement...")

    # Stockage des caractéristiques pour chaque classe
    class_features = defaultdict(list)
    total_images = 0

    # Parcourir les classes dans le dataset
    for class_name in os.listdir(dataset_path):
        if class_name not in allowed_classes:
            continue  # Ignorer les classes non autorisées

        class_folder_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_folder_path):
            continue  # Ignorer les fichiers qui ne sont pas des dossiers

        # Ajouter la classe au modèle si elle n'est pas encore listée
        if class_name not in bayesian_model.classes:
            bayesian_model.classes.append(class_name)

        # Parcourir les images dans le dossier de la classe
        for image_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_name)
            image = cv2.imread(image_path)

            if image is not None:
                # Extraire les caractéristiques de l'image
                features = bayesian_model.extract_features(image)
                for feature in features:
                    class_features[class_name].append(feature)
                total_images += 1

    # Calculer les statistiques pour chaque classe
    for class_name in bayesian_model.classes:
        if class_name in class_features:
            features = np.array(class_features[class_name])
            bayesian_model.feature_means[class_name] = np.mean(features, axis=0)
            bayesian_model.feature_variances[class_name] = np.var(features, axis=0) + 1e-6  # Éviter la division par zéro
            bayesian_model.class_priors[class_name] = len(features) / total_images

    print("Entraînement terminé avec succès.")

    # Sauvegarder le modèle entraîné
    bayesian_model.save_model(model_path)
    print(f"Le modèle a été sauvegardé à l'emplacement : {model_path}")