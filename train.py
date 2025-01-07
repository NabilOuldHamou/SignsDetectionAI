from main import analysis_mode

if analysis_mode == "plan":
    dataset_path = "data/catalogueSymbol"
    allowed_classes = ['Figure1', 'Figure2', 'Figure3', 'Figure4', 'Figure5', 'Figure6']
    model_path = "models/bayesian_modelPLAN.pth"
else:
    dataset_path = "data/catalogue"
    allowed_classes = ['2', 'd', 'I', 'n', 'o', 'u']
    model_path = "models/bayesian_modelPAGE.pth"

from src.classifiers.bayesian import BayesianClassifier
from collections import defaultdict
import os
import cv2
import numpy as np

# Initialisation
bayesian_model = BayesianClassifier()

print("Début de l'entraînement...")
class_features = defaultdict(list)
total_images = 0

# Parcours des classes dans le dataset
for class_name in os.listdir(dataset_path):
    if class_name not in allowed_classes:
        continue

    class_folder_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder_path):
        continue

    if class_name not in bayesian_model.classes:
        bayesian_model.classes.append(class_name)

    for image_name in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, image_name)
        image = cv2.imread(image_path)

        if image is not None:
            features = bayesian_model.extract_features(image)
            for feature in features:
                class_features[class_name].append(feature)
            total_images += 1

# Calcul des statistiques pour chaque classe
for class_name in bayesian_model.classes:
    if class_name in class_features:
        features = np.array(class_features[class_name])
        bayesian_model.feature_means[class_name] = np.mean(features, axis=0)
        bayesian_model.feature_variances[class_name] = np.var(features, axis=0) + 1e-6
        bayesian_model.class_priors[class_name] = len(features) / total_images

print("Entraînement terminé.")
bayesian_model.save_model(model_path)
print(f"Modèle sauvegardé dans : {model_path}")
