import os
import cv2
from src.pipeline import ObjectDetectionPipeline
from src.classifiers.bayesian import BayesianClassifier
from collections import defaultdict

if __name__ == "__main__":
    # Chemin vers le modèle entraîné
    model_path = "models/bayesian_modelPAGE.pth"

    # Chargement du modèle bayésien
    print(f"Chargement du modèle bayésien depuis {model_path}")
    bayesian_model = BayesianClassifier()
    try:
        bayesian_model.load_model(model_path)
        print(f"Modèle bayésien chargé depuis {model_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        exit(1)

    # Chemin de l'image de test
    image_path = "data/page.png"
    if not os.path.exists(image_path):
        print(f"L'image de test {image_path} n'existe pas.")
        exit(1)

    # Initialisation du dossier de sortie
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialisation de la pipeline
    print("Initialisation de la pipeline...")
    pipeline = ObjectDetectionPipeline(image_path=image_path, model=bayesian_model, output_dir=output_dir)

    # Chargement de l'image
    print("Chargement de l'image...")
    try:
        pipeline.load_image()
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Détection et classification des objets
    print("Détection et classification des objets...")
    try:
        class_counts, detected_objects = pipeline.detect_and_classify_objects()
    except Exception as e:
        print(f"Erreur lors de la détection/classification : {e}")
        exit(1)

    # Sauvegarde et affichage des résultats
    print("Sauvegarde et affichage des résultats...")
    pipeline.display_results(class_counts, detected_objects)

    # Chargement des comptes réels manuels avec distinction entre minuscule et majuscule
    true_counts_manual = {
        'A_': 30, 'A': 30, 'B_': 4, 'B': 0, 'C_': 14, 'C': 14, 'D_': 17, 'D': 17,
        'E_': 68, 'E': 69, 'F_': 2, 'F': 2, 'G_': 8, 'G': 8, 'H_': 9, 'H': 9,
        'I_': 26, 'I': 25, 'J_': 1, 'J': 0, 'K_': 0, 'K': 0, 'L_': 20, 'L': 19,
        'M_': 15, 'M': 15, 'N_': 30, 'N': 29, 'O_': 37, 'O': 37, 'P_': 23, 'P': 22,
        'Q_': 5, 'Q': 4, 'R_': 28, 'R': 27, 'S_': 26, 'S': 25, 'T_': 38, 'T': 38,
        'U_': 25, 'U': 25, 'V_': 7, 'V': 6, 'W_': 1, 'W': 0, 'X_': 2, 'X': 2,
        'Y_': 6, 'Y': 5, 'Z_': 3, 'Z': 2,
        '1': 8, '2': 11, '3': 2, '4': 1, '5': 2, '6': 1, '7': 1, '8': 3, '9': 3
    }

    # Chargement des résultats détectés depuis results.txt
    results_path = "output/results.txt"
    detected_counts = defaultdict(int)
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            for line in f:
                char, count = line.strip().split(":")
                detected_counts[char.strip()] = int(count.strip())
    else:
        print(f"Le fichier {results_path} n'existe pas.")
        exit(1)

    # Calcul du pourcentage de précision
    print("Calcul du pourcentage de précision...")
    total_true = sum(true_counts_manual.values())
    common_keys = set(true_counts_manual.keys()) & set(detected_counts.keys())

    correctly_detected = sum(min(detected_counts[char], true_counts_manual[char]) for char in common_keys)
    precision = (correctly_detected / total_true) * 100 if total_true > 0 else 0

    # Afficher les résultats
    print("\nRésultats de comparaison :")
    for char in sorted(common_keys):
        print(f"{char}: True={true_counts_manual[char]}, Detected={detected_counts[char]}")

    print(f"\nPrécision globale : {precision:.2f}%")
    print(f"Les résultats ont été sauvegardés dans le dossier : {output_dir}")
