import os
import subprocess
from src.pipeline import ObjectDetectionPipeline
from src.classifiers.bayesian import BayesianClassifier

# Définissez le mode d'analyse ici : "plan" ou "page"
analysis_mode = "plan"

if __name__ == "__main__":
    # Configuration en fonction du mode sélectionné
    if analysis_mode == "plan":
        model_path = "models/bayesian_modelPLAN.pth"
        image_path = "data/plan.png"
    else:
        model_path = "models/bayesian_modelPAGE.pth"
        image_path = "data/page.png"

    # Lancement de l'entraînement avec le mode choisi
    print(f"Entraînement en cours avec le mode {analysis_mode}...")
    try:
        subprocess.run(["python", "train.py", "--mode", analysis_mode], check=True)
        print("Entraînement terminé avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Une erreur s'est produite pendant l'entraînement : {e}")
        exit(1)

    # Chargement du modèle bayésien
    print(f"Chargement du modèle depuis {model_path}...")
    bayesian_model = BayesianClassifier(mode=analysis_mode)
    try:
        bayesian_model.load_model(model_path)
        print(f"Modèle chargé depuis {model_path}.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        exit(1)

    # Vérification de l'existence de l'image de test
    if not os.path.exists(image_path):
        print(f"L'image spécifiée ({image_path}) n'existe pas.")
        exit(1)

    # Création du dossier de sortie si nécessaire
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialisation de la pipeline
    print("Initialisation de la pipeline...")
    pipeline = ObjectDetectionPipeline(image_path=image_path, model=bayesian_model, output_dir=output_dir)

    # Configuration du mode d'analyse dans la pipeline
    pipeline.set_mode(analysis_mode)

    # Chargement de l'image de test
    print("Chargement de l'image...")
    try:
        pipeline.load_image()
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Détection et classification des objets
    print("Détection et classification des objets en cours...")
    try:
        class_counts, detected_objects, total_objects, ignored_objects, identified_objects = pipeline.detect_and_classify_objects()
        print(f"Objets détectés par classe : {class_counts}")
        print("Résumé de la détection :")
        print(f"- Nombre total d'objets : {total_objects}")
        print(f"- Objets identifiés : {identified_objects}")
        print(f"- Objets ignorés : {ignored_objects}")
    except Exception as e:
        print(f"Erreur pendant la détection/classification : {e}")
        exit(1)

    # Sauvegarde et visualisation des résultats
    print("Sauvegarde et affichage des résultats...")
    pipeline.display_results(class_counts, detected_objects)

    # Affichage de l'histogramme des classes détectées
    print("Affichage de l'histogramme des résultats...")
    try:
        pipeline.display_histogram(class_counts)
    except Exception as e:
        print(f"Erreur lors de l'affichage de l'histogramme : {e}")

    # Affichage du nuage de points
    print("Affichage du nuage de points...")
    try:
        pipeline.display_scatter_plot(class_counts)
    except Exception as e:
        print(f"Erreur lors de l'affichage du nuage de points : {e}")

    print(f"Tous les résultats sont sauvegardés dans le dossier : {output_dir}")
