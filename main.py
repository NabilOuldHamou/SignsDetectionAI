import os
from src.pipeline import ObjectDetectionPipeline
from src.classifiers.bayesian import BayesianClassifier

# Définissez le mode d'analyse ici : "plan" ou "page"
analysis_mode = "plan"

if __name__ == "__main__":
    # Configuration basée sur le mode
    if analysis_mode == "plan":
        model_path = "models/bayesian_modelPLAN.pth"
        image_path = "data/plan.png"
    else:
        model_path = "models/bayesian_modelPAGE.pth"
        image_path = "data/page.png"

    # Chargement du modèle bayésien
    print(f"Chargement du modèle bayésien depuis {model_path}")
    bayesian_model = BayesianClassifier(mode=analysis_mode)
    try:
        bayesian_model.load_model(model_path)
        print(f"Modèle bayésien chargé depuis {model_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        exit(1)

    # Vérification de l'existence de l'image
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

    # Définition du mode (plan ou page)
    pipeline.set_mode(analysis_mode)

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
        print("Classes détectées :", class_counts)
    except Exception as e:
        print(f"Erreur lors de la détection/classification : {e}")
        exit(1)

    # Sauvegarde et affichage des résultats
    print("Sauvegarde et affichage des résultats...")
    pipeline.display_results(class_counts, detected_objects)

    print(f"Les résultats ont été sauvegardés dans le dossier : {output_dir}")
