import time
from src.classifiers.bayesian import BayesianClassifier
from src.pipeline import ObjectDetectionPipeline

if __name__ == "__main__":
    # Chemin de l'image à traiter
    image_path = "data/page.png"

    # Initialisation du pipeline avec le chemin de l'image
    pipeline = ObjectDetectionPipeline(image_path)

    # Initialisation et chargement du modèle Bayésien
    bayesian_model = BayesianClassifier()
    model_path = "models/bayesian_model.pth"
    pipeline.load_model(model_path, bayesian_model)

    # Chargement de l'image
    pipeline.load_image()

    # Mesure du temps d'exécution pour la détection et classification
    start_time = time.time()
    class_counts, detected_objects = pipeline.detect_and_classify_objects()
    end_time = time.time()

    # Résultats
    print(f"Temps d'exécution: {end_time - start_time:.2f} secondes")
    print("Comptage des classes :", class_counts)
    print("Nombre d'objets détectés :", len(detected_objects))

    # Affichage des résultats
    pipeline.display_results(class_counts, detected_objects)
