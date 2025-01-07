import cv2
import os
from matplotlib import pyplot as plt
from collections import defaultdict


class ObjectDetectionPipeline:
    def __init__(self, image_path, model=None, output_dir="output", min_contour_area=50, binary_threshold=127):
        """
        Initialisation de la pipeline de détection d'objets.

        :param image_path: Chemin de l'image à traiter
        :param model: Modèle de classification à utiliser
        :param output_dir: Dossier où les résultats seront sauvegardés
        :param min_contour_area: Aire minimale des contours à prendre en compte
        :param binary_threshold: Seuil de binarisation pour les canaux
        """
        self.image_path = image_path
        self.image = None
        self.binary_image = None
        self.model = model
        self.output_dir = output_dir
        self.min_contour_area = min_contour_area
        self.binary_threshold = binary_threshold

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_image(self):
        """Charge l'image spécifiée."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"L'image {self.image_path} est introuvable.")
        return self.image

    def preprocess_image(self):
        """Prétraite l'image pour la préparer à l'inférence."""
        channels = cv2.split(self.image)
        binary_images = []

        for channel in channels:
            _, binary_channel = cv2.threshold(channel, self.binary_threshold, 255, cv2.THRESH_BINARY_INV)
            binary_images.append(binary_channel)

        binary_image = cv2.bitwise_or(binary_images[0], binary_images[1])
        binary_image = cv2.bitwise_or(binary_image, binary_images[2])
        self.binary_image = binary_image
        return binary_image

    def detect_and_classify_objects(self):
        """Détecte et classe les objets présents dans l'image."""
        if self.model is None:
            raise ValueError("Aucun modèle de classification fourni.")

        self.binary_image = self.preprocess_image()
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        class_counts = defaultdict(int)
        detected_objects = []

        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            letter_image = self.image[y:y + h, x:x + w]

            predicted_class = self.model.predict(letter_image)
            if predicted_class is None:
                print("Skipping object with invalid prediction.")
                continue

            class_counts[predicted_class] += 1
            detected_objects.append((x, y, w, h, predicted_class))

        return dict(sorted(class_counts.items())), detected_objects

    def save_results(self, class_counts, detected_objects):
        """Sauvegarde les résultats de détection et classification."""
        # Sauvegarder l'image binaire
        binary_output_path = os.path.join(self.output_dir, "binary_image.jpg")
        cv2.imwrite(binary_output_path, self.binary_image)

        # Sauvegarder l'image annotée
        annotated_image = self.image.copy()
        for (x, y, w, h, predicted_class) in detected_objects:
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, str(predicted_class), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        annotated_output_path = os.path.join(self.output_dir, "annotated_page.jpg")
        cv2.imwrite(annotated_output_path, annotated_image)

        # Sauvegarder les classes et leurs occurrences
        results_text_path = os.path.join(self.output_dir, "results.txt")
        with open(results_text_path, "w") as f:
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")

    def display_results(self, class_counts, detected_objects):
        """Affiche et sauvegarde les résultats."""
        self.save_results(class_counts, detected_objects)

        plt.figure(figsize=(10, 5))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel("Classes")
        plt.ylabel("Nombre d'objets")
        plt.title("Distribution des classes détectées")
        plt.show()
