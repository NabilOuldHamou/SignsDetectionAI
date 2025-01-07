import cv2
import os
from matplotlib import pyplot as plt
from collections import defaultdict


class ObjectDetectionPipeline:
    def __init__(self, image_path, model=None, output_dir="output", min_contour_area=20, binary_threshold=None):
        """
        Initialise le pipeline de détection et classification d'objets.
        """
        self.image_path = image_path
        self.image = None
        self.binary_image = None
        self.model = model
        self.output_dir = output_dir
        self.min_contour_area = min_contour_area
        self.binary_threshold = binary_threshold
        self.mode = None  # Défini par le main.py ("plan" ou "page")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def set_mode(self, mode):
        """
        Configure le mode d'analyse (plan ou page).
        """
        self.mode = mode
        if self.mode == "plan":
            self.annotated_output_path = os.path.join(self.output_dir, "annotated_plan.jpg")
            self.detection_threshold = -395000  # Seuil pour le mode plan
        elif self.mode == "page":
            self.annotated_output_path = os.path.join(self.output_dir, "annotated_page.jpg")
            self.detection_threshold = -65000  # Seuil pour le mode page
        else:
            raise ValueError(f"Mode inconnu : {mode}")

    def load_image(self):
        """
        Charge l'image spécifiée.
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image {self.image_path} non trouvée.")
        return self.image

    def preprocess_image(self):
        """
        Prétraite l'image pour la détection.
        """
        channels = cv2.split(self.image)
        binary_images = []

        for channel in channels:
            if self.binary_threshold is None:
                _, binary_channel = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, binary_channel = cv2.threshold(channel, self.binary_threshold, 255, cv2.THRESH_BINARY_INV)
            binary_images.append(binary_channel)

        binary_image = cv2.bitwise_or(binary_images[0], binary_images[1])
        binary_image = cv2.bitwise_or(binary_image, binary_images[2])
        self.binary_image = binary_image
        return binary_image

    def detect_and_classify_objects(self):
        """
        Détecte et classe les objets dans l'image.
        """
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

            # Prédit la classe de l'objet détecté
            predicted_class = self.model.predict(letter_image, threshold=self.detection_threshold)
            if predicted_class is None:
                print("Objet ignoré en raison d'une faible ressemblance.")
                continue

            class_counts[predicted_class] += 1
            detected_objects.append((x, y, w, h, predicted_class))

        return dict(sorted(class_counts.items())), detected_objects

    def save_results(self, class_counts, detected_objects):
        """
        Sauvegarde les résultats de la détection et de la classification.
        """
        binary_output_path = os.path.join(self.output_dir, "binary_image.jpg")
        cv2.imwrite(binary_output_path, self.binary_image)

        annotated_image = self.image.copy()
        for (x, y, w, h, predicted_class) in detected_objects:
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, str(predicted_class), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite(self.annotated_output_path, annotated_image)

        results_text_path = os.path.join(self.output_dir, "results.txt")
        with open(results_text_path, "w") as f:
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")

    def display_results(self, class_counts, detected_objects):
        """
        Affiche et sauvegarde les résultats.
        """
        self.save_results(class_counts, detected_objects)

        plt.figure(figsize=(10, 5))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel("Classes")
        plt.ylabel("Nombre d'objets détectés")
        plt.title("Distribution des classes détectées")
        plt.show()
