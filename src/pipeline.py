import cv2
import os
from matplotlib import pyplot as plt
from collections import defaultdict

class ObjectDetectionPipeline:
    def __init__(self, image_path, model=None):
        """
        Initialisation de la pipeline de détection d'objets avec le chemin de l'image et le modèle.

        :param image_path: Chemin de l'image à traiter
        :param model: Modèle de classification à utiliser (par défaut, aucun modèle n'est chargé)
        """
        self.image_path = image_path
        self.image = None
        self.binary_image = None
        self.model = model  # Le modèle personnalisé à utiliser

    def load_model(self, model_path: str, instance_classifier=None):
        """
        Charger un modèle pré-entraîné ou un classifieur bayésien.

        :param model_path: Chemin du fichier du modèle
        :param instance_classifier: Instance du classifieur à utiliser
        """
        if os.path.exists(model_path):
            self.model = instance_classifier  # Créer une instance du classifieur
            print(f"Chargement du modèle bayésien depuis {model_path}")
            self.model.load_model(model_path)  # Charger les paramètres du modèle
            print(f"Modèle bayésien chargé depuis {model_path}")
        else:
            print(f"Aucun modèle trouvé à {model_path}. Un nouveau modèle sera créé.")

    def load_image(self):
        """
        Charger l'image spécifiée par le chemin d'accès.

        :return: Image chargée
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"L'image {self.image_path} est introuvable.")
        return self.image

    def preprocess_image(self):
        """
        Prétraiter l'image pour la préparer à l'inférence.

        :return: Image binarisée
        """
        # Binarisation de l'image par canaux
        channels = cv2.split(self.image)
        binary_images = []

        for channel in channels:
            _, binary_channel = cv2.threshold(channel, 127, 255, cv2.THRESH_BINARY_INV)
            binary_images.append(binary_channel)

        # Fusionner les canaux binarisés
        binary_image = cv2.bitwise_or(binary_images[0], binary_images[1])
        binary_image = cv2.bitwise_or(binary_image, binary_images[2])
        self.binary_image = binary_image
        return binary_image

    def detect_and_classify_objects(self):
        """
        Détecter et classer les objets présents dans l'image en fonction des contours détectés.

        :return: Dictionnaire des classes détectées et objets détectés
        """
        if self.model is None:
            print("Aucun modèle de classification fourni.")
            return {}

        self.binary_image = self.preprocess_image()

        # Trouver les contours dans l'image binarisée
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        class_counts = defaultdict(int)
        detected_objects = []

        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            letter_image = self.image[y:y + h, x:x + w]

            # Prédiction avec le modèle
            predicted_class = self.model.predict(letter_image)

            # Incrémenter le comptage de la classe prédite
            class_counts[predicted_class] += 1

            # Ajouter les coordonnées et la classe prédite
            detected_objects.append((x, y, w, h, predicted_class))

        return dict(sorted(class_counts.items())), detected_objects

    def display_results(self, class_counts, detected_objects):
        """
        Afficher les résultats de la détection et classification.

        :param class_counts: Dictionnaire des classes détectées et leurs occurrences
        :param detected_objects: Liste des objets détectés avec leurs coordonnées et classes prédites
        """
        self.display_binary_image()
        self.display_image_with_classes(detected_objects)
        self.display_image_with_annotations(detected_objects)
        self.display_classes_count(class_counts)

    def display_binary_image(self):
        """
        Afficher l'image binaire résultant du prétraitement.
        """
        plt.figure(figsize=(self.binary_image.shape[1] / 100, self.binary_image.shape[0] / 100))
        plt.imshow(self.binary_image, cmap='gray')
        plt.axis('off')
        plt.show()

    def display_image_with_classes(self, detected_objects):
        """
        Afficher l'image avec les classes prédites annotées.

        :param detected_objects: Liste des objets détectés avec leurs coordonnées et classes prédites
        """
        image_with_classes_only = self.image.copy()

        for (x, y, w, h, predicted_class) in detected_objects:
            if predicted_class[-1] == "_":
                text = predicted_class.split("_")[0].upper()
            else:
                text = predicted_class.lower()

            cv2.rectangle(image_with_classes_only, (x, y), (x + w, y + h), (255, 255, 255), -1)
            font_scale = 0.7
            font_thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2

            cv2.putText(image_with_classes_only, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        fig = plt.figure(figsize=(image_with_classes_only.shape[1] / 100, image_with_classes_only.shape[0] / 100))
        plt.imshow(cv2.cvtColor(image_with_classes_only, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def display_image_with_annotations(self, detected_objects):
        """
        Afficher l'image avec les annotations des objets détectés (rectangles et classes).

        :param detected_objects: Liste des objets détectés avec leurs coordonnées et classes prédites
        """
        image_with_annotations = self.image.copy()
        for (x, y, w, h, predicted_class) in detected_objects:
            if predicted_class[-1] == "_":
                text = predicted_class.split("_")[0].upper()
            else:
                text = predicted_class.lower()

            cv2.rectangle(image_with_annotations, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_with_annotations, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        fig = plt.figure(figsize=(image_with_annotations.shape[1] / 100, image_with_annotations.shape[0] / 100))
        plt.imshow(cv2.cvtColor(image_with_annotations, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def display_classes_count(self, class_counts):
        """
        Afficher le nombre d'objets détectés par classe.

        :param class_counts: Dictionnaire des classes détectées et leurs occurrences
        """
        plt.figure(figsize=(10, 5))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel("Classes")
        plt.ylabel("Nombre de lettres")
        plt.title("Classes détectées et leur nombre")
        plt.show()
