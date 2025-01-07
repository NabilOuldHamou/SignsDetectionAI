import cv2
import os
from matplotlib import pyplot as plt
from collections import defaultdict


class ObjectDetectionPipeline:
    def __init__(self, image_path, model=None, output_dir="output", mode="page", min_contour_area=20, binary_threshold=None):
        self.image_path = image_path
        self.image = None
        self.binary_image = None
        self.model = model
        self.output_dir = output_dir
        self.min_contour_area = min_contour_area
        self.binary_threshold = binary_threshold
        self.mode = mode  # Default mode is "page"
        self.annotated_output_path = os.path.join(self.output_dir, f"annotated_{os.path.basename(image_path)}")
        self.threshold = -395000 if mode == "plan" else -65000

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def set_mode(self, mode):
        """Set the detection mode (page or plan)."""
        if mode not in ["page", "plan"]:
            raise ValueError("Mode must be 'page' or 'plan'.")
        self.mode = mode
        self.threshold = -395000 if mode == "plan" else -65000
        print(f"Mode set to: {self.mode}, Threshold set to: {self.threshold}")

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Image {self.image_path} not found.")
        return self.image

    def preprocess_image(self):
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
        if self.model is None:
            raise ValueError("No classification model provided.")

        self.binary_image = self.preprocess_image()
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        class_counts = defaultdict(int)
        detected_objects = []
        total_objects = 0
        ignored_objects = 0
        identified_objects = 0

        for contour in contours:
            total_objects += 1  # Compteur total des objets

            if cv2.contourArea(contour) < self.min_contour_area:
                ignored_objects += 1
                continue

            x, y, w, h = cv2.boundingRect(contour)
            letter_image = self.image[y:y + h, x:x + w]

            predicted_class = self.model.predict(letter_image, threshold=self.threshold)
            if predicted_class is None:
                ignored_objects += 1
                continue

            identified_objects += 1
            class_counts[predicted_class] += 1
            detected_objects.append((x, y, w, h, predicted_class))

        return dict(sorted(class_counts.items())), detected_objects, total_objects, ignored_objects, identified_objects

    def save_results(self, class_counts, detected_objects):
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
        self.save_results(class_counts, detected_objects)

        plt.figure(figsize=(10, 5))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel("Classes")
        plt.ylabel("Object count")
        plt.title("Detected Class Distribution")
        plt.show()
