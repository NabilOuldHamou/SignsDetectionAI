import os
import cv2
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt


class BayesianClassifier:
    def __init__(self):
        self.feature_means = {}
        self.feature_variances = {}
        self.class_priors = {}
        self.classes = []

        # Initialize HOG descriptor with standard parameters
        self.hog = cv2.HOGDescriptor(
            _winSize=(28, 28),
            _blockSize=(8, 8),
            _blockStride=(4, 4),
            _cellSize=(8, 8),
            _nbins=9
        )

    def extract_features(self, image):
        try:
            # Convert image to grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Apply adaptive thresholding
            binary_image = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("No contours found.")
                return np.array([])

            features = []
            for contour in contours:
                if cv2.contourArea(contour) < 22:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                letter_image = gray_image[y:y + h, x:x + w]
                letter_image = cv2.resize(letter_image, (28, 28))

                # Compute HOG features
                hog_features = self.hog.compute(letter_image)
                features.append(hog_features.flatten())

            features = np.array(features)
            if features.size == 0:
                print("No features extracted.")
                return np.array([])

            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.where(norms > 1e-6, norms, 1)

            return features
        except Exception as e:
            print(f"Error in extract_features: {e}")
            return np.array([])

    def train(self, dataset_path):
        class_features = defaultdict(list)
        total_images = 0

        for class_name in os.listdir(dataset_path):
            class_folder_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_folder_path):
                if class_name not in self.classes:
                    self.classes.append(class_name)

                for img_name in os.listdir(class_folder_path):
                    img_path = os.path.join(class_folder_path, img_name)
                    if os.path.isfile(img_path):
                        try:
                            image = cv2.imread(img_path)
                            if image is not None:
                                features = self.extract_features(image)
                                if features.size > 0:
                                    for feature in features:
                                        class_features[class_name].append(feature)
                                    total_images += 1
                                else:
                                    print(f"No features extracted for {img_path}")
                            else:
                                print(f"Failed to load image: {img_path}")
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")

        for class_name in self.classes:
            if class_name in class_features:
                features = np.array(class_features[class_name])
                self.feature_means[class_name] = np.mean(features, axis=0)
                self.feature_variances[class_name] = np.var(features, axis=0) + 1e-6
                self.class_priors[class_name] = len(features) / total_images

        print("Training completed for classes:", self.classes)

    def save_model(self, model_path):
        model_data = {
            "feature_means": self.feature_means,
            "feature_variances": self.feature_variances,
            "class_priors": self.class_priors,
            "classes": self.classes
        }
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(model_data, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        if os.path.exists(model_path):
            model_data = torch.load(model_path, weights_only=False)
            self.feature_means = model_data["feature_means"]
            self.feature_variances = model_data["feature_variances"]
            self.class_priors = model_data["class_priors"]
            self.classes = model_data["classes"]
            print(f"Model loaded from {model_path}")
        else:
            print(f"No model found at {model_path}.")

    def predict(self, image):
        try:
            features = self.extract_features(image)
            if features.size == 0:
                print("Empty features, skipping prediction.")
                return None

            posteriors = {}
            for class_name in self.classes:
                mean = self.feature_means[class_name]
                variance = self.feature_variances[class_name]
                prior = self.class_priors[class_name]

                likelihood = -0.5 * np.sum(((features - mean) ** 2) / variance + np.log(2 * np.pi * variance))
                posterior = likelihood + np.log(prior)
                posteriors[class_name] = posterior

            return max(posteriors, key=posteriors.get)
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

    def visualize(self):
        if not self.classes:
            print("No classes to visualize.")
            return

        for class_name in self.classes:
            mean_features = self.feature_means[class_name]

            plt.figure(figsize=(10, 4))
            plt.title(f"Mean features for class: {class_name}")
            plt.plot(mean_features)
            plt.xlabel("Feature Index")
            plt.ylabel("Mean Value")
            plt.grid(True)
            plt.show()
