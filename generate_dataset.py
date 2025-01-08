from PIL import Image, ImageDraw, ImageFont
import os

# Répertoire pour sauvegarder les images générées
output_dir = "data/catalogue"

# Définir la taille de la police et de l'image
font_size = 20  # Ajustez pour la taille souhaitée
image_size = (28, 28)  # Taille de l'image pour chaque caractère

# Listes des caractères à générer
uppercase_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowercase_letters = "abcdefghijklmnopqrstuvwxyz"
numbers = "0123456789"

# Chemin vers le fichier de police (à mettre à jour avec un chemin valide sur votre système)
font_path = "arial.ttf"  # Assurez-vous que cette police est disponible

# Créer le répertoire de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Fonction pour créer des images de caractères
def create_character_image(character, output_path):
    """
    Crée une image contenant un caractère spécifique et la sauvegarde dans le chemin donné.

    :param character: Caractère à dessiner
    :param output_path: Chemin où sauvegarder l'image
    """
    # Créer une image vierge avec un fond blanc
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    # Charger la police
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Fichier de police introuvable : {font_path}")
        return

    # Calculer la position du texte pour centrer le caractère
    bbox = font.getbbox(character)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (image_size[0] - text_width) // 2
    text_y = (image_size[1] - text_height) // 2

    # Dessiner le caractère sur l'image
    draw.text((text_x, text_y), character, font=font, fill="black")

    # Sauvegarder l'image
    img.save(output_path)

# Générer des images pour les lettres majuscules et minuscules
for upper, lower in zip(uppercase_letters, lowercase_letters):
    upper_dir = os.path.join(output_dir, f"{upper}_")  # Sous-dossier pour les majuscules
    lower_dir = os.path.join(output_dir, upper)         # Sous-dossier pour les minuscules

    os.makedirs(upper_dir, exist_ok=True)  # Créer le sous-dossier pour les majuscules
    os.makedirs(lower_dir, exist_ok=True)  # Créer le sous-dossier pour les minuscules

    # Sauvegarder l'image de la lettre majuscule
    upper_image_path = os.path.join(upper_dir, f"{upper}.png")
    create_character_image(upper, upper_image_path)

    # Sauvegarder l'image de la lettre minuscule
    lower_image_path = os.path.join(lower_dir, f"{lower}.png")
    create_character_image(lower, lower_image_path)

# Générer des images pour les chiffres
for num in numbers:
    num_dir = os.path.join(output_dir, num)  # Sous-dossier pour chaque chiffre
    os.makedirs(num_dir, exist_ok=True)     # Créer le sous-dossier

    num_image_path = os.path.join(num_dir, f"{num}.png")
    create_character_image(num, num_image_path)

print(f"Les images des lettres et des chiffres ont été générées dans le répertoire : {output_dir}")
