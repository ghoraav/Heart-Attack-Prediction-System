import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
import os

# Function to add random noise to the image
def add_noise(image):
    noise = np.random.normal(0, 25, image.size)  # Gaussian noise
    noisy_image = np.array(image) + noise.reshape(image.size[::-1])
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Function to apply random brightness adjustment
def adjust_brightness(image, factor=1.2):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# Function to rotate the image by a random angle
def random_rotation(image, angle_range=(0, 20)):
    angle = random.uniform(*angle_range)
    return image.rotate(angle)

# Function to flip the image horizontally or vertically
def random_flip(image):
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

# Function to resize/zoom the image
def random_zoom(image, scale_range=(0.8, 1.2)):
    scale = random.uniform(*scale_range)
    width, height = image.size
    new_width, new_height = int(width * scale), int(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Crop to original size if scaled up, pad if scaled down
    if scale >= 1.0:
        left = (new_width - width) // 2
        top = (new_height - height) // 2
        image = image.crop((left, top, left + width, top + height))
    else:
        image = image.resize((width, height))
    return image

# Main function to apply all augmentations to each image in the folder
def augment_images_in_folder(input_folder, output_folder, augmentations_per_image=5):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if not os.path.isfile(image_path):
            continue

        # Open the image and apply augmentations
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        for i in range(augmentations_per_image):
            # Apply random augmentations
            aug_image = random_rotation(image)
            aug_image = random_flip(aug_image)
            aug_image = random_zoom(aug_image)
            aug_image = adjust_brightness(aug_image, factor=random.uniform(0.7, 1.5))
            aug_image = add_noise(aug_image)

            # Save the augmented image
            output_path = os.path.join(output_folder, f"augmented_{i}_{image_name}")
            aug_image.save(output_path)
            print(f"Saved: {output_path}")

# Directory paths
input_folder = r"C:\Users\ghora\Desktop\AI-PJ\dataset\normal"    # Folder with original ECG images
output_folder = r"C:\Users\ghora\Desktop\AI-PJ\DaA-normal\output"       # Folder to save augmented images

# Perform augmentation on all images in the folder
augment_images_in_folder(input_folder, output_folder, augmentations_per_image=5)
