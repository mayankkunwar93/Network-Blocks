# -*- coding: utf-8 -*-
"""
Author: Mayank Kunwar
Find Me: https://in.linkedin.com/in/mayankkunwar93
"""

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Load the original image
image_path = '/content/test_image_2.jpg'  # Replace with your image path
original_image = Image.open(image_path)

# Define the transformations
def increase_contrast(img):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(2.0)  # Increase contrast by a factor of 2.0

def change_hue(img):
    return ImageOps.colorize(img.convert('L'), black="red", white="yellow")  # Example hue change

def posterize(img):
    return ImageOps.posterize(img, bits=4)  # Posterize the image to 16 colors

def blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=2))

def rotate(img):
    return img.rotate(45)  # Rotate by 45 degrees

# Apply transformations
contrast_image = increase_contrast(original_image)
hue_image = change_hue(original_image)
posterized_image = posterize(original_image)
blurred_image = blur(original_image)
rotated_image = rotate(original_image)

# Plotting the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Original Image
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Increased Contrast
axes[1].imshow(contrast_image)
axes[1].set_title('Increased Contrast')
axes[1].axis('off')

# Changed Hue
axes[2].imshow(hue_image)
axes[2].set_title('Changed Hue')
axes[2].axis('off')

# Posterized (Reduce the number of colors)
axes[3].imshow(posterized_image)
axes[3].set_title('Posterized')
axes[3].axis('off')

# Blurred (Apply Gaussian blur)
axes[4].imshow(blurred_image)
axes[4].set_title('Blurred')
axes[4].axis('off')

# Rotated
axes[5].imshow(rotated_image)
axes[5].set_title('Rotated')
axes[5].axis('off')

plt.tight_layout()
plt.show()

















