import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from config import CONFIG

def get_sample_images(directory, num_samples):
    """
    Randomly select a sample of images from the specified directory.
    """
    images = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(images) <= num_samples:
        return [os.path.join(directory, img) for img in images]
    return [os.path.join(directory, random.choice(images)) for _ in range(num_samples)]

def display_images(images, title):
    """
    Display a list of images in a grid.
    """
    plt.figure(figsize=(15, 10))
    plt.suptitle(title, fontsize=16)
    for i, img_path in enumerate(images):
        try:
            img = Image.open(img_path)
            plt.subplot(3, 4, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(img_path))
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    # Paths from CONFIG
    original_images_dir = CONFIG['sliced_images_dir']
    transformed_images_dir = CONFIG['transformed_images_dir']
    
    # Number of samples to display
    num_samples = 12
    
    # Get sample images
    original_images = get_sample_images(original_images_dir, num_samples)
    transformed_images = get_sample_images(transformed_images_dir, num_samples)
    
    # Display images
    display_images(original_images, "Sample Original Images")
    display_images(transformed_images, "Sample Transformed Images")

if __name__ == "__main__":
    main()
