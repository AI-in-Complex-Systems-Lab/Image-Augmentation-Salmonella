import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

def get_image_paths(sliced_dir, transformed_dir):
    """Retrieve the paths for images: one from the sliced directory and five from the transformed directory."""
    # Get all images from the sliced directory
    sliced_images = sorted(glob.glob(os.path.join(sliced_dir, '*')))
    
    # Get all images from the transformed directory
    transformed_images = sorted(glob.glob(os.path.join(transformed_dir, '*')))

    if len(sliced_images) == 0 or len(transformed_images) < 5:
        raise ValueError("Not enough images in one of the directories.")

    return sliced_images, transformed_images

def create_image_grid(image_paths, output_path):
    """Create a 2x3 grid of images with custom text below each and save it to a file with high resolution."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Define text labels
    texts = ['Original Image'] + ['Transformed Image'] * 5
    
    for i, ax in enumerate(axs.flat):
        img = mpimg.imread(image_paths[i])
        ax.imshow(img)
        ax.axis('off')  # Hide the axis
        # Add text below the image
        ax.set_title(texts[i], fontsize=12, pad=10)

    plt.tight_layout()

    # Save the figure as an image file with high resolution
    dpi = 300  # Increase DPI for higher resolution
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    plt.close()

def main():
    sliced_dir = '/Users/mikaildemir/Desktop/Image-Augmentation-Salmonella/sample_data/sliced_images_sample'
    transformed_dir = '/Users/mikaildemir/Desktop/Image-Augmentation-Salmonella/sample_data/transformed_images_sample'
    
    # Retrieve all images from both directories
    sliced_images, transformed_images = get_image_paths(sliced_dir, transformed_dir)
    
    # Ensure output directory exists
    output_dir = '/Users/mikaildemir/Desktop/Image-Augmentation-Salmonella/output_grids'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each batch
    for i in range(min(len(sliced_images), len(transformed_images) // 5)):
        # Get the first image from the sliced directory
        first_image_path = sliced_images[i]
        
        # Get the next 5 images from the transformed directory
        other_images_paths = transformed_images[i * 5:(i + 1) * 5]
        
        # Combine the images
        image_paths = [first_image_path] + other_images_paths
        
        # Extract the base file name of the first image
        base_file_name = os.path.basename(first_image_path)
        file_name_without_ext = os.path.splitext(base_file_name)[0]
        
        # Define the output path for the grid image
        output_path = os.path.join(output_dir, f'{file_name_without_ext}.png')
        
        # Create and save the image grid
        create_image_grid(image_paths, output_path)
        
        print(f"Image grid saved to {output_path}")

if __name__ == "__main__":
    main()
