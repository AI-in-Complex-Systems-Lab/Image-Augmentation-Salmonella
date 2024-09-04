# config.py

import os

print("Starting...")


CONFIG = {
    'original_images': [
        ('data/original_images/image1.png', 9),
        ('data/original_images/image2.png', 9),
        ('data/original_images/image3.png', 6),
        ('data/original_images/image4.jpg', 9),
        ('data/original_images/image5.jpg', 6)
    ],
    'sliced_images_dir': 'data/sliced_images',
    'transformed_images_dir': 'data/transformed_images',
    'label_file': 'data/labels.csv',
    'zip_filename': 'data/sliced_images.zip',
    'min_augmentations': 15,
    'max_augmentations': 25,
}

# main.py

import os
from config import CONFIG
# ... (other imports)

def get_project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))

def get_abs_path(rel_path):
    """Convert a relative path to an absolute path based on the project root."""
    return os.path.join(get_project_root(), rel_path)

# Update the CONFIG paths to be absolute
for key in ['sliced_images_dir', 'transformed_images_dir', 'label_file', 'zip_filename']:
    CONFIG[key] = get_abs_path(CONFIG[key])

CONFIG['original_images'] = [(get_abs_path(path), sections) for path, sections in CONFIG['original_images']]

# ... (rest of your functions remain the same)

def main():
    # Slice images
    all_section_paths = []
    for image_path, num_sections in CONFIG['original_images']:
        output_prefix = os.path.splitext(os.path.basename(image_path))[0]
        section_paths = slice_image(image_path, num_sections, CONFIG['sliced_images_dir'], output_prefix)
        all_section_paths.extend(section_paths)
    print("Image slicing completed.")

    # ... (rest of main function remains the same)

if __name__ == "__main__":
    main()
