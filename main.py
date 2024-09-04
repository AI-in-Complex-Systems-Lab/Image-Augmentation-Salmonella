import os
import random
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import zipfile
from config import CONFIG

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

def slice_image(image_path, num_sections, output_dir, output_prefix):
    """
    Slice an image into multiple sections and save them.
    """
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            section_width = img_width // num_sections
            
            os.makedirs(output_dir, exist_ok=True)
            
            section_paths = []
            for i in range(num_sections):
                left = i * section_width
                right = (i + 1) * section_width
                box = (left, 0, right, img_height)
                section = img.crop(box)
                section_path = os.path.join(output_dir, f'{output_prefix}_section_{i+1}.png')
                section.save(section_path)
                section_paths.append(section_path)
            
            return section_paths
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return []

def label_images(output_dir, label_file):
    """
    Create labels for the sliced images and save them to a CSV file.
    """
    labels = []
    for filename in os.listdir(output_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            parts = filename.split('_')
            if len(parts) >= 3:
                original_image = '_'.join(parts[:-2])
                section_number = parts[-1].split('.')[0]
                labels.append({
                    'Original Image': original_image,
                    'Section Number': section_number,
                    'File Name': filename,
                })

    df = pd.DataFrame(labels)
    df['Section Number'] = df['Section Number'].astype(int)
    df = df.sort_values(['Original Image', 'Section Number'])
    df.to_csv(label_file, index=False)
    print(f'Labels saved to {label_file}')

def augment_images(input_dir, output_dir):
    """
    Augment the sliced images and save the augmented versions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8,1.2],
        fill_mode='nearest',
        rescale=1./255
    )

    total_images = sum(1 for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    for index, image_name in enumerate(os.listdir(input_dir), 1):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, image_name)
            print(f"Processing image {index}/{total_images}: {image_name}")
            try:
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                num_augmentations = random.randint(CONFIG['min_augmentations'], CONFIG['max_augmentations'])

                for i, batch in enumerate(datagen.flow(x, batch_size=1,
                                                       save_to_dir=output_dir,
                                                       save_prefix=os.path.splitext(image_name)[0],
                                                       save_format='png')):
                    if i >= num_augmentations:
                        break
                print(f"  Created {num_augmentations} augmented images")
            except UnidentifiedImageError:
                print(f"  Cannot identify image file: {img_path}")
        else:
            print(f"Skipping non-image file: {image_name}")

def create_metadata(transformed_dir):
    """
    Create metadata for the transformed images.
    """
    image_data = []
    for image_name in os.listdir(transformed_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(image_name)[0]
            original_image_name, section = base_name.split('_', 1)
            image_data.append({
                'Original Image': original_image_name,
                'Section': section,
                'Filename': image_name
            })
    return pd.DataFrame(image_data)

def create_zip(input_dir, zip_filename):
    """
    Create a zip file of the sliced images.
    """
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, input_dir))
    print(f'Zip file created: {zip_filename}')

def main():
    # Slice images
    all_section_paths = []
    for image_path, num_sections in CONFIG['original_images']:
        output_prefix = os.path.splitext(os.path.basename(image_path))[0]
        section_paths = slice_image(image_path, num_sections, CONFIG['sliced_images_dir'], output_prefix)
        all_section_paths.extend(section_paths)
    print("Image slicing completed.")

    # Label images
    label_images(CONFIG['sliced_images_dir'], CONFIG['label_file'])
    print("Image labeling completed.")

    # Augment images
    augment_images(CONFIG['sliced_images_dir'], CONFIG['transformed_images_dir'])
    print("Image augmentation completed.")

    # Create metadata
    metadata_df = create_metadata(CONFIG['transformed_images_dir'])
    print("Metadata creation completed.")
    print(metadata_df)

    # Create zip file
    create_zip(CONFIG['sliced_images_dir'], CONFIG['zip_filename'])
    print("Zip file creation completed.")

if __name__ == "__main__":
    main()