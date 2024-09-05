import os
import random
import shutil

def create_sample_data(config, output_dir, num_transformed=5):
    sliced_output_dir = os.path.join(output_dir, "sliced_images_sample")
    transformed_output_dir = os.path.join(output_dir, "transformed_images_sample")
    
    os.makedirs(sliced_output_dir, exist_ok=True)
    os.makedirs(transformed_output_dir, exist_ok=True)

    sliced_dir = config['sliced_images_dir']
    transformed_dir = config['transformed_images_dir']

    for picture_path, _ in config['original_images']:
        picture_name = os.path.splitext(os.path.basename(picture_path))[0]
        
        sliced_sections = [f for f in os.listdir(sliced_dir) if f.startswith(f"{picture_name}_")]
        if not sliced_sections:
            continue
        
        selected_section = random.choice(sliced_sections)
        
        # Copy only to the subfolder
        shutil.copy(os.path.join(sliced_dir, selected_section), os.path.join(sliced_output_dir, selected_section))
        
        section_prefix = selected_section.split('.')[0]
        transformed_versions = [f for f in os.listdir(transformed_dir) if f.startswith(section_prefix)]
        
        selected_transformed = random.sample(transformed_versions, min(num_transformed, len(transformed_versions)))
        
        for transformed_file in selected_transformed:
            # Copy only to the subfolder
            shutil.copy(os.path.join(transformed_dir, transformed_file), os.path.join(transformed_output_dir, transformed_file))

    print("Sample data creation completed")

# Usage
CONFIG = {
    'original_images': [
        ('data/original_images/Picture1.png', 6),
        ('data/original_images/Picture2.jpg', 9),
        ('data/original_images/Picture3.png', 6),
        ('data/original_images/Picture4.png', 9),
        ('data/original_images/Picture5.png', 9)
    ],
    'sliced_images_dir': 'data/sliced_images',
    'transformed_images_dir': 'data/transformed_images',
    'label_file': 'data/labels.csv',
    'zip_filename': 'data/sliced_images.zip',
    'min_augmentations': 15,
    'max_augmentations': 25,
}

output_dir = "sample_data"
create_sample_data(CONFIG, output_dir)

# Print the contents of the directories
print(f"\nContents of {output_dir}:")
print(os.listdir(output_dir))

for subdir in ['sliced_images_sample', 'transformed_images_sample']:
    subdir_path = os.path.join(output_dir, subdir)
    if os.path.exists(subdir_path):
        print(f"\nContents of {subdir}:")
        print(os.listdir(subdir_path))
    else:
        print(f"\n{subdir} does not exist")