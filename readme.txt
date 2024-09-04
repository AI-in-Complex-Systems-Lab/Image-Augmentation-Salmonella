# Salmonella Image Processing Pipeline

This project provides a pipeline for processing and augmenting salmonella images.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/salmonella-image-processing.git
   cd salmonella-image-processing
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Prepare your data:
   - Place your original images in the `data/original_images/` directory.
   - Update the `config.py` file with the correct filenames and section counts for your images.

5. Run the script:
   ```
   python main.py
   ```

## Configuration

You can modify the `config.py` file to change the following settings:
- Paths for input and output directories
- Number of augmentations per image
- Names and section counts of original images

## Output

The script will generate:
- Sliced images in `data/sliced_images/`
- Augmented images in `data/transformed_images/`
- A CSV file with labels in `data/labels.csv`
- A zip file of sliced images in `data/sliced_images.zip`

## Project Structure

```
salmonella-image-processing/
│
├── data/
│   ├── original_images/
│   ├── sliced_images/
│   ├── transformed_images/
│   ├── labels.csv
│   └── sliced_images.zip
│
├── main.py
├── config.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
