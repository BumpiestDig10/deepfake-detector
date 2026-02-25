# Inception V3 Feature Extraction Tool

## Installation

1. Install Python 3.7 or higher
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
```bash
python inception_feature_extractor.py --dir "/path/to/folder/containing/images/"
```

### Custom Output Path
```bash
python inception_feature_extractor.py --dir "/path/to/images/" --out "/path/to/output/features.csv"
```

## Features

1. **Command Line Arguments**:
   - `--dir`: Mandatory path to folder containing images
   - `--out`: Optional output CSV path (defaults to ../results/Inception_[timestamp].csv)

2. **Batch Processing**: Processes images in batches of 100 for memory efficiency

3. **Graceful Termination**: Handles Ctrl+C to save progress before exiting

4. **Resource Monitoring**: Limits CPU usage to 75% and monitors memory usage

5. **Supported Image Formats**: JPG, JPEG, PNG, BMP, TIFF, WEBP

6. **Feature Extraction**: Uses Inception V3's global average pooling layer (2048 dimensions)

## Output Format

CSV file with columns:
- `image_path`: Full path to the processed image
- `feature_0` to `feature_2047`: 2048-dimensional feature vector

## Error Handling

- Skips corrupted or unreadable images
- Continues processing even if individual images fail
- Saves progress on graceful termination
