#!/usr/bin/env python3
"""
Inception V3 Feature Extraction Tool

This tool extracts features from images using the Inception V3 model's
global average pooling layer and saves them to a CSV file.

Requirements:
- TensorFlow/Keras
- PIL (Pillow)
- NumPy
- argparse
"""

import argparse
import csv
import os
import signal
import sys
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image


class GracefulKiller:
    """Handle graceful termination signals"""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        """Set flag for graceful termination"""
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.kill_now = True


class InceptionV3FeatureExtractor:
    """Extract features from images using Inception V3 model"""

    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.load_model()

    def load_model(self):
        """Load Inception V3 model and create feature extractor"""
        try:
            print("Loading Inception V3 model...")

            # Load the base Inception V3 model
            base_model = InceptionV3(
                weights='imagenet',
                include_top=False,
                pooling='avg',  # Use global average pooling
                input_shape=(299, 299, 3)
            )

            # The model with global average pooling already applied
            self.feature_extractor = base_model

            print("Model loaded successfully!")
            print(f"Feature vector size: {base_model.output_shape[1]} dimensions")

        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess a single image for Inception V3"""
        try:
            # Load and resize image to 299x299 (Inception V3 input size)
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize((299, 299), Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Apply Inception V3 preprocessing
            img_array = preprocess_input(img_array)

            return img_array

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None

    def extract_features_batch(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Extract features from a batch of images"""
        features = []
        successful_paths = []

        print(f"Processing batch of {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths):
            try:
                # Preprocess image
                img_array = self.preprocess_image(image_path)
                if img_array is None:
                    continue

                # Extract features
                feature_vector = self.feature_extractor.predict(img_array, verbose=0)
                features.append(feature_vector.flatten())
                successful_paths.append(image_path)

                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(image_paths)} images in batch")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        return features, successful_paths


def get_image_files(directory: str) -> List[str]:
    """Get list of image files from directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []

    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))

    return sorted(image_files)


def write_features_to_csv(csv_path: str, features: List[np.ndarray], 
                         image_paths: List[str], mode: str = 'w'):
    """Write features to CSV file"""
    try:
        with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header only for new files
            if mode == 'w':
                feature_size = len(features[0]) if features else 2048  # Default Inception V3 size
                header = ['image_path'] + [f'feature_{i}' for i in range(feature_size)]
                writer.writerow(header)

            # Write features
            for img_path, feature_vector in zip(image_paths, features):
                row = [img_path] + feature_vector.tolist()
                writer.writerow(row)

        print(f"Successfully wrote {len(features)} feature vectors to {csv_path}")

    except Exception as e:
        print(f"Error writing to CSV: {e}")
        raise


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract Inception V3 features from images and save to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inception_extractor.py --dir /path/to/images/
  python inception_extractor.py --dir /path/to/images/ --out /path/to/output.csv
        """
    )

    parser.add_argument(
        '--dir',
        required=True,
        help='Path to folder containing images (mandatory)'
    )

    parser.add_argument(
        '--out',
        default=None,
        help='Path to output CSV file (optional, defaults to ../results/Inception_[timestamp].csv)'
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        args.out = str(results_dir / f"Inception_{timestamp}.csv")

    # Validate input directory
    if not os.path.isdir(args.dir):
        print(f"Error: Directory does not exist: {args.dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.out).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {args.dir}")
    print(f"Output CSV file: {args.out}")

    # Initialize graceful killer
    killer = GracefulKiller()

    try:
        # Get list of image files
        print("\nScanning for image files...")
        image_files = get_image_files(args.dir)

        if not image_files:
            print("No image files found in the specified directory!")
            sys.exit(1)

        print(f"Found {len(image_files)} image files")

        # Initialize feature extractor
        feature_extractor = InceptionV3FeatureExtractor()

        # Process images in batches of 100
        batch_size = 100
        total_batches = (len(image_files) + batch_size - 1) // batch_size
        processed_count = 0

        print(f"\nProcessing {len(image_files)} images in {total_batches} batches of {batch_size}...")

        for batch_idx in range(total_batches):
            # Check for graceful termination
            if killer.kill_now:
                print("\nGraceful termination requested. Saving progress...")
                break

            # Get batch of image files
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_files))
            batch_files = image_files[start_idx:end_idx]

            print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")

            # Extract features for batch
            features, successful_paths = feature_extractor.extract_features_batch(batch_files)

            if features:
                # Write to CSV (append mode for subsequent batches)
                mode = 'w' if batch_idx == 0 else 'a'
                write_features_to_csv(args.out, features, successful_paths, mode=mode)
                processed_count += len(features)

            # Force garbage collection to manage memory
            gc.collect()

            print(f"Batch {batch_idx + 1} completed. Total processed: {processed_count}/{len(image_files)}")

        # Final summary
        print(f"\n{'='*50}")
        print(f"Feature extraction completed!")
        print(f"Total images processed: {processed_count}/{len(image_files)}")
        print(f"Output saved to: {args.out}")
        print(f"{'='*50}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
    except Exception as e:
        print(f"\nError during processing: {e}")
        sys.exit(1)
    finally:
        print("\nCleanup completed.")


if __name__ == "__main__":
    main()
