#!/usr/bin/env python3
"""
ResNet50 Feature Extraction Tool

This tool extracts features from images using the ResNet50 model's
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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

import centralLogging as cl
logger = cl.get_logger(console_level="INFO", file_level="DEBUG")


class GracefulKiller:
    """Handle graceful termination signals"""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        """Set flag for graceful termination"""
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.kill_now = True


class ResNet50FeatureExtractor:
    """Extract features from images using ResNet50 model"""

    def __init__(self, model_weights: str = 'imagenet'):
        self.model = None
        self.feature_extractor = None
        self.load_model(model_weights)

    def load_model(self, model_weights: str = 'imagenet'):
        """Load ResNet50 model and create feature extractor"""
        try:
            logger.debug("Loading ResNet50 model...")

            # Load the base ResNet50 model
            base_model = ResNet50(
                weights=model_weights,
                include_top=False,
                pooling='avg',  # Use global average pooling
                input_shape=(224, 224, 3)
            )

            # The model with global average pooling already applied
            self.feature_extractor = base_model

            logger.info(f"Model loaded successfully! Using weights: {model_weights}")
            logger.info(f"Feature vector size: {base_model.output_shape[1]} dimensions")

        except Exception as e:
            logger.critical(f"Error loading model: {e}")
            sys.exit(1)

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess a single image for ResNet50"""
        try:
            # Load and resize image to 224x224 (ResNet50 input size)
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize((224, 224), Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Apply ResNet50 preprocessing
            img_array = preprocess_input(img_array)

            return img_array

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None

    def extract_features_batch(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Extract features from a batch of images"""
        features = []
        successful_filenames = []

        logger.debug(f"Processing batch of {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths):
            try:
                # Preprocess image
                img_array = self.preprocess_image(image_path)
                logger.debug(f"Preprocessed image: {image_path}")
                if img_array is None:
                    continue

                # Extract features
                feature_vector = self.feature_extractor.predict(img_array, verbose=0)
                features.append(feature_vector.flatten())

                # Store only the filename, not the full path
                filename = os.path.basename(image_path)
                successful_filenames.append(filename)
                logger.debug(f"Extracted features for: {image_path}")

                # Progress indicator
                if (i + 1) % 10 == 0:
                    logger.debug(f"  Processed {i + 1}/{len(image_paths)} images in batch")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue

        return features, successful_filenames


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
                         image_filenames: List[str], mode: str = 'w'):
    """Write features to CSV file"""
    try:
        with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header only for new files
            if mode == 'w':
                feature_size = len(features[0]) if features else 2048  # Default ResNet50 size
                header = ['image_filename'] + [f'feature_{i}' for i in range(feature_size)]
                writer.writerow(header)

            # Write features with filenames only
            for filename, feature_vector in zip(image_filenames, features):
                row = [filename] + feature_vector.tolist()
                writer.writerow(row)

        logger.info(f"Successfully wrote {len(features)} feature vectors to {csv_path}")

    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")
        raise


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract ResNet50 features from images and save to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resnet50_extractor.py --input /path/to/images/
  python resnet50_extractor.py --input /path/to/images/ --output /path/to/output.csv
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Path to folder containing images (mandatory)'
    )

    parser.add_argument(
        '--output',
        default=None,
        help='Path to output CSV file (optional, defaults to results/image_features/ResNet50_[timestamp].csv)'
    )
    
    parser.add_argument(
        '--weights',
        default='imagenet',
        help='model weights to use (optional, defaults to imagenet)'
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/image_features")
        results_dir.mkdir(exist_ok=True)
        args.output = str(results_dir / f"ResNet50_{timestamp}.csv")

    # Validate input directory
    if not os.path.isdir(args.input):
        logger.critical(f"Error: Directory does not exist: {args.input}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output CSV file: {args.output}")

    # Initialize graceful killer
    killer = GracefulKiller()

    try:
        # Get list of image files
        logger.debug("Scanning for image files...")
        image_files = get_image_files(args.input)

        if not image_files:
            logger.critical("No image files found in the specified directory!")
            sys.exit(1)

        logger.info(f"Found {len(image_files)} image files")

        # Initialize feature extractor
        feature_extractor = ResNet50FeatureExtractor(args.weights)

        # Process images in batches of 100
        batch_size = 100
        total_batches = (len(image_files) + batch_size - 1) // batch_size
        processed_count = 0

        logger.info(f"Processing {len(image_files)} images in {total_batches} batches of {batch_size}...")

        for batch_idx in range(total_batches):
            # Check for graceful termination
            if killer.kill_now:
                logger.warning("Graceful termination requested. Saving progress...")
                break

            # Get batch of image files
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_files))
            batch_files = image_files[start_idx:end_idx]

            logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}")

            # Extract features for batch
            features, successful_filenames = feature_extractor.extract_features_batch(batch_files)

            if features:
                # Write to CSV (append mode for subsequent batches)
                mode = 'w' if batch_idx == 0 else 'a'
                write_features_to_csv(args.output, features, successful_filenames, mode=mode)
                processed_count += len(features)

            # Force garbage collection to manage memory
            gc.collect()

            logger.debug(f"Batch {batch_idx + 1} completed. Total processed: {processed_count}/{len(image_files)}")

        # Final summary
        logger.info(f"{'='*50}")
        logger.info(f"Feature extraction completed!")
        logger.info(f"Total images processed: {processed_count}/{len(image_files)}")
        logger.info(f"Output saved to: {args.output}")
        logger.info(f"{'='*50}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving progress...")
    except Exception as e:
        logger.critical(f"Error during processing: {e}")
        sys.exit(1)
    finally:
        logger.info("Cleanup completed.")


if __name__ == "__main__":
    main()
