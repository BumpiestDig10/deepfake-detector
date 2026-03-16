"""
Integrated Image Classification and Feature Extraction Tool

This tool extracts features from images using ResNet50 or InceptionV3,
performs predictions using a pre-trained joblib model, and saves/visualizes results.
"""

import argparse
import csv
import os
import sys
import gc
import joblib
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

import centralLogging  as cl
logger = cl.get_logger(console_level="INFO", file_level="DEBUG")

# --- Feature Extractor Classes ---

class BaseFeatureExtractor:
    def __init__(self, model_name: str, weights: str = 'imagenet'):
        self.model_name = model_name
        self.feature_extractor = None
        self.input_size = (224, 224) if model_name == "ResNet50" else (299, 299)
        self.load_model(weights)

    def load_model(self, weights: str):
        try:
            logger.info(f"Loading {self.model_name} with weights: {weights}...")
            if self.model_name == "ResNet50":
                self.feature_extractor = ResNet50(weights=weights, include_top=False, pooling='avg', input_shape=(224, 224, 3))
            else:
                self.feature_extractor = InceptionV3(weights=weights, include_top=False, pooling='avg', input_shape=(299, 299, 3))
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            sys.exit(1)

    def preprocess(self, image_path: str) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.input_size, Image.Resampling.LANCZOS)
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            if self.model_name == "ResNet50":
                return resnet_preprocess(img_array)
            return inception_preprocess(img_array)
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            return None

    def extract_batch(self, paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        features, valid_paths = [], []
        for path in paths:
            preprocessed = self.preprocess(path)
            if preprocessed is not None:
                feat = self.feature_extractor.predict(preprocessed, verbose=0)
                features.append(feat.flatten())
                valid_paths.append(path)
        return features, valid_paths

# --- Main Tool Logic ---

class PredictionTool:
    def __init__(self, args):
        self.args = args
        self.feature_cols = [f'feature_{i}' for i in range(2048)]
        self.model = self.load_prediction_model()
        self.extractor = None

    def load_prediction_model(self):
        try:
            logger.info(f"Loading prediction model: {self.args.modelPath}")
            return joblib.load(self.args.modelPath)
        except Exception as e:
            logger.error(f"Error loading joblib model: {e}")
            sys.exit(1)

    def get_input_files(self) -> List[str]:
        """
        Parses the --input argument. 
        If a directory is found, it scans it recursively for image files.
        """
        input_list = self.args.input # This is a list due to nargs='+'
        
        # Check if the single input is a CSV
        if len(input_list) == 1 and input_list[0].lower().endswith('.csv'):
            return [input_list[0]]
        
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        all_image_paths = []
        
        for item in input_list:
            p = Path(item)
            if not p.exists():
                logger.warning(f"Input path does not exist: {item}")
                continue
                
            if p.is_dir():
                logger.info(f"Scanning directory for images: {item}")
                # Recursively find all image files
                found_files = [str(f) for f in p.rglob('*') if f.is_file() and f.suffix.lower() in image_exts]
                all_image_paths.extend(found_files)
                logger.info(f"Found {len(found_files)} image(s) in directory: {item}")
            elif p.is_file():
                if p.suffix.lower() in image_exts:
                    all_image_paths.append(str(p))
                else:
                    logger.warning(f"File is not a supported image format: {item}")
        
        # Remove duplicates and sort
        unique_paths = sorted(list(set(all_image_paths)))
        logger.info(f"Total unique images to process: {len(unique_paths)}")
        return unique_paths

    def process_csv(self, file_path: str):
        logger.info(f"Processing features from CSV: {file_path}")
        df = pd.read_csv(file_path)
        
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.error(f"CSV missing {len(missing)} feature columns.")
            return None
        
        # Pass the slice of the dataframe directly to maintain feature names
        X_df = df[self.feature_cols]
        filenames = df['image_filename'] if 'image_filename' in df.columns else df.index.astype(str)
        
        return self.run_prediction(X_df, filenames, is_csv=True)

    def run_prediction(self, X_input, filenames, is_csv=False):
        """
        X_input can be a numpy array or a pandas DataFrame.
        To avoid UserWarning, we ensure it's a DataFrame with correct feature names.
        """
        logger.info("Running model predictions...")
        
        # Ensure input is a DataFrame with feature names to match model training
        if isinstance(X_input, np.ndarray):
            X_df = pd.DataFrame(X_input, columns=self.feature_cols)
        else:
            X_df = X_input

        preds = self.model.predict(X_df)
        probs = self.model.predict_proba(X_df)
        
        results = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            conf = prob[1] if pred == 1 else prob[0]
            label = "REAL" if pred == 1 else "FAKE"
            results.append({
                'filepath': filenames[i],
                'prediction': label,
                'confidence': conf,
                'model_used': Path(self.args.modelPath).name,
                'extractor': self.args.featureExtractor if not is_csv else "Pre-extracted",
                'weights': self.args.weights
            })
            
        res_df = pd.DataFrame(results)
        # Reset index to ensure concat works correctly if X_df came from a filtered CSV
        res_df.index = X_df.index 
        return pd.concat([res_df, X_df], axis=1)

    def visualize(self, df):
        display_count = min(len(df), 12)
        if display_count == 0: return

        # Check if the first input was a CSV
        input_val = self.args.input[0] if isinstance(self.args.input, list) else self.args.input
        is_csv_input = str(input_val).lower().endswith('.csv')
        
        cols = 3
        rows = (display_count + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)
        
        # Flatten the 2D array of axes returned by squeeze=False
        axes_flat = axes.flatten()

        for i in range(display_count):
            row = df.iloc[i]
            title = f"{row['prediction']} ({row['confidence']:.2%})\n{Path(row['filepath']).name}"
            
            if not is_csv_input and os.path.exists(row['filepath']):
                try:
                    img = Image.open(row['filepath'])
                    axes_flat[i].imshow(img)
                except Exception:
                    axes_flat[i].text(0.5, 0.5, "Error Loading\nImage", ha='center')
            else:
                axes_flat[i].text(0.5, 0.5, "CSV Input\nNo Preview", ha='center')
            
            axes_flat[i].set_title(title, color='green' if row['prediction'] == 'REAL' else 'red', fontsize=10)
            axes_flat[i].axis('off')

        # Turn off remaining empty subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout()
        plt.show()

    def run(self):
        input_paths = self.get_input_files()
        if not input_paths:
            logger.error("No valid inputs found.")
            return

        final_df = None

        if input_paths[0].lower().endswith('.csv'):
            final_df = self.process_csv(input_paths[0])
        else:
            self.extractor = BaseFeatureExtractor(self.args.featureExtractor, self.args.weights)
            
            batch_size = 50
            all_results = []
            
            for i in range(0, len(input_paths), batch_size):
                batch = input_paths[i:i+batch_size]
                logger.info(f"Processing image batch {i//batch_size + 1}...")
                
                feats, valid_batch_paths = self.extractor.extract_batch(batch)
                if feats:
                    batch_res = self.run_prediction(np.array(feats), valid_batch_paths)
                    all_results.append(batch_res)
                gc.collect()

            if all_results:
                final_df = pd.concat(all_results, ignore_index=True)

        if final_df is not None:
            out_path = Path(self.args.output)
            out_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_name = out_path / f"results_{timestamp}.csv"
            final_df.to_csv(csv_name, index=False)
            logger.info(f"Results saved to: {csv_name}")
            
            self.visualize(final_df)

def main():
    parser = argparse.ArgumentParser(description="Image Feature Extraction & Prediction Tool")
    parser.add_argument('--input', required=True, nargs='+', help='Image file, directory, or feature CSV')
    parser.add_argument('--modelPath', default="results/imageModels/ResNet50_imagenet/32kModel/best_random_forest_model.joblib", help='Path to .joblib model')
    parser.add_argument('--featureExtractor', choices=['ResNet50', 'InceptionV3'], default='ResNet50')
    parser.add_argument('--weights', default='imagenet', help='Extractor weights')
    parser.add_argument('--output', default="results/detections/", help='Output directory')

    args = parser.parse_args()

    # Dynamic default output
    if args.output is None:
        input_val = args.input[0] if isinstance(args.input, list) else args.input
        folder_name = Path(input_val).stem
        args.output = f"results/detections/{folder_name}"

    tool = PredictionTool(args)
    tool.run()

if __name__ == "__main__":
    main()