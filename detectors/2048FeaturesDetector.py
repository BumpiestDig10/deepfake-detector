#!/usr/bin/env python3
"""
DeepVerify: Modular Service-Oriented DeepFake Detection Architecture
Handles image feature extraction, Keras session clearing, memory management,
Random Forest inference, and a human-in-the-loop TkAgg verification gallery.
"""

import os
import sys
import gc
import re
import csv
import math
import signal
import argparse
import datetime
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import joblib

import matplotlib
# Force TkAgg backend as requested for local execution scrollable UI
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image

# Import provided extractors
try:
    from utils.featureExtractor.ResNet50_image_feature_extractor import ResNet50FeatureExtractor
    from utils.featureExtractor.InceptionV3_image_feature_extractor import InceptionV3FeatureExtractor
except ImportError as e:
    print(f"CRITICAL: Could not import feature extractors. Ensure 'utils.featureExtractor' exists. Error: {e}")
    sys.exit(1)

# Import central logging (mocked fallback if not present)
try:
    import centralLogging as cl
    logger = cl.get_logger(console_level="INFO", file_level="DEBUG")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)


class GracefulKiller:
    """Handle graceful termination signals across the pipeline"""
    def __init__(self):
        self.kill_now = False
        self.kill_count = 0
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_count += 1
        if self.kill_count > 1:
            logger.critical("Force quitting immediately due to multiple interrupts.")
            sys.exit(1)
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown... (Press again to force quit)")
        self.kill_now = True


def get_all_image_paths(input_path: str) -> List[str]:
    """Identify all target image files whether input is a single file or directory."""
    target_path = Path(input_path)
    if target_path.is_file():
        return [str(target_path)]
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    for file_path in target_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    return sorted(image_files)


def generate_output_path(input_arg: str) -> str:
    """Generate deterministic output path based on input context."""
    out_dir = Path("results/detections")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = Path(input_arg)
    if target_path.is_file():
        return str(out_dir / f"{target_path.stem}.csv")
    else:
        return str(out_dir / f"{target_path.name}.csv")


def extract_features_to_temp(image_paths: List[str], extractor_type: str, weights: str, temp_csv: str, killer: GracefulKiller):
    """Phase I: Initialize CNN, extract raw features, save to temp, and manage memory."""
    logger.info(f"Phase I: Extracting features using {extractor_type} ({weights} weights)")
    
    # 1. Dynamic Instance Initialization
    if extractor_type.lower() == 'inceptionv3':
        extractor = InceptionV3FeatureExtractor(weights=weights)
    else:
        extractor = ResNet50FeatureExtractor(model_weights=weights)

    batch_size = 100
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    # 2. Intermediate Storage
    with open(temp_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath'] + [f'feature_{i}' for i in range(2048)])

        for batch_idx in range(total_batches):
            if killer.kill_now:
                break
                
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_files = image_paths[start_idx:end_idx]
            
            # The extractors return (features, paths/filenames). 
            # We strictly track our own batch_files to maintain accurate full paths.
            features, _ = extractor.extract_features_batch(batch_files)
            
            # We must map the generated features back to their original file paths
            # In case the extractor dropped an image due to a read error, we iterate safely.
            if features:
                for img_path, feature_vec in zip(batch_files, features):
                    writer.writerow([img_path] + feature_vec.tolist())

            logger.info(f"  Processed batch {batch_idx + 1}/{total_batches}")

    # 4. The "Clean Break" (Memory Firewall)
    logger.info("Executing Memory Guard: Clearing Keras session and dropping CNN instance.")
    del extractor
    
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()


def display_gallery(df: pd.DataFrame):
    """Phase V: Launch the human-in-the-loop scrollable verification UI."""
    logger.info("Launching TkAgg Visualization UI...")
    
    root = tk.Tk()
    root.title("DeepVerify - Prediction Gallery")
    
    # Ensure window is large enough
    root.geometry("1000x800")
    
    # Scrollable Canvas setup
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Mouse wheel binding (Cross-platform)
    def _on_mousewheel(event):
        if event.num == 4 or event.delta > 0:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            canvas.yview_scroll(1, "units")
            
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel) # Linux scroll up
    canvas.bind_all("<Button-5>", _on_mousewheel) # Linux scroll down

    # Matplotlib UI Grid Construction
    total_images = len(df)
    ncols = 4
    nrows = math.ceil(total_images / ncols)
    
    # Limit nrows to avoid Tkinter height limits breaking internally, though RAM OOM will crash first
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    if total_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, row in enumerate(df.itertuples()):
        ax = axes[idx]
        img_path = row.filepath
        try:
            img = Image.open(img_path)
            ax.imshow(img)
        except Exception as e:
            logger.warning(f"Could not load {img_path} for UI: {e}")
            ax.text(0.5, 0.5, 'Image Load Error', ha='center', va='center')
        
        # Color & label coding
        pred = row.prediction
        conf = row.confidence_score
        color = 'red' if pred == "FAKE" else 'green'
        
        ax.set_title(f"{pred} ({conf:.2f}%)", color=color, fontweight='bold')
        ax.axis('off')

    # Blank out any remaining empty plots
    for idx in range(total_images, len(axes)):
        axes[idx].axis('off')

    fig.tight_layout()
    canvas_agg = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_agg.draw()
    canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="DeepVerify: AI DeepFake Verification System")
    parser.add_argument('--input', required=True, help='Path to an image, folder, or extracted CSV.')
    parser.add_argument('--output', default=None, help='Output path for predictions (auto-generated if omitted).')
    parser.add_argument('--modelPath', default='results/imageModels/ResNet50_imagenet/32kModel/randomForest/best_random_forest_model.joblib', help='Path to Random Forest joblib model.')
    parser.add_argument('--featureExtractor', default='ResNet50', choices=['ResNet50', 'InceptionV3'], help='CNN Base Extractor.')
    parser.add_argument('--weights', default='imagenet', help='Weights for CNN Extractor.')
    parser.add_argument('--showOutput', action='store_true', help='Display scrollable gallery at completion.')
    
    args = parser.parse_args()
    logger.debug(f"Input: {args.input} | Output: {args.output} | Model: {args.modelPath} | Extractor: {args.featureExtractor} | Weights: {args.weights} | Show UI: {args.showOutput}")
    killer = GracefulKiller()
    
    is_csv_input = str(args.input).lower().endswith('.csv')
    temp_csv_file = "temp_raw_features.csv"
    logger.debug(f"Is CSV Input: {is_csv_input} | Temp CSV: {temp_csv_file}")
    
    if not args.output:
        args.output = generate_output_path(args.input)
        logger.info(f"No output path provided. Auto-generated output path: {args.output}")

    try:
        # ---------------------------------------------------------------------------
        # Phase I: Environment Prep & Feature Extraction
        # ---------------------------------------------------------------------------
        if not is_csv_input:
            image_paths = get_all_image_paths(args.input)
            if not image_paths:
                logger.critical(f"No valid images found in {args.input}")
                sys.exit(1)
                
            logger.debug(f"Found image paths: {image_paths}")
            
            logger.info(f"Extracting features for {len(image_paths)} images...")
            extract_features_to_temp(
                image_paths=image_paths, 
                extractor_type=args.featureExtractor, 
                weights=args.weights, 
                temp_csv=temp_csv_file,
                killer=killer
            )
            working_csv = temp_csv_file
        else:
            working_csv = args.input

        was_interrupted = False
        if killer.kill_now:
            was_interrupted = True
            logger.warning("Feature extraction interrupted. Proceeding to inference with extracted features...")
            killer.kill_now = False  # Reset to allow inference on partial data

        # ---------------------------------------------------------------------------
        # Phase II: Data Conditioning & Validation
        # ---------------------------------------------------------------------------
        logger.info("Phase II: Conditioning feature data...")
        df = pd.read_csv(working_csv)
        
        # Explicit Regex Filter for features
        feature_cols = [col for col in df.columns if re.match(r'^feature_\d+$', col)]
        
        # Custom Numerical Extraction Sort to ensure strict .joblib alignment
        # (prevents "feature_10" from sorting before "feature_2")
        feature_cols = sorted(feature_cols, key=lambda x: int(x.split('_')[1]))
        logger.debug(f"Identified feature columns: {feature_cols[:3]} ... {feature_cols[-3:]} (Total: {len(feature_cols)})")
        
        # Integrity check required by specification
        if len(feature_cols) != 2048:
            logger.critical(f"Integrity Check Failed: Model requires exactly 2048 columns. Found {len(feature_cols)}.")
            sys.exit(1)
        logger.debug("Integrity check passed: 2048 feature columns found.")

        # ---------------------------------------------------------------------------
        # Phase III: Inference & Scoring
        # ---------------------------------------------------------------------------
        logger.info(f"Phase III: Loading Inference Engine ({args.modelPath})...")
        try:
            model = joblib.load(args.modelPath)
        except Exception as e:
            logger.critical(f"Failed to load RF Model: {e}")
            sys.exit(1)

        predictions = []
        confidences = []
        batch_size = 200  # Adjust batch size based on expected memory constraints
        total_rows = len(df)
        
        # Chunked iteration to minimize peak memory during predict_proba
        for i in range(0, total_rows, batch_size):
            if killer.kill_now:
                was_interrupted = True
                logger.warning("Inference interrupted. Saving completed predictions...")
                break
            chunk = df[feature_cols].iloc[i:i + batch_size]
            probs = model.predict_proba(chunk)
            
            for prob in probs:
                # 0 -> FAKE, 1 -> REAL mapped by spec definition
                pred_idx = 0 if prob[0] > prob[1] else 1
                pred_label = "FAKE" if pred_idx == 0 else "REAL"
                
                # Confidence Calculation
                conf = max(prob[0], prob[1]) * 100
                
                logger.debug(f"Predicted: {pred_label} | Confidence: {conf:.2f}%")
                
                predictions.append(pred_label)
                confidences.append(conf)

        # ---------------------------------------------------------------------------
        # Phase IV: Result Consolidation
        # ---------------------------------------------------------------------------
        logger.info("Phase IV: Consolidating final report...")
        df['prediction'] = predictions
        df['confidence_score'] = confidences
        df['model_used'] = Path(args.modelPath).name
        df['extractor'] = 'Pre-Extracted CSV' if is_csv_input else args.featureExtractor
        df['weights'] = 'N/A' if is_csv_input else args.weights

        # Find original filename/filepath column gracefully
        path_col = next((col for col in ['filepath', 'image_filename', 'file_name'] if col in df.columns), None)
        if not path_col:
            # Fallback if somehow missing
            df['filepath'] = [f"item_{i}" for i in range(len(df))]
            path_col = 'filepath'
        elif path_col != 'filepath':
            df.rename(columns={path_col: 'filepath'}, inplace=True)

        # Enforce column order: filepath | prediction | confidence score | model used | extractor | weights | feature_0 ... feature_2047
        final_cols = ['filepath', 'prediction', 'confidence_score', 'model_used', 'extractor', 'weights'] + feature_cols
        df = df[final_cols]
        
        df.to_csv(args.output, index=False)
        logger.info(f"Report fully compiled and saved to: {args.output}")

        # Cleanup intermediate file
        if not is_csv_input and os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)

        # ---------------------------------------------------------------------------
        # Phase V: Visualization
        # ---------------------------------------------------------------------------
        if was_interrupted:
            logger.info("Skipping Visualization UI due to user interruption. Exiting cleanly.")
        elif args.showOutput and not is_csv_input:
            display_gallery(df)
        elif args.showOutput and is_csv_input:
            logger.warning("Visualization UI is disabled for raw CSV inputs as image paths are likely disconnected.")

    except Exception as e:
        logger.critical(f"Unhandled pipeline exception: {e}")
        sys.exit(1)
    finally:
        if not is_csv_input and os.path.exists(temp_csv_file):
            try:
                os.remove(temp_csv_file)
            except Exception:
                pass


if __name__ == "__main__":
    main()