import argparse
import logging
import sys
import os
import csv
from datetime import datetime
import io

# Set up logging
# Configure logging to output to stdout for CLI visibility
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Global variable for the deep learning model (will be loaded lazily)
base_model = None

# Placeholder for lazy-loaded modules
Image = None
cv2 = None
np = None
graycomatrix = None
graycoprops = None
ResNet50 = None
preprocess_input = None
image = None

def calculate_skewness(hist):
    """Calculates skewness from a histogram."""
    total_pixels = np.sum(hist)
    if total_pixels == 0:
        return 0.0
    intensities = np.arange(len(hist))
    mean = np.sum(intensities * hist.flatten()) / total_pixels
    std_dev = np.sqrt(np.sum(np.power(intensities - mean, 2) * hist.flatten()) / total_pixels)
    if std_dev == 0:
        return 0.0
    skewness = np.sum(np.power(intensities - mean, 3) * hist.flatten()) / (total_pixels * np.power(std_dev, 3))
    return float(skewness)


def calculate_kurtosis(hist):
    """Calculates kurtosis from a histogram."""
    total_pixels = np.sum(hist)
    if total_pixels == 0:
        return 0.0
    intensities = np.arange(len(hist))
    mean = np.sum(intensities * hist.flatten()) / total_pixels
    std_dev = np.sqrt(np.sum(np.power(intensities - mean, 2) * hist.flatten()) / total_pixels)
    if std_dev == 0:
        return 0.0
    kurtosis = np.sum(np.power(intensities - mean, 4) * hist.flatten()) / (total_pixels * np.power(std_dev, 4)) - 3
    return float(kurtosis)


def extract_features_from_image(image_bytes, image_filename="unknown_image"):
    """
    Extracts various features from image bytes.
    Args:
        image_bytes (bytes): The raw bytes of the image file.
        image_filename (str): The name of the image file, used for logging.
    Returns:
        dict: A dictionary containing the extracted features.
    """
    features = {"file_name": image_filename} # Initialize with file name
    try:
        # Use lazy-loaded Image
        img_pil = Image.open(io.BytesIO(image_bytes))

        # Convert PIL image to OpenCV format (numpy array)
        img_np = np.array(img_pil)
        
        num_channels = 1
        if img_np.ndim == 2:  # Grayscale image
            img_cv = img_np
        elif img_np.shape[2] == 4: # RGBA image
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            num_channels = 3
        elif img_np.shape[2] == 3: # RGB image
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            num_channels = 3
        else:
            logger.warning(f"Unsupported image channel count for {image_filename}: {img_np.shape[2]}. Proceeding with best guess.")
            img_cv = img_np # Fallback, may cause issues later

        # --- Part I: Foundational and Low-Level Image Metrics ---
        # 1.1. Pixel-Level and Global Statistics
        height, width = img_cv.shape[:2]
        features['resolution'] = f"{width}x{height}"
        features['width_pixels'] = width
        features['height_pixels'] = height
        features['aspect_ratio'] = round(width / height, 2)
        features['file_size_kb'] = round(len(image_bytes) / 1024, 2)

        bit_depth_per_channel = img_cv.dtype.itemsize * 8
        features['num_channels'] = num_channels
        features['bit_depth_total'] = bit_depth_per_channel * num_channels

        # 1.2. Intensity and Contrast Metrics
        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if num_channels > 1 else img_cv.copy()
        
        if gray_img.size == 0:
            raise ValueError("Grayscale image is empty after conversion.")

        hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        features['overall_brightness_mean_gray'] = round(float(np.mean(gray_img)), 2)
        features['overall_contrast_std_dev_gray'] = round(float(np.std(gray_img)), 2)
        features['skewness_gray'] = round(calculate_skewness(hist_gray), 2)
        features['kurtosis_gray'] = round(calculate_kurtosis(hist_gray), 2)

        if num_channels > 1:
            b_channel, g_channel, r_channel = cv2.split(img_cv)
            hist_r = cv2.calcHist([r_channel], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([g_channel], [0], None, [256], [0, 256])
            hist_b = cv2.calcHist([b_channel], [0], None, [256], [0, 256])

            features['red_channel_mean'] = round(float(np.mean(r_channel)), 2)
            features['red_channel_std_dev'] = round(float(np.std(r_channel)), 2)
            features['red_channel_skewness'] = round(calculate_skewness(hist_r), 2)
            features['red_channel_kurtosis'] = round(calculate_kurtosis(hist_r), 2)

            features['green_channel_mean'] = round(float(np.mean(g_channel)), 2)
            features['green_channel_std_dev'] = round(float(np.std(g_channel)), 2)
            features['green_channel_skewness'] = round(calculate_skewness(hist_g), 2)
            features['green_channel_kurtosis'] = round(calculate_kurtosis(hist_g), 2)

            features['blue_channel_mean'] = round(float(np.mean(b_channel)), 2)
            features['blue_channel_std_dev'] = round(float(np.std(b_channel)), 2)
            features['blue_channel_skewness'] = round(calculate_skewness(hist_b), 2)
            features['blue_channel_kurtosis'] = round(calculate_kurtosis(hist_b), 2)
        
        # --- Part II: Classical Feature Extraction for Machine Learning ---
        # 2.1. Color Histograms (32 Bins)
        hist_size = 32
        ranges = [0, 256]

        if num_channels > 1:
            hist_r_32 = cv2.calcHist([r_channel], [0], None, [hist_size], ranges)
            hist_g_32 = cv2.calcHist([g_channel], [0], None, [hist_size], ranges)
            hist_b_32 = cv2.calcHist([b_channel], [0], None, [hist_size], ranges)
            for i, val in enumerate(hist_r_32.flatten()):
                features[f'color_hist_red_{i}'] = float(val)
            for i, val in enumerate(hist_g_32.flatten()):
                features[f'color_hist_green_{i}'] = float(val)
            for i, val in enumerate(hist_b_32.flatten()):
                features[f'color_hist_blue_{i}'] = float(val)
        else:
            hist_gray_32 = cv2.calcHist([gray_img], [0], None, [hist_size], ranges)
            for i, val in enumerate(hist_gray_32.flatten()):
                features[f'color_hist_gray_{i}'] = float(val)

        # 3.1 & 3.2. Gray-Level Co-occurrence Matrix (GLCM) and Haralick Texture Features
        img_glcm = (gray_img / 32).astype(np.uint8)
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        if img_glcm.dtype != np.uint8:
            img_glcm = cv2.normalize(img_glcm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        glcm_all = graycomatrix(img_glcm, distances=distances, angles=angles, levels=8,
                                symmetric=True, normed=True)

        props = ['contrast', 'correlation', 'energy', 'homogeneity']
        for p in props:
            features[f'haralick_texture_{p}'] = round(float(np.mean(graycoprops(glcm_all, p))), 4)

        # 4.1. Edge and Corner Detection
        edges = cv2.Canny(gray_img, 100, 200)
        features['canny_edge_count'] = int(np.sum(edges > 0))
        features['canny_edge_density'] = round(features['canny_edge_count'] / (width * height), 4)

        block_size = 2
        aperture_size = 3
        k = 0.04
        dst = cv2.cornerHarris(gray_img, block_size, aperture_size, k)
        dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
        features['harris_corner_count'] = int(np.sum(dst > 0.01 * dst.max()))

        # 4.2 & 4.3 Geometric/Morphological & Hu Moments
        _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(binary_img)
        hu_moments = cv2.HuMoments(moments)
        for i, m in enumerate(hu_moments.flatten()):
            features[f'hu_moment_{i}'] = round(float(m), 6)

        # 4.4. Frequency-Domain Features via Fourier Transform
        dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-10)

        rows, cols = gray_img.shape
        crow, ccol = rows // 2, cols // 2

        r_low = min(crow, ccol) * 0.1
        mask_low = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask_low, (ccol, crow), int(r_low), 255, -1)

        r_high_inner = min(crow, ccol) * 0.5
        r_high_outer = min(crow, ccol) * 0.9
        mask_high = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask_high, (ccol, crow), int(r_high_outer), 255, -1)
        cv2.circle(mask_high, (ccol, crow), int(r_high_inner), 0, -1)

        mask_mid = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask_mid, (ccol, crow), int(r_high_inner), 255, -1)
        cv2.circle(mask_mid, (ccol, crow), int(r_low), 0, -1)

        total_spectrum_energy = float(np.sum(np.exp(magnitude_spectrum / 20)))
        low_freq_energy = float(np.sum(np.exp(magnitude_spectrum[mask_low == 255] / 20)))
        mid_freq_energy = float(np.sum(np.exp(magnitude_spectrum[mask_mid == 255] / 20)))
        high_freq_energy = float(np.sum(np.exp(magnitude_spectrum[mask_high == 255] / 20)))

        features['freq_domain_total_energy'] = round(total_spectrum_energy, 2)
        features['freq_domain_low_energy'] = round(low_freq_energy, 2)
        features['freq_domain_mid_energy'] = round(mid_freq_energy, 2)
        features['freq_domain_high_energy'] = round(high_freq_energy, 2)

        # --- Part III: Modern and Application-Specific Feature Extraction ---
        # 5.1. CNNs as Feature Extractors / 5.2. The Embedding Vector
        global base_model # Declare as global to modify the outer scope variable
        if base_model:
            try:
                # Use lazy-loaded image and preprocess_input
                img_resized = img_pil.resize((224, 224))
                x = image.img_to_array(img_resized)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                embedding = base_model.predict(x, verbose=0)
                full_embedding = embedding.flatten()
                
                for i in range(len(full_embedding)):
                    features[f'deep_embed_{i}'] = float(full_embedding[i])

            except Exception as e:
                logger.error(f"Error generating deep learning embedding for {image_filename}: {e}")
                features['deep_embed_error'] = f"Error generating embedding: {e}"
        else:
            features['deep_embed_note'] = "Deep learning model not loaded."

    except Exception as e:
        logger.error(f"An unexpected error occurred while processing image {image_filename}: {e}", exc_info=True)
        features['processing_error'] = f"Processing error: {e}"

    return features


def get_all_possible_headers(list_of_feature_dicts):
    """
    Analyzes a list of feature dictionaries to determine all unique keys,
    including flattened nested keys and expanded list keys, to form a comprehensive CSV header.
    """
    all_keys = set()
    for features_dict in list_of_feature_dicts:
        for key, value in features_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (list, np.ndarray)):
                        max_list_len = 32 if 'hist' in sub_key else (7 if 'hu_moment' in sub_key else 2048) # Assuming 2048 for full deep embed
                        for i in range(max_list_len):
                            all_keys.add(f"{key}_{sub_key}_{i}")
                    else:
                        all_keys.add(f"{key}_{sub_key}")
            elif isinstance(value, (list, np.ndarray)):
                max_list_len = 32 if 'hist' in key else (7 if 'hu_moment' in key else 2048)
                for i in range(max_list_len):
                    all_keys.add(f"{key}_{i}")
            else:
                all_keys.add(key)
    
    sorted_keys = sorted(list(all_keys))
    if 'file_name' in sorted_keys:
        sorted_keys.remove('file_name')
        sorted_keys.insert(0, 'file_name')
    return sorted_keys


def main():
    parser = argparse.ArgumentParser(description="CLI tool to extract features from images.")
    parser.add_argument("--dir", type=str, required=True,
                        help="Input directory path containing images.")
    parser.add_argument("--out", type=str,
                        help="Output CSV file path. Defaults to ../results/[timestamp].csv.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output to the terminal (prints all extracted features for each image).")

    args = parser.parse_args()

    input_dir = args.dir
    output_file_path = args.out
    verbose = args.verbose

    # --- Step 1: Validate input arguments and directory first ---
    if not os.path.isdir(input_dir):
        logger.error(f"Error: Input directory not found: '{input_dir}'")
        sys.exit(1)

    if output_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
        os.makedirs(default_output_dir, exist_ok=True)
        output_file_path = os.path.join(default_output_dir, f"features_{timestamp}.csv")
        logger.info(f"Output file path not specified. Defaulting to: '{output_file_path}'")
    else:
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: '{output_dir}'")

    # --- Step 2: If validation passes, then import heavy libraries ---
    global Image, cv2, np, graycomatrix, graycoprops, ResNet50, preprocess_input, image, base_model
    try:
        from PIL import Image as PIL_Image # Renamed to avoid conflict
        import cv2 as OpenCV_cv2 # Renamed to avoid conflict
        import numpy as np_import # Renamed to avoid conflict
        from skimage.feature import graycomatrix as sk_graycomatrix, graycoprops as sk_graycoprops
        import tensorflow as tf
        from tensorflow.keras.applications.resnet50 import ResNet50 as Keras_ResNet50, preprocess_input as Keras_preprocess_input
        from tensorflow.keras.preprocessing import image as Keras_image # Renamed to avoid conflict

        Image = PIL_Image
        cv2 = OpenCV_cv2
        np = np_import
        graycomatrix = sk_graycomatrix
        graycoprops = sk_graycoprops
        preprocess_input = Keras_preprocess_input
        image = Keras_image # Assign the imported Keras image module

        # Load pre-trained ResNet50 model for feature extraction (moved here)
        logger.info("Attempting to load ResNet50 model (this may take a moment and requires internet for first download)...")
        base_model = Keras_ResNet50(weights='imagenet', include_top=False, pooling='avg')
        logger.info("ResNet50 model loaded successfully for feature extraction.")

    except ImportError as e:
        logger.error(f"Error importing required libraries: {e}. "
                     "Please ensure all dependencies are installed: "
                     "'pip install scikit-image tensorflow opencv-python Pillow numpy'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during library import or model loading: {e}", exc_info=True)
        sys.exit(1)

    # List of image file extensions to process
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp')

    all_extracted_features_raw = []
    
    logger.info(f"Scanning directory for images: '{input_dir}'")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        logger.warning(f"No image files found in '{input_dir}' with extensions {image_extensions}. Exiting.")
        sys.exit(0)

    for i, filename in enumerate(image_files):
        file_path = os.path.join(input_dir, filename)
        logger.info(f"Processing image {i+1}/{len(image_files)}: '{filename}'")
        try:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            
            features = extract_features_from_image(image_bytes, filename)
            all_extracted_features_raw.append(features)

            if verbose:
                logger.info(f"--- Features for '{filename}' ---")
                import json
                logger.info(json.dumps(features, indent=4))
                logger.info("-" * (len(filename) + 25))

        except IOError as e:
            logger.error(f"Failed to read image file '{filename}': {e}. Skipping.")
            all_extracted_features_raw.append({"file_name": filename, "error": f"Failed to read file: {e}"})
        except Exception as e:
            logger.error(f"An unexpected error occurred during processing of '{filename}': {e}. Skipping.", exc_info=True)
            all_extracted_features_raw.append({"file_name": filename, "error": f"Unexpected error during processing: {e}"})

    csv_headers = get_all_possible_headers(all_extracted_features_raw)

    logger.info(f"Writing results to CSV: '{output_file_path}'")
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            
            for features_dict in all_extracted_features_raw:
                flat_row = {}
                for header in csv_headers:
                    flat_row[header] = None
                
                for key, value in features_dict.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (list, np.ndarray)):
                                for i, item in enumerate(sub_value):
                                    header_key = f"{key}_{sub_key}_{i}"
                                    if header_key in csv_headers:
                                        flat_row[header_key] = item
                            else:
                                header_key = f"{key}_{sub_key}"
                                if header_key in csv_headers:
                                    flat_row[header_key] = sub_value
                    elif isinstance(value, (list, np.ndarray)):
                        for i, item in enumerate(value):
                            header_key = f"{key}_{i}"
                            if header_key in csv_headers:
                                flat_row[header_key] = item
                    else:
                        if key in csv_headers:
                            flat_row[key] = value
                
                writer.writerow(flat_row)
        logger.info(f"Feature extraction complete. Results saved to '{output_file_path}'.")
    except IOError as e:
        logger.error(f"Could not write to output file '{output_file_path}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing CSV: {e}", exc_info=True)

if __name__ == "__main__":
    main()
