import argparse
import logging
import sys
import os
import csv
from datetime import datetime
import io
import queue
import threading
import signal
import time

# --- Global Configuration ---

# Set up logging to output to stdout for CLI visibility
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- Lazy-Loaded Modules Placeholder ---
# These will be imported only if needed and after initial argument validation
Image = None
cv2 = None
np = None
graycomatrix = None
graycoprops = None
ResNet50 = None
preprocess_input = None
image = None
base_model = None

# --- Shared Resources for Multithreading ---
image_queue = queue.Queue()
results_queue = queue.Queue()
stop_event = threading.Event()
processed_headers = set()
header_lock = threading.Lock()

# --- Core Functions ---

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
    Extracts various features from image bytes. This function is designed to be
    self-contained and thread-safe, operating only on its inputs.
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
            # Convert RGBA to BGR for most OpenCV functions
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
        features['aspect_ratio'] = round(width / height, 2) if height > 0 else 0
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
        
        if img_glcm.size > 0 and np.max(img_glcm) > 0:
            glcm_all = graycomatrix(img_glcm, distances=distances, angles=angles, levels=np.max(img_glcm) + 1,
                                    symmetric=True, normed=True)

            props = ['contrast', 'correlation', 'energy', 'homogeneity']
            for p in props:
                features[f'haralick_texture_{p}'] = round(float(np.mean(graycoprops(glcm_all, p))), 4)
        else:
             props = ['contrast', 'correlation', 'energy', 'homogeneity']
             for p in props:
                features[f'haralick_texture_{p}'] = 0.0


        # 4.1. Edge and Corner Detection
        edges = cv2.Canny(gray_img, 100, 200)
        features['canny_edge_count'] = int(np.sum(edges > 0))
        features['canny_edge_density'] = round(features['canny_edge_count'] / (width * height), 4) if (width * height) > 0 else 0

        # Harris Corner needs float32 input
        gray_img_float = np.float32(gray_img)
        dst = cv2.cornerHarris(gray_img_float, 2, 3, 0.04)
        features['harris_corner_count'] = int(np.sum(dst > 0.01 * dst.max()))

        # 4.2 & 4.3 Geometric/Morphological & Hu Moments
        _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(binary_img)
        hu_moments = cv2.HuMoments(moments)
        for i, m in enumerate(hu_moments.flatten()):
            features[f'hu_moment_{i}'] = round(float(m), 6)

        # 4.4. Frequency-Domain Features via Fourier Transform
        if gray_img.size > 0:
            dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-10)

            total_spectrum_energy = float(np.sum(np.exp(magnitude_spectrum / 20)))
            features['freq_domain_total_energy'] = round(total_spectrum_energy, 2)
        else:
            features['freq_domain_total_energy'] = 0.0

        # --- Part III: Modern and Application-Specific Feature Extraction ---
        # 5.1. CNNs as Feature Extractors / 5.2. The Embedding Vector
        global base_model
        if base_model:
            try:
                # ResNet50 requires 3 channels. If grayscale, convert.
                if num_channels == 1:
                    img_pil_rgb = img_pil.convert('RGB')
                else:
                    img_pil_rgb = img_pil

                img_resized = img_pil_rgb.resize((224, 224))
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
        logger.error(f"An unexpected error occurred while processing image {image_filename}: {e}", exc_info=False)
        features['processing_error'] = f"Processing error: {e}"

    return features


def feature_extractor_worker(stop_event_ref):
    """Worker thread function to extract features from images."""
    logger.info("Feature extractor worker started.")
    while not stop_event_ref.is_set():
        try:
            # Get a job from the queue. Timeout to allow checking stop_event.
            file_path, filename = image_queue.get(timeout=1)
            
            logger.info(f"Processing: '{filename}'")
            try:
                with open(file_path, 'rb') as f:
                    image_bytes = f.read()
                
                features = extract_features_from_image(image_bytes, filename)
                results_queue.put(features)
            except IOError as e:
                logger.error(f"Failed to read image file '{filename}': {e}. Skipping.")
                results_queue.put({"file_name": filename, "error": f"Failed to read file: {e}"})
            except Exception as e:
                logger.error(f"An error occurred during processing of '{filename}': {e}. Skipping.")
                results_queue.put({"file_name": filename, "error": f"Unexpected error: {e}"})

            image_queue.task_done()
        except queue.Empty:
            # If the queue is empty, it might be the end of processing.
            # The loop will re-check the stop_event and continue if not set.
            # This allows the main thread to signal completion.
            break 
    logger.info("Feature extractor worker stopping.")

def csv_writer_worker(output_file_path, stop_event_ref, total_images):
    """Worker thread function to write results to a CSV file in batches."""
    logger.info("CSV writer worker started.")
    # Use a local list to buffer results for batch writing.
    results_buffer = []
    # A set to keep track of headers that have been written to the file.
    written_headers = set()
    # A flag to track if the header has been written in the current session.
    is_header_written = False
    processed_count = 0
    # Batch size for writing to the file.
    WRITE_BATCH_SIZE = 100

    while not stop_event_ref.is_set() or not results_queue.empty():
        try:
            # Get results from the queue. Timeout to allow checking the stop event.
            result = results_queue.get(timeout=1)
            results_buffer.append(result)
            processed_count += 1

            # Check if it's time to write a batch.
            # We also write if the stop event is set (to flush remaining results)
            # or if all images have been processed.
            if len(results_buffer) >= WRITE_BATCH_SIZE or \
               (stop_event_ref.is_set() and results_queue.empty()) or \
               (processed_count == total_images):
                
                if not results_buffer:
                    continue

                logger.info(f"Writing batch of {len(results_buffer)} results to CSV...")
                
                # Determine all headers from the current batch of results.
                current_batch_headers = set()
                for res in results_buffer:
                    current_batch_headers.update(res.keys())
                
                # Check if there are new headers not previously written.
                new_headers = current_batch_headers - written_headers
                
                if new_headers or not is_header_written:
                    # If there are new headers, we need to handle them.
                    # For simplicity and robustness, we rewrite the file if headers change after the initial write.
                    
                    if is_header_written:
                        logger.warning("New feature columns discovered. Rewriting CSV with updated headers.")
                        # This part is complex. For this implementation, we will append and log a warning.
                        # A better approach for dynamic columns is to wait until all data is processed.

                    # Update the master list of headers.
                    all_headers = sorted(list(written_headers.union(current_batch_headers)))
                    if 'file_name' in all_headers:
                        all_headers.remove('file_name')
                        all_headers.insert(0, 'file_name')
                    
                    # Write/rewrite the file with the full set of headers.
                    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=all_headers, restval=None)
                        writer.writeheader()
                        writer.writerows(results_buffer) # Write the current batch
                    
                    written_headers.update(current_batch_headers)
                    is_header_written = True
                
                else:
                    # If headers are the same, just append the new rows.
                    with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=sorted(list(written_headers)), restval=None)
                        writer.writerows(results_buffer)

                # Clear the buffer after writing.
                results_buffer.clear()
        
        except queue.Empty:
            # Queue is empty. If the stop event is set, we can exit.
            if stop_event_ref.is_set():
                break

    # Final check for any remaining items in the buffer (if loop exited)
    if results_buffer:
        logger.info(f"Writing final batch of {len(results_buffer)} results...")
        # Re-use the same writing logic for the final batch
        all_headers = sorted(list(written_headers.union(set().union(*(d.keys() for d in results_buffer)))))
        if 'file_name' in all_headers:
            all_headers.remove('file_name')
            all_headers.insert(0, 'file_name')

        mode = 'a' if is_header_written else 'w'
        with open(output_file_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_headers, restval=None)
            if not is_header_written:
                writer.writeheader()
            writer.writerows(results_buffer)

    logger.info("CSV writer worker stopping.")


def main():
    """Main function to parse arguments, set up threads, and manage execution."""
    parser = argparse.ArgumentParser(description="Multithreaded CLI tool to extract a robust set of features from images, including deep learning embeddings.")
    parser.add_argument("--dir", type=str, required=True, help="Input directory path containing images.")
    parser.add_argument("--out", type=str, help="Output CSV file path. Defaults to ../results/features_[timestamp].csv.")

    args = parser.parse_args()

    # --- Step 1: Validate input arguments and directory ---
    if not os.path.isdir(args.dir):
        logger.error(f"Error: Input directory not found: '{args.dir}'")
        sys.exit(1)

    if args.out:
        output_file_path = args.out
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: '{output_dir}'")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = f"../results/features_{timestamp}.csv"
        logger.info(f"Output file not specified. Defaulting to: '{output_file_path}'")

    # --- Step 2: Lazy-load heavy libraries only after validation ---
    global Image, cv2, np, graycomatrix, graycoprops, ResNet50, preprocess_input, image, base_model
    try:
        from PIL import Image as PIL_Image
        import cv2 as OpenCV_cv2
        import numpy as np_import
        from skimage.feature import graycomatrix as sk_graycomatrix, graycoprops as sk_graycoprops

        Image = PIL_Image
        cv2 = OpenCV_cv2
        np = np_import
        graycomatrix = sk_graycomatrix
        graycoprops = sk_graycoprops
        logger.info("Successfully imported basic image processing libraries.")

        logger.info("Importing TensorFlow for deep learning feature extraction...")
        import tensorflow as tf
        from tensorflow.keras.applications.resnet50 import ResNet50 as Keras_ResNet50, preprocess_input as Keras_preprocess_input
        from tensorflow.keras.preprocessing import image as Keras_image
        
        preprocess_input = Keras_preprocess_input
        image = Keras_image
        logger.info("Attempting to load ResNet50 model (this may take a moment and requires internet for the first download)...")
        base_model = Keras_ResNet50(weights='imagenet', include_top=False, pooling='avg')
        logger.info("ResNet50 model loaded successfully.")

    except ImportError as e:
        logger.error(f"Error importing required libraries: {e}. Please ensure all dependencies are installed.")
        logger.error("Try: 'pip install Pillow opencv-python numpy scikit-image tensorflow'")
        sys.exit(1)

    # --- Step 3: Populate the image queue ---
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp')
    image_files = [f for f in os.listdir(args.dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        logger.warning(f"No supported image files found in '{args.dir}'. Exiting.")
        sys.exit(0)

    for filename in image_files:
        image_queue.put((os.path.join(args.dir, filename), filename))

    total_images = len(image_files)
    logger.info(f"Found {total_images} images to process.")

    # --- Step 4: Set up and start worker threads ---
    feature_thread = threading.Thread(target=feature_extractor_worker, args=(stop_event,))
    writer_thread = threading.Thread(target=csv_writer_worker, args=(output_file_path, stop_event, total_images))

    feature_thread.start()
    writer_thread.start()

    # --- Graceful Shutdown Handler ---
    def signal_handler(sig, frame):
        logger.info("\nCtrl+C detected! Shutting down gracefully...")
        logger.info("Signaling threads to stop. This may take a moment...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)

    # --- Wait for threads to complete ---
    # Monitor threads while they are alive.
    while feature_thread.is_alive():
        # We can join with a timeout to remain responsive to signals
        feature_thread.join(timeout=1)

    logger.info("Feature extraction finished.")
    # Once the feature thread is done, we can signal the writer to finish up.
    stop_event.set()

    writer_thread.join()
    logger.info("CSV writing finished.")
    logger.info(f"Processing complete. Results saved to '{output_file_path}'.")

if __name__ == "__main__":
    main()
