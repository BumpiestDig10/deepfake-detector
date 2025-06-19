from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from PIL import Image, ExifTags
import io
import base64
import os
import math
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import exifread # A more robust EXIF reader for some tags
import logging
import sys

try:
    sys.path.append(os.path.abspath("../../metadata-parser/src"))
    import metadata_parser
except ImportError:
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical("CRITICAL ERROR: Could not import 'metadata_parser'. Make sure 'metadata_parser.py' is in the correct path.")
    sys.exit(1)
    
try:
    sys.path.append(os.path.abspath("../../metadata-parser/src"))
    from fileTypeIdentifier import FileTypeIdentifier
except ImportError:
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical("CRITICAL ERROR: Could not import 'FileTypeIdentifier'. Make sure 'fileTypeIdentifier.py' is in the same directory.")
    sys.exit(1)


app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = 'uploads'
#os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained ResNet50 model for feature extraction
# We will remove the final classification layer
try:
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    # The output of this model will be our image embedding
    print("ResNet50 model loaded successfully for feature extraction.")
except Exception as e:
    print(f"Error loading ResNet50 model: {e}")
    base_model = None # Handle cases where model cannot be loaded (e.g., no internet)


@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/extract_features', methods=['POST'])
def extract_features():
    """
    Handles image upload and extracts various features from the image.
    Returns JSON response with extracted features.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    if file:
        try:
            # Read image data as bytes
            image_bytes = file.read()
            img_pil = Image.open(io.BytesIO(image_bytes))

            # Convert PIL image to OpenCV format (numpy array)
            # OpenCV uses BGR by default, PIL uses RGB
            img_np = np.array(img_pil)
            if img_np.ndim == 2: # Grayscale image
                img_cv = img_np
            elif img_np.shape[2] == 4: # RGBA image
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else: # RGB image
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            features = {}

            # --- Part I: Foundational and Low-Level Image Metrics ---
            # 1.1. Pixel-Level and Global Statistics
            height, width = img_cv.shape[:2]
            features['resolution'] = f"{width} x {height} pixels"
            features['aspect_ratio'] = float(round(width / height, 2)) # Ensure float
            features['file_size_kb'] = float(round(len(image_bytes) / 1024, 2)) # Ensure float

            # Determine bit depth and number of channels
            num_channels = 1 if img_cv.ndim == 2 else img_cv.shape[2]
            bit_depth_per_channel = img_cv.dtype.itemsize * 8
            features['num_channels'] = int(num_channels) # Ensure int
            features['bit_depth'] = f"{bit_depth_per_channel * num_channels} bits"


            # 1.2. Intensity and Contrast Metrics
            # Convert to grayscale for overall metrics
            gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if num_channels > 1 else img_cv.copy()

            # Calculate statistics for grayscale image
            hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            mean_gray = float(np.mean(gray_img)) # Explicitly cast
            std_dev_gray = float(np.std(gray_img)) # Explicitly cast
            skewness_gray = calculate_skewness(hist_gray)
            kurtosis_gray = calculate_kurtosis(hist_gray)

            features['intensity_metrics'] = {
                'overall_brightness_mean_gray': round(mean_gray, 2),
                'overall_contrast_std_dev_gray': round(std_dev_gray, 2),
                'skewness_gray': round(skewness_gray, 2),
                'kurtosis_gray': round(kurtosis_gray, 2),
            }

            if num_channels > 1:
                # Calculate statistics for each color channel
                b_channel, g_channel, r_channel = cv2.split(img_cv)
                hist_b = cv2.calcHist([b_channel], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([g_channel], [0], None, [256], [0, 256])
                hist_r = cv2.calcHist([r_channel], [0], None, [256], [0, 256])

                features['intensity_metrics']['red_channel'] = {
                    'mean': round(float(np.mean(r_channel)), 2), # Explicitly cast
                    'std_dev': round(float(np.std(r_channel)), 2), # Explicitly cast
                    'skewness': round(calculate_skewness(hist_r), 2),
                    'kurtosis': round(calculate_kurtosis(hist_r), 2),
                }
                features['intensity_metrics']['green_channel'] = {
                    'mean': round(float(np.mean(g_channel)), 2), # Explicitly cast
                    'std_dev': round(float(np.std(g_channel)), 2), # Explicitly cast
                    'skewness': round(calculate_skewness(hist_g), 2),
                    'kurtosis': round(calculate_kurtosis(hist_g), 2),
                }
                features['intensity_metrics']['blue_channel'] = {
                    'mean': round(float(np.mean(b_channel)), 2), # Explicitly cast
                    'std_dev': round(float(np.std(b_channel)), 2), # Explicitly cast
                    'skewness': round(calculate_skewness(hist_b), 2),
                    'kurtosis': round(calculate_kurtosis(hist_b), 2),
                }

            # 1.3. EXIF Data
            exif_data = {}
            try:
                # Use PIL for common EXIF tags
                info = img_pil._getexif()
                if info:
                    for tag, value in info.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        exif_data[decoded] = str(value) # Convert all values to string for JSON

                # Use exifread for more detailed EXIF and GPS if available
                # Re-open file for exifread as it needs a file-like object
                f = io.BytesIO(image_bytes)
                tags = exifread.process_file(f)
                if 'Image Make' in tags: exif_data['Make'] = str(tags['Image Make'])
                if 'Image Model' in tags: exif_data['Model'] = str(tags['Image Model'])
                if 'EXIF DateTimeOriginal' in tags: exif_data['DateTimeOriginal'] = str(tags['EXIF DateTimeOriginal'])
                if 'EXIF FNumber' in tags: exif_data['FNumber'] = str(tags['EXIF FNumber'])
                if 'EXIF ExposureTime' in tags: exif_data['ExposureTime'] = str(tags['EXIF ExposureTime'])
                if 'EXIF ISOSpeedRatings' in tags: exif_data['ISOSpeedRatings'] = str(tags['EXIF ISOSpeedRatings'])
                if 'EXIF Flash' in tags: exif_data['Flash'] = str(tags['EXIF Flash'])

                if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                    lat_dms = [float(f.num) / float(f.den) for f in tags['GPS GPSLatitude'].values]
                    lon_dms = [float(f.num) / float(f.den) for f in tags['GPS GPSLongitude'].values]
                    lat_ref = str(tags['GPS GPSLatitudeRef'].values)
                    lon_ref = str(tags['GPS GPSLongitudeRef'].values)
                    lat_dd = convert_dms_to_dd(lat_dms, lat_ref)
                    lon_dd = convert_dms_to_dd(lon_dms, lon_ref)
                    exif_data['GPSCoordinates'] = f"{lat_dd:.6f}, {lon_dd:.6f}"

            except Exception as e:
                print(f"Error extracting EXIF data: {e}")
                exif_data['Note'] = "Could not extract EXIF data or no EXIF data present."
            features['exif_data'] = exif_data


            # --- Part II: Classical Feature Extraction for Machine Learning ---

            # 2.1. Color Histograms (32 Bins)
            # Histograms for 32 bins, normalized
            hist_size = 32
            ranges = [0, 256]

            if num_channels > 1:
                hist_r_32 = cv2.calcHist([r_channel], [0], None, [hist_size], ranges)
                hist_g_32 = cv2.calcHist([g_channel], [0], None, [hist_size], ranges)
                hist_b_32 = cv2.calcHist([b_channel], [0], None, [hist_size], ranges)
                features['color_histograms'] = {
                    'red_hist_32_bins': hist_r_32.flatten().astype(float).tolist(), # Ensure float list
                    'green_hist_32_bins': hist_g_32.flatten().astype(float).tolist(), # Ensure float list
                    'blue_hist_32_bins': hist_b_32.flatten().astype(float).tolist(), # Ensure float list
                }
            else:
                hist_gray_32 = cv2.calcHist([gray_img], [0], None, [hist_size], ranges)
                features['color_histograms'] = {
                    'gray_hist_32_bins': hist_gray_32.flatten().astype(float).tolist() # Ensure float list
                }

            # 2.2. Color Moments - already covered in 1.2 and directly outputted

            # 3.1 & 3.2. Gray-Level Co-occurrence Matrix (GLCM) and Haralick Texture Features
            # GLCM requires grayscale image
            # Normalize image to 8 gray levels for GLCM
            img_glcm = (gray_img / 32).astype(np.uint8) # 256/32 = 8 levels (0-7)

            # Distances: 1 pixel, Angles: 0, 45, 90, 135 degrees
            # For simplicity, calculate for (1, 0) and (1, 45) and average as common practice
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees

            # Ensure image depth is sufficient for GLCM (e.g., 8-bit)
            if img_glcm.dtype != np.uint8:
                img_glcm = cv2.normalize(img_glcm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            glcm_all = graycomatrix(img_glcm, distances=distances, angles=angles, levels=8,
                                    symmetric=True, normed=True)

            # Calculate Haralick features and average them
            props = ['contrast', 'correlation', 'energy', 'homogeneity']
            haralick_avg = {p: 0.0 for p in props} # Initialize with float

            for p in props:
                # Ensure the result of np.mean is converted to a standard float
                haralick_avg[p] = float(np.mean(graycoprops(glcm_all, p)))

            features['haralick_texture_features'] = {k: round(v, 4) for k, v in haralick_avg.items()}


            # 4.1. Edge and Corner Detection
            edges = cv2.Canny(gray_img, 100, 200) # Canny edge detector (low_threshold, high_threshold)
            features['canny_edge_count'] = int(np.sum(edges > 0)) # Count non-zero pixels (edges)
            features['canny_edge_density'] = float(round(features['canny_edge_count'] / (width * height), 4)) # Ensure float

            # Harris Corner Detector
            block_size = 2
            aperture_size = 3
            k = 0.04
            dst = cv2.cornerHarris(gray_img, block_size, aperture_size, k)
            dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX) # Normalize for visualization/thresholding
            # Threshold for corners (choose a suitable value based on image type)
            features['harris_corner_count'] = int(np.sum(dst > 0.01 * dst.max()))


            # 4.2 & 4.3 Geometric/Morphological & Hu Moments (Simplified)
            # Hu Moments require a binary image (segmented object).
            # For a general image without specific segmentation, we can calculate them
            # on a thresholded version of the whole image as a basic example.
            _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
            # Calculate moments
            moments = cv2.moments(binary_img)
            # Calculate Hu Moments
            hu_moments = cv2.HuMoments(moments)
            features['hu_moments'] = [round(float(m), 6) for m in hu_moments.flatten()]


            # 4.4. Frequency-Domain Features via Fourier Transform
            # Compute DFT
            dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
            # Shift the zero-frequency component to the center
            dft_shift = np.fft.fftshift(dft)
            # Compute magnitude spectrum
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1e-10) # Add epsilon to avoid log(0)

            # Quantify energy distribution (simplified: overall, low, mid, high frequency bands)
            rows, cols = gray_img.shape
            crow, ccol = rows // 2 , cols // 2     # Center of the spectrum

            # Define masks for low, mid, high frequencies
            # Low frequencies (central circle)
            r_low = min(crow, ccol) * 0.1 # 10% radius
            mask_low = np.zeros((rows, cols), np.uint8)
            cv2.circle(mask_low, (ccol, crow), int(r_low), 255, -1)

            # High frequencies (outer ring, excluding mid)
            r_high_inner = min(crow, ccol) * 0.5 # 50% radius for inner high-freq boundary
            r_high_outer = min(crow, ccol) * 0.9 # 90% radius for outer high-freq boundary
            mask_high = np.zeros((rows, cols), np.uint8)
            cv2.circle(mask_high, (ccol, crow), int(r_high_outer), 255, -1)
            cv2.circle(mask_high, (ccol, crow), int(r_high_inner), 0, -1) # Remove mid-inner portion

            # Mid frequencies (between low and high inner)
            mask_mid = np.zeros((rows, cols), np.uint8)
            cv2.circle(mask_mid, (ccol, crow), int(r_high_inner), 255, -1)
            cv2.circle(mask_mid, (ccol, crow), int(r_low), 0, -1) # Remove low freq portion

            # Apply masks to magnitude spectrum and calculate average energy (sum)
            total_spectrum_energy = float(np.sum(np.exp(magnitude_spectrum/20))) # Convert back from log scale, ensure float
            low_freq_energy = float(np.sum(np.exp(magnitude_spectrum[mask_low == 255]/20))) # Ensure float
            mid_freq_energy = float(np.sum(np.exp(magnitude_spectrum[mask_mid == 255]/20))) # Ensure float
            high_freq_energy = float(np.sum(np.exp(magnitude_spectrum[mask_high == 255]/20))) # Ensure float

            features['frequency_domain_features'] = {
                'total_spectrum_energy': round(total_spectrum_energy, 2),
                'low_frequency_energy': round(low_freq_energy, 2),
                'mid_frequency_energy': round(mid_freq_energy, 2),
                'high_frequency_energy': round(high_freq_energy, 2),
            }


            # --- Part III: Modern and Application-Specific Feature Extraction ---
            # 5.1. CNNs as Feature Extractors / 5.2. The Embedding Vector
            if base_model:
                try:
                    # Resize image to target size for ResNet (224x224)
                    img_resized = img_pil.resize((224, 224))
                    x = image.img_to_array(img_resized)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x) # Preprocess for ResNet50

                    # Get embedding
                    embedding = base_model.predict(x)
                    # Convert all elements to standard Python floats for JSON serialization
                    full_embedding = [float(val) for val in embedding.flatten()]
                    features['deep_learning_embedding_shape'] = list(embedding.shape)
                    features['deep_learning_embedding_snippet'] = full_embedding[:10]
                    # Note: full_embedding might be very large, consider sending only snippet if bandwidth is an issue
                    # features['deep_learning_embedding_full'] = full_embedding

                except Exception as e:
                    features['deep_learning_embedding_error'] = f"Error generating embedding: {e}"
            else:
                features['deep_learning_embedding_note'] = "Deep learning model not loaded (possibly due to network or installation issues)."

            # Encode the image for preview in HTML
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            features['image_preview_base64'] = img_str

            return jsonify(features)

        except Exception as e:
            print(f"Server-side error during feature extraction: {e}")
            return jsonify({'error': f'Server-side error during feature extraction: {e}'}), 500

# Helper functions for statistical calculations (skewness and kurtosis from histogram)
def calculate_skewness(hist):
    """Calculates skewness from a histogram."""
    total_pixels = np.sum(hist)
    if total_pixels == 0:
        return 0.0 # Ensure float return
    intensities = np.arange(len(hist))
    mean = np.sum(intensities * hist.flatten()) / total_pixels
    std_dev = np.sqrt(np.sum(np.power(intensities - mean, 2) * hist.flatten()) / total_pixels)
    if std_dev == 0:
        return 0.0 # Ensure float return
    skewness = np.sum(np.power(intensities - mean, 3) * hist.flatten()) / (total_pixels * np.power(std_dev, 3))
    return float(skewness) # Explicitly cast to float

def calculate_kurtosis(hist):
    """Calculates kurtosis from a histogram."""
    total_pixels = np.sum(hist)
    if total_pixels == 0:
        return 0.0 # Ensure float return
    intensities = np.arange(len(hist))
    mean = np.sum(intensities * hist.flatten()) / total_pixels
    std_dev = np.sqrt(np.sum(np.power(intensities - mean, 2) * hist.flatten()) / total_pixels)
    if std_dev == 0:
        return 0.0 # Ensure float return
    kurtosis = np.sum(np.power(intensities - mean, 4) * hist.flatten()) / (total_pixels * np.power(std_dev, 4)) - 3 # Excess kurtosis
    return float(kurtosis) # Explicitly cast to float

def convert_dms_to_dd(dms, ref):
    """Converts Degrees, Minutes, Seconds to Decimal Degrees."""
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    dd = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if ref in ['S', 'W']:
        dd *= -1
    return float(dd) # Ensure float return


if __name__ == '__main__':
    # Using threaded=True is for development, for production use a WSGI server like Gunicorn
    app.run(debug=True, threaded=True)

