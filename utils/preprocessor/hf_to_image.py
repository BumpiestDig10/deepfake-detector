import os
import csv
import json
from datasets import load_dataset

import centralLogging as cl
logger = cl.get_logger(console_level="INFO", file_level="DEBUG")

# Load the dataset
try:
    logger.info("Loading dataset")
    dataset = load_dataset('mkhLlamaLearn/dfdcpics2', split='train') 
    logger.info("Dataset loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load dataset: {e}")
    exit(1)

output_dir = "D:/02_Deepfake/dfdcpics2"
os.makedirs(output_dir, exist_ok=True)

# Prepare metadata files
csv_file = os.path.join(output_dir, "metadata.csv")
json_file = os.path.join(output_dir, "metadata.json")

metadata_list = []
csv_rows = []

for i, item in enumerate(dataset):
    if 'image' in item and item['image'] is not None:
        # Save the image
        image = item['image']
        image_filename = f"image_{i}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        
        # Collect metadata including labels ← THIS WAS MISSING!
        metadata = {
            "image_filename": image_filename,
            "index": i,
            "class": item.get('label', None),           # FAKE or REAL
            "original": item.get('original', None),
            "source": item.get('source', None)
        }
        
        metadata_list.append(metadata)
        csv_rows.append([image_filename, i, item.get('label', ''), 
                        item.get('original', ''), item.get('source', '')])
        
        logger.debug(f"Saved {image_path} with label: {item.get('label', 'N/A')}")
    else:
        logger.warning(f"Item {i} does not contain an image or image is None.")

# Save metadata as CSV
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image_filename', 'index', 'label', 'original', 'source'])
    writer.writerows(csv_rows)

# Save metadata as JSON
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(metadata_list, f, indent=2, ensure_ascii=False)

logger.info(f"Saved {len(metadata_list)} images with metadata")
logger.info(f"Metadata saved to: {csv_file} and {json_file}")
