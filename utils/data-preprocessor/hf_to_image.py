import os
import csv
import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('JamieWithofs/Deepfake-and-real-images-4', split='train') 
output_dir = "./Deepfake-and-real-images-4"
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
        
        print(f"Saved {image_path} with label: {item.get('label', 'N/A')}")
    else:
        print(f"Item {i} does not contain an image or image is None.")

# Save metadata as CSV
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image_filename', 'index', 'label', 'original', 'source'])
    writer.writerows(csv_rows)

# Save metadata as JSON
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(metadata_list, f, indent=2, ensure_ascii=False)

print(f"Saved {len(metadata_list)} images with metadata")
print(f"Metadata saved to: {csv_file} and {json_file}")
