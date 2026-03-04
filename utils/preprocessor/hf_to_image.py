import os
import csv
import json
import argparse
from datasets import load_dataset
from huggingface_hub import login

import centralLogging as cl
logger = cl.get_logger(console_level="INFO", file_level="DEBUG")

def main():
    # ── Argument parsing ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Download and export a Hugging Face dataset.")

    parser.add_argument(
        "--dataset", "-d", "-input", "-i",
        type=str,
        help="Hugging Face dataset name (mandatory, e.g. 'prithivMLmods/Deepfake-vs-Real')"
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        default='train',
        help="Split of the dataset to download (e.g. 'train', 'test'). Defaults to 'train'."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for images and metadata. "
            "Defaults to the dataset name with '/' replaced by '_'."
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="Hugging Face access token."
    )

    args = parser.parse_args()

    # ── Optional HuggingFace authentication ──────────────────────────────────────
    if args.token:
        try:
            logger.info(f"Logging in to Hugging Face")
            login(token=args.token)
            logger.info("Hugging Face login successful.")
        except Exception as e:
            logger.error(f"Hugging Face login failed: {e}")
            exit(1)
    else:
        logger.info("No credentials provided — attempting unauthenticated access.")

    # ── Resolve output directory ──────────────────────────────────────────────────
    output_dir = args.output if args.output else args.dataset.replace("/", "_")

    # ── Load dataset ──────────────────────────────────────────────────────────────
    try:
        logger.info(f"Loading dataset: '{args.dataset}'")
        dataset = load_dataset(args.dataset, split=args.split)
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load dataset: {e}")
        exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # ── Prepare metadata files ────────────────────────────────────────────────────
    csv_file = os.path.join(output_dir, "metadata.csv")
    metadata_list = []
    csv_rows = []

    for i, item in enumerate(dataset):
        if 'image' in item and item['image'] is not None:
            image = item['image']
            image_filename = f"image_{i}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)

            metadata = {
                "image_filename": image_filename,
                "index": i,
                "class": item.get('label', None),
                "original": item.get('original', None),
                "source": item.get('source', None)
            }

            metadata_list.append(metadata)
            csv_rows.append([image_filename, i, item.get('label', ''),
                            item.get('original', ''), item.get('source', '')])

            logger.debug(f"Saved {image_path} with label: {item.get('label', 'N/A')}")
        else:
            logger.warning(f"Item {i} does not contain an image or image is None.")

    # ── Save metadata ─────────────────────────────────────────────────────────────
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_filename', 'index', 'label', 'original', 'source'])
        writer.writerows(csv_rows)

    logger.info(f"Saved {len(metadata_list)} images to: {output_dir}")
    logger.info(f"Metadata saved to: {csv_file}")
    
if __name__ == "__main__":
    main()