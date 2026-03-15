import hashlib
import argparse
import os
import centralLogging as cl

def get_file_hash(file_path):
    hash_sha = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(), b""):
                hash_sha.update(chunk)
    except Exception as e:
        logger.error(f"Error occurred while reading file: {e}")
    return hash_sha.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Calculate the SHA-256 hash of a file.")
    parser.add_argument("--input", required=True, help="Path to the file to be hashed.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        logger.error(f"File not found: {args.input}")
        return

    file_hash = get_file_hash(args.input)
    logger.info(f"SHA-256 hash of {args.input}: {file_hash}")
    
    # Save hash to file
    hash_output_path = f"{args.input}.sha256"
    try:
        with open(hash_output_path, "w") as f:
            f.write(file_hash)
        logger.info(f"Hash saved to: {hash_output_path}")
    except Exception as e:
        logger.error(f"Failed to save hash to file: {e}")
        
if __name__ == "__main__":
    logger = cl.get_logger(console_level="INFO", file_level="DEBUG")
    main()