'''
might need to build sthing like this
'''
try:
    import centralLogging as cl
    logger = cl.get_logger(console_level = "INFO", file_level = "DEBUG")
except Exception as e:
    print(f"Could not import Central Logger: {e}")
    exit(1)

logger.info("=== Image RGB Data Extractor ===")

try:
    import numpy as np
    from PIL import Image
    
    import os
    import argparse
    
    logger.debug("Imported all libraries successfully.")
except ImportError as e:
    logger.critical(f"Could not import necessary libraries. {e}")

# Load images from a directory
def load_images(filepath):
    loaded_images = []
    for file in filepath:
        loaded_image = np.array(Image.open(file))
        loaded_images.append(loaded_image)
    
    return loaded_images

# Writing the RGB values to Test.txt as matrix of dimensions [{(r,g,b)*no. of horizontal pixels} * no. of vertical pixels]
def matrix(loaded_images, output):
    for loaded_image in loaded_images:
        with open(f'{output}.txt', 'w') as f:
            for row in loaded_image:
                for pixel in row:
                    f.write(f'({pixel[0]},{pixel[1]},{pixel[2]}) ')
                f.write('\n')
                
    logger.info(f"RGB values have been written to {output}.txt")
    
def main():
    parser = argparse.ArgumentParser(
        description="Convert Image to (r,g,b) data in matrix of image dimensions"
    )

    # Add arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input folder. (Mandatory)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default='results/imageRGB/',
        help="Path to the output folder. Defaults to 'results/imageRGB/'"
    )
    
    args = parser.parse_args()
    
    try:
        loaded_images = load_images(args.input)
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        exit(1)
    
    try:
        matrix(loaded_images, args.output)
    except Exception as e:
        logger.error(f"Error writing as matrix: {e}")
        exit(1)
        
    logger.info("DONE!")
    exit(0)
    
if __name__ == "__main__":
    main()