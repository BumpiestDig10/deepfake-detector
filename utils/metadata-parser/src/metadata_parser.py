#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metadata_parser.py

This script serves as the main orchestrator for a multi-layered metadata
extraction process. It processes files from a target directory and extracts
metadata using Layer 0 (OS), Layer 1 (ExifTool, Tika), Layer 2 (specialized
Python libraries), and Layer 3 (Hachoir for binary analysis) techniques. All
extracted data is consolidated and saved to a single CSV file.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
Run from the command line, specifying the input directory.

Required:
  --dir DIR    Path to the directory with files to process.

Optional:
  --output OUTPUT_CSV  Path to save the output CSV file.
                       (Default: ../results/metadata_results.csv)
  -v, --verbose      Increase verbosity. -v for progress bar, -vv for detailed logs.

Example:
  python metadata_parser.py --dir /path/to/my_documents --output /path/to/results.csv -vv

-------------------------------------------------------------------------------
PREREQUISITES
-------------------------------------------------------------------------------
1. REQUIRED MODULE:
   - This script requires the 'fileTypeIdentifier.py' file to be present
     in the same directory.

2. PYTHON LIBRARIES:
   - pip install python-magic pyexiftool tika-client Pillow mutagen pypdf python-docx hachoir tqdm

3. EXTERNAL TOOLS:
   (See previous versions for detailed installation instructions)
   - libmagic
   - ExifTool
   - Java & Apache Tika Server (must be running)

-------------------------------------------------------------------------------
"""
import os
import sys
import logging
import datetime
import time
import collections.abc
import argparse
import csv
from pathlib import Path
from tqdm import tqdm

# --- Dependency Imports & Checks ---

# Custom Module
try:
    from fileTypeIdentifier import FileTypeIdentifier
except ImportError:
    # Use a basic logger until setup is complete
    logging.basicConfig(level=logging.CRITICAL)
    logging.critical("CRITICAL ERROR: Could not import 'FileTypeIdentifier'. Make sure 'fileTypeIdentifier.py' is in the same directory.")
    sys.exit(1)

# Layer 1 Tools
try:
    import exiftool
    EXIFTOOL_AVAILABLE = True
except ImportError:
    EXIFTOOL_AVAILABLE = False
try:
    from tika import parser as tika_parser
    TIKA_AVAILABLE = True
except ImportError:
    TIKA_AVAILABLE = False

# Layer 2 Libraries
try:
    from PIL import Image, ExifTags
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
try:
    import mutagen
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Layer 3 Library
try:
    import hachoir.metadata
    import hachoir.stream
    HACHOIR_AVAILABLE = True
except ImportError:
    HACHOIR_AVAILABLE = False

# Platform-specific imports
try:
    import pwd
    import grp
    UNIX_SYSTEM = True
except ImportError:
    UNIX_SYSTEM = False

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(verbosity: int):
    """Configures logging based on the verbosity level."""
    # Determine console logging level
    if verbosity >= 2:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING

    # Create logs directory in the project's base directory (one level up)
    script_path = Path(__file__).resolve()
    log_dir = script_path.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Generate a timestamped log file name
    log_file_name = f"extraction_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = log_dir / log_file_name
    
    # Configure root logger to capture everything at INFO level for the file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create file handler - always logs at INFO level
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Create console handler - logs at the level determined by verbosity
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)-8s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Log initial warnings for missing dependencies now that logging is configured
    if not EXIFTOOL_AVAILABLE: logging.warning("pyexiftool not found. Layer 1 ExifTool extraction skipped.")
    if not TIKA_AVAILABLE: logging.warning("tika-client not found. Layer 1 Tika extraction skipped.")
    if not PILLOW_AVAILABLE: logging.warning("Pillow not found. Layer 2 image extraction skipped.")
    if not MUTAGEN_AVAILABLE: logging.warning("Mutagen not found. Layer 2 audio extraction skipped.")
    if not PYPDF_AVAILABLE: logging.warning("pypdf not found. Layer 2 PDF extraction skipped.")
    if not DOCX_AVAILABLE: logging.warning("python-docx not found. Layer 2 DOCX extraction skipped.")
    if not HACHOIR_AVAILABLE: logging.warning("hachoir not found. Layer 3 binary analysis skipped.")
    if not UNIX_SYSTEM: logging.info("Not a UNIX-like system. File owner/group names not extracted.")

# =============================================================================
# METADATA PARSER FUNCTIONS (Layers 0, 1, 2, 3)
# =============================================================================

# --- Layer 0 ---
def extract_os_metadata(filepath: str) -> dict:
    try:
        stat_info = os.stat(filepath)
        metadata = {
            "OS.FileName": os.path.basename(filepath),
            "OS.FilePath": os.path.abspath(filepath),
            "OS.FileSize_Bytes": stat_info.st_size,
            "OS.ModTime_UTC": datetime.datetime.fromtimestamp(stat_info.st_mtime, datetime.timezone.utc).isoformat(),
            "OS.AccessTime_UTC": datetime.datetime.fromtimestamp(stat_info.st_atime, datetime.timezone.utc).isoformat(),
            "OS.CreateTime_UTC": datetime.datetime.fromtimestamp(stat_info.st_ctime, datetime.timezone.utc).isoformat(),
            "OS.Permissions": oct(stat_info.st_mode)[-3:],
        }
        if UNIX_SYSTEM:
            metadata["OS.Owner_Name"] = pwd.getpwuid(stat_info.st_uid).pw_name
            metadata["OS.Group_Name"] = grp.getgrgid(stat_info.st_gid).gr_name
        return metadata
    except Exception as e:
        return {"Error.Layer0": f"OS metadata extraction failed: {e}"}

# --- Layer 1 ---
def extract_exiftool_metadata(filepath: str) -> dict:
    if not EXIFTOOL_AVAILABLE: return {}
    try:
        with exiftool.ExifTool() as et:
            return et.execute_json(filepath)[0]
    except Exception as e:
        return {"Error.ExifTool": str(e)}

def extract_tika_metadata(filepath: str) -> dict:
    if not TIKA_AVAILABLE: return {}
    try:
        parsed = tika_parser.from_file(filepath)
        return parsed.get("metadata", {})
    except Exception as e:
        if "ConnectionRefusedError" in str(e):
             logging.critical("CRITICAL: Tika connection failed. Is Tika Server running? Aborting.")
             sys.exit(1)
        return {"Error.Tika": str(e)}

# --- Layer 2 ---
def extract_pillow_metadata(filepath: str) -> dict:
    if not PILLOW_AVAILABLE: return {}
    try:
        with Image.open(filepath) as img:
            pillow_meta = {"Pillow.Format": img.format, "Pillow.Mode": img.mode, "Pillow.Size": img.size}
            if hasattr(img, '_getexif'):
                exif_data = img._getexif()
                if exif_data:
                    decoded_exif = {ExifTags.TAGS.get(key, key): val for key, val in exif_data.items()}
                    pillow_meta["Pillow.EXIF"] = decoded_exif
            return pillow_meta
    except Exception as e:
        return {"Error.Pillow": str(e)}

def extract_mutagen_metadata(filepath: str) -> dict:
    if not MUTAGEN_AVAILABLE: return {}
    try:
        audio = mutagen.File(filepath, easy=True)
        return {"Mutagen": dict(audio)}
    except Exception as e:
        return {"Error.Mutagen": str(e)}

def extract_pypdf_metadata(filepath: str) -> dict:
    if not PYPDF_AVAILABLE: return {}
    try:
        reader = PdfReader(filepath)
        return {"PyPDF": {k:v for k,v in reader.metadata.items()}}
    except Exception as e:
        return {"Error.PyPDF": str(e)}

def extract_docx_metadata(filepath: str) -> dict:
    if not DOCX_AVAILABLE: return {}
    try:
        props = docx.Document(filepath).core_properties
        prop_dict = {p: getattr(props, p) for p in dir(props) if not p.startswith('_') and not callable(getattr(props, p))}
        return {"Docx": prop_dict}
    except Exception as e:
        return {"Error.Docx": str(e)}

# --- Layer 3 ---
def extract_hachoir_metadata(filepath: str) -> dict:
    if not HACHOIR_AVAILABLE: return {}
    logging.info(f"[Layer 3] Routing to Hachoir for binary analysis: {os.path.basename(filepath)}")
    hachoir_meta = {}
    try:
        stream = hachoir.stream.FileInputStream(filepath)
        metadata = hachoir.metadata.extractMetadata(stream)
        if metadata:
            for item in metadata:
                if item.values:
                    key = item.key.replace('-', '_')
                    hachoir_meta[key] = [v.text for v in item.values]
            return {"Hachoir": hachoir_meta}
        return {}
    except Exception as e:
        return {"Error.Hachoir": str(e)}


# =============================================================================
# METADATA CONSOLIDATION & ORCHESTRATION
# =============================================================================
def flatten_dict(d: collections.abc.Mapping, parent_key: str = '', sep: str = '.') -> dict:
    """Flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        clean_k = str(k).replace(':', '_').replace(' ', '_').replace('/', '_')
        new_key = f"{parent_key}{sep}{clean_k}" if parent_key else clean_k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, '; '.join(map(str, v))))
        elif isinstance(v, bytes):
            items.append((new_key, v.decode('utf-8', errors='ignore')))
        else:
            items.append((new_key, v))
    return dict(items)

def process_file(filepath: str, file_identifier: FileTypeIdentifier) -> dict:
    """Processes a single file through all relevant extraction layers."""
    logging.info(f"---------------- Processing file: {filepath} ----------------")

    all_metadata = {}
    mime_type = file_identifier.identify_file_type(filepath)
    logging.info(f"Identified MIME Type for '{os.path.basename(filepath)}' as '{mime_type}'.")
    
    # Layer 0 (Always runs)
    all_metadata.update(extract_os_metadata(filepath))
    all_metadata['IdentifiedMIMEType'] = mime_type
    
    main_category, sub_type = mime_type.split('/')[0], mime_type.split('/')[-1]

    # Layer 1 (Broad-spectrum)
    if main_category in ['image', 'video', 'audio']:
        all_metadata.update(extract_exiftool_metadata(filepath))
    elif main_category in ['application', 'text']:
        all_metadata.update(extract_tika_metadata(filepath))
    else:
        logging.warning(f"No specific Layer 1 tool for MIME category '{main_category}'.")
        all_metadata.update(extract_hachoir_metadata(filepath))

    # Layer 2 (Specialized refinement)
    logging.info(f"[Layer 2] Checking for specialized parsers for {mime_type}...")
    if main_category == 'image':
        all_metadata.update(extract_pillow_metadata(filepath))
    elif main_category == 'audio':
        all_metadata.update(extract_mutagen_metadata(filepath))
    elif sub_type == 'pdf':
        all_metadata.update(extract_pypdf_metadata(filepath))
    elif 'wordprocessingml' in sub_type:
        all_metadata.update(extract_docx_metadata(filepath))
    else:
        logging.info(f"[Layer 2] No specialized parser for this subtype.")

    return flatten_dict(all_metadata)


def process_directory_to_csv(directory: str, output_csv: str, verbosity: int):
    """Processes all files in a directory and writes results to a CSV."""
    if not os.path.isdir(directory):
        logging.error(f"Provided path is not a directory: {directory}")
        return

    identifier = FileTypeIdentifier()
    all_results = []
    
    logging.info(f"Starting to process directory: {directory}")
    
    files_to_process = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Setup progress bar, disable if verbosity is 0
    progress_iterator = tqdm(files_to_process, desc="Extracting Metadata", unit="file", disable=(verbosity < 1))

    for filename in progress_iterator:
        filepath = os.path.join(directory, filename)
        try:
            file_metadata = process_file(filepath, identifier)
            all_results.append(file_metadata)
        except Exception as e:
            logging.error(f"An unhandled error occurred processing {filepath}: {e}")
            all_results.append({"OS.FileName": filename, "Error.Processing": str(e)})
    
    if not all_results:
        logging.warning("No files were processed.")
        return

    all_keys = set()
    for res in all_results:
        all_keys.update(res.keys())
    
    sorted_fieldnames = sorted(list(all_keys))
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Writing {len(all_results)} records to {output_csv}")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=sorted_fieldnames, extrasaction='ignore', escapechar='\\')
            writer.writeheader()
            writer.writerows(all_results)
        logging.info(f"Successfully created metadata CSV file at {os.path.abspath(output_csv)}")
    except Exception as e:
        logging.error(f"Failed to write CSV file: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A multi-layered metadata parser.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help="Path to the directory with files to process."
    )
    parser.add_argument(
        '--output',
        type=str,
        default=os.path.join('..', 'results', f"metadata_{time.strftime('%Y%m%d_%H%M%S')}.csv"),
        help="Path to save the output CSV file.\n(Default: ../results/metadata_[timestamp].csv)"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help="Increase console verbosity. -v for progress bar, -vv for detailed info logs."
    )
    args = parser.parse_args()

    # Setup logging as the first step after parsing args
    setup_logging(args.verbose)

    target_directory = args.dir
    output_csv_file = args.output

    # Use print for the final summary as it should always be visible
    print("\n" + "="*70)
    print("      METADATA PARSER/EXTRACTOR")
    print(f"      Input Directory: '{target_directory}'")
    print(f"      Output CSV: '{output_csv_file}'")
    print(f"      Verbosity Level: {args.verbose}")
    print("="*70 + "\n")
    
    # Run the main processing function
    process_directory_to_csv(target_directory, output_csv_file, args.verbose)
    
    print("\n" + "="*70)
    print("      Processing Complete.")
    print(f"      Check '{os.path.abspath(output_csv_file)}' for results.")
    print(f"      A detailed log file has been saved in the ../logs/ directory.")
    print("="*70 + "\n")
