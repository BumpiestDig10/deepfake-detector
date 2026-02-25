import argparse
import sys
import os
import csv
from datetime import datetime
import time
import queue
import threading
import signal
import collections.abc
from pathlib import Path
from tqdm import tqdm

import centralLogging as centralLogging
logger = centralLogging.get_logger(console_level="DEBUG", file_level="INFO")

# --- Shared Resources for Multithreading ---
file_queue = queue.Queue()
results_queue = queue.Queue()
stop_event = threading.Event()

# --- Dependency Imports & Checks ---

# Custom Module
try:
    from utils.metadata.fileTypeIdentifier import FileTypeIdentifier
except ImportError:
    logger.critical("CRITICAL ERROR: Could not import 'FileTypeIdentifier'. Make sure 'fileTypeIdentifier.py' is in 'utils/metadata/' directory.")
    sys.exit(1)

# Layer 1 Tools
try:
    import exiftool
    EXIFTOOL_AVAILABLE = True
except ImportError:
    EXIFTOOL_AVAILABLE = False
    logger.warning("pyexiftool not found. Layer 1 ExifTool extraction skipped.")
    
try:
    from tika import parser as tika_parser
    TIKA_AVAILABLE = True
except ImportError:
    TIKA_AVAILABLE = False
    logger.warning("tika-client not found. Layer 1 Tika extraction skipped.")

# Layer 2 Libraries
try:
    from PIL import Image, ExifTags
    PILLOW_AVAILABLE = True
    logger.info("Pillow library found. Layer 2 image extraction enabled.")
except ImportError:
    PILLOW_AVAILABLE = False
    logger.warning("Pillow not found. Layer 2 image extraction skipped.")
    
try:
    import mutagen
    MUTAGEN_AVAILABLE = True
    logger.info("Mutagen library found. Layer 2 audio extraction enabled.")
except ImportError:
    MUTAGEN_AVAILABLE = False
    logger.warning("Mutagen not found. Layer 2 audio extraction skipped.")
    
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
    logger.info("PyPDF library found. Layer 2 PDF extraction enabled.")
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("pypdf not found. Layer 2 PDF extraction skipped.")
    
try:
    import docx
    DOCX_AVAILABLE = True
    logger.info("python-docx library found. Layer 2 DOCX extraction enabled.")
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not found. Layer 2 DOCX extraction skipped.")

# Layer 3 Library
try:
    import hachoir.metadata
    import hachoir.stream
    HACHOIR_AVAILABLE = True
    logger.info("Hachoir library found. Layer 3 binary analysis enabled.")
except ImportError:
    HACHOIR_AVAILABLE = False
    logger.warning("hachoir not found. Layer 3 binary analysis skipped.")

# Platform-specific imports
try:
    import pwd
    import grp
    UNIX_SYSTEM = True
    logger.info("UNIX-like system detected. File owner/group names will be extracted.")
except ImportError:
    UNIX_SYSTEM = False
    logger.info("Not a UNIX-like system. File owner/group names not extracted.")

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
            "OS.ModTime_Local": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "OS.AccessTime_Local": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
            "OS.CreateTime_Local": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            "OS.Permissions": oct(stat_info.st_mode)[-3:],
        }
        if UNIX_SYSTEM:
            try:
                metadata["OS.Owner_Name"] = pwd.getpwuid(stat_info.st_uid).pw_name
            except KeyError:
                metadata["OS.Owner_Name"] = f"UID_{stat_info.st_uid}" # Handle unknown UID
            try:
                metadata["OS.Group_Name"] = grp.getgrgid(stat_info.st_gid).gr_name
            except KeyError:
                 metadata["OS.Group_Name"] = f"GID_{stat_info.st_gid}" # Handle unknown GID
        return metadata
    except Exception as e:
        return {"Error.Layer0": f"OS metadata extraction failed: {e}"}

# --- Layer 1 ---
def extract_exiftool_metadata(filepath: str) -> dict:
    if not EXIFTOOL_AVAILABLE: return {}
    try:
        with exiftool.ExifTool() as et:
            # Use -G to get group names for keys, and -j for json
            return et.execute_json(filepath, "-G", "-j")[0]
    except Exception as e:
        return {"Error.ExifTool": str(e)}

def extract_tika_metadata(filepath: str) -> dict:
    if not TIKA_AVAILABLE: return {}
    try:
        parsed = tika_parser.from_file(filepath)
        return parsed.get("metadata", {})
    except Exception as e:
        if "ConnectionRefusedError" in str(e):
             logger.critical("CRITICAL: Tika connection failed. Is Tika Server running? Aborting.")
             stop_event.set() # Signal all threads to stop
             # We can't sys.exit here as it kills only this thread.
             # The main thread will handle the exit.
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
        return {"Mutagen": dict(audio)} if audio else {}
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
    logger.info(f"[Layer 3] Routing to Hachoir for binary analysis: {os.path.basename(filepath)}")
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
    logger.info(f"--- Processing: {os.path.basename(filepath)} ---")
    all_metadata = {}
    mime_type = file_identifier.identify_file_type(filepath)
    logger.info(f"Identified MIME Type for '{os.path.basename(filepath)}' as '{mime_type}'.")
    
    # Layer 0 (Always runs)
    all_metadata.update(extract_os_metadata(filepath))
    all_metadata['IdentifiedMIMEType'] = mime_type
    
    main_category, sub_type = (mime_type.split('/')[0], mime_type.split('/')[-1]) if '/' in mime_type else (mime_type, '')

    # Layer 1 (Broad-spectrum)
    all_metadata.update(extract_exiftool_metadata(filepath))
    all_metadata.update(extract_tika_metadata(filepath))

    # Layer 2 (Specialized refinement)
    logger.info(f"[Layer 2] Checking for specialized parsers for {mime_type}...")
    if main_category == 'image' and PILLOW_AVAILABLE:
        all_metadata.update(extract_pillow_metadata(filepath))
    elif main_category == 'audio' and MUTAGEN_AVAILABLE:
        all_metadata.update(extract_mutagen_metadata(filepath))
    elif sub_type == 'pdf' and PYPDF_AVAILABLE:
        all_metadata.update(extract_pypdf_metadata(filepath))
    elif 'wordprocessingml' in sub_type and DOCX_AVAILABLE:
        all_metadata.update(extract_docx_metadata(filepath))
    else:
        logger.warning("[Layer 2] No specialized parser for this subtype.")
    
    # Layer 3 (Generic fallback)
    all_metadata.update(extract_hachoir_metadata(filepath))

    return flatten_dict(all_metadata)


# =============================================================================
# WORKER THREAD FUNCTIONS
# =============================================================================

def metadata_extractor_worker(stop_event_ref):
    """Worker thread to pull files from queue and extract metadata."""
    logger.info("Extractor worker started.")
    identifier = FileTypeIdentifier()
    while not stop_event_ref.is_set():
        try:
            filepath = file_queue.get(timeout=1)
            try:
                metadata = process_file(filepath, identifier)
                results_queue.put(metadata)
            except Exception as e:
                logger.error(f"Unhandled error processing {os.path.basename(filepath)}: {e}")
                results_queue.put({"OS.FileName": os.path.basename(filepath), "Error.Processing": str(e)})
            finally:
                file_queue.task_done()
        except queue.Empty:
            if file_queue.qsize() == 0:
                logger.info("File queue is empty, extractor worker is finishing.")
                break # Exit if the queue is truly empty
    logger.info("Extractor worker stopped.")

def csv_writer_worker(output_csv, stop_event_ref, progress_bar):
    """Worker thread to write metadata results to CSV in batches."""
    logger.info("CSV writer worker started.")
    results_buffer = []
    all_fieldnames = set()
    is_header_written = False
    WRITE_BATCH_SIZE = 100
    
    while not stop_event_ref.is_set() or not results_queue.empty():
        try:
            result = results_queue.get(timeout=1)
            results_buffer.append(result)
            progress_bar.update(1)

            should_write = (len(results_buffer) >= WRITE_BATCH_SIZE or
                           (stop_event_ref.is_set() and results_queue.empty()))

            if should_write and results_buffer:
                logger.info(f"Writing batch of {len(results_buffer)} results to CSV.")
                
                # Check for new headers
                current_keys = set()
                for res in results_buffer:
                    current_keys.update(res.keys())
                
                new_fieldnames = current_keys - all_fieldnames
                
                # Write mode is 'a' (append) unless headers change or it's the first write
                write_mode = 'w' if (new_fieldnames or not is_header_written) else 'a'
                
                # If we have new headers, we need to rewrite the whole file
                # To do this safely, we would need to read the old file, combine, and write
                # For this script, we will just append new columns, which may result in a non-uniform CSV
                # The safest approach is to just write everything at the end, but this meets the incremental requirement
                if new_fieldnames:
                    all_fieldnames.update(new_fieldnames)
                
                sorted_fieldnames = sorted(list(all_fieldnames))

                try:
                    with open(output_csv, write_mode, newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=sorted_fieldnames, extrasaction='ignore')
                        if write_mode == 'w' or not is_header_written:
                            writer.writeheader()
                            is_header_written = True
                        writer.writerows(results_buffer)
                    results_buffer.clear()
                except IOError as e:
                    logger.error(f"Could not write to CSV file {output_csv}: {e}")
                    # Don't clear buffer, try again on next iteration
                    
        except queue.Empty:
            # This is the normal exit condition when processing is done
            pass

    # Final write for any remaining items in the buffer
    if results_buffer:
        logger.info(f"Writing final batch of {len(results_buffer)} results.")
        # Final write is always append unless it's the very first write
        write_mode = 'a' if is_header_written else 'w'
        all_fieldnames.update(*(res.keys() for res in results_buffer))
        sorted_fieldnames = sorted(list(all_fieldnames))
        with open(output_csv, write_mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fieldnames, extrasaction='ignore')
            if not is_header_written:
                writer.writeheader()
            writer.writerows(results_buffer)

    logger.info("CSV writer worker stopped.")
    progress_bar.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="A multi-layered, multithreaded metadata parser.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True, help="Path to the directory with files to process.")
    parser.add_argument('--output', type=str,
        default=os.path.join('results', 'metadata', f"metadata_{time.strftime('%Y%m%d_%H%M%S')}.csv"),
        help="Path to save the output CSV file.\n(Default: ../results/metadata_[timestamp].csv)")
    args = parser.parse_args()

    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Print Header ---
    logger.info("="*70 + "\n")
    logger.info(f"Input Directory: '{args.input}'")
    logger.info(f"Output CSV: '{args.output}'")
    logger.info("="*70 + "\n")

    # --- Populate file queue ---
    if not os.path.isdir(args.input):
        logger.critical(f"Provided path is not a directory: {args.input}")
        sys.exit(1)
        
    files_to_process = [os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
    if not files_to_process:
        logger.warning("No files found in the specified directory. Exiting.")
        sys.exit(0)
    
    for filepath in files_to_process:
        file_queue.put(filepath)
    
    total_files = len(files_to_process)
    logger.info(f"Found {total_files} files to process.")
    
    # --- Setup Progress Bar ---
    progress_bar = tqdm(total=total_files, desc="Extracting Metadata", unit="file")

    # --- Setup and Start Threads ---
    extractor_thread = threading.Thread(target=metadata_extractor_worker, args=(stop_event,))
    writer_thread = threading.Thread(target=csv_writer_worker, args=(args.output, stop_event, progress_bar))
    
    extractor_thread.start()
    writer_thread.start()

    # --- Graceful Shutdown Handler ---
    def signal_handler(sig, frame):
        logger.warning("\nCtrl+C detected! Shutting down gracefully...")
        stop_event.set()
    signal.signal(signal.SIGINT, signal_handler)

    # --- Wait for threads to complete ---
    while extractor_thread.is_alive():
        extractor_thread.join(timeout=1)

    # Once the extractor is done, wait for the processing queue to be fully empty
    file_queue.join()
    
    # Signal the writer thread that no more items are coming
    stop_event.set()
    writer_thread.join()

    # --- Final Summary ---
    logger.info("="*70 + "\n")
    logger.info("Processing Complete.")
    logger.info(f"Check '{os.path.abspath(args.output)}' for results.")
    logger.info(f"A detailed log file has been saved in the ../logs/ directory.")
    logger.info("="*70 + "\n")
if __name__ == '__main__':
    main()
