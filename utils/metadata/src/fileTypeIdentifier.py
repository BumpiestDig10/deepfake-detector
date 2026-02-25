#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
file_type_identifier.py

This module provides a class to identify the MIME type of a file.

It implements the first step of the comprehensive metadata extraction strategy.
The primary method uses the 'python-magic' library to identify files based
on their binary signatures (magic numbers), which is more reliable than
relying on file extensions. A fallback method using file extensions is
provided in case 'python-magic' is not installed or fails.

Dependencies:
- python-magic: (Optional, but highly recommended for accuracy).
  Install using pip: `pip install python-magic`
  Note: This library also requires the `libmagic` C library to be present
  on your system.
  - On macOS (using Homebrew): `brew install libmagic`
  - On Debian/Ubuntu: `sudo apt-get install libmagic1`
  - On Windows: Installation can be more complex. You might need to
    install it via a package manager like Chocolatey or find pre-compiled
    binaries.
"""

import os
import centralLogging as centralLogging

# --- Configuration for Logging ---
logger = centralLogging.get_logger(console_level="WARNING", file_level="DEBUG")

# --- Attempt to import the magic library ---
try:
    import magic
    MAGIC_AVAILABLE = True
    logger.info("Successfully imported the 'magic' library. Will use it for file type identification.")
except ImportError:
    MAGIC_AVAILABLE = False
    logger.warning("The 'magic' library is not installed. Falling back to file extension-based identification.")
    logger.warning("For more accurate results, please install python-magic and its dependency, libmagic.")
except Exception as e:
    MAGIC_AVAILABLE = False
    logger.error(f"An unexpected error occurred while importing 'magic': {e}")
    logger.error("Proceeding with fallback to file extension-based identification.")


class FileTypeIdentifier:
    """
    Identifies the MIME type of a file using libmagic with a fallback to file extensions.
    """

    def __init__(self):
        """
        Initializes the FileTypeIdentifier.
        It checks if the 'magic' library is available for use.
        """
        self.magic_available = MAGIC_AVAILABLE
        self.mime_map = self._get_common_mime_map()

    def _get_common_mime_map(self):
        """
        Returns a dictionary mapping common file extensions to MIME types.
        This is used as a fallback mechanism.
        """
        return {
            # Images
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.gif': 'image/gif', '.bmp': 'image/bmp', '.tiff': 'image/tiff',
            '.tif': 'image/tiff', '.webp': 'image/webp', '.svg': 'image/svg+xml',
            '.heic': 'image/heic', '.heif': 'image/heif',
            # Audio
            '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg',
            '.flac': 'audio/flac', '.m4a': 'audio/mp4', '.aac': 'audio/aac',
            # Video
            '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska', '.wmv': 'video/x-ms-wmv', '.flv': 'video/x-flv',
            '.webm': 'video/webm',
            # Documents
            '.pdf': 'application/pdf', '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.txt': 'text/plain', '.csv': 'text/csv', '.html': 'text/html',
            '.htm': 'text/html', '.xml': 'application/xml', '.json': 'application/json',
            '.rtf': 'application/rtf',
            # Archives
            '.zip': 'application/zip', '.gz': 'application/gzip',
            '.rar': 'application/vnd.rar', '.tar': 'application/x-tar',
            '.7z': 'application/x-7z-compressed'
        }


    def identify_by_magic(self, filepath: str) -> str | None:
        """
        Identifies the file's MIME type using the python-magic library.

        Args:
            filepath: The path to the file.

        Returns:
            The identified MIME type as a string, or None if identification fails.
        """
        if not self.magic_available:
            logger.debug("Magic library not available, skipping magic-based identification.")
            return None
        try:
            mime_type = magic.from_file(filepath, mime=True)
            logger.debug(f"Magic identified '{filepath}' as '{mime_type}'.")
            return mime_type
        except magic.MagicException as e:
            logger.error(f"A magic-related error occurred for '{filepath}': {e}")
            return None
        except FileNotFoundError:
            logger.error(f"File not found for magic identification: '{filepath}'")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during magic identification of '{filepath}': {e}")
            return None

    def identify_by_extension(self, filepath: str) -> str:
        """
        Identifies the file's MIME type based on its file extension as a fallback.

        Args:
            filepath: The path to the file.

        Returns:
            A guessed MIME type string, or 'application/octet-stream' if unknown.
        """
        try:
            _, file_extension = os.path.splitext(filepath)
            file_extension = file_extension.lower()
            mime_type = self.mime_map.get(file_extension, 'application/octet-stream')
            logger.debug(f"Identified '{filepath}' as '{mime_type}' by extension.")
            return mime_type
        except Exception as e:
            logger.error(f"Could not identify by extension for '{filepath}': {e}")
            return 'application/octet-stream' # Default for unknown binary files

    def identify_file_type(self, filepath: str) -> str:
        """
        Orchestrates the file type identification process.

        It first attempts to use the 'magic' library for an accurate identification.
        If that is not possible or fails, it falls back to using the file extension.

        Args:
            filepath: The absolute or relative path to the file.

        Returns:
            The identified MIME type string.
        """
        if not os.path.exists(filepath):
            logger.error(f"File does not exist: {filepath}")
            raise FileNotFoundError(f"The file '{filepath}' was not found.")
        
        logger.info(f"Identifying file type for: {filepath}")

        # Primary method: using magic numbers
        mime_type = self.identify_by_magic(filepath)

        if mime_type:
            logger.info(f"SUCCESS (Magic): Identified '{os.path.basename(filepath)}' as '{mime_type}'.")
            return mime_type
        
        # Fallback method: using file extension
        logger.warning(f"Could not identify with magic, falling back to file extension for '{filepath}'.")
        mime_type = self.identify_by_extension(filepath)
        logger.info(f"SUCCESS (Fallback): Identified '{os.path.basename(filepath)}' as '{mime_type}'.")
        return mime_type

def create_dummy_files():
    """Creates a few dummy files for testing purposes."""
    logger.info("Creating dummy files for testing...")
    os.makedirs("test_files", exist_ok=True)
    
    # Text file
    with open("test_files/sample.txt", "w") as f:
        f.write("This is a test text file.")
        
    # Fake JPEG with a .png extension
    with open("test_files/fake_image.png", "w") as f:
        # A simple text file is enough to fail magic's JPEG test
        f.write("I am not a real png file.")

    # A file with no extension
    with open("test_files/filewithnoextension", "w") as f:
        f.write("Some data")
        
    logger.info("Dummy files created in 'test_files' directory.")


if __name__ == '__main__':
    # --- Setup and Demonstration ---
    # create_dummy_files()
    
    identifier = FileTypeIdentifier()
    
    # A list of file paths to test
    test_filepaths = [
        "../test_files/sample.txt",
        "../test_files/fake_image.png", # Magic should identify this incorrectly for a PNG
        "../test_files/filewithnoextension",
        "../test_files/non_existent_file.xyz", # To test error handling
        "../test_files/001_fe3347c0_0.png",
        "../test_files/002_8f8da10e_1.png"
    ]

    print("\n" + "="*50)
    print("      Starting File Type Identification Test")
    print("="*50 + "\n")
    
    for path in test_filepaths:
        try:
            result_mime_type = identifier.identify_file_type(path)
            print(f"-> Final determined MIME type for '{path}': {result_mime_type}\n")
        except FileNotFoundError as e:
            print(f"-> ERROR for '{path}': {e}\n")
        except Exception as e:
            print(f"-> An unexpected error occurred for '{path}': {e}\n")
    
    print("\n" + "="*50)
    print("          Test Complete")
    print("="*50)


