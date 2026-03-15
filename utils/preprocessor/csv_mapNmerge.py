#!/usr/bin/env python3
"""
CSV Class Label Merger Tool

This tool merges class labels from a metadata CSV file into a main CSV file
by matching filenames and mapping labels to standardized values.

Requirements:
- pandas
- argparse
"""

import argparse
import csv
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

import centralLogging as cl

logger = cl.get_logger(console_level = "INFO", file_level = "DEBUG")


class GracefulKiller:
    """Handle graceful termination signals"""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        """Set flag for graceful termination"""
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.kill_now = True


class CSVLabelMerger:
    """Merge class labels from metadata CSV into main CSV"""

    def __init__(self, base_csv: str, label_csv: str):
        self.base_csv = base_csv
        self.label_csv = label_csv
        self.base_df = None
        self.label_df = None
        self.base_filename_col = None
        self.label_filename_col = None
        self.label_class_col = None
        self.filename_to_label = {}

    def extract_filename(self, filepath: str) -> str:
        """Extract filename from full path or return as-is if already filename"""
        return Path(filepath).name

    def normalize_label(self, label) -> int:
        """Normalize label to 0 or 1"""
        if pd.isna(label):
            return None
        
        label_str = str(label).lower().strip()
        
        # Category 1: real or 1 -> 1
        if label_str in ['real', '1', '1.0']:
            return 1
        # Category 2: fake or 0 -> 0
        elif label_str in ['fake', '0', '0.0']:
            return 0
        else:
            logger.warning(f"!! Unknown label '{label}' - skipping !!")
            return None

    def load_and_validate_csv(self, filepath: str, csv_type: str) -> pd.DataFrame:
        """Load and validate CSV file"""
        if not os.path.exists(filepath):
            logger.critical(f"{csv_type.capitalize()} CSV file not found: {filepath}")
            exit(1)
        
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                logger.critical(f"{csv_type.capitalize()} CSV file is empty: {filepath}")
                exit(1)
            return df
        except Exception as e:
            logger.error(f"Error reading {csv_type} CSV file: {e}")
            exit(1)

    def get_column_choice(self, df: pd.DataFrame, csv_type: str, default_name: str) -> str:
        """Get user's choice for filename column"""
        columns = list(df.columns)
        
        logger.info(f"Available columns in {csv_type} CSV:")
        for i, col in enumerate(columns, 1):
            logger.info(f"  {i}. {col}")
        
        prompt = f"Enter the header name for filename in {csv_type} CSV (default: '{default_name}'): "
        user_input = input(prompt).strip()
        
        # Use default if empty
        if not user_input:
            if default_name in columns:
                return default_name
            else:
                logger.warning(f"Default '{default_name}' not found, using first column: '{columns[0]}'")
                return columns[0]
        
        # Validate user input
        if user_input in columns:
            return user_input
        else:
            logger.warning(f"Column '{user_input}' not found, using first column: '{columns[0]}'")
            return columns[0]

    def get_label_column_choice(self, df: pd.DataFrame) -> str:
        """Get user's choice for label column in metadata CSV"""
        columns = list(df.columns)
        
        logger.info(f"Available columns in metadata CSV for labels:")
        for i, col in enumerate(columns, 1):
            logger.info(f"  {i}. {col}")
        
        prompt = "Enter the header name for class labels in metadata CSV: "
        user_input = input(prompt).strip()
        
        if user_input in columns:
            return user_input
        else:
            logger.warning(f"Column '{user_input}' not found. Please try again.")
            return self.get_label_column_choice(df)

    def load_data(self):
        """Load both CSV files and get column selections"""
        logger.info("Loading CSV files...")
        
        # Load base CSV
        self.base_df = self.load_and_validate_csv(self.base_csv, "base")
        logger.info(f"Base CSV loaded: {len(self.base_df)} rows, {len(self.base_df.columns)} columns")
        
        # Load label CSV
        self.label_df = self.load_and_validate_csv(self.label_csv, "metadata")
        logger.info(f"Metadata CSV loaded: {len(self.label_df)} rows, {len(self.label_df.columns)} columns")
        
        # Get column selections
        self.base_filename_col = self.get_column_choice(self.base_df, "base", "filename")
        self.label_filename_col = self.get_column_choice(self.label_df, "metadata", "filename")
        self.label_class_col = self.get_label_column_choice(self.label_df)
        
        logger.info(f"Selected columns:")
        logger.info(f"  Base filename column: '{self.base_filename_col}'")
        logger.info(f"  Metadata filename column: '{self.label_filename_col}'")
        logger.info(f"  Metadata label column: '{self.label_class_col}'")

    def build_filename_to_label_mapping(self):
        """Build mapping from filename to label"""
        logger.debug("Building filename to label mapping...")
        
        for idx, row in self.label_df.iterrows():
            filename = self.extract_filename(str(row[self.label_filename_col]))
            label = self.normalize_label(row[self.label_class_col])
            
            if label is not None:
                self.filename_to_label[filename] = label
        
        logger.debug(f"Created mapping for {len(self.filename_to_label)} filenames")
        
        # Show label distribution
        label_counts = {}
        for label in self.filename_to_label.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info("Label distribution:")
        for label, count in label_counts.items():
            label_name = "real" if label == 1 else "fake"
            logger.info(f"  {label_name} ({label}): {count}")

    def process_and_save(self, output_path: str):
        """Process base CSV and save with class labels"""
        logger.info(f"Processing base CSV and saving to: {output_path}")
        
        # Prepare output dataframe
        output_df = self.base_df.copy()
        
        # Extract filenames and create class column
        output_df['filename'] = output_df[self.base_filename_col].apply(
            lambda x: self.extract_filename(str(x))
        )
        
        # Map labels
        output_df['class'] = output_df['filename'].map(self.filename_to_label)
        
        # Reorder columns: filename, class, then other columns
        other_cols = [col for col in output_df.columns if col not in ['filename', 'class']]
        output_df = output_df[['filename', 'class'] + other_cols]
        
        # Count matches and misses
        matched = output_df['class'].notna().sum()
        total = len(output_df)
        
        logger.info(f"Matched labels for {matched}/{total} files")
        if matched < total:
            logger.warning(f"Warning: {total - matched} files have no matching labels")
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        
        return matched, total

    def process_and_save_batch(self, output_path: str, killer: GracefulKiller):
        """Process base CSV and save in batches of 100"""
        logger.info(f"Processing base CSV in batches and saving to: {output_path}")
        
        batch_size = 100
        total_rows = len(self.base_df)
        processed_rows = 0
        matched_count = 0
        
        # Prepare output file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process in batches
        with open(output_path, 'w', newline='', encoding='utf-8') as output_file:
            writer = None
            
            for start_idx in range(0, total_rows, batch_size):
                # Check for graceful termination
                if killer.kill_now:
                    logger.info(f"Graceful termination requested. Processed {processed_rows}/{total_rows} rows.")
                    break
                
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = self.base_df.iloc[start_idx:end_idx].copy()
                
                # Extract filenames and create class column
                batch_df['filename'] = batch_df[self.base_filename_col].apply(
                    lambda x: self.extract_filename(str(x))
                )
                
                # Map labels
                batch_df['class'] = batch_df['filename'].map(self.filename_to_label)
                
                # Reorder columns: filename, class, then other columns
                other_cols = [col for col in batch_df.columns if col not in ['filename', 'class']]
                batch_df = batch_df[['filename', 'class'] + other_cols]
                
                # Count matches in this batch
                batch_matched = batch_df['class'].notna().sum()
                matched_count += batch_matched
                
                # Write to CSV
                if writer is None:
                    # Write header for first batch
                    writer = csv.DictWriter(output_file, fieldnames=batch_df.columns)
                    writer.writeheader()
                
                # Write batch data
                batch_df.to_csv(output_file, header=False, index=False, mode='a')
                
                processed_rows += len(batch_df)
                
                logger.info(f"Processed batch: {processed_rows}/{total_rows} rows (matched: {batch_matched}/{len(batch_df)})")
        
        logger.info(f"Processing completed!")
        logger.info(f" Total processed: {processed_rows}/{total_rows}")
        logger.info(f" Total matched: {matched_count}/{processed_rows}")
        logger.info(f" Results saved to: {output_path}")
        
        return matched_count, processed_rows


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Merge class labels from metadata CSV into main CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python csv_label_merger.py --base features.csv --label metadata.csv
  python csv_label_merger.py --base /path/to/main.csv --label /path/to/labels.csv
        """
    )
    
    parser.add_argument(
        '--base',
        required=True,
        help='Path to the main/base CSV file (mandatory)'
    )
    
    parser.add_argument(
        '--label',
        required=True,
        help='Path to the metadata/label CSV file (mandatory)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.isfile(args.base):
        logger.error(f"Base CSV file does not exist: {args.base}")
        sys.exit(1)
    
    if not os.path.isfile(args.label):
        logger.error(f"Label CSV file does not exist: {args.label}")
        sys.exit(1)
    
    # Generate output filename
    base_path = Path(args.base)
    output_path = base_path.parent / f"{base_path.stem}_combined.csv"
    
    logger.info(f"Base CSV file: {args.base}")
    logger.info(f"Label CSV file: {args.label}")
    logger.info(f"Output file: {output_path}")
    
    # Initialize graceful killer
    killer = GracefulKiller()
    
    try:
        # Initialize merger
        merger = CSVLabelMerger(args.base, args.label)
        
        # Load data and get user input
        merger.load_data()
        
        # Build mapping
        merger.build_filename_to_label_mapping()
        
        # Process and save with batch processing
        matched, total = merger.process_and_save_batch(str(output_path), killer)
        
        logger.info(f"{'='*50}")
        logger.info(f"Label merging completed!")
        logger.info(f"Files processed: {total}")
        logger.info(f"Labels matched: {matched}")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"{'='*50}")
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Progress has been saved.")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)
    finally:
        logger.info("Cleanup completed.")


if __name__ == "__main__":
    main()