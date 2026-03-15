import pandas as pd
import argparse
import os
from datetime import datetime

import centralLogging as cl

logger = cl.get_logger(console_level = "INFO", file_level = "DEBUG")

def merge_csv_with_class(real_csv_path, fake_csv_path, output_csv_path=None):
    """
    Merges two CSV files, adding a 'class' column to each before merging.

    Args:
        real_csv_path (str): Path to the CSV file representing 'real' data.
        fake_csv_path (str): Path to the CSV file representing 'fake' data.
        output_csv_path (str, optional): Path to save the merged CSV file.
                                         Defaults to './combined_[timestamp].csv'.
    """
    try:
        # Load the real CSV file
        df_real = pd.read_csv(real_csv_path)
        logger.info(f"Successfully loaded real CSV from: {real_csv_path}")

        # Add 'class' column with value 1 to the real DataFrame
        df_real['class'] = 1
        logger.info("Added 'class' column with value 1 to real data.")

        # Load the fake CSV file
        df_fake = pd.read_csv(fake_csv_path)
        logger.info(f"Successfully loaded fake CSV from: {fake_csv_path}")

        # Add 'class' column with value 0 to the fake DataFrame
        df_fake['class'] = 0
        logger.info("Added 'class' column with value 0 to fake data.")

        # Concatenate (merge) the two DataFrames
        # The 'ignore_index=True' ensures a new, continuous index for the merged DataFrame
        merged_df = pd.concat([df_real, df_fake], ignore_index=True)
        logger.debug("Successfully merged real and fake dataframes.")

        # Determine the output path
        if output_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv_path = f"./combined_{timestamp}.csv"
            logger.warning(f"Output path not specified. Defaulting to: {output_csv_path}")

        # Save the merged DataFrame to the specified output path
        merged_df.to_csv(output_csv_path, index=False) # index=False prevents writing DataFrame index as a column
        logger.info(f"Merged CSV saved successfully to: {output_csv_path}")

    except FileNotFoundError as e:
        logger.critical(f"Error: One of the specified CSV files was not found. {e}")
        exit(1)  # Exit with a non-zero code to indicate an error
    except pd.errors.EmptyDataError:
        logger.critical(f"Error: One of the CSV files is empty. {e}")
        exit(1)  # Exit with a non-zero code to indicate an error
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        exit(1)  # Exit with a non-zero code to indicate an error

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Merge two CSV files, adding a 'class' column (1 for real, 0 for fake)."
    )

    # Add arguments
    parser.add_argument(
        "--real",
        type=str,
        required=True,
        help="Path to the 'real' CSV file. (Mandatory)"
    )
    parser.add_argument(
        "--fake",
        type=str,
        required=True,
        help="Path to the 'fake' CSV file. (Mandatory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Path to save the merged output CSV file. "
             "If not specified, saves to ./combined_[timestamp].csv in the current directory."
    )

    # Parse arguments from the command line
    args = parser.parse_args()

    # Call the merge function with the parsed arguments
    merge_csv_with_class(args.real, args.fake, args.output)
