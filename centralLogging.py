import logging
import os
import sys
from datetime import datetime
import inspect

def get_logger(console_level="INFO", file_level="DEBUG", log_to_console=True, log_to_file=True):
    """
    Creates and returns a logger instance with independent levels for terminal and file storage.
    
    Args:
        console_level (int): Minimum level for terminal output (e.g., logging.WARNING).
        file_level (int): Minimum level for the saved file (e.g., logging.DEBUG).
        log_to_console (bool): Whether to enable terminal output.
        log_to_file (bool): Whether to enable file storage.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    # 1. Identify the calling module's name
    caller_frame = inspect.stack()[1]
    module = inspect.getmodule(caller_frame[0])
    
    if module and hasattr(module, '__file__'):
        module_name = os.path.basename(module.__file__).replace('.py', '')
    else:
        module_name = "main_execution"

    # 2. Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{module_name}.log"
    
    # Create a unique logger for this session
    logger = logging.getLogger(f"{module_name}")
    
    # IMPORTANT: The master logger level must be the lowest of the two levels 
    # to allow messages to reach the handlers for individual filtering.
    logger.setLevel(min(console_level, file_level))
    
    logger.propagate = False

    # 3. Define the log format
    log_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 4. Set up File Handler (e.g., set to DEBUG to catch everything)
    if log_to_file:
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        file_path = os.path.join('logs', log_filename)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(log_format)
        file_handler.setLevel(file_level) # Apply specific level for file
        logger.addHandler(file_handler)
        print(f"Logging to file: {file_path}")

    # 5. Set up Console Handler (e.g., set to WARNING to keep terminal clean)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        console_handler.setLevel(console_level) # Apply specific level for console
        logger.addHandler(console_handler)

    return logger

# Demonstration of the specific dual-level requirement
if __name__ == "__main__":
    logger = get_logger()
    
    print("--- Starting Logging Test (Check console vs the generated file in /logs) ---")
    
    logger.debug("This is DEBUG: Only appears in the FILE.")
    logger.info("This is INFO: Only appears in the FILE.")
    logger.warning("This is WARNING: Appears in BOTH Terminal and File.")
    logger.error("This is ERROR: Appears in BOTH Terminal and File.")
    logger.critical("This is CRITICAL: Appears in BOTH Terminal and File.")