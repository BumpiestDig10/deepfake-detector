import logging
import os
import sys
from datetime import datetime
import inspect

# 1. Save the original standard outputs to prevent infinite logging loops.
# If the logger's console handler writes to the intercepted sys.stdout, 
# it will endlessly trigger the logger. We must bind the console handler to these originals.
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr

# 2. Define custom log levels to fit between standard levels
# INFO (20) < STDOUT (25) < WARNING (30) < ERROR (40) < STDERR (45) < CRITICAL (50)
LOG_LEVEL_STDOUT = 25
LOG_LEVEL_STDERR = 45

# Register the level names so they appear in the log output formatting
logging.addLevelName(LOG_LEVEL_STDOUT, "STDOUT")
logging.addLevelName(LOG_LEVEL_STDERR, "STDERR")

class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.buffer = ""

    def write(self, message):
        """Intercept the write command and send it to the logger."""
        self.buffer += message
        # Extract and log lines as they get a newline character
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self.logger.log(self.log_level, line)

    def flush(self):
        """Flush the buffer when the stream is flushed."""
        if self.buffer:
            self.logger.log(self.log_level, self.buffer)
            self.buffer = ""


def _parse_level(level):
    """Helper to convert string level names ('INFO') to integer constants (logging.INFO)."""
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return level


def get_logger(console_level="INFO", file_level="DEBUG", log_to_console=True, log_to_file=True, capture_output=True):
    """
    Creates and returns a logger instance with independent levels for terminal and file storage.
    
    Args:
        console_level (str|int): Minimum level for terminal output.
        file_level (str|int): Minimum level for the saved file.
        log_to_console (bool): Whether to enable terminal output.
        log_to_file (bool): Whether to enable file storage.
        capture_output (bool): Whether to capture standard print() and sys.stderr.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    console_lvl = _parse_level(console_level)
    file_lvl = _parse_level(file_level)

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
    logger.setLevel(min(console_lvl, file_lvl))
    logger.propagate = False

    # Prevent adding duplicate handlers if the function is called multiple times
    if logger.handlers:
        return logger

    # 3. Define the log format
    log_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 4. Set up File Handler
    if log_to_file:
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        file_path = os.path.join('logs', log_filename)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(log_format)
        file_handler.setLevel(file_lvl) 
        logger.addHandler(file_handler)

    # 5. Set up Console Handler
    if log_to_console:
        # Crucial: Use _ORIGINAL_STDOUT to prevent recursion!
        console_handler = logging.StreamHandler(_ORIGINAL_STDOUT)
        console_handler.setFormatter(log_format)
        console_handler.setLevel(console_lvl) 
        logger.addHandler(console_handler)

    # 6. Intercept Standard Output/Error with custom levels
    if capture_output:
        # Avoid overriding if it's already intercepted
        if not isinstance(sys.stdout, StreamToLogger):
            sys.stdout = StreamToLogger(logger, LOG_LEVEL_STDOUT)
        if not isinstance(sys.stderr, StreamToLogger):
            sys.stderr = StreamToLogger(logger, LOG_LEVEL_STDERR)

    return logger

# Demonstration
if __name__ == "__main__":
    # Create the logger, capturing standard outputs by default
    logger = get_logger(console_level="INFO", file_level="DEBUG")
    
    _ORIGINAL_STDOUT.write("\n--- Starting Logging Capture Test ---\n")
    
    # 1. Standard Logging Test
    logger.debug("1. Native Logger DEBUG: This should only go to the file.")
    logger.info("2. Native Logger INFO: Standard native log message.")
    
    # 2. Captured Stdout Test (print statements)
    print("3. Captured Print: This standard print statement is intercepted as an INFO log!")
    
    # 3. Captured Stderr Test (errors/exceptions)
    sys.stderr.write("4. Captured Stderr Write: This standard error write is intercepted as an ERROR log!\n")
    
    # 4. Exception Traceback Test
    try:
        raise ValueError("Oops! A serious error occurred.")
    except Exception as e:
        # Standard sys.excepthook relies on sys.stderr, so traceback prints are automatically captured!
        import traceback
        traceback.print_exc() 
        
    print("5. Captured Print: Test completed successfully.")