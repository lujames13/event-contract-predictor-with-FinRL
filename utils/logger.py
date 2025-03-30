import os
import logging
import logging.handlers
from datetime import datetime

# Constants
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO
LOG_DIRECTORY = "logs"


def ensure_log_directory():
    """Ensure the log directory exists"""
    os.makedirs(LOG_DIRECTORY, exist_ok=True)


def configure_logging(log_to_console=True, log_to_file=True, log_level=DEFAULT_LOG_LEVEL):
    """
    Configure the root logger with handlers for console and/or file output.
    
    Args:
        log_to_console (bool): Whether to log to console
        log_to_file (bool): Whether to log to file
        log_level (int): Logging level (e.g., logging.INFO)
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        ensure_log_directory()
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(LOG_DIRECTORY, f"{today}.log")
        
        # Use a rotating file handler to prevent logs from growing too large
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name, log_level=None):
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Logger name (typically the module name)
        log_level (int, optional): Specific log level for this logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set specific log level if provided
    if log_level is not None:
        logger.setLevel(log_level)
    
    return logger


# Configure logging when the module is imported
configure_logging(log_to_console=True, log_to_file=True)


def configure_logging(log_to_console=True, log_to_file=True, log_level=DEFAULT_LOG_LEVEL):
    """
    Configure the root logger with handlers for console and/or file output.
    
    Args:
        log_to_console (bool): Whether to log to console
        log_to_file (bool): Whether to log to file
        log_level (int): Logging level (e.g., logging.INFO)
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        ensure_log_directory()
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(LOG_DIRECTORY, f"{today}.log")
        
        # Use a rotating file handler to prevent logs from growing too large
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name, log_level=None):
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Logger name (typically the module name)
        log_level (int, optional): Specific log level for this logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set specific log level if provided
    if log_level is not None:
        logger.setLevel(log_level)
    
    return logger


# Configure logging when the module is imported
configure_logging(log_to_console=True, log_to_file=True)


# Example usage
if __name__ == "__main__":
    # Get logger
    logger = get_logger("logger_test")
    
    # Log some messages
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test a different log level
    debug_logger = get_logger("debug_logger", logging.DEBUG)
    debug_logger.debug("This debug message should be visible")