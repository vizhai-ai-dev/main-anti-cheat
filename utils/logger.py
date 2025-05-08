import logging
import sys
import os
from datetime import datetime
from typing import Optional
import traceback
from functools import wraps

# Configure logging
def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """Setup and configure logger"""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create handlers
    handlers = []
    
    if log_to_file:
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        log_file = os.path.join(
            log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    if log_to_console:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger

# Error handling decorator
def handle_errors(logger: Optional[logging.Logger] = None):
    """Decorator for handling and logging errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.error(traceback.format_exc())
                raise
        return wrapper
    return decorator

# Custom exception classes
class VideoProcessingError(Exception):
    """Base exception for video processing errors"""
    pass

class AudioProcessingError(Exception):
    """Base exception for audio processing errors"""
    pass

class FileOperationError(Exception):
    """Base exception for file operation errors"""
    pass

class ValidationError(Exception):
    """Base exception for validation errors"""
    pass

# Error handling context manager
class ErrorHandler:
    """Context manager for handling errors with logging"""
    def __init__(self, logger: logging.Logger, error_message: str):
        self.logger = logger
        self.error_message = error_message
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"{self.error_message}: {str(exc_val)}")
            self.logger.error(traceback.format_exc())
            return True  # Suppress the exception
        return False

# Logging context manager
class LoggingContext:
    """Context manager for temporary logging level changes"""
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.old_level = logger.level
        
    def __enter__(self):
        self.logger.setLevel(self.level)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
        return False 