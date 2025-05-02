# utils/logging_utils.py
import logging
import os
from datetime import datetime
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
import traceback

import config.config as config

# Global logger configuration state
_is_configured = False

# Utility functions
def get_timestamp():
    """Return current timestamp string for file naming."""
    return datetime.now().strftime("%d%m%Y_%H%M%S")

def setup_logging(
    name='logging',
    type='logging_utils',
    log_level=config.DEFAULT_LOG_LEVEL,
    log_format=config.DEFAULT_LOG_FORMAT,
    date_format=config.DEFAULT_DATE_FORMAT,
    log_to_console=True,
    log_to_file=True,
    log_dir=config.BASE_LOGS_DIR,
    max_file_size=10*1024*1024,  # 10 MB - remove?
    backup_count=5,
    include_trade_logs=False,
    include_performance_logs=False,
    environment=config.DEV_ENV
):
    """
    Configure the global logging for the entire application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_format: Log message format
        date_format: Date format in log messages
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        max_file_size: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        include_trade_logs: Whether to create separate trade log
        include_performance_logs: Whether to create separate performance log
        environment: Application environment (development, production)
    """
    global _is_configured
    
    # Avoid reconfiguring if already set up
    if _is_configured:
        return get_logger(type)
    
    # Create logs directory if it doesn't exist
    if log_to_file:
        for subdir in config.DIFFERENT_LOG_DIRS:
            full_path = os.path.join(log_dir, subdir)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates if this function is called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    timestamp = get_timestamp()
    # Add main file handler
    if log_to_file:
        for subdir in config.DIFFERENT_LOG_DIRS:
            if type is subdir:
                full_path = os.path.join(log_dir, subdir)
                log_dir = full_path
        main_log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        main_file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        main_file_handler.setFormatter(formatter)
        root_logger.addHandler(main_file_handler)
    
    # Set the configured flag
    _is_configured = True
    
    # Log startup
    logger = get_logger(type)
    logger.info(f"Logging system initialized. Environment: {environment}")
    
    # Add an exception hook to log uncaught exceptions
    sys.excepthook = _log_uncaught_exception
    
    return logger

def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions by logging them."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupts
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = get_logger("uncaught_exception")
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def get_logger(name=None):
    """
    Get a logger instance. If logging system isn't configured yet, it will be configured
    with default settings.
    
    Args:
        name: Logger name (usually __name__ of the calling module)
        
    Returns:
        Logger instance
    """
    global _is_configured
    
    if not _is_configured:
        setup_logging()
    
    return logging.getLogger(name)
