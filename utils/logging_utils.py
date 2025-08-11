# utils/logging_utils.py
import logging
import os
from datetime import datetime
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
import traceback
import inspect
from typing import Optional, Dict, Any

import config.config as config
from utils.hypothesis_log_categorizer import get_dynamic_log_path, get_category_description

# Global logger configuration state
_is_configured = False

# Utility functions
def get_timestamp():
    """Return current timestamp string for file naming."""
    return datetime.now().strftime("%d%m%Y_%H%M%S")

def setup_logging(
    name='logging',
    type='logging_utils',
    log_level=config.DEBUG_LOG_LEVEL,
    log_format=config.DEFAULT_LOG_FORMAT,
    date_format=config.DEFAULT_DATE_FORMAT,
    log_to_console=True,
    log_to_file = False,  # save log file or not 
    log_dir=config.BASE_LOGS_DIR,
    max_file_size=90*1024*1024,  # 10 MB - remove?
    backup_count=5,
    include_trade_logs=False,
    include_performance_logs=False,
    environment=config.DEV_ENV,
    hypothesis_config_path=None
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
        hypothesis_config_path: Path to hypothesis config file for dynamic log categorization
    """
    global _is_configured
    
    # Avoid reconfiguring if already set up
    if _is_configured:
        return get_logger(type)
    
    # Determine log subdirectory based on hypothesis config
    dynamic_log_subdir = None
    if hypothesis_config_path:
        try:
            dynamic_log_subdir = get_dynamic_log_path(hypothesis_config_path)
        except Exception:
            dynamic_log_subdir = 'mixed'  # Fallback
    
    # Create logs directory if it doesn't exist
    if log_to_file:
        # Create standard log directories
        for subdir in config.DIFFERENT_LOG_DIRS:
            full_path = os.path.join(log_dir, subdir)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
        
        # Create dynamic log directory if needed
        if dynamic_log_subdir and dynamic_log_subdir not in config.DIFFERENT_LOG_DIRS:
            dynamic_path = os.path.join(log_dir, dynamic_log_subdir)
            if not os.path.exists(dynamic_path):
                os.makedirs(dynamic_path)
    
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
        # Determine the final log directory path
        final_log_dir = log_dir
        if dynamic_log_subdir:
            final_log_dir = os.path.join(log_dir, dynamic_log_subdir)
        else:
            for subdir in config.DIFFERENT_LOG_DIRS:
                if type == subdir:
                    final_log_dir = os.path.join(log_dir, subdir)
                    break
        
        main_log_file = os.path.join(final_log_dir, f"{name}_{timestamp}.log")
        
        main_file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        main_file_handler.setFormatter(formatter)
        root_logger.addHandler(main_file_handler)
    
    # Set the configured flag
    _is_configured = True
    
    # Suppress noisy loggers from external libraries
    noisy_loggers = [
        'asyncio',
        'kaleido',
        'choreographer',
        'urllib3',
        'matplotlib',
        'plotly',
        'selenium',
        'yfinance',
        'peewee',
        'browser_proc'
        #'root'  # optional: this avoids logs from libraries using root logger
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Log startup
    logger = get_logger(type)
    startup_msg = f"Logging system initialized. Environment: {environment}"
    if dynamic_log_subdir:
        category_desc = get_category_description(dynamic_log_subdir)
        startup_msg += f", Log category: {dynamic_log_subdir} ({category_desc})"
    logger.info(startup_msg)
    
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

def get_logger(name: Optional[str] = None, auto_setup: bool = True) -> logging.Logger:
    """
    Get a logger instance with auto-detection of caller module name.
    If logging system isn't configured yet and auto_setup is True, 
    it will be configured with default settings.
    
    Args:
        name: Logger name. If None, automatically detects the calling module name
        auto_setup: Whether to automatically setup logging if not configured
        
    Returns:
        Logger instance configured with centralized settings
        
    Examples:
        # Auto-detect caller module name
        logger = get_logger()
        
        # Explicit name
        logger = get_logger(__name__)
        logger = get_logger('custom_logger')
        
        # Don't auto-setup logging
        logger = get_logger(auto_setup=False)
    """
    global _is_configured
    
    if not _is_configured and auto_setup:
        setup_logging()
    
    # Auto-detect caller module name if not provided
    if name is None:
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the calling module
            caller_frame = frame.f_back
            name = caller_frame.f_globals.get('__name__', 'unknown')
        finally:
            del frame  # Prevent reference cycles
    
    return logging.getLogger(name)

def log_with_context(logger: logging.Logger, level: str, message: str, 
                    context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a message with additional context information.
    
    Args:
        logger: Logger instance to use
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        message: Main log message
        context: Dictionary of additional context information
        
    Example:
        log_with_context(logger, 'info', 'Trade executed', {
            'symbol': 'BTCUSD',
            'size': 0.1,
            'price': 45000.00
        })
    """
    context = context or {}
    
    # Format context as key=value pairs
    context_str = ' '.join([f"{k}={v}" for k, v in context.items()])
    full_message = f"{message} | {context_str}" if context_str else message
    
    # Get the appropriate logging method
    log_method = getattr(logger, level.lower())
    log_method(full_message)
