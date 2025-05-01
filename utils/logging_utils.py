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
    
    # Set up trade logger if requested
    if include_trade_logs and log_to_file:
        _setup_trade_logger(log_dir, max_file_size, backup_count)
    
    # Set up performance logger if requested
    if include_performance_logs and log_to_file:
        _setup_performance_logger(log_dir, max_file_size, backup_count)
    
    # Set the configured flag
    _is_configured = True
    
    # Log startup
    logger = get_logger(type)
    logger.info(f"Logging system initialized. Environment: {environment}")
    
    # Add an exception hook to log uncaught exceptions
    sys.excepthook = _log_uncaught_exception
    
    return logger

#
def _setup_trade_logger(log_dir, max_file_size, backup_count):
    """Set up a separate logger for trade events."""
    trade_logger = logging.getLogger("trades")
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False  # Don't propagate to root logger
    
    trade_log_file = os.path.join(log_dir, "trades", f"trades_{get_timestamp()}.log")
    
    # Create a JSON formatter for structured trade logs
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            # If the message is already a dict, use it directly
            if isinstance(record.msg, dict):
                log_data = record.msg
            else:
                log_data = {
                    "message": record.getMessage(),
                }
            
            log_data.update({
                "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S.%f"),
                "level": record.levelname,
                "logger": record.name
            })
            
            return json.dumps(log_data)
    
    trade_handler = RotatingFileHandler(
        trade_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    trade_handler.setFormatter(JsonFormatter())
    trade_logger.addHandler(trade_handler)

def _setup_performance_logger(log_dir, max_file_size, backup_count):
    """Set up a separate logger for performance metrics."""
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False  # Don't propagate to root logger
    
    perf_log_file = os.path.join(log_dir, "performance", f"performance_{get_timestamp()}.log")
    
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            if isinstance(record.msg, dict):
                log_data = record.msg
            else:
                log_data = {"message": record.getMessage()}
            
            log_data.update({
                "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S.%f"),
                "level": record.levelname,
                "logger": record.name
            })
            
            return json.dumps(log_data)
    
    perf_handler = RotatingFileHandler(
        perf_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    perf_handler.setFormatter(JsonFormatter())
    perf_logger.addHandler(perf_handler)

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

# Trade-specific logging functions
def log_trade(trade_data):
    """
    Log structured trade information to the trade log.
    
    Args:
        trade_data: Dictionary with trade information
    """
    trade_logger = logging.getLogger("trades")
    trade_logger.info(trade_data)
    
    # Also log a simplified version to the main log
    main_logger = get_logger("trades")
    main_logger.info(f"TRADE: {trade_data['side']} {trade_data['symbol']} "
                    f"@ {trade_data['price']} Qty: {trade_data['quantity']}")

def log_order(order_data):
    """
    Log structured order information to the trade log.
    
    Args:
        order_data: Dictionary with order information
    """
    trade_logger = logging.getLogger("trades")
    
    order_log = {
        "event_type": "order",
        **order_data
    }
    
    trade_logger.info(order_log)
    
    # Also log a simplified version to the main log
    main_logger = get_logger("orders")
    main_logger.info(f"ORDER: {order_data['side']} {order_data['symbol']} "
                    f"@ {order_data.get('price', 'MARKET')} Type: {order_data['type']}")

def log_performance(metric_name, value, context=None):
    """
    Log performance metrics.
    
    Args:
        metric_name: Name of the performance metric
        value: Value of the metric
        context: Additional context information
    """
    perf_logger = logging.getLogger("performance")
    
    perf_data = {
        "metric": metric_name,
        "value": value
    }
    
    if context:
        perf_data["context"] = context
    
    perf_logger.info(perf_data)

def log_error(error, context=None, include_traceback=True):
    """
    Enhanced error logging with context and optional traceback.
    
    Args:
        error: Error message or exception object
        context: Additional context information
        include_traceback: Whether to include traceback
    """
    logger = get_logger("error")
    
    if isinstance(error, Exception):
        error_msg = str(error)
        exc_info = error if include_traceback else None
    else:
        error_msg = error
        exc_info = None
    
    if context:
        logger.error(f"{error_msg} | Context: {context}", exc_info=exc_info)
    else:
        logger.error(error_msg, exc_info=exc_info)