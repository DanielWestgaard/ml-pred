# utils/logging_examples.py
"""
Examples demonstrating proper usage of the centralized logging system.
This file shows the recommended patterns for logging throughout the codebase.
"""

from utils.logging_utils import get_logger, log_with_context

# Example 1: Auto-detect module name (recommended)
logger = get_logger()

# Example 2: Explicit module name  
# logger = get_logger(__name__)

# Example 3: Custom logger name
# logger = get_logger('custom_trading_logger')

def example_basic_logging():
    """Basic logging examples with different levels."""
    
    # Different log levels
    logger.debug("Detailed diagnostic information")
    logger.info("General operational information")  
    logger.warning("Something unexpected but not critical")
    logger.error("An error occurred but system continues")
    logger.critical("System-breaking error")

def example_trading_operations():
    """Examples of logging for trading operations with context."""
    
    # Trade execution with context
    log_with_context(
        logger, 'info', 'Position opened',
        {'symbol': 'BTCUSD', 'size': 0.1, 'direction': 'long', 'price': 45000.00}
    )
    
    # Position update
    log_with_context(
        logger, 'info', 'Position updated',
        {'deal_id': 'ABC123', 'stop_loss': 44000.00, 'take_profit': 46000.00}
    )
    
    # Error with context
    log_with_context(
        logger, 'error', 'Position creation failed',
        {'symbol': 'BTCUSD', 'error_code': 'INSUFFICIENT_MARGIN', 'account_balance': 1000.00}
    )

def example_data_processing():
    """Examples for data processing operations."""
    
    logger.info("Starting data processing pipeline")
    
    # Processing progress
    for i in range(1, 101):
        if i % 25 == 0:  # Log every 25%
            logger.info(f"Data processing progress: {i}%")
    
    # Processing complete with metrics
    log_with_context(
        logger, 'info', 'Data processing completed',
        {'total_records': 10000, 'processing_time_ms': 1234, 'errors': 0}
    )

def example_api_interactions():
    """Examples for API interactions and responses."""
    
    # API request
    logger.debug("Making API request to /api/v1/positions")
    
    # API response (don't log sensitive data)
    log_with_context(
        logger, 'debug', 'API response received',
        {'endpoint': '/api/v1/positions', 'status_code': 200, 'response_time_ms': 150}
    )
    
    # API error
    log_with_context(
        logger, 'error', 'API request failed',
        {'endpoint': '/api/v1/positions', 'status_code': 429, 'error': 'Rate limit exceeded'}
    )

def example_websocket_events():
    """Examples for WebSocket event logging."""
    
    # Connection events
    logger.info("WebSocket connection established")
    
    # Market data updates (use debug level as these are frequent)
    log_with_context(
        logger, 'debug', 'Market data update received',
        {'symbol': 'BTCUSD', 'price': 45000.00, 'volume': 1.5}
    )
    
    # Connection issues
    log_with_context(
        logger, 'warning', 'WebSocket connection lost, attempting reconnect',
        {'reconnect_attempt': 1, 'last_message_time': '2025-01-15T10:30:00Z'}
    )

if __name__ == "__main__":
    # Run examples
    print("Running logging examples...")
    
    example_basic_logging()
    example_trading_operations()
    example_data_processing()
    example_api_interactions()
    example_websocket_events()
    
    print("Examples completed. Check logs for output.")