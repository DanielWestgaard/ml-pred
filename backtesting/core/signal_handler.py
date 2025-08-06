import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from utils.logging_utils import get_logger

logger = get_logger()

class BacktestSignalHandler:
    """
    Handles signal processing coordination for backtesting.
    Manages the interaction between model predictions, signal processors,
    and hypothesis-specific logic.
    """
    
    def __init__(self, config: Dict[str, Any], model_type, hypothesis_plugin):
        """
        Initialize signal handler
        
        Args:
            config: Hypothesis configuration
            model_type: Model type from model handler
            hypothesis_plugin: Hypothesis plugin instance
        """
        self.config = config
        self.model_type = model_type
        self.hypothesis_plugin = hypothesis_plugin
        
        # Initialize signal processor
        self._initialize_signal_processor()
        
        # Initialize hypothesis-specific components
        self.exit_strategy = hypothesis_plugin.get_exit_strategy()
        self.signal_tracker = hypothesis_plugin.get_signal_tracker()
        
        # Statistics tracking
        self.signals_generated = 0
        self.signals_processed = 0
        
        logger.info("BacktestSignalHandler initialized successfully")
    
    def _initialize_signal_processor(self):
        """Initialize the appropriate signal processor based on configuration"""
        # Import signal processors
        from strategy.components.signal_processor import (
            SignalProcessor, AdvancedSignalProcessor, 
            SmartHybridProcessor, UltraPermissiveProcessor
        )
        
        signal_config = self.config.get('backtesting', {}).get('signal_processing', {})
        
        # Choose processor type
        processor_type = signal_config.get('processor_type', 'basic')
        
        # Create processor with appropriate parameters
        if processor_type == 'ultra':
            self.signal_processor = UltraPermissiveProcessor(
                confidence_threshold=signal_config.get('confidence_threshold', 0.30),
                signal_filters=signal_config.get('signal_filters', []),
                model_type=self.model_type,
                max_signals_per_second=signal_config.get('max_signals_per_second', 10)
            )
        elif processor_type == 'smart':
            self.signal_processor = SmartHybridProcessor(
                confidence_threshold=signal_config.get('confidence_threshold', 0.35),
                signal_filters=signal_config.get('signal_filters', []),
                model_type=self.model_type,
                min_direction_change_seconds=signal_config.get('min_direction_change_seconds', 30),
                same_direction_cooldown_seconds=signal_config.get('same_direction_cooldown_seconds', 120),
                confidence_percentile_filter=signal_config.get('confidence_percentile_filter', 0.2),
                max_signals_per_minute=signal_config.get('max_signals_per_minute', 3)
            )
        elif processor_type == 'advanced' or signal_config.get('use_advanced_filtering', False):
            self.signal_processor = AdvancedSignalProcessor(
                confidence_threshold=signal_config.get('confidence_threshold', 0.49),
                signal_filters=signal_config.get('signal_filters', []),
                model_type=self.model_type,
                cooldown_minutes=signal_config.get('cooldown_minutes', 30),
                min_confidence_delta=signal_config.get('min_confidence_delta', 0.05),
                max_signals_per_hour=signal_config.get('max_signals_per_hour', 2),
                confidence_window_minutes=signal_config.get('confidence_window_minutes', 10)
            )
        else:
            self.signal_processor = SignalProcessor(
                confidence_threshold=signal_config.get('confidence_threshold', 0.6),
                signal_filters=signal_config.get('signal_filters', []),
                model_type=self.model_type
            )
        
        logger.info(f"Initialized signal processor: {self.signal_processor.__class__.__name__}")
    
    def process_predictions(self, predictions: np.ndarray, probabilities: Optional[np.ndarray] = None,
                          market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process model predictions into trading signals
        
        Args:
            predictions: Model predictions
            probabilities: Model probabilities (if available)
            market_data: Current market data context
            
        Returns:
            Processed signals dictionary
        """
        try:
            # Process through signal processor
            signals = self.signal_processor.process(
                predictions=predictions,
                market_data=None,  # Signal processor doesn't need full market buffer for backtesting
                model=None,  # Not needed as model_type is already set
                prediction_probabilities=probabilities
            )
            
            # Apply hypothesis-specific feature customization
            if market_data:
                enhanced_market_data = self.hypothesis_plugin.customize_features(market_data)
                signals['enhanced_market_data'] = enhanced_market_data
            
            # Update statistics
            self.signals_generated += len(predictions)
            self.signals_processed += len(signals['processed_signals'])
            
            return signals
            
        except Exception as e:
            logger.error(f"Error processing predictions: {e}")
            return {
                'raw_predictions': predictions,
                'prediction_probabilities': probabilities,
                'processed_signals': [],
                'signal_strength': [],
                'signal_direction': [],
                'confidence_scores': [],
                'error': str(e)
            }
    
    def enhance_signal_with_market_data(self, signal_info: Dict[str, Any], 
                                      current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance signal with current market data and metadata
        
        Args:
            signal_info: Raw signal information
            current_data: Current market data
            
        Returns:
            Enhanced signal dictionary
        """
        enhanced_signal = {
            **signal_info,
            'symbol': 'GBPUSD',  # TODO: Make configurable
            'current_price': current_data['close'],
            'timestamp': current_data['timestamp'],
            'market_data': current_data
        }
        
        # Apply hypothesis-specific enhancements
        try:
            enhanced_market_data = self.hypothesis_plugin.customize_features(current_data)
            enhanced_signal['enhanced_market_data'] = enhanced_market_data
        except Exception as e:
            logger.debug(f"Error enhancing signal with hypothesis features: {e}")
        
        return enhanced_signal
    
    def should_exit_position(self, position: Dict[str, Any], current_data: Dict[str, Any],
                           predictions: Optional[np.ndarray] = None, 
                           probabilities: Optional[np.ndarray] = None) -> tuple:
        """
        Determine if a position should be exited using hypothesis-specific logic
        
        Args:
            position: Position information
            current_data: Current market data
            predictions: Latest model predictions
            probabilities: Latest model probabilities
            
        Returns:
            Tuple of (should_exit: bool, exit_reason: str)
        """
        try:
            if predictions is None or len(predictions) == 0:
                return False, None
            
            # Use the latest prediction
            latest_prediction = predictions[0]
            
            # Handle probabilities based on model type
            if probabilities is not None:
                if self.model_type.value == 'multiclass_classification':
                    latest_probability = probabilities[0]  # Full array
                else:
                    # Binary classification - extract scalar probability
                    latest_probability = probabilities[0] if isinstance(probabilities[0], (int, float)) else probabilities[0][1]
            else:
                latest_probability = None
            
            # Use hypothesis-specific exit strategy
            should_exit, exit_reason = self.exit_strategy.should_exit_position(
                position=position,
                current_data=current_data,
                prediction=latest_prediction,
                probability=latest_probability
            )
            
            return should_exit, exit_reason
            
        except Exception as e:
            logger.error(f"Error in exit decision: {e}")
            return False, None
    
    def track_entry_signal(self, position_id: str, entry_data: Dict[str, Any], 
                          confidence: float):
        """
        Track signal at position entry
        
        Args:
            position_id: Position identifier
            entry_data: Market data at entry
            confidence: Signal confidence
        """
        try:
            self.signal_tracker.track_entry_signal(position_id, entry_data, confidence)
        except Exception as e:
            logger.debug(f"Error tracking entry signal: {e}")
    
    def update_signal_tracking(self, position_id: str, current_data: Dict[str, Any],
                             current_confidence: float):
        """
        Update signal tracking for an open position
        
        Args:
            position_id: Position identifier
            current_data: Current market data
            current_confidence: Current signal confidence
        """
        try:
            self.signal_tracker.update_signal_strength(position_id, current_data, current_confidence)
        except Exception as e:
            logger.debug(f"Error updating signal tracking: {e}")
    
    def get_signal_history(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get signal tracking history for a position"""
        try:
            return self.signal_tracker.get_signal_history(position_id)
        except Exception as e:
            logger.debug(f"Error getting signal history: {e}")
            return None
    
    def cleanup_closed_position(self, position_id: str):
        """Clean up tracking data for a closed position"""
        try:
            self.signal_tracker.cleanup_closed_position(position_id)
        except Exception as e:
            logger.debug(f"Error cleaning up position tracking: {e}")
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """
        Get signal processor statistics
        
        Returns:
            Dictionary with processor-specific statistics
        """
        stats = {
            'processor_type': self.signal_processor.__class__.__name__,
            'signals_generated': self.signals_generated,
            'signals_processed': self.signals_processed,
            'processing_rate': self.signals_processed / max(1, self.signals_generated) * 100
        }
        
        # Get processor-specific stats
        if hasattr(self.signal_processor, 'get_filter_stats'):
            stats.update(self.signal_processor.get_filter_stats())
        elif hasattr(self.signal_processor, 'get_smart_stats'):
            stats.update(self.signal_processor.get_smart_stats())
        elif hasattr(self.signal_processor, 'get_ultra_stats'):
            stats.update(self.signal_processor.get_ultra_stats())
        
        return stats
    
    def get_exit_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the exit strategy being used
        
        Returns:
            Dictionary with exit strategy information
        """
        return {
            'exit_strategy_class': self.exit_strategy.__class__.__name__,
            'confidence_exit_threshold': getattr(self.exit_strategy, 'confidence_exit_threshold', 'not set'),
            'minimum_signal_life_minutes': getattr(self.exit_strategy, 'minimum_signal_life_minutes', 'not set'),
            'hypothesis_specific': self.exit_strategy.__class__.__name__ != 'DefaultExitStrategy'
        }
    
    def get_signal_tracker_info(self) -> Dict[str, Any]:
        """
        Get information about the signal tracker being used
        
        Returns:
            Dictionary with signal tracker information
        """
        return {
            'signal_tracker_class': self.signal_tracker.__class__.__name__,
            'track_compression_features': getattr(self.signal_tracker, 'track_compression_features', False),
            'track_volume_patterns': getattr(self.signal_tracker, 'track_volume_patterns', False),
            'tracked_positions_count': len(getattr(self.signal_tracker, 'tracked_signals', {})),
            'hypothesis_specific': self.signal_tracker.__class__.__name__ != 'DefaultSignalTracker'
        }