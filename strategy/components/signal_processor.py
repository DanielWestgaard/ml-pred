import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class SignalProcessor:
    """
    Converts model predictions into actionable trading signals.
    Applies filters and confidence thresholds to raw predictions.
    """
    
    def __init__(self, confidence_threshold: float = 0.6, 
                 signal_filters: List[str] = None, **kwargs):
        """
        Initialize signal processor
        
        Args:
            confidence_threshold: Minimum confidence required for signal generation
            signal_filters: List of filter names to apply to signals
        """
        self.confidence_threshold = confidence_threshold
        self.signal_filters = signal_filters or []
        
        logging.info(f"Signal processor initialized with confidence threshold: {confidence_threshold}")
    
    def process(self, predictions: np.ndarray, market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Process raw model predictions into trading signals
        
        Args:
            predictions: Raw model predictions
            market_data: Current market data for context
            
        Returns:
            Dictionary containing signal information
        """
        signals = {
            'raw_predictions': predictions,
            'processed_signals': [],
            'signal_strength': [],
            'signal_direction': [],
            'confidence_scores': [],
            'timestamp': pd.Timestamp.now()
        }
        
        # Convert predictions to signals
        for i, prediction in enumerate(predictions):
            signal_info = self._convert_prediction_to_signal(prediction, market_data, i)
            
            if signal_info['confidence'] >= self.confidence_threshold:
                signals['processed_signals'].append(signal_info)
                signals['signal_strength'].append(signal_info['strength'])
                signals['signal_direction'].append(signal_info['direction'])
                signals['confidence_scores'].append(signal_info['confidence'])
        
        # Apply filters
        signals = self._apply_filters(signals, market_data)
        
        logging.info(f"Generated {len(signals['processed_signals'])} signals from {len(predictions)} predictions")
        return signals
    
    def _convert_prediction_to_signal(self, prediction: float, market_data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """
        Convert a single prediction to signal information
        
        Args:
            prediction: Single prediction value
            market_data: Market data context
            index: Index of current prediction
            
        Returns:
            Signal information dictionary
        """
        # Basic signal conversion logic (to be customized per hypothesis)
        if prediction > 0.5:
            direction = 'BUY'
            strength = abs(prediction - 0.5) * 2  # Normalize to 0-1
        else:
            direction = 'SELL'
            strength = abs(0.5 - prediction) * 2  # Normalize to 0-1
        
        # Calculate confidence (simplified - can be enhanced)
        confidence = strength
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'prediction_value': prediction,
            'index': index
        }
    
    def _apply_filters(self, signals: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply configured filters to signals
        
        Args:
            signals: Signal dictionary
            market_data: Market data for filter context
            
        Returns:
            Filtered signals
        """
        for filter_name in self.signal_filters:
            if hasattr(self, f'_apply_{filter_name}'):
                filter_method = getattr(self, f'_apply_{filter_name}')
                signals = filter_method(signals, market_data)
            else:
                logging.warning(f"Unknown signal filter: {filter_name}")
        
        return signals
    
    def _apply_volatility_filter(self, signals: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Filter signals based on market volatility
        """
        # Example volatility filter - only trade in moderate volatility
        if market_data is not None and len(market_data) > 20:
            recent_volatility = market_data['close'].rolling(20).std().iloc[-1]
            vol_threshold_high = market_data['close'].rolling(100).std().mean() * 2
            vol_threshold_low = market_data['close'].rolling(100).std().mean() * 0.5
            
            if recent_volatility > vol_threshold_high or recent_volatility < vol_threshold_low:
                # Filter out signals in extreme volatility conditions
                signals['processed_signals'] = []
                signals['signal_strength'] = []
                signals['signal_direction'] = []
                signals['confidence_scores'] = []
                logging.info("Signals filtered out due to extreme volatility")
        
        return signals
    
    def _apply_session_filter(self, signals: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Filter signals based on trading session
        """
        # Example session filter - placeholder for session-specific logic
        current_hour = pd.Timestamp.now().hour
        
        # Only trade during active market hours (example: 8-16 UTC)
        if not (8 <= current_hour <= 16):
            signals['processed_signals'] = []
            signals['signal_strength'] = []
            signals['signal_direction'] = []
            signals['confidence_scores'] = []
            logging.info("Signals filtered out due to inactive trading session")
        
        return signals