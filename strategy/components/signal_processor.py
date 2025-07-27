import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from enum import Enum

from utils.logging_utils import get_logger
from utils.model.model_type_detector import ModelType

logger = get_logger()

class SignalProcessor:
    """
    Converts model predictions into actionable trading signals.
    Supports both regression and classification models with automatic detection.
    Applies filters and confidence thresholds to raw predictions.
    """
    
    def __init__(self, confidence_threshold: float = 0.6, 
                 signal_filters: List[str] = None, 
                 model_type: Optional[ModelType] = None,
                 **kwargs):
        """
        Initialize signal processor
        
        Args:
            confidence_threshold: Minimum confidence required for signal generation
            signal_filters: List of filter names to apply to signals
            model_type: Type of model (auto-detected if not specified)
        """
        self.confidence_threshold = confidence_threshold
        self.signal_filters = signal_filters or []
        self.model_type = model_type
        
        logger.info(f"Signal processor initialized with confidence threshold: {confidence_threshold}, model type: {model_type}")
    
    def process(self, predictions: np.ndarray, market_data: pd.DataFrame = None, 
                model=None, prediction_probabilities: np.ndarray = None) -> Dict[str, Any]:
        """
        Process raw model predictions into trading signals
        
        Args:
            predictions: Raw model predictions (class labels for classification, values for regression)
            market_data: Current market data for context
            model: Model object for auto-detection of type
            prediction_probabilities: Probability scores for classification models
            
        Returns:
            Dictionary containing signal information
        """
        # Auto-detect model type if not specified
        if self.model_type is None and model is not None:
            self.model_type = self._detect_model_type(model)
        
        signals = {
            'raw_predictions': predictions,
            'prediction_probabilities': prediction_probabilities,
            'processed_signals': [],
            'signal_strength': [],
            'signal_direction': [],
            'confidence_scores': [],
            'timestamp': pd.Timestamp.now(),
            'model_type': self.model_type.value if self.model_type else 'unknown'
        }
        
        # Convert predictions to signals
        for i, prediction in enumerate(predictions):
            prob = prediction_probabilities[i] if prediction_probabilities is not None else None
            signal_info = self._convert_prediction_to_signal(prediction, market_data, i, prob)
            
            # Debug logger for confidence filtering
            logger.debug(f"Prediction {i}: value={prediction}, confidence={signal_info['confidence']:.4f}, threshold={self.confidence_threshold}")
            
            if signal_info['confidence'] >= self.confidence_threshold:
                signals['processed_signals'].append(signal_info)
                signals['signal_strength'].append(signal_info['strength'])
                signals['signal_direction'].append(signal_info['direction'])
                signals['confidence_scores'].append(signal_info['confidence'])
            else:
                logger.debug(f"Signal rejected: confidence {signal_info['confidence']:.4f} < threshold {self.confidence_threshold}")
        
        # Apply filters
        signals = self._apply_filters(signals, market_data)
        
        logger.info(f"Generated {len(signals['processed_signals'])} signals from {len(predictions)} predictions (model: {self.model_type})")
        return signals
    
    def _convert_prediction_to_signal(self, prediction: float, market_data: pd.DataFrame, 
                                    index: int, probability: Optional[float] = None) -> Dict[str, Any]:
        """
        Convert a single prediction to signal information
        
        Args:
            prediction: Single prediction value (class label for classification, value for regression)
            market_data: Market data context
            index: Index of current prediction
            probability: Probability score for classification models
            
        Returns:
            Signal information dictionary
        """
        if self.model_type == ModelType.BINARY_CLASSIFICATION:
            return self._convert_classification_prediction(prediction, probability, index)
        elif self.model_type == ModelType.REGRESSION:
            return self._convert_regression_prediction(prediction, index)
        else:
            # Fallback to regression logic for unknown models
            logger.warning(f"Unknown model type {self.model_type}, defaulting to regression logic")
            return self._convert_regression_prediction(prediction, index)
    
    def _convert_classification_prediction(self, prediction: float, probability: Optional[float], index: int) -> Dict[str, Any]:
        """
        Convert classification prediction to signal
        
        For binary classification models like spread compression:
        - Class 1 (positive) → BUY signal with high confidence
        - Class 0 (negative) → NO SIGNAL (low confidence, filtered out)
        
        Args:
            prediction: Class prediction (0 or 1)
            probability: Probability of positive class
            index: Prediction index
        """
        # Use probability if available, otherwise use binary prediction
        if probability is not None:
            # Only generate signals for positive class predictions
            if probability > 0.5:  # Class 1 prediction
                confidence = probability  # Use actual probability as confidence
                direction = 'BUY'
                strength = probability  # Probability itself is the strength
            else:  # Class 0 prediction
                # For negative predictions, set very low confidence to filter them out
                confidence = probability  # Use the low probability of positive class
                direction = 'BUY'  # Default direction, but will be filtered out
                strength = probability  # Very low strength for negative predictions
        else:
            # Binary prediction without probabilities
            if prediction == 1:
                confidence = 0.8  # High confidence for positive prediction
                direction = 'BUY'
                strength = 0.8
            else:
                # For class 0, set low confidence to ensure filtering
                confidence = 0.05  # Very low confidence to trigger filtering
                direction = 'BUY'  # Default direction, but will be filtered out
                strength = 0.05
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'prediction_value': prediction,
            'probability': probability,
            'index': index,
            'signal_type': 'classification'
        }
    
    def _convert_regression_prediction(self, prediction: float, index: int) -> Dict[str, Any]:
        """
        Convert regression prediction to signal (original logic)
        
        Args:
            prediction: Regression prediction value
            index: Prediction index
        """
        # Original regression logic
        if prediction > 0.5:
            direction = 'BUY'
            strength = abs(prediction - 0.5) * 2  # Normalize to 0-1
        else:
            direction = 'SELL'
            strength = abs(0.5 - prediction) * 2  # Normalize to 0-1
        
        confidence = strength
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'prediction_value': prediction,
            'index': index,
            'signal_type': 'regression'
        }
    
    def _detect_model_type(self, model) -> ModelType:
        """
        Auto-detect model type from model object
        
        Args:
            model: Trained model object
            
        Returns:
            Detected model type
        """
        model_class_name = model.__class__.__name__.lower()
        
        # Classification model detection
        classification_keywords = ['classifier', 'classification']
        if any(keyword in model_class_name for keyword in classification_keywords):
            # Check if it's binary classification by looking for predict_proba
            if hasattr(model, 'predict_proba'):
                return ModelType.BINARY_CLASSIFICATION
            return ModelType.BINARY_CLASSIFICATION  # Default to binary for classifiers
        
        # Regression model detection
        regression_keywords = ['regressor', 'regression']
        if any(keyword in model_class_name for keyword in regression_keywords):
            return ModelType.REGRESSION
        
        # Default fallback based on available methods
        if hasattr(model, 'predict_proba'):
            return ModelType.BINARY_CLASSIFICATION
        else:
            return ModelType.REGRESSION
    
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
                logger.warning(f"Unknown signal filter: {filter_name}")
        
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
                logger.info("Signals filtered out due to extreme volatility")
        
        return signals
    
    def _apply_session_filter(self, signals: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Filter signals based on trading session
        """
        # Extended session filter for testing - covers main FX trading hours
        current_hour = pd.Timestamp.now().hour
        
        # Trade during extended market hours (covers major FX sessions)
        # Asian: 21-06 UTC, London: 07-16 UTC, NY: 12-21 UTC
        # This allows trading almost 24/7 except for brief quiet periods
        if not (6 <= current_hour <= 22):  # Allow trading 6 AM - 10 PM UTC
            signals['processed_signals'] = []
            signals['signal_strength'] = []
            signals['signal_direction'] = []
            signals['confidence_scores'] = []
            logger.info("Signals filtered out due to inactive trading session")
        
        return signals