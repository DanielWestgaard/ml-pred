import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

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
        elif self.model_type == ModelType.MULTICLASS_CLASSIFICATION:
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
        
        For three-class directional models:
        - Class 0 (hold) → NO SIGNAL (low confidence, filtered out)
        - Class 1 (buy) → BUY signal with high confidence  
        - Class 2 (sell) → SELL signal with high confidence
        
        For binary classification models:
        - Class 1 (positive) → BUY signal with high confidence
        - Class 0 (negative) → NO SIGNAL (low confidence, filtered out)
        
        Args:
            prediction: Class prediction (0, 1, or 2 for three-class; 0 or 1 for binary)
            probability: Probability array or single probability
            index: Prediction index
        """
        # Handle three-class directional predictions
        if hasattr(probability, '__len__') and len(probability) == 3:
            # Three-class model: [prob_hold, prob_buy, prob_sell]
            prob_hold, prob_buy, prob_sell = probability
            predicted_class = int(prediction)
            
            if predicted_class == 1:  # BUY signal (Class 1)
                confidence = prob_buy
                direction = 'BUY'
                strength = prob_buy
            elif predicted_class == 2:  # SELL signal (Class 2)
                confidence = prob_sell  
                direction = 'SELL'
                strength = prob_sell
            else:  # HOLD signal (Class 0) - should NOT generate trading signals
                confidence = prob_hold  # Use actual model confidence for HOLD predictions
                direction = 'HOLD'  # Proper direction for HOLD
                strength = prob_hold  # Use actual strength for HOLD signals
                
        # Handle binary classification (legacy support)
        elif probability is not None:
            if isinstance(probability, (list, tuple)):
                probability = probability[1] if len(probability) > 1 else probability[0]
                
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


class SmartHybridProcessor(SignalProcessor):
    """
    Smart hybrid processor that balances signal quality with quantity.
    Uses intelligent filtering without over-restricting valid trading opportunities.
    """
    
    def __init__(self, confidence_threshold: float = 0.35,
                 signal_filters: List[str] = None,
                 model_type: Optional[ModelType] = None,
                 min_direction_change_seconds: int = 30,
                 same_direction_cooldown_seconds: int = 120,
                 confidence_percentile_filter: float = 0.2,
                 max_signals_per_minute: int = 3,
                 **kwargs):
        """
        Initialize smart hybrid processor
        
        Args:
            confidence_threshold: Base confidence threshold
            min_direction_change_seconds: Minimum seconds between direction changes (30s)
            same_direction_cooldown_seconds: Cooldown for same direction signals (2min)
            confidence_percentile_filter: Filter bottom X% of predictions (0.2 = bottom 20%)
            max_signals_per_minute: Maximum signals per minute to prevent spam
        """
        super().__init__(confidence_threshold, signal_filters, model_type, **kwargs)
        
        self.min_direction_change_seconds = min_direction_change_seconds
        self.same_direction_cooldown_seconds = same_direction_cooldown_seconds
        self.confidence_percentile_filter = confidence_percentile_filter
        self.max_signals_per_minute = max_signals_per_minute
        
        # Smart state tracking
        self.last_signal_time = {}
        self.last_direction = None
        self.confidence_buffer = deque(maxlen=100)  # Rolling confidence history
        self.minute_signal_count = deque(maxlen=60)  # Track per-minute signals
        
        logger.info(f"SmartHybridProcessor initialized:")
        logger.info(f"  - Direction change cooldown: {min_direction_change_seconds}s")
        logger.info(f"  - Same direction cooldown: {same_direction_cooldown_seconds}s")
        logger.info(f"  - Confidence percentile filter: {confidence_percentile_filter}")
        logger.info(f"  - Max signals/minute: {max_signals_per_minute}")
    
    def process(self, predictions: np.ndarray, market_data: pd.DataFrame = None,
                model=None, prediction_probabilities: np.ndarray = None) -> Dict[str, Any]:
        """
        Process predictions with smart hybrid filtering
        """
        # Get base signals first
        signals = super().process(predictions, market_data, model, prediction_probabilities)
        
        # Apply smart filtering
        signals = self._apply_smart_filters(signals, market_data)
        
        return signals
    
    def _apply_smart_filters(self, signals: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply smart filtering that prioritizes quality while maintaining reasonable volume
        """
        if not signals['processed_signals']:
            return signals
        
        current_time = datetime.now()
        filtered_signals = []
        filtered_strengths = []
        filtered_directions = []
        filtered_confidences = []
        
        # Build confidence distribution for percentile filtering
        all_confidences = [s['confidence'] for s in signals['processed_signals']]
        if len(all_confidences) > 5:  # Need reasonable sample size
            confidence_threshold_dynamic = np.percentile(all_confidences, 
                                                       self.confidence_percentile_filter * 100)
        else:
            confidence_threshold_dynamic = self.confidence_threshold
        
        for i, signal in enumerate(signals['processed_signals']):
            direction = signal['direction']
            confidence = signal['confidence']
            
            # Update confidence buffer for analysis
            self.confidence_buffer.append({
                'timestamp': current_time,
                'confidence': confidence,
                'direction': direction
            })
            
            # Smart Layer 1: Dynamic Confidence Filter
            if confidence < max(confidence_threshold_dynamic, self.confidence_threshold):
                logger.debug(f"Signal filtered by dynamic confidence: {confidence:.4f} < {confidence_threshold_dynamic:.4f}")
                continue
                
            # Smart Layer 2: Intelligent Temporal Filter
            if not self._passes_smart_temporal_filter(direction, current_time):
                logger.debug(f"Signal filtered by smart temporal filter: {direction}")
                continue
            
            # Smart Layer 3: Rate Limiting (per minute, not hour)
            if not self._passes_rate_limit(current_time):
                logger.debug(f"Signal filtered by rate limit")
                continue
            
            # Signal accepted
            filtered_signals.append(signal)
            filtered_strengths.append(signals['signal_strength'][i])
            filtered_directions.append(signals['signal_direction'][i])
            filtered_confidences.append(signals['confidence_scores'][i])
            
            # Update state
            self.last_signal_time[direction] = current_time
            self.last_direction = direction
            self._update_minute_counter(current_time)
            
            logger.info(f"Smart signal accepted: {direction} confidence={confidence:.4f}")
        
        # Update signals with filtered results
        signals['processed_signals'] = filtered_signals
        signals['signal_strength'] = filtered_strengths
        signals['signal_direction'] = filtered_directions
        signals['confidence_scores'] = filtered_confidences
        
        filter_summary = f"Smart filtering: {len(signals['raw_predictions'])} predictions → {len(filtered_signals)} signals"
        logger.info(filter_summary)
        
        return signals
    
    def _passes_smart_temporal_filter(self, direction: str, current_time: datetime) -> bool:
        """
        Smart temporal filter: Short cooldown for direction changes, longer for same direction
        """
        if direction not in self.last_signal_time:
            return True
        
        time_since_last = current_time - self.last_signal_time[direction]
        seconds_since = time_since_last.total_seconds()
        
        # Direction change gets priority with shorter cooldown
        if self.last_direction and direction != self.last_direction:
            return seconds_since >= self.min_direction_change_seconds
        else:
            # Same direction needs longer cooldown
            return seconds_since >= self.same_direction_cooldown_seconds
    
    def _passes_rate_limit(self, current_time: datetime) -> bool:
        """
        Rate limiting per minute instead of per hour
        """
        # Clean old entries
        cutoff_time = current_time - timedelta(minutes=1)
        while self.minute_signal_count and self.minute_signal_count[0] < cutoff_time:
            self.minute_signal_count.popleft()
        
        return len(self.minute_signal_count) < self.max_signals_per_minute
    
    def _update_minute_counter(self, current_time: datetime):
        """Update per-minute signal counter"""
        self.minute_signal_count.append(current_time)
    
    def get_smart_stats(self) -> Dict[str, Any]:
        """Get smart filtering statistics"""
        current_time = datetime.now()
        
        # Recent confidence stats
        recent_confidences = [h['confidence'] for h in self.confidence_buffer]
        avg_confidence = np.mean(recent_confidences) if recent_confidences else 0.0
        
        return {
            'signals_last_minute': len(self.minute_signal_count),
            'avg_confidence': avg_confidence,
            'confidence_buffer_size': len(self.confidence_buffer),
            'last_signal_times': dict(self.last_signal_time),
            'last_direction': self.last_direction
        }


class UltraPermissiveProcessor(SignalProcessor):
    """
    Ultra-permissive processor that maximizes signal generation with minimal filtering.
    Only applies essential quality filters to unleash your model's full potential.
    """
    
    def __init__(self, confidence_threshold: float = 0.30,
                 signal_filters: List[str] = None,
                 model_type: Optional[ModelType] = None,
                 max_signals_per_second: int = 10,
                 **kwargs):
        """
        Initialize ultra-permissive processor with minimal restrictions
        
        Args:
            confidence_threshold: Very low confidence threshold
            max_signals_per_second: Prevent system overload only
        """
        super().__init__(confidence_threshold, signal_filters, model_type, **kwargs)
        
        self.max_signals_per_second = max_signals_per_second
        self.second_signal_count = deque(maxlen=10)  # Track per-second signals
        
        logger.info(f"UltraPermissiveProcessor initialized:")
        logger.info(f"  - Confidence threshold: {confidence_threshold}")
        logger.info(f"  - Max signals/second: {max_signals_per_second}")
        logger.info(f"  - NO temporal cooldowns")
        logger.info(f"  - NO confidence delta requirements")
    
    def process(self, predictions: np.ndarray, market_data: pd.DataFrame = None,
                model=None, prediction_probabilities: np.ndarray = None) -> Dict[str, Any]:
        """
        Process predictions with ultra-permissive filtering
        """
        # Get base signals first
        signals = super().process(predictions, market_data, model, prediction_probabilities)
        
        # Apply minimal filtering
        signals = self._apply_minimal_filters(signals, market_data)
        
        return signals
    
    def _apply_minimal_filters(self, signals: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply only essential filters to prevent system overload
        """
        if not signals['processed_signals']:
            return signals
        
        current_time = datetime.now()
        filtered_signals = []
        filtered_strengths = []
        filtered_directions = []
        filtered_confidences = []
        
        for i, signal in enumerate(signals['processed_signals']):
            direction = signal['direction']
            confidence = signal['confidence']
            
            # ONLY filter: System overload protection
            if not self._passes_overload_protection(current_time):
                logger.debug(f"Signal filtered by overload protection")
                continue
            
            # Accept everything else!
            filtered_signals.append(signal)
            filtered_strengths.append(signals['signal_strength'][i])
            filtered_directions.append(signals['signal_direction'][i])
            filtered_confidences.append(signals['confidence_scores'][i])
            
            # Update overload tracking
            self._update_second_counter(current_time)
            
            logger.debug(f"Ultra signal accepted: {direction} confidence={confidence:.4f}")
        
        # Update signals with filtered results
        signals['processed_signals'] = filtered_signals
        signals['signal_strength'] = filtered_strengths
        signals['signal_direction'] = filtered_directions
        signals['confidence_scores'] = filtered_confidences
        
        filter_summary = f"Ultra filtering: {len(signals['raw_predictions'])} predictions → {len(filtered_signals)} signals"
        logger.info(filter_summary)
        
        return signals
    
    def _passes_overload_protection(self, current_time: datetime) -> bool:
        """
        Only prevent system overload - allow everything else
        """
        # Clean old entries
        cutoff_time = current_time - timedelta(seconds=1)
        while self.second_signal_count and self.second_signal_count[0] < cutoff_time:
            self.second_signal_count.popleft()
        
        return len(self.second_signal_count) < self.max_signals_per_second
    
    def _update_second_counter(self, current_time: datetime):
        """Update per-second signal counter"""
        self.second_signal_count.append(current_time)
    
    def get_ultra_stats(self) -> Dict[str, Any]:
        """Get ultra processor statistics"""
        return {
            'signals_last_second': len(self.second_signal_count),
            'max_signals_per_second': self.max_signals_per_second
        }


class AdvancedSignalProcessor(SignalProcessor):
    """
    Advanced signal processor with intelligent filtering to reduce signal noise.
    Implements temporal cooldown, confidence delta detection, and directional change filters.
    """
    
    def __init__(self, confidence_threshold: float = 0.49,
                 signal_filters: List[str] = None,
                 model_type: Optional[ModelType] = None,
                 cooldown_minutes: int = 30,
                 min_confidence_delta: float = 0.05,
                 max_signals_per_hour: int = 2,
                 confidence_window_minutes: int = 10,
                 **kwargs):
        """
        Initialize advanced signal processor with intelligent filtering
        
        Args:
            confidence_threshold: Minimum confidence required for signal generation
            signal_filters: List of filter names to apply to signals
            model_type: Type of model (auto-detected if not specified)
            cooldown_minutes: Minimum minutes between signals in same direction
            min_confidence_delta: Minimum confidence increase required vs recent average
            max_signals_per_hour: Maximum signals allowed per hour
            confidence_window_minutes: Window for calculating confidence average
        """
        super().__init__(confidence_threshold, signal_filters, model_type, **kwargs)
        
        # Advanced filtering parameters
        self.cooldown_minutes = cooldown_minutes
        self.min_confidence_delta = min_confidence_delta
        self.max_signals_per_hour = max_signals_per_hour
        self.confidence_window_minutes = confidence_window_minutes
        
        # State tracking
        self.last_signal_time = {}  # Per direction: {'BUY': timestamp, 'SELL': timestamp}
        self.confidence_history = deque(maxlen=100)  # Rolling confidence history
        self.hourly_signal_count = deque(maxlen=60)  # Track signals per minute for hourly count
        self.last_signal_direction = None
        
        logger.info(f"AdvancedSignalProcessor initialized:")
        logger.info(f"  - Cooldown: {cooldown_minutes}min")
        logger.info(f"  - Min confidence delta: {min_confidence_delta}")
        logger.info(f"  - Max signals/hour: {max_signals_per_hour}")
    
    def process(self, predictions: np.ndarray, market_data: pd.DataFrame = None,
                model=None, prediction_probabilities: np.ndarray = None) -> Dict[str, Any]:
        """
        Process predictions with advanced filtering
        """
        # Get base signals first
        signals = super().process(predictions, market_data, model, prediction_probabilities)
        
        # Apply advanced filtering
        signals = self._apply_advanced_filters(signals, market_data)
        
        return signals
    
    def _apply_advanced_filters(self, signals: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply advanced filtering layers to reduce signal noise
        """
        if not signals['processed_signals']:
            return signals
        
        current_time = datetime.now()
        filtered_signals = []
        filtered_strengths = []
        filtered_directions = []
        filtered_confidences = []
        
        for i, signal in enumerate(signals['processed_signals']):
            direction = signal['direction']
            confidence = signal['confidence']
            
            # Update confidence history for delta calculation
            self.confidence_history.append({
                'timestamp': current_time,
                'confidence': confidence,
                'direction': direction
            })
            
            # Layer 1: Temporal Cooldown Filter
            if not self._passes_temporal_filter(direction, current_time):
                logger.debug(f"Signal filtered by temporal cooldown: {direction}")
                continue
            
            # Layer 2: Confidence Delta Filter  
            if not self._passes_confidence_delta_filter(confidence, direction, current_time):
                logger.debug(f"Signal filtered by confidence delta: {confidence:.4f}")
                continue
            
            # Layer 3: Hourly Signal Limit
            if not self._passes_hourly_limit_filter(current_time):
                logger.debug(f"Signal filtered by hourly limit")
                continue
            
            # Layer 4: Directional Change Priority
            priority_boost = self._calculate_directional_change_priority(direction)
            signal['priority_boost'] = priority_boost
            
            # Signal passed all filters
            filtered_signals.append(signal)
            filtered_strengths.append(signals['signal_strength'][i])
            filtered_directions.append(signals['signal_direction'][i])
            filtered_confidences.append(signals['confidence_scores'][i])
            
            # Update tracking state
            self.last_signal_time[direction] = current_time
            self.last_signal_direction = direction
            self._update_hourly_counter(current_time)
            
            logger.info(f"Advanced signal accepted: {direction} confidence={confidence:.4f} (delta priority: {priority_boost:.2f})")
        
        # Update signals with filtered results
        signals['processed_signals'] = filtered_signals
        signals['signal_strength'] = filtered_strengths
        signals['signal_direction'] = filtered_directions
        signals['confidence_scores'] = filtered_confidences
        
        filter_summary = f"Advanced filtering: {len(signals['raw_predictions'])} predictions → {len(filtered_signals)} signals"
        logger.info(filter_summary)
        
        return signals
    
    def _passes_temporal_filter(self, direction: str, current_time: datetime) -> bool:
        """
        Check if enough time has passed since last signal in this direction
        """
        if direction not in self.last_signal_time:
            return True
        
        time_since_last = current_time - self.last_signal_time[direction]
        return time_since_last.total_seconds() >= (self.cooldown_minutes * 60)
    
    def _passes_confidence_delta_filter(self, confidence: float, direction: str, current_time: datetime) -> bool:
        """
        Check if confidence represents significant improvement over recent average
        """
        if len(self.confidence_history) < 3:  # Need some history
            return True
        
        # Calculate recent average confidence for same direction
        cutoff_time = current_time - timedelta(minutes=self.confidence_window_minutes)
        
        recent_same_direction = [
            h['confidence'] for h in self.confidence_history
            if h['timestamp'] > cutoff_time and h['direction'] == direction
        ]
        
        if not recent_same_direction:
            # No recent signals in same direction, allow this one
            return True
        
        recent_avg = np.mean(recent_same_direction)
        confidence_delta = confidence - recent_avg
        
        logger.debug(f"Confidence delta check: {confidence:.4f} vs avg {recent_avg:.4f} = delta {confidence_delta:.4f}")
        
        return confidence_delta >= self.min_confidence_delta
    
    def _passes_hourly_limit_filter(self, current_time: datetime) -> bool:
        """
        Check if we haven't exceeded hourly signal limit
        """
        # Clean old entries from hourly counter
        cutoff_time = current_time - timedelta(hours=1)
        while self.hourly_signal_count and self.hourly_signal_count[0] < cutoff_time:
            self.hourly_signal_count.popleft()
        
        return len(self.hourly_signal_count) < self.max_signals_per_hour
    
    def _calculate_directional_change_priority(self, direction: str) -> float:
        """
        Calculate priority boost for signals that change direction
        """
        if self.last_signal_direction is None:
            return 1.0  # No previous direction
        
        if direction != self.last_signal_direction:
            return 2.0  # Direction change gets priority boost
        else:
            return 0.5  # Same direction gets lower priority
    
    def _update_hourly_counter(self, current_time: datetime):
        """
        Update hourly signal counter
        """
        self.hourly_signal_count.append(current_time)
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """
        Get current filtering statistics for monitoring
        """
        current_time = datetime.now()
        
        # Recent signals in last hour
        hour_ago = current_time - timedelta(hours=1)
        recent_signals = len([t for t in self.hourly_signal_count if t > hour_ago])
        
        # Confidence history stats
        recent_confidences = [h['confidence'] for h in self.confidence_history]
        avg_confidence = np.mean(recent_confidences) if recent_confidences else 0.0
        
        return {
            'signals_last_hour': recent_signals,
            'avg_confidence': avg_confidence,
            'confidence_history_size': len(self.confidence_history),
            'last_signal_times': dict(self.last_signal_time),
            'last_direction': self.last_signal_direction
        }