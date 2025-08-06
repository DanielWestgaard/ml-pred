from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from utils.logging_utils import get_logger

logger = get_logger()

class ExitStrategy(ABC):
    """
    Base class for hypothesis-specific exit strategies.
    
    Exit strategies determine when to close positions based on model predictions,
    market conditions, and hypothesis-specific logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exit strategy
        
        Args:
            config: Exit strategy configuration from hypothesis config
        """
        self.config = config
        
        # Common exit parameters
        self.confidence_exit_threshold = config.get('confidence_exit_threshold', 0.30)
        self.minimum_signal_life_minutes = config.get('minimum_signal_life', 3)
        
        logger.info(f"Initialized {self.__class__.__name__} exit strategy")
    
    @abstractmethod
    def should_exit_position(self, position: Dict[str, Any], current_data: Dict[str, Any],
                           prediction: int, probability: Optional[float] = None) -> Tuple[bool, str]:
        """
        Determine if a position should be exited
        
        Args:
            position: Current position information
            current_data: Current market data and features
            prediction: Latest model prediction
            probability: Latest model probability (if available)
            
        Returns:
            Tuple of (should_exit: bool, exit_reason: str)
        """
        pass
    
    def check_minimum_life(self, position: Dict[str, Any], current_data: Dict[str, Any]) -> bool:
        """
        Check if position has been open for minimum required time
        
        Args:
            position: Position information
            current_data: Current market data
            
        Returns:
            True if position has exceeded minimum life, False otherwise
        """
        import pandas as pd
        
        entry_time = position.get('entry_time')
        if not entry_time:
            return True  # No entry time recorded, allow exit
        
        if isinstance(entry_time, str):
            entry_time = pd.to_datetime(entry_time)
        
        current_time = pd.to_datetime(current_data['timestamp'], utc=True)
        position_age_minutes = (current_time - entry_time).total_seconds() / 60
        
        return position_age_minutes >= self.minimum_signal_life_minutes


class DefaultExitStrategy(ExitStrategy):
    """
    Default exit strategy for models without specific hypothesis logic.
    
    Uses simple model-based exits:
    - Exit when model predicts HOLD with high confidence
    - No feature-specific logic
    """
    
    def should_exit_position(self, position: Dict[str, Any], current_data: Dict[str, Any],
                           prediction: int, probability: Optional[float] = None) -> Tuple[bool, str]:
        """
        Default exit logic: Exit on strong HOLD predictions
        """
        # Check minimum life first
        if not self.check_minimum_life(position, current_data):
            return False, None
        
        # Exit on HOLD prediction
        if prediction == 0:
            if probability is None:
                return True, 'model_exit_signal'
            elif hasattr(probability, '__len__') and len(probability) == 3:
                # Multiclass: check HOLD confidence (class 0)
                hold_confidence = probability[0]
                return hold_confidence > 0.7, 'model_exit_signal'
            else:
                # Binary: use inverse probability
                return (1 - probability) > 0.7, 'model_exit_signal'
        
        return False, None


class SpreadCompressionExitStrategy(ExitStrategy):
    """
    Specialized exit strategy for spread compression pattern detection.
    
    Uses intelligent exits based on:
    - Model prediction confidence for trading direction
    - Feature-based compression reversal detection
    - Contrary signal detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Spread compression specific parameters
        self.signal_decay_threshold = config.get('signal_strength_decay', 0.25)
        self.volatility_exit_multiplier = config.get('volatility_exit_multiplier', 1.5)
        
        logger.info(f"SpreadCompressionExitStrategy configured:")
        logger.info(f"  - Confidence exit threshold: {self.confidence_exit_threshold}")
        logger.info(f"  - Signal decay threshold: {self.signal_decay_threshold}")
        logger.info(f"  - Minimum signal life: {self.minimum_signal_life_minutes}min")
    
    def should_exit_position(self, position: Dict[str, Any], current_data: Dict[str, Any],
                           prediction: int, probability: Optional[float] = None) -> Tuple[bool, str]:
        """
        Intelligent exit logic specific to spread compression patterns
        """
        position_direction = position.get('direction', 'BUY')
        
        # Calculate proper confidence based on model output
        current_confidence, hold_confidence = self._calculate_confidences(
            prediction, probability, position_direction
        )
        
        # Check minimum signal life first (ENFORCED)
        if not self.check_minimum_life(position, current_data):
            # Only exit on EXTREMELY strong contrary signals during minimum life
            if self._is_emergency_contrary_signal(position_direction, prediction, current_confidence):
                logger.debug(f"Emergency contrary exit during minimum life")
                return True, "emergency_contrary_signal"
            return False, None
        
        # Exit Strategy 1: Strong contrary signal
        if self._is_contrary_signal(position_direction, prediction, current_confidence):
            logger.debug(f"Contrary signal exit: {position_direction} position, model predicts opposite")
            return True, "contrary_signal"
        
        # Exit Strategy 2: Extremely strong HOLD with very weak original signal
        if prediction == 0 and hold_confidence > 0.98 and current_confidence < 0.1:
            logger.debug(f"Market change exit: HOLD {hold_confidence:.4f} > 0.98")
            return True, "market_change"
        
        # Exit Strategy 3: Feature-based compression reversal
        if self._check_compression_reversal(position, current_data):
            return True, "compression_reversal"
        
        return False, None
    
    def _calculate_confidences(self, prediction: int, probability: Optional[float], 
                             position_direction: str) -> Tuple[float, float]:
        """Calculate trading confidence and hold confidence from model output"""
        if probability is not None:
            if hasattr(probability, '__len__') and len(probability) == 3:
                # Three-class model: [prob_hold, prob_buy, prob_sell]
                prob_hold, prob_buy, prob_sell = probability
                
                if position_direction == 'BUY':
                    current_confidence = prob_buy
                elif position_direction == 'SELL':
                    current_confidence = prob_sell
                else:
                    current_confidence = max(prob_buy, prob_sell)
                
                hold_confidence = prob_hold
            else:
                # Binary model
                if prediction == 1:
                    current_confidence = probability
                    hold_confidence = 1 - probability
                else:
                    current_confidence = 1 - probability if probability is not None else 0
                    hold_confidence = probability
        else:
            # No probability available
            current_confidence = 0.6 if prediction > 0 else 0.4
            hold_confidence = 0.4
        
        return current_confidence, hold_confidence
    
    def _is_contrary_signal(self, position_direction: str, prediction: int, confidence: float) -> bool:
        """Check if model predicts opposite direction with high confidence"""
        return ((position_direction == 'BUY' and prediction == 2 and confidence > 0.6) or 
                (position_direction == 'SELL' and prediction == 1 and confidence > 0.6))
    
    def _is_emergency_contrary_signal(self, position_direction: str, prediction: int, confidence: float) -> bool:
        """Check for extremely strong contrary signals that override minimum life"""
        return ((position_direction == 'BUY' and prediction == 2 and confidence > 0.8) or 
                (position_direction == 'SELL' and prediction == 1 and confidence > 0.8))
    
    def _check_compression_reversal(self, position: Dict[str, Any], current_data: Dict[str, Any]) -> bool:
        """Check if core compression features have reversed significantly"""
        try:
            # Check compression ratio deterioration
            current_compression = current_data.get('compression_ratio_micro', 0)
            entry_compression = position.get('entry_features', {}).get('compression_ratio_micro', 0)
            
            # Check volume confirmation loss
            current_volume_ratio = current_data.get('volume_ratio', 0)
            entry_volume_ratio = position.get('entry_features', {}).get('volume_ratio', 0)
            
            # Exit if compression has reversed by more than 15%
            if entry_compression != 0:
                compression_change = (current_compression - entry_compression) / abs(entry_compression)
                if compression_change < -0.15:
                    logger.debug(f"Compression reversal: {compression_change:.3f}")
                    return True
            
            # Exit if volume confirmation has significantly weakened
            if entry_volume_ratio != 0:
                volume_change = (current_volume_ratio - entry_volume_ratio) / abs(entry_volume_ratio)
                if volume_change < -0.3:
                    logger.debug(f"Volume confirmation loss: {volume_change:.3f}")
                    return True
        
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        
        return False


class ExitStrategyFactory:
    """Factory for creating hypothesis-specific exit strategies"""
    
    _strategies = {
        'default': DefaultExitStrategy,
        'spread_compression': SpreadCompressionExitStrategy
    }
    
    @classmethod
    def create(cls, hypothesis_name: str, config: Dict[str, Any]) -> ExitStrategy:
        """
        Create exit strategy for hypothesis
        
        Args:
            hypothesis_name: Name of hypothesis
            config: Exit strategy configuration
            
        Returns:
            Appropriate ExitStrategy instance
        """
        # Map hypothesis names to strategy types
        if 'spread_compression' in hypothesis_name.lower():
            strategy_type = 'spread_compression'
        else:
            strategy_type = 'default'
        
        strategy_class = cls._strategies.get(strategy_type, DefaultExitStrategy)
        return strategy_class(config)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """Register a new exit strategy type"""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered exit strategy: {name}")
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """Get list of available exit strategy types"""
        return list(cls._strategies.keys())