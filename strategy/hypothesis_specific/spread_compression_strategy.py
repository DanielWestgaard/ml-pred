"""
Spread Compression Strategy
===========================

Intelligent exit strategy specifically designed for spread compression patterns.
Implements confidence-based exits, signal strength decay detection, and volatility-adaptive logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy
from utils.logging_utils import get_logger

logger = get_logger()

class SpreadCompressionSignalTracker:
    """Track signal evolution for intelligent exits specific to spread compression patterns."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.signal_history = {}
        self.entry_signals = {}
        
        # Configuration parameters
        self.signal_strength_features = self.config.get('signal_strength_features', [
            'compression_ratio_micro', 'volume_ratio', 'signal_strength_v2', 'compression_consistency'
        ])
        
    def track_entry_signal(self, position_id: str, features: Dict, model_confidence: float):
        """Track signal characteristics at entry."""
        self.entry_signals[position_id] = {
            'entry_time': datetime.now(),
            'entry_confidence': model_confidence,
            'entry_features': {feat: features.get(feat, 0) for feat in self.signal_strength_features},
            'peak_signal_strength': model_confidence,
            'signal_history': []
        }
        logger.debug(f"Tracking entry signal for {position_id}: confidence={model_confidence:.4f}")
    
    def update_signal_strength(self, position_id: str, features: Dict, model_confidence: float):
        """Update current signal strength for a position."""
        if position_id not in self.entry_signals:
            return
        
        entry_data = self.entry_signals[position_id]
        current_time = datetime.now()
        
        # Calculate signal decay metrics
        current_features = {feat: features.get(feat, 0) for feat in self.signal_strength_features}
        
        # Update peak signal strength
        if model_confidence > entry_data['peak_signal_strength']:
            entry_data['peak_signal_strength'] = model_confidence
        
        # Add to history
        entry_data['signal_history'].append({
            'time': current_time,
            'confidence': model_confidence,
            'features': current_features
        })
        
        # Keep only recent history (last 10 minutes)
        cutoff_time = current_time - timedelta(minutes=10)
        entry_data['signal_history'] = [
            h for h in entry_data['signal_history'] if h['time'] > cutoff_time
        ]
    
    def calculate_signal_decay(self, position_id: str) -> float:
        """Calculate how much the signal has decayed from peak."""
        if position_id not in self.entry_signals:
            return 0.0
        
        entry_data = self.entry_signals[position_id]
        if not entry_data['signal_history']:
            return 0.0
        
        peak_strength = entry_data['peak_signal_strength']
        current_strength = entry_data['signal_history'][-1]['confidence']
        
        if peak_strength == 0:
            return 0.0
        
        decay = (peak_strength - current_strength) / peak_strength
        return max(0.0, decay)
    
    def cleanup_position(self, position_id: str):
        """Clean up tracking data when position is closed."""
        if position_id in self.entry_signals:
            del self.entry_signals[position_id]

class SpreadCompressionStrategy(BaseStrategy):
    """
    Spread compression specific trading strategy with intelligent exit logic.
    """
    
    def __init__(self, model_directory: str = None, save_results: bool = True, 
                 model_type: str = None, hypothesis_config: str = None):
        super().__init__(model_directory, save_results, model_type, hypothesis_config)
        
        # Initialize spread compression specific components
        self.signal_tracker = SpreadCompressionSignalTracker(
            self.strategy_config.get('signal_tracking', {})
        )
        
        # Exit strategy configuration
        self.exit_config = self.strategy_config.get('exit_strategy', {})
        self.confidence_exit_threshold = self.exit_config.get('confidence_exit_threshold', 0.30)
        self.signal_decay_threshold = self.exit_config.get('signal_strength_decay', 0.25)
        self.volatility_exit_multiplier = self.exit_config.get('volatility_exit_multiplier', 1.5)
        self.minimum_signal_life = self.exit_config.get('minimum_signal_life', 3)  # minutes
        
        logger.info(f"SpreadCompressionStrategy initialized with intelligent exits:")
        logger.info(f"  - Confidence exit threshold: {self.confidence_exit_threshold}")
        logger.info(f"  - Signal decay threshold: {self.signal_decay_threshold}")
        logger.info(f"  - Minimum signal life: {self.minimum_signal_life} minutes")
    
    def should_exit_position(self, position: Dict, current_data: Dict, model_prediction: Dict) -> Tuple[bool, str]:
        """
        Intelligent exit logic specific to spread compression patterns.
        
        Args:
            position: Position information including entry details
            current_data: Current market data and features
            model_prediction: Current model prediction with confidence
            
        Returns:
            Tuple of (should_exit: bool, exit_reason: str)
        """
        position_id = position.get('id', 'unknown')
        current_confidence = model_prediction.get('confidence', 0.0)
        current_prediction = model_prediction.get('prediction', 0)
        
        # Update signal tracking
        self.signal_tracker.update_signal_strength(position_id, current_data, current_confidence)
        
        # Check minimum signal life (not a hard constraint, just a filter)
        entry_time = position.get('entry_time')
        if entry_time:
            position_age_minutes = (datetime.now() - entry_time).total_seconds() / 60
            if position_age_minutes < self.minimum_signal_life:
                # Only exit on very strong contrary signals during minimum life period
                if current_prediction == 0 and current_confidence > 0.9:
                    return True, "strong_contrary_signal"
                return False, None
        
        # 1. Confidence-based exit: DISABLED FOR TESTING - model makes too confident predictions
        # if current_prediction == 0 and current_confidence > self.confidence_exit_threshold:
        #     logger.debug(f"Confidence exit triggered: {current_confidence:.4f} > {self.confidence_exit_threshold}")
        #     return True, "confidence_drop"
        
        # 2. Signal strength decay: Exit when signal deteriorates significantly
        signal_decay = self.signal_tracker.calculate_signal_decay(position_id)
        if signal_decay > self.signal_decay_threshold:
            logger.debug(f"Signal decay exit triggered: {signal_decay:.4f} > {self.signal_decay_threshold}")
            return True, "signal_decay"
        
        # 3. Volatility-based exit: Exit when volatility spikes (spread compression ending)
        if self._check_volatility_exit(position, current_data):
            return True, "volatility_spike"
        
        # 4. Feature-based exit: Exit when core compression features reverse
        if self._check_compression_reversal(position, current_data):
            return True, "compression_reversal"
        
        return False, None
    
    def _check_volatility_exit(self, position: Dict, current_data: Dict) -> bool:
        """Check if volatility has spiked beyond acceptable levels."""
        try:
            entry_volatility = position.get('entry_volatility', 0)
            current_volatility = current_data.get('volatility_short', 0)
            
            if entry_volatility > 0 and current_volatility > entry_volatility * self.volatility_exit_multiplier:
                logger.debug(f"Volatility exit: {current_volatility:.6f} > {entry_volatility * self.volatility_exit_multiplier:.6f}")
                return True
        except (KeyError, TypeError, ValueError):
            pass
        
        return False
    
    def _check_compression_reversal(self, position: Dict, current_data: Dict) -> bool:
        """Check if core compression features have reversed significantly."""
        try:
            # Check if compression ratio has deteriorated
            current_compression = current_data.get('compression_ratio_micro', 0)
            entry_compression = position.get('entry_features', {}).get('compression_ratio_micro', 0)
            
            # Check if volume confirmation has disappeared
            current_volume_ratio = current_data.get('volume_ratio', 0)
            entry_volume_ratio = position.get('entry_features', {}).get('volume_ratio', 0)
            
            # Exit if compression has reversed by more than 15%
            if entry_compression != 0:
                compression_change = (current_compression - entry_compression) / abs(entry_compression)
                if compression_change < -0.15:  # 15% deterioration
                    logger.debug(f"Compression reversal: {compression_change:.3f}")
                    return True
            
            # Exit if volume confirmation has significantly weakened
            if entry_volume_ratio != 0:
                volume_change = (current_volume_ratio - entry_volume_ratio) / abs(entry_volume_ratio)
                if volume_change < -0.3:  # 30% deterioration in volume confirmation
                    logger.debug(f"Volume confirmation loss: {volume_change:.3f}")
                    return True
            
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        
        return False
    
    def on_position_entry(self, position: Dict, market_data: Dict, model_prediction: Dict):
        """Called when a new position is entered - track entry signals."""
        position_id = position.get('id')
        if position_id:
            # Store entry features for comparison
            position['entry_features'] = {
                'compression_ratio_micro': market_data.get('compression_ratio_micro', 0),
                'volume_ratio': market_data.get('volume_ratio', 0),
                'signal_strength_v2': market_data.get('signal_strength_v2', 0),
                'compression_consistency': market_data.get('compression_consistency', 0)
            }
            position['entry_volatility'] = market_data.get('volatility_short', 0)
            position['entry_time'] = datetime.now()
            
            # Track with signal tracker
            self.signal_tracker.track_entry_signal(
                position_id, market_data, model_prediction.get('confidence', 0)
            )
            
            logger.info(f"Position {position_id} entered with compression features tracked")
    
    def on_position_exit(self, position: Dict, exit_reason: str):
        """Called when a position is exited - cleanup tracking."""
        position_id = position.get('id')
        if position_id:
            self.signal_tracker.cleanup_position(position_id)
            logger.info(f"Position {position_id} exited: {exit_reason}")
    
    def generate_signals(self, market_data):
        """Generate trading signals from market data."""
        # Use the base signal processor but with spread compression specific logic
        signals = self.signal_processor.process_signals(
            model=self.model,
            market_data=market_data
        )
        
        return signals
    
    def execute_strategy(self):
        """Execute the complete strategy logic."""
        # This would be implemented based on your specific strategy execution needs
        # For now, return a placeholder result
        return {
            'strategy_type': 'spread_compression',
            'exit_strategy': 'intelligent',
            'executed_at': datetime.now().isoformat()
        }