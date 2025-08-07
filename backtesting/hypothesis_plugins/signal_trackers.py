from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from utils.logging_utils import get_logger

logger = get_logger()

class SignalTracker(ABC):
    """
    Base class for hypothesis-specific signal tracking.
    
    Signal trackers monitor signal evolution and provide additional context
    for exit decisions and performance analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize signal tracker
        
        Args:
            config: Signal tracking configuration
        """
        self.config = config
        self.tracked_signals = {}
        
        logger.info(f"Initialized {self.__class__.__name__} signal tracker")
    
    @abstractmethod
    def track_entry_signal(self, position_id: str, entry_data: Dict[str, Any], 
                          confidence: float) -> None:
        """
        Track a signal at position entry
        
        Args:
            position_id: Unique position identifier
            entry_data: Market data at entry
            confidence: Signal confidence at entry
        """
        pass
    
    @abstractmethod
    def update_signal_strength(self, position_id: str, current_data: Dict[str, Any],
                             current_confidence: float) -> None:
        """
        Update signal strength for an open position
        
        Args:
            position_id: Position identifier
            current_data: Current market data
            current_confidence: Current signal confidence
        """
        pass
    
    def get_signal_history(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get signal tracking history for a position
        
        Args:
            position_id: Position identifier
            
        Returns:
            Signal history dictionary or None if not found
        """
        return self.tracked_signals.get(position_id)
    
    def cleanup_closed_position(self, position_id: str) -> None:
        """
        Clean up tracking data for closed position
        
        Args:
            position_id: Position identifier
        """
        if position_id in self.tracked_signals:
            del self.tracked_signals[position_id]


class DefaultSignalTracker(SignalTracker):
    """
    Default signal tracker with basic functionality.
    
    Provides minimal tracking suitable for most hypotheses.
    """
    
    def track_entry_signal(self, position_id: str, entry_data: Dict[str, Any], 
                          confidence: float) -> None:
        """Track basic entry signal information"""
        self.tracked_signals[position_id] = {
            'entry_confidence': confidence,
            'entry_timestamp': entry_data.get('timestamp'),
            'confidence_history': [confidence],
            'timestamp_history': [entry_data.get('timestamp')]
        }
        
        logger.debug(f"Tracking entry signal for {position_id}: confidence={confidence:.4f}")
    
    def update_signal_strength(self, position_id: str, current_data: Dict[str, Any],
                             current_confidence: float) -> None:
        """Update basic signal tracking"""
        if position_id in self.tracked_signals:
            tracking = self.tracked_signals[position_id]
            tracking['confidence_history'].append(current_confidence)
            tracking['timestamp_history'].append(current_data.get('timestamp'))
            
            # Keep only recent history (last 50 updates)
            if len(tracking['confidence_history']) > 50:
                tracking['confidence_history'] = tracking['confidence_history'][-50:]
                tracking['timestamp_history'] = tracking['timestamp_history'][-50:]


class SpreadCompressionSignalTracker(SignalTracker):
    """
    Specialized signal tracker for spread compression patterns.
    
    Tracks compression-specific metrics and signal evolution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Spread compression specific parameters
        self.track_compression_features = config.get('track_compression_features', True)
        self.track_volume_patterns = config.get('track_volume_patterns', True)
        
        logger.info("SpreadCompressionSignalTracker configured with compression feature tracking")
    
    def track_entry_signal(self, position_id: str, entry_data: Dict[str, Any], 
                          confidence: float) -> None:
        """Track spread compression entry signal with feature details"""
        tracking_data = {
            'entry_confidence': confidence,
            'entry_timestamp': entry_data.get('timestamp'),
            'confidence_history': [confidence],
            'timestamp_history': [entry_data.get('timestamp')],
        }
        
        # Track compression-specific features
        if self.track_compression_features:
            tracking_data.update({
                'compression_ratio_history': [entry_data.get('compression_ratio_micro', 0)],
                'signal_strength_history': [entry_data.get('signal_strength_v2', 0)],
                'compression_consistency_history': [entry_data.get('compression_consistency', 0)]
            })
        
        # Track volume patterns
        if self.track_volume_patterns:
            tracking_data.update({
                'volume_ratio_history': [entry_data.get('volume_ratio', 0)],
                'volume_ma_history': [entry_data.get('volume_ma_ratio', 0)]
            })
        
        self.tracked_signals[position_id] = tracking_data
        
        logger.debug(f"Tracking spread compression entry for {position_id}: "
                    f"confidence={confidence:.4f}, compression={entry_data.get('compression_ratio_micro', 0):.4f}")
    
    def update_signal_strength(self, position_id: str, current_data: Dict[str, Any],
                             current_confidence: float) -> None:
        """Update spread compression signal tracking with feature evolution"""
        if position_id not in self.tracked_signals:
            return
        
        tracking = self.tracked_signals[position_id]
        
        # Update basic tracking
        tracking['confidence_history'].append(current_confidence)
        tracking['timestamp_history'].append(current_data.get('timestamp'))
        
        # Update compression features
        if self.track_compression_features:
            tracking['compression_ratio_history'].append(current_data.get('compression_ratio_micro', 0))
            tracking['signal_strength_history'].append(current_data.get('signal_strength_v2', 0))
            tracking['compression_consistency_history'].append(current_data.get('compression_consistency', 0))
        
        # Update volume patterns
        if self.track_volume_patterns:
            tracking['volume_ratio_history'].append(current_data.get('volume_ratio', 0))
            tracking['volume_ma_history'].append(current_data.get('volume_ma_ratio', 0))
        
        # Maintain reasonable history size
        max_history = 100
        for key in tracking:
            if isinstance(tracking[key], list) and len(tracking[key]) > max_history:
                tracking[key] = tracking[key][-max_history:]
        
        logger.debug(f"Updated signal tracking for {position_id}: confidence={current_confidence:.4f}")
    
    def get_compression_metrics(self, position_id: str) -> Optional[Dict[str, float]]:
        """
        Get current compression metrics for a position
        
        Args:
            position_id: Position identifier
            
        Returns:
            Dictionary with compression metrics or None
        """
        if position_id not in self.tracked_signals:
            return None
        
        tracking = self.tracked_signals[position_id]
        
        if not self.track_compression_features:
            return None
        
        # Calculate current vs entry metrics
        try:
            entry_compression = tracking['compression_ratio_history'][0] if tracking['compression_ratio_history'] else 0
            current_compression = tracking['compression_ratio_history'][-1] if tracking['compression_ratio_history'] else 0
            
            entry_signal_strength = tracking['signal_strength_history'][0] if tracking['signal_strength_history'] else 0
            current_signal_strength = tracking['signal_strength_history'][-1] if tracking['signal_strength_history'] else 0
            
            return {
                'entry_compression': entry_compression,
                'current_compression': current_compression,
                'compression_change': current_compression - entry_compression,
                'compression_change_pct': ((current_compression - entry_compression) / abs(entry_compression) * 100) if entry_compression != 0 else 0,
                'entry_signal_strength': entry_signal_strength,
                'current_signal_strength': current_signal_strength,
                'signal_strength_change': current_signal_strength - entry_signal_strength
            }
        except (IndexError, KeyError, TypeError):
            return None


class SignalTrackerFactory:
    """Factory for creating hypothesis-specific signal trackers"""
    
    _trackers = {
        'default': DefaultSignalTracker,
        'spread_compression': SpreadCompressionSignalTracker
    }
    
    @classmethod
    def create(cls, hypothesis_name: str, config: Dict[str, Any]) -> SignalTracker:
        """
        Create signal tracker for hypothesis
        
        Args:
            hypothesis_name: Name of hypothesis
            config: Signal tracking configuration
            
        Returns:
            Appropriate SignalTracker instance
        """
        # Map hypothesis names to tracker types
        if 'spread_compression' in hypothesis_name.lower():
            tracker_type = 'spread_compression'
        else:
            tracker_type = 'default'
        
        tracker_class = cls._trackers.get(tracker_type, DefaultSignalTracker)
        return tracker_class(config)
    
    @classmethod
    def register_tracker(cls, name: str, tracker_class: type):
        """Register a new signal tracker type"""
        cls._trackers[name] = tracker_class
        logger.info(f"Registered signal tracker: {name}")
    
    @classmethod
    def get_available_trackers(cls) -> list:
        """Get list of available signal tracker types"""
        return list(cls._trackers.keys())