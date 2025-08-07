from typing import Dict, Any, Tuple
from ..base_plugin import HypothesisPlugin
from ..exit_strategies import SpreadCompressionExitStrategy
from ..signal_trackers import SpreadCompressionSignalTracker
from utils.logging_utils import get_logger

logger = get_logger()

class SpreadCompressionPlugin(HypothesisPlugin):
    """
    Hypothesis plugin for spread compression pattern detection strategies.
    
    This plugin provides specialized exit strategies and signal tracking
    optimized for spread compression patterns in FX markets.
    """
    
    def __init__(self, hypothesis_name: str, config: Dict[str, Any]):
        super().__init__(hypothesis_name, config)
        
        # Extract spread compression specific config
        self.compression_config = config.get('spread_compression', {})
        
        logger.info(f"SpreadCompressionPlugin initialized for: {hypothesis_name}")
    
    def get_exit_strategy(self):
        """Get spread compression specialized exit strategy"""
        return SpreadCompressionExitStrategy(self.exit_config)
    
    def get_signal_tracker(self):
        """Get spread compression specialized signal tracker"""
        return SpreadCompressionSignalTracker(self.signal_tracking_config)
    
    def customize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Customize features for spread compression patterns
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Enhanced feature dictionary with compression-specific metrics
        """
        # Add any spread compression specific feature transformations
        enhanced_features = features.copy()
        
        # Example: Enhance compression ratio with normalized values
        if 'compression_ratio_micro' in features:
            compression_raw = features['compression_ratio_micro']
            enhanced_features['compression_ratio_normalized'] = min(max(compression_raw, -2.0), 2.0)
        
        # Example: Create composite compression signal
        if all(key in features for key in ['compression_ratio_micro', 'volume_ratio', 'signal_strength_v2']):
            composite_signal = (
                features['compression_ratio_micro'] * 0.4 +
                features['volume_ratio'] * 0.3 +
                features['signal_strength_v2'] * 0.3
            )
            enhanced_features['compression_composite_signal'] = composite_signal
        
        return enhanced_features
    
    def validate_configuration(self) -> Tuple[bool, list]:
        """
        Validate spread compression specific configuration
        
        Returns:
            Tuple of (is_valid: bool, error_messages: list)
        """
        # Start with base validation
        is_valid, errors = super().validate_configuration()
        
        if not is_valid:
            return is_valid, errors
        
        # Validate spread compression specific requirements
        backtest_config = self.config.get('backtesting', {})
        
        # Check for compression-specific exit strategy parameters
        exit_config = backtest_config.get('exit_strategy', {})
        
        # Validate exit strategy thresholds
        confidence_threshold = exit_config.get('confidence_exit_threshold')
        if confidence_threshold is not None:
            try:
                confidence_threshold = float(confidence_threshold)
                if not (0.0 <= confidence_threshold <= 1.0):
                    errors.append("exit_strategy.confidence_exit_threshold must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                errors.append("exit_strategy.confidence_exit_threshold must be numeric")
        
        # Validate minimum signal life
        min_life = exit_config.get('minimum_signal_life')
        if min_life is not None:
            try:
                min_life = float(min_life)
                if not (0.1 <= min_life <= 60):  # 0.1 to 60 minutes
                    errors.append("exit_strategy.minimum_signal_life must be between 0.1 and 60 minutes")
            except (ValueError, TypeError):
                errors.append("exit_strategy.minimum_signal_life must be numeric")
        
        # Check for required compression features in data source
        data_source = backtest_config.get('data_source')
        if data_source:
            required_features = [
                'compression_ratio_micro',
                'volume_ratio',
                'signal_strength_v2',
                'compression_consistency'
            ]
            
            # Note: We can't validate actual data file contents here,
            # but we can check if the path exists and add warnings
            import os
            if not os.path.exists(data_source):
                errors.append(f"Data source file not found: {data_source}")
            else:
                # Add info message about required features
                logger.info(f"Spread compression requires features: {required_features}")
        
        # Validate signal processing parameters for compression patterns
        signal_config = backtest_config.get('signal_processing', {})
        
        # Check confidence threshold is appropriate for compression detection
        signal_confidence = signal_config.get('confidence_threshold')
        if signal_confidence is not None:
            try:
                signal_confidence = float(signal_confidence)
                if signal_confidence < 0.3:
                    logger.warning("Low confidence threshold (<0.3) may generate too many false signals for compression patterns")
                elif signal_confidence > 0.8:
                    logger.warning("High confidence threshold (>0.8) may miss valid compression opportunities")
            except (ValueError, TypeError):
                errors.append("signal_processing.confidence_threshold must be numeric")
        
        # Validate risk management for compression trading
        risk_config = backtest_config.get('risk_management', {})
        
        # Check if stop loss is appropriate for compression patterns (typically tighter)
        stop_loss = risk_config.get('stop_loss_pct')
        if stop_loss is not None:
            try:
                stop_loss = float(stop_loss)
                if stop_loss > 0.03:  # 3%
                    logger.warning("Stop loss >3% may be too wide for compression patterns")
                elif stop_loss < 0.005:  # 0.5%
                    logger.warning("Stop loss <0.5% may be too tight and cause premature exits")
            except (ValueError, TypeError):
                errors.append("risk_management.stop_loss_pct must be numeric")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Spread compression configuration validation passed")
        else:
            logger.error(f"Spread compression configuration validation failed: {errors}")
        
        return is_valid, errors
    
    def get_hypothesis_info(self) -> Dict[str, Any]:
        """Get spread compression plugin information"""
        base_info = super().get_hypothesis_info()
        
        base_info.update({
            'specialization': 'spread_compression_patterns',
            'required_features': [
                'compression_ratio_micro',
                'volume_ratio', 
                'signal_strength_v2',
                'compression_consistency'
            ],
            'exit_strategy_type': 'SpreadCompressionExitStrategy',
            'signal_tracker_type': 'SpreadCompressionSignalTracker',
            'supports_feature_enhancement': True,
            'optimal_timeframes': ['1m', '5m'],
            'market_sessions': ['London', 'NY', 'Overlap'],
            'risk_profile': 'medium_high_frequency'
        })
        
        return base_info