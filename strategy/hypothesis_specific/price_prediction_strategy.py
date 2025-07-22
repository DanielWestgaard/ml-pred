import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from strategy.hypothesis_specific.base_strategy import BaseStrategy
from utils.logging_utils import get_logger

logger = get_logger()

class PricePredictionStrategy(BaseStrategy):
    """
    Strategy implementation for price prediction hypothesis.
    Focuses on directional price movement predictions.
    """
    
    def __init__(self, **kwargs):
        """Initialize price prediction strategy"""
        super().__init__(**kwargs)
        
        # Price prediction specific configuration
        self.prediction_config = self._setup_prediction_config()
        
        logger.info("Price prediction strategy initialized")
    
    def _setup_prediction_config(self) -> Dict[str, Any]:
        """Setup price prediction specific configuration"""
        return {
            'min_price_movement': 0.0005,  # Minimum price movement to consider (50 pips for GBPUSD)
            'prediction_horizon': 5,       # Number of periods ahead
            'trend_confirmation_periods': 3,  # Periods to confirm trend
            'volatility_filter': True,
            'momentum_filter': True
        }
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals for price prediction
        
        Args:
            market_data: Current market data
            
        Returns:
            Price prediction signals
        """
        try:
            # Prepare features for prediction
            feature_columns = [col for col in market_data.columns 
                             if not col.startswith('target_') and col not in ['date', 'close_og']]
            
            if not feature_columns:
                logger.error("No valid feature columns found in market data")
                return {'signals': [], 'prediction_info': {}}
            
            X = market_data[feature_columns].iloc[-1:].values
            predictions = self.model.predict(X)
            
            # Convert predictions to signals
            signals = self.signal_processor.process(predictions, market_data)
            
            # Apply price prediction specific logic
            prediction_signals = self._apply_price_prediction_logic(signals, market_data)
            
            return prediction_signals
            
        except Exception as e:
            logger.error(f"Error generating price prediction signals: {e}")
            return {'signals': [], 'prediction_info': {}, 'error': str(e)}
    
    def _apply_price_prediction_logic(self, signals: Dict[str, Any], 
                                    market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply price prediction specific logic to signals
        
        Args:
            signals: Base signals from signal processor
            market_data: Market data for context
            
        Returns:
            Price prediction filtered signals
        """
        prediction_info = {
            'current_price': market_data['close'].iloc[-1] if len(market_data) > 0 else 0,
            'prediction_horizon': self.prediction_config['prediction_horizon'],
            'trend_analysis': {},
            'volatility_analysis': {}
        }
        
        # Analyze current market conditions
        if len(market_data) >= 20:
            prediction_info['trend_analysis'] = self._analyze_trend(market_data)
            prediction_info['volatility_analysis'] = self._analyze_volatility(market_data)
        
        # Filter signals based on price prediction criteria
        filtered_signals = []
        for signal in signals.get('processed_signals', []):
            if self._validate_price_prediction_signal(signal, market_data, prediction_info):
                # Add price prediction specific information
                signal['predicted_price_movement'] = self._calculate_expected_movement(
                    signal, prediction_info['current_price']
                )
                signal['trend_alignment'] = self._check_trend_alignment(
                    signal, prediction_info['trend_analysis']
                )
                filtered_signals.append(signal)
        
        signals['processed_signals'] = filtered_signals
        signals['prediction_info'] = prediction_info
        return signals
    
    def _analyze_trend(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current price trend
        
        Args:
            market_data: Market data
            
        Returns:
            Trend analysis results
        """
        prices = market_data['close']
        
        # Short-term trend (5 periods)
        short_trend = prices.iloc[-5:].pct_change().mean() if len(prices) >= 5 else 0
        
        # Medium-term trend (20 periods)
        medium_trend = prices.iloc[-20:].pct_change().mean() if len(prices) >= 20 else 0
        
        # Trend strength
        trend_strength = abs(medium_trend) * 1000  # Convert to basis points
        
        # Trend direction
        if medium_trend > 0.0001:  # 1 basis point threshold
            trend_direction = 'bullish'
        elif medium_trend < -0.0001:
            trend_direction = 'bearish'
        else:
            trend_direction = 'neutral'
        
        return {
            'short_term_trend': short_trend,
            'medium_term_trend': medium_trend,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'is_trending': trend_strength > 5  # 5 basis points threshold
        }
    
    def _analyze_volatility(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current market volatility
        
        Args:
            market_data: Market data
            
        Returns:
            Volatility analysis results
        """
        prices = market_data['close']
        
        # Current volatility (20-period rolling std)
        current_vol = prices.rolling(20).std().iloc[-1] if len(prices) >= 20 else 0
        
        # Average volatility (100-period rolling std mean)
        avg_vol = prices.rolling(100).std().mean() if len(prices) >= 100 else current_vol
        
        # Volatility regime
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        if vol_ratio > 1.5:
            vol_regime = 'high'
        elif vol_ratio < 0.7:
            vol_regime = 'low'
        else:
            vol_regime = 'normal'
        
        return {
            'current_volatility': current_vol,
            'average_volatility': avg_vol,
            'volatility_ratio': vol_ratio,
            'volatility_regime': vol_regime,
            'is_volatile': vol_ratio > 1.2
        }
    
    def _validate_price_prediction_signal(self, signal: Dict[str, Any], 
                                        market_data: pd.DataFrame,
                                        prediction_info: Dict[str, Any]) -> bool:
        """
        Validate signal against price prediction criteria
        
        Args:
            signal: Trading signal
            market_data: Market data
            prediction_info: Prediction analysis info
            
        Returns:
            True if signal passes validation
        """
        # Check minimum predicted price movement
        expected_movement = self._calculate_expected_movement(
            signal, prediction_info['current_price']
        )
        
        if abs(expected_movement) < self.prediction_config['min_price_movement']:
            logger.debug("Signal filtered out: insufficient predicted price movement")
            return False
        
        # Apply volatility filter
        if self.prediction_config['volatility_filter']:
            vol_analysis = prediction_info.get('volatility_analysis', {})
            if vol_analysis.get('volatility_regime') == 'high':
                # Require higher confidence in high volatility
                if signal['confidence'] < 0.8:
                    logger.debug("Signal filtered out: insufficient confidence in high volatility")
                    return False
        
        # Apply momentum filter
        if self.prediction_config['momentum_filter']:
            trend_analysis = prediction_info.get('trend_analysis', {})
            if not trend_analysis.get('is_trending', False):
                # Require higher confidence in non-trending markets
                if signal['confidence'] < 0.75:
                    logger.debug("Signal filtered out: insufficient confidence in non-trending market")
                    return False
        
        return True
    
    def _calculate_expected_movement(self, signal: Dict[str, Any], current_price: float) -> float:
        """
        Calculate expected price movement based on signal
        
        Args:
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            Expected price movement
        """
        # Simple calculation: signal strength * typical movement
        # This would be enhanced with actual model output interpretation
        typical_movement = current_price * 0.001  # 0.1% typical movement
        
        direction_multiplier = 1 if signal['direction'] == 'BUY' else -1
        expected_movement = typical_movement * signal['strength'] * direction_multiplier
        
        return expected_movement
    
    def _check_trend_alignment(self, signal: Dict[str, Any], 
                             trend_analysis: Dict[str, Any]) -> bool:
        """
        Check if signal aligns with current trend
        
        Args:
            signal: Trading signal
            trend_analysis: Trend analysis results
            
        Returns:
            True if signal aligns with trend
        """
        signal_direction = signal['direction']
        trend_direction = trend_analysis.get('trend_direction', 'neutral')
        
        if trend_direction == 'neutral':
            return True  # No trend bias
        
        if signal_direction == 'BUY' and trend_direction == 'bullish':
            return True
        
        if signal_direction == 'SELL' and trend_direction == 'bearish':
            return True
        
        return False
    
    def execute_strategy(self) -> Dict[str, Any]:
        """
        Execute the complete price prediction strategy
        
        Returns:
            Strategy execution results
        """
        strategy_results = {
            'strategy_type': 'price_prediction',
            'execution_time': datetime.now(),
            'signals_generated': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'errors': [],
            'prediction_analysis': {}
        }
        
        try:
            logger.info("Price prediction strategy execution started")
            
            # Placeholder execution logic
            # In a real implementation, this would:
            # 1. Get live market data
            # 2. Generate price prediction signals
            # 3. Validate against trend and volatility
            # 4. Execute trades with proper risk management
            # 5. Monitor positions and adjust as needed
            
            strategy_results.update({
                'status': 'executed',
                'message': 'Price prediction strategy executed successfully (placeholder)',
                'recommendations': [
                    'Implement live price feed integration',
                    'Add prediction confidence calibration',
                    'Enhance trend analysis algorithms'
                ]
            })
            
            logger.info("Price prediction strategy execution completed")
            
        except Exception as e:
            strategy_results['errors'].append(str(e))
            strategy_results['status'] = 'failed'
            logger.error(f"Price prediction strategy execution failed: {e}")
        
        return strategy_results
    
    def get_prediction_accuracy_analysis(self) -> Dict[str, Any]:
        """
        Analyze prediction accuracy over time
        
        Returns:
            Prediction accuracy metrics
        """
        # Get closed positions from position manager
        closed_positions = self.position_manager.closed_positions
        
        if not closed_positions:
            return {'message': 'No closed positions for prediction accuracy analysis'}
        
        # Analyze prediction vs actual outcomes
        correct_predictions = 0
        total_predictions = len(closed_positions)
        
        direction_accuracy = {'BUY': {'correct': 0, 'total': 0}, 
                            'SELL': {'correct': 0, 'total': 0}}
        
        for position in closed_positions:
            direction = position['direction']
            entry_price = position['entry_price']
            close_price = position['close_price']
            
            # Check if prediction was correct
            if direction == 'BUY':
                prediction_correct = close_price > entry_price
            else:  # SELL
                prediction_correct = close_price < entry_price
            
            if prediction_correct:
                correct_predictions += 1
                direction_accuracy[direction]['correct'] += 1
            
            direction_accuracy[direction]['total'] += 1
        
        # Calculate accuracy metrics
        overall_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        buy_accuracy = (direction_accuracy['BUY']['correct'] / direction_accuracy['BUY']['total'] * 100) \
                      if direction_accuracy['BUY']['total'] > 0 else 0
        
        sell_accuracy = (direction_accuracy['SELL']['correct'] / direction_accuracy['SELL']['total'] * 100) \
                       if direction_accuracy['SELL']['total'] > 0 else 0
        
        return {
            'overall_accuracy_pct': overall_accuracy,
            'buy_accuracy_pct': buy_accuracy,
            'sell_accuracy_pct': sell_accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'direction_breakdown': direction_accuracy
        }