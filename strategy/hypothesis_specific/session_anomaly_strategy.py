import logging
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Any
from strategy.hypothesis_specific.base_strategy import BaseStrategy

class SessionAnomalyStrategy(BaseStrategy):
    """
    Strategy implementation for session-based volume anomaly detection hypothesis.
    Focuses on trading during specific market sessions with anomaly detection.
    """
    
    def __init__(self, **kwargs):
        """Initialize session anomaly strategy"""
        super().__init__(**kwargs)
        
        # Session-specific configuration
        self.trading_sessions = self._setup_trading_sessions()
        self.session_filters = self._setup_session_filters()
        
        logging.info("Session anomaly strategy initialized")
    
    def _setup_trading_sessions(self) -> Dict[str, Any]:
        """Setup trading session definitions"""
        return {
            'Asian': {
                'start_time': time(0, 0),   # 00:00 UTC
                'end_time': time(9, 0),     # 09:00 UTC
                'volatility_multiplier': 0.8,  # Lower volatility session
                'min_confidence': 0.7
            },
            'European': {
                'start_time': time(7, 0),   # 07:00 UTC
                'end_time': time(16, 0),    # 16:00 UTC
                'volatility_multiplier': 1.2,  # Higher volatility session
                'min_confidence': 0.6
            },
            'US': {
                'start_time': time(13, 0),  # 13:00 UTC
                'end_time': time(22, 0),    # 22:00 UTC
                'volatility_multiplier': 1.1,  # High volatility session
                'min_confidence': 0.6
            },
            'Overlap_EU_US': {
                'start_time': time(13, 0),  # 13:00 UTC
                'end_time': time(16, 0),    # 16:00 UTC
                'volatility_multiplier': 1.5,  # Highest volatility
                'min_confidence': 0.5
            }
        }
    
    def _setup_session_filters(self) -> Dict[str, Any]:
        """Setup session-specific filters"""
        return {
            'preferred_sessions': ['European', 'US', 'Overlap_EU_US'],
            'avoid_sessions': [],
            'volume_threshold_multiplier': 1.2,  # Require 20% above average volume
            'anomaly_strength_threshold': 0.6
        }
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals specific to session anomaly detection
        
        Args:
            market_data: Current market data 
            
        Returns:
            Session-specific signals
        """
        # Make predictions using the loaded model
        # Assume market_data has the same features as training data
        try:
            # Prepare features for prediction (exclude target columns and non-feature columns)
            feature_columns = [col for col in market_data.columns 
                             if not col.startswith('target_') and col not in ['date', 'close_og']]
            
            if not feature_columns:
                logging.error("No valid feature columns found in market data")
                return {'signals': [], 'session_info': {}}
            
            X = market_data[feature_columns].iloc[-1:].values  # Get latest row
            predictions = self.model.predict(X)
            
            # Convert predictions to signals using signal processor
            signals = self.signal_processor.process(predictions, market_data)
            
            # Apply session-specific logic
            session_signals = self._apply_session_logic(signals, market_data)
            
            return session_signals
            
        except Exception as e:
            logging.error(f"Error generating signals: {e}")
            return {'signals': [], 'session_info': {}, 'error': str(e)}
    
    def _apply_session_logic(self, signals: Dict[str, Any], 
                           market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply session-specific logic to signals
        
        Args:
            signals: Base signals from signal processor
            market_data: Market data for context
            
        Returns:
            Session-filtered signals
        """
        current_time = datetime.now().time()
        current_session = self._get_current_session(current_time)
        
        session_info = {
            'current_session': current_session,
            'session_config': self.trading_sessions.get(current_session, {}),
            'is_preferred_session': current_session in self.session_filters['preferred_sessions'],
            'should_trade': False
        }
        
        # Check if we should trade in current session
        if current_session and current_session not in self.session_filters['avoid_sessions']:
            session_config = self.trading_sessions[current_session]
            
            # Apply session-specific confidence threshold
            min_confidence = session_config.get('min_confidence', 0.6)
            
            # Filter signals by session confidence
            filtered_signals = []
            for signal in signals.get('processed_signals', []):
                if signal['confidence'] >= min_confidence:
                    # Adjust signal strength by session volatility multiplier
                    adjusted_strength = signal['strength'] * session_config.get('volatility_multiplier', 1.0)
                    signal['adjusted_strength'] = min(1.0, adjusted_strength)  # Cap at 1.0
                    signal['session'] = current_session
                    
                    # Apply additional session filters
                    if self._validate_session_signal(signal, market_data, session_config):
                        filtered_signals.append(signal)
            
            session_info['should_trade'] = len(filtered_signals) > 0
            signals['processed_signals'] = filtered_signals
        else:
            # No trading outside preferred sessions
            signals['processed_signals'] = []
            session_info['should_trade'] = False
        
        signals['session_info'] = session_info
        return signals
    
    def _get_current_session(self, current_time: time) -> str:
        """
        Determine current trading session
        
        Args:
            current_time: Current time
            
        Returns:
            Session name or None
        """
        for session_name, session_config in self.trading_sessions.items():
            start_time = session_config['start_time']
            end_time = session_config['end_time']
            
            # Handle sessions that cross midnight
            if start_time <= end_time:
                if start_time <= current_time <= end_time:
                    return session_name
            else:
                if current_time >= start_time or current_time <= end_time:
                    return session_name
        
        return None
    
    def _validate_session_signal(self, signal: Dict[str, Any], market_data: pd.DataFrame,
                                session_config: Dict[str, Any]) -> bool:
        """
        Validate signal against session-specific criteria
        
        Args:
            signal: Trading signal
            market_data: Market data
            session_config: Session configuration
            
        Returns:
            True if signal passes validation
        """
        # Check volume requirements
        if len(market_data) >= 20:
            recent_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            volume_multiplier = self.session_filters['volume_threshold_multiplier']
            
            if recent_volume < avg_volume * volume_multiplier:
                logging.debug("Signal filtered out: insufficient volume")
                return False
        
        # Check anomaly strength
        anomaly_threshold = self.session_filters['anomaly_strength_threshold']
        if signal.get('adjusted_strength', signal['strength']) < anomaly_threshold:
            logging.debug("Signal filtered out: insufficient anomaly strength")
            return False
        
        return True
    
    def execute_strategy(self) -> Dict[str, Any]:
        """
        Execute the complete session anomaly strategy
        
        Returns:
            Strategy execution results
        """
        strategy_results = {
            'strategy_type': 'session_anomaly',
            'execution_time': datetime.now(),
            'signals_generated': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'errors': [],
            'session_analysis': {}
        }
        
        try:
            # Get current market data (this would typically come from a data feed)
            # For now, we'll simulate with placeholder logic
            logging.info("Session anomaly strategy execution started")
            
            # In a real implementation, this would:
            # 1. Get live market data
              # 1.5. Feed live data to generate features (the ones model is trained on)? - comment added by me
            # 2. Generate signals
            # 3. Check risk management
            # 4. Execute trades
            # 5. Monitor positions
            
            # Placeholder execution logic
            strategy_results.update({
                'status': 'executed',
                'message': 'Session anomaly strategy executed successfully (placeholder)',
                'recommendations': [
                    'Integrate with live data feed',
                    'Implement position monitoring loop',
                    'Add session transition handling'
                ]
            })
            
            logging.info("Session anomaly strategy execution completed")
            
        except Exception as e:
            strategy_results['errors'].append(str(e))
            strategy_results['status'] = 'failed'
            logging.error(f"Strategy execution failed: {e}")
        
        return strategy_results
    
    def get_session_performance_analysis(self) -> Dict[str, Any]:
        """
        Analyze performance by trading session
        
        Returns:
            Session-based performance metrics
        """
        # Get closed positions from position manager
        closed_positions = self.position_manager.closed_positions
        
        if not closed_positions:
            return {'message': 'No closed positions for session analysis'}
        
        # Group positions by session
        session_performance = {}
        for position in closed_positions:
            session = position.get('session', 'unknown')
            
            if session not in session_performance:
                session_performance[session] = {
                    'trades': 0,
                    'total_pnl': 0.0,
                    'winning_trades': 0,
                    'losing_trades': 0
                }
            
            session_performance[session]['trades'] += 1
            session_performance[session]['total_pnl'] += position['final_pnl']
            
            if position['final_pnl'] > 0:
                session_performance[session]['winning_trades'] += 1
            else:
                session_performance[session]['losing_trades'] += 1
        
        # Calculate session metrics
        for session, metrics in session_performance.items():
            if metrics['trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['trades'] * 100
                metrics['avg_pnl'] = metrics['total_pnl'] / metrics['trades']
            else:
                metrics['win_rate'] = 0
                metrics['avg_pnl'] = 0
        
        return {
            'session_performance': session_performance,
            'total_sessions_traded': len(session_performance),
            'best_session': max(session_performance.items(), 
                              key=lambda x: x[1]['total_pnl'])[0] if session_performance else None
        }