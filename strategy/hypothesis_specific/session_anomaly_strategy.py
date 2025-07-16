import logging
import pandas as pd
import time
from datetime import datetime, time as dt_time
from typing import Dict, List, Any
from strategy.hypothesis_specific.base_strategy import BaseStrategy
from data.live.live_data_manager import LiveDataManager
from strategy.monitoring.position_monitor import PositionMonitor

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
        
        # Live data manager
        self.live_data_manager = None
        self._initialize_live_data()
        
        # Position monitor
        self.position_monitor = None
        self._initialize_position_monitor()
        
        logging.info("Session anomaly strategy initialized")
    
    def _initialize_live_data(self):
        """Initialize live data manager with strategy configuration"""
        try:
            # Get data configuration from strategy config
            data_config = self.strategy_config.get('data', {})
            
            self.live_data_manager = LiveDataManager(
                symbol=data_config.get('symbol', 'GBPUSD'),
                timeframe=data_config.get('timeframe', 'MINUTE'),
                buffer_size=data_config.get('buffer_size', 500),
                feature_window=data_config.get('feature_window', 200),
                selected_features=data_config.get('selected_features', None),
                use_websocket=data_config.get('use_websocket', True)
            )
            
            # Add callback for new data notifications
            self.live_data_manager.add_data_callback(self._on_new_data)
            
            logging.info("Live data manager initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize live data manager: {e}")
            self.live_data_manager = None
    
    def _on_new_data(self, latest_features: pd.DataFrame):
        """Callback for when new processed data is available"""
        try:
            # This could trigger signal generation in real-time
            logging.debug("New market data received and processed")
            
            # In a more advanced implementation, this could:
            # 1. Generate signals immediately
            # 2. Check for position updates
            # 3. Trigger risk management checks
            
        except Exception as e:
            logging.error(f"Error processing new data callback: {e}")
    
    def _initialize_position_monitor(self):
        """Initialize position monitoring system"""
        try:
            # Get monitoring configuration
            monitoring_config = self.strategy_config.get('monitoring', {})
            
            self.position_monitor = PositionMonitor(
                update_interval=monitoring_config.get('update_interval', 1.0)
            )
            
            # Add callbacks for position events
            self.position_monitor.add_position_callback(self._on_position_event)
            self.position_monitor.add_risk_callback(self._on_risk_event)
            
            logging.info("Position monitor initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize position monitor: {e}")
            self.position_monitor = None
    
    def _on_position_event(self, event_type: str, position_data: Dict, extra_data: Dict = None):
        """Handle position events from monitor"""
        try:
            logging.info(f"Position event: {event_type} for {position_data.get('position_id')}")
            
            if event_type == 'close_requested':
                # Handle automatic position closing
                position_id = position_data['position_id']
                reason = extra_data.get('reason', 'unknown')
                
                # Execute close order via order manager
                close_order = {
                    'action': 'close',
                    'position_id': position_id,
                    'reason': reason
                }
                
                if self.order_manager.close_position(close_order):
                    logging.info(f"Successfully closed position {position_id} - Reason: {reason}")
                else:
                    logging.error(f"Failed to close position {position_id}")
            
        except Exception as e:
            logging.error(f"Error handling position event: {e}")
    
    def _on_risk_event(self, event_type: str, data: Dict):
        """Handle risk events from monitor"""
        try:
            logging.warning(f"Risk event: {event_type} - Data: {data}")
            
            if event_type == 'max_drawdown':
                # Stop trading if max drawdown reached
                logging.critical("Maximum drawdown reached - halting new positions")
                # Could implement emergency stop here
                
            elif event_type == 'high_exposure':
                # Reduce position sizes if exposure too high
                logging.warning("High exposure detected - consider reducing position sizes")
                
            elif event_type == 'too_many_positions':
                # Stop opening new positions if too many active
                logging.warning("Too many active positions - pausing new entries")
            
        except Exception as e:
            logging.error(f"Error handling risk event: {e}")
    
    def _setup_trading_sessions(self) -> Dict[str, Any]:
        """Setup trading session definitions"""
        return {
            'Asian': {
                'start_time': dt_time(0, 0),   # 00:00 UTC
                'end_time': dt_time(9, 0),     # 09:00 UTC
                'volatility_multiplier': 0.8,  # Lower volatility session
                'min_confidence': 0.7
            },
            'European': {
                'start_time': dt_time(7, 0),   # 07:00 UTC
                'end_time': dt_time(16, 0),    # 16:00 UTC
                'volatility_multiplier': 1.2,  # Higher volatility session
                'min_confidence': 0.6
            },
            'US': {
                'start_time': dt_time(13, 0),  # 13:00 UTC
                'end_time': dt_time(22, 0),    # 22:00 UTC
                'volatility_multiplier': 1.1,  # High volatility session
                'min_confidence': 0.6
            },
            'Overlap_EU_US': {
                'start_time': dt_time(13, 0),  # 13:00 UTC
                'end_time': dt_time(16, 0),    # 16:00 UTC
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
            # Prepare features for prediction using model-aware feature selection
            # Load the model features directly from metadata file
            from utils.model.model_metadata_loader import load_model_metadata
            loader = load_model_metadata(self.model_directory)
            expected_features = loader.get_features()
            feature_columns = [col for col in expected_features if col in market_data.columns]
            
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
    
    def _get_current_session(self, current_time: dt_time) -> str:
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
        Execute the complete session anomaly strategy with live data integration
        
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
            'session_analysis': {},
            'status': 'starting'
        }
        
        try:
            logging.info(f"{'='*10} Session anomaly strategy execution started {'='*10}")
            
            # 1. Initialize and start live data feed
            if not self._start_live_data_feed():
                strategy_results['errors'].append("Failed to start live data feed")
                strategy_results['status'] = 'failed'
                return strategy_results
            
            # 2. Start position monitoring
            if not self._start_position_monitoring():
                strategy_results['errors'].append("Failed to start position monitoring")
                strategy_results['status'] = 'failed'
                return strategy_results
            
            # 3. Main trading loop
            strategy_results = self._run_trading_loop(strategy_results)
            
            logging.info("Session anomaly strategy execution completed")
            
        except Exception as e:
            strategy_results['errors'].append(str(e))
            strategy_results['status'] = 'failed'
            logging.error(f"Strategy execution failed: {e}")
        finally:
            # Always cleanup
            self._cleanup_resources()
        
        return strategy_results
    
    def _start_live_data_feed(self) -> bool:
        """Start live data feed and wait for initial data"""
        if not self.live_data_manager:
            logging.error("Live data manager not initialized")
            return False
        
        try:
            # Start streaming with jumpstart for faster startup
            dry_run = self.strategy_config.get('data', {}).get('dry_run', True)
            jumpstart = self.strategy_config.get('data', {}).get('jumpstart', True)
            
            if not self.live_data_manager.start_streaming(dry_run=dry_run, jumpstart=jumpstart):
                logging.error("Failed to start data streaming")
                return False
            
            # Wait for initial data buffer to fill
            logging.info("Waiting for initial data buffer to fill...")
            max_wait_time = 60  # seconds
            start_time = datetime.now()
            
            while not self.live_data_manager.is_ready_for_trading():
                if (datetime.now() - start_time).total_seconds() > max_wait_time:
                    logging.error("Timeout waiting for initial data")
                    return False
                
                status = self.live_data_manager.get_buffer_status()
                logging.info(f"Buffer status: {status['buffer_size']}/{status['max_buffer_size']} "
                           f"({status['buffer_utilization']:.1%})")
                
                time.sleep(2)
            
            logging.info("Live data feed ready for trading")
            return True
            
        except Exception as e:
            logging.error(f"Error starting live data feed: {e}")
            return False
    
    def _start_position_monitoring(self) -> bool:
        """Start position monitoring system"""
        if not self.position_monitor:
            logging.error("Position monitor not initialized")
            return False
        
        try:
            if not self.position_monitor.start_monitoring():
                logging.error("Failed to start position monitoring")
                return False
            
            logging.info("Position monitoring started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting position monitoring: {e}")
            return False
    
    def _run_trading_loop(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Main trading loop with live data processing"""
        loop_count = 0
        max_loops = self.strategy_config.get('execution', {}).get('max_loops', 100)
        loop_interval = self.strategy_config.get('execution', {}).get('loop_interval', 5)
        
        strategy_results['status'] = 'running'
        
        while loop_count < max_loops and self.live_data_manager.is_ready_for_trading():
            try:
                loop_start = datetime.now()
                logging.debug(f"Trading loop iteration {loop_count + 1}")
                
                # 1. Get latest processed market data
                latest_features = self.live_data_manager.get_latest_features()
                if latest_features is None:
                    logging.warning("No processed data available, skipping iteration")
                    time.sleep(loop_interval)
                    continue
                
                # 2. Generate trading signals
                signals = self.generate_signals(latest_features)
                strategy_results['signals_generated'] += len(signals.get('processed_signals', []))
                
                # 3. Process signals and manage positions
                if signals.get('session_info', {}).get('should_trade', False):
                    position_actions = self._process_trading_signals(signals)
                    strategy_results['positions_opened'] += position_actions.get('opened', 0)
                    strategy_results['positions_closed'] += position_actions.get('closed', 0)
                
                # 4. Monitor existing positions
                self._monitor_positions()
                
                # 5. Update session analysis
                current_session = signals.get('session_info', {}).get('current_session')
                if current_session:
                    if current_session not in strategy_results['session_analysis']:
                        strategy_results['session_analysis'][current_session] = {
                            'signals': 0, 'trades': 0
                        }
                    strategy_results['session_analysis'][current_session]['signals'] += len(
                        signals.get('processed_signals', []))
                
                # 6. Rate limiting
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, loop_interval - loop_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                loop_count += 1
                
            except Exception as e:
                strategy_results['errors'].append(f"Trading loop error: {str(e)}")
                logging.error(f"Error in trading loop iteration {loop_count}: {e}")
                time.sleep(loop_interval)
                loop_count += 1
        
        strategy_results['status'] = 'completed'
        strategy_results['total_loops'] = loop_count
        logging.info(f"Trading loop completed after {loop_count} iterations")
        
        return strategy_results
    
    def _process_trading_signals(self, signals: Dict[str, Any]) -> Dict[str, int]:
        """Process trading signals and execute orders"""
        position_actions = {'opened': 0, 'closed': 0}
        
        try:
            processed_signals = signals.get('processed_signals', [])
            
            for signal in processed_signals:
                # Check risk management
                if not self.risk_manager.can_open_position(signal):
                    logging.debug("Signal rejected by risk management")
                    continue
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    signal_strength=signal.get('confidence', 0.5),
                    account_balance=self._get_account_balance(),
                    current_price=signal.get('current_price', 1.0)
                )
                
                # Extract order parameters
                symbol = signal.get('symbol', 'GBPUSD')
                direction = signal.get('direction', 'BUY')
                size_usd = position_size.get('position_size_usd', 0)
                current_price = signal.get('current_price', 1.0)
                stop_loss = self.risk_manager.calculate_stop_loss(signal)
                take_profit = self.risk_manager.calculate_take_profit(signal)
                
                # Create order details for logging
                order_details = {
                    'symbol': symbol,
                    'direction': direction,
                    'size_usd': size_usd,
                    'current_price': current_price,
                    'signal_confidence': signal.get('confidence'),
                    'session': signal.get('session'),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
                # Execute order with correct arguments
                order_result = self.order_manager.place_order(
                    symbol=symbol,
                    direction=direction,
                    size_usd=size_usd,
                    current_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if order_result and order_result.get('status') == 'executed':
                    position_actions['opened'] += 1
                    logging.info(f"Order placed: {order_details}")
                    
                    # Add position to monitoring
                    if self.position_monitor:
                        position_id = f"{order_details['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        self.position_monitor.add_position(
                            position_id=position_id,
                            symbol=order_details['symbol'],
                            direction=order_details['direction'],
                            size=order_details['size_usd'],
                            entry_price=self._get_current_price(order_details['symbol']),
                            stop_loss=order_details.get('stop_loss'),
                            take_profit=order_details.get('take_profit'),
                            session=order_details.get('session'),
                            signal_confidence=order_details.get('signal_confidence')
                        )
                else:
                    logging.warning(f"Failed to place order: {order_details}")
            
        except Exception as e:
            logging.error(f"Error processing trading signals: {e}")
        
        return position_actions
    
    def _monitor_positions(self):
        """Monitor and manage existing positions"""
        try:
            # Update positions with current market prices
            if self.position_monitor:
                latest_features = self.live_data_manager.get_latest_features()
                
                if latest_features is not None and not latest_features.empty:
                    # Get current price from latest market data
                    current_price = self._get_current_price_from_features(latest_features)
                    
                    # Update all active positions with current price
                    portfolio_summary = self.position_monitor.get_portfolio_summary()
                    
                    for position_id in portfolio_summary.get('positions', {}):
                        self.position_monitor.update_position_price(position_id, current_price)
                    
                    # Log portfolio status periodically
                    if portfolio_summary['active_positions'] > 0:
                        logging.debug(f"Portfolio: {portfolio_summary['active_positions']} positions, "
                                    f"PnL: {portfolio_summary['unrealized_pnl']:.4f}")
        
        except Exception as e:
            logging.error(f"Error monitoring positions: {e}")
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        try:
            latest_features = self.live_data_manager.get_latest_features()
            if latest_features is not None and not latest_features.empty:
                return self._get_current_price_from_features(latest_features)
            else:
                # Fallback to a default price if no data available
                return 1.2500  # Default for GBPUSD
        except Exception as e:
            logging.error(f"Error getting current price for {symbol}: {e}")
            return 1.2500  # Default fallback
    
    def _get_current_price_from_features(self, features: pd.DataFrame) -> float:
        """Extract current price from processed features"""
        try:
            # Look for close price in the features
            if 'close_og' in features.columns:  # Original close price before normalization
                return float(features['close_og'].iloc[-1])
            elif 'close' in features.columns:
                return float(features['close'].iloc[-1])
            else:
                # Try to find any price-related column
                price_cols = [col for col in features.columns if 'price' in col.lower() or 'close' in col.lower()]
                if price_cols:
                    return float(features[price_cols[0]].iloc[-1])
                else:
                    logging.warning("No price column found in features, using default")
                    return 1.2500  # Default fallback
        except Exception as e:
            logging.error(f"Error extracting price from features: {e}")
            return 1.2500  # Default fallback
    
    def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            # TODO: This would typically call the broker API
            # For now, return a default value
            return 10000.0
        except Exception as e:
            logging.error(f"Error getting account balance: {e}")
            return 10000.0
    
    def _cleanup_resources(self):
        """Cleanup resources when strategy stops"""
        try:
            if self.live_data_manager:
                self.live_data_manager.stop_streaming()
                logging.info("Live data streaming stopped")
            
            if self.position_monitor:
                self.position_monitor.stop_monitoring()
                logging.info("Position monitoring stopped")
                
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
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