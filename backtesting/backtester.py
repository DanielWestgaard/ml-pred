import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import time
from datetime import datetime

from utils.logging_utils import get_logger
from utils.hypothesis.config_loader import load_hypothesis_config
import pickle
from utils.model.model_metadata_loader import load_model_metadata
from data.features.enhanced_generation import ModelAwareFeatureGenerator

# Import existing strategy components (reuse!)
from strategy.components.signal_processor import SignalProcessor
from strategy.components.risk_manager import RiskManager
from strategy.components.position_manager import PositionManager

# Import backtesting-specific components
from backtesting.simulation_broker import SimulationBroker
from backtesting.backtest_data_manager import BacktestDataManager

logger = get_logger()

class BacktestOrchestrator:
    """
    Main backtesting orchestrator that leverages existing strategy components
    for rapid and accurate simulation of trading strategies.
    """
    
    def __init__(self, hypothesis_config_path: str = None, model_folder: str = None, **kwargs):
        """
        Initialize backtest orchestrator
        
        Args:
            hypothesis_config_path: Path to hypothesis configuration file
            model_folder: Path to trained model folder (alternative to hypothesis config)
        """
        self.hypothesis_config_path = hypothesis_config_path
        self.model_folder = model_folder
        
        # Configuration and model
        self.config = None
        self.model = None
        self.model_metadata = None
        self.feature_generator = None
        
        # Backtesting components
        self.data_manager = None
        self.simulation_broker = None
        self.signal_processor = None
        self.risk_manager = None
        self.position_manager = None
        
        # Backtesting state
        self.backtest_results = {}
        self.is_running = False
        self.start_time = None
        
        # Load configuration and initialize components
        self._load_configuration()
        self._load_model()
        self._initialize_components()
        
        logger.info("Backtest orchestrator initialized successfully")
    
    def _load_configuration(self):
        """Load hypothesis configuration using existing config_loader utility"""
        try:
            if self.hypothesis_config_path:
                # Use existing config loader utility for consistency
                config_loader = load_hypothesis_config(self.hypothesis_config_path)
                self.config = config_loader.handle_hypothesis_source()
                logger.info(f"Loaded hypothesis config using utils/hypothesis/config_loader: {self.hypothesis_config_path}")
                logger.info(f"Config keys: {list(self.config.keys())}")
                self.model_folder = self.config.get("backtesting_model_folder")
                logger.debug(f"Model Folder is {self.model_folder}")
            else:
                # Create minimal config for model-only backtesting
                self.config = {
                    'hypothesis_name': 'model_backtest',
                    'backtesting': {
                        'enabled': True,
                        'initial_capital': 10000,
                        'transaction_costs': 0.0013,
                        'slippage': 0.0002
                    }
                }
                logger.info("Created minimal config for model-only backtesting")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_model(self):
        """Load trained model and metadata using existing model utilities"""
        try:
            # Determine model path
            if self.model_folder:
                model_path = self.model_folder
            elif self.config.get('model_training', {}).get('model_folder'):
                model_path = self.config['model_training']['model_folder']
            else:
                # Use existing utility to find most recent model for the hypothesis
                from utils.model.model_utils import get_latest_model
                hypothesis_name = self.config.get('hypothesis_name', 'spread_compression_patterns')
                model_path = get_latest_model("storage/models", model_type=None)
                
                # If no latest model found, try the old method as fallback
                if not model_path:
                    model_path = self._find_latest_model(hypothesis_name)
            
            if not model_path:
                raise ValueError("No model path specified and no trained models found")
            
            # Use existing model metadata loader utility
            loader = load_model_metadata(model_path)
            self.model_metadata = loader.get_full_metadata()
            
            # Load model using same pattern as base_strategy
            model_type = loader.get_model_type()
            model_file = Path(model_path) / f"{model_type}.pkl"
            
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"Loaded model using utils/model utilities: {model_path}")
            logger.info(f"Model type: {model_type}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _find_latest_model(self, hypothesis_name: str) -> Optional[str]:
        """
        Legacy method - Find the latest trained model for a hypothesis.
        
        Note: This method is kept as fallback. Consider using utils.model.model_utils.get_latest_model() 
        for new implementations as it provides more robust model discovery.
        """
        models_dir = Path("storage/models")
        if not models_dir.exists():
            return None
        
        # Find models with matching hypothesis
        matching_models = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "model_metadata.yml"
                if metadata_path.exists():
                    try:
                        import yaml
                        with open(metadata_path, 'r') as f:
                            metadata = yaml.safe_load(f)
                        if metadata.get('hypothesis', {}).get('name') == hypothesis_name:
                            matching_models.append(model_dir)
                    except:
                        continue
        
        if matching_models:
            # Return most recent model (sorted by directory name which includes timestamp)
            latest_model = sorted(matching_models, key=lambda x: x.name)[-1]
            return str(latest_model)
        
        return None
    
    def _initialize_components(self):
        """Initialize backtesting components"""
        try:
            # Get backtesting config
            backtest_config = self.config.get('backtesting', {})
            
            # Data manager
            data_source = backtest_config.get('data_source')
            if not data_source:
                # Try to get from model metadata
                data_source = self.model_metadata.get('dataset', {}).get('dataset_path')
            
            if not data_source:
                raise ValueError("No data source specified in config or model metadata")
            
            self.data_manager = BacktestDataManager(
                data_path=data_source,
                start_date=backtest_config.get('start_date'),
                end_date=backtest_config.get('end_date'),
                buffer_size=backtest_config.get('buffer_size', 200)
            )
            
            # Simulation broker
            self.simulation_broker = SimulationBroker(
                initial_capital=backtest_config.get('initial_capital', 10000),
                transaction_cost_pct=backtest_config.get('transaction_costs', 0.0013),
                slippage_pct=backtest_config.get('slippage', 0.0002)
            )
            
            # Feature generator (reuse existing system!)
            self.feature_generator = None  # Will be initialized per-iteration
            
            # Strategy components (reuse existing components!)
            signal_config = self.config.get('signal_processing', {})
            self.signal_processor = SignalProcessor(
                confidence_threshold=signal_config.get('confidence_threshold', 0.6),
                signal_filters=signal_config.get('signal_filters', [])
            )
            
            risk_config = self.config.get('risk_management', {})
            self.risk_manager = RiskManager(
                max_position_size=risk_config.get('max_position_size', 0.02),
                stop_loss_pct=risk_config.get('stop_loss_pct', 0.01),
                take_profit_pct=risk_config.get('take_profit_pct', 0.02),
                max_drawdown=risk_config.get('max_drawdown', 0.05),
                max_concurrent_positions=risk_config.get('max_concurrent_positions', 3)
            )
            
            position_config = self.config.get('position_management', {})
            self.position_manager = PositionManager(
                max_concurrent_positions=position_config.get('max_concurrent_positions', 3),
                position_tracking=position_config.get('position_tracking', 'detailed')
            )
            
            logger.info("All backtesting components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the complete backtesting simulation
        
        Returns:
            Backtest results dictionary
        """
        try:
            logger.info("🚀 Starting backtest simulation...")
            self.is_running = True
            self.start_time = time.time()
            
            # Initialize tracking
            processed_signals = 0
            executed_trades = 0
            signals_generated = 0
            progress_interval = 1000  # Log progress every 1000 iterations
            
            # Main backtesting loop
            iteration = 0
            while self.data_manager.has_more_data() and self.is_running:
                iteration += 1
                
                # Get next data point
                current_data = self.data_manager.get_next_data_point()
                if not current_data:
                    break
                
                # Get data buffer for feature generation
                data_buffer = self.data_manager.get_current_data_buffer()
                
                # Skip if buffer is too small for feature generation
                if len(data_buffer) < 50:  # Minimum buffer for stable feature generation
                    continue
                
                # Generate features for current data point
                try:
                    # Get required features from model metadata
                    required_features = self._get_required_features()
                    
                    # Initialize feature generator with current data buffer
                    if self.feature_generator is None or len(data_buffer) > 50:
                        self.feature_generator = ModelAwareFeatureGenerator(
                            data_buffer,
                            selected_features=required_features
                        )
                    
                    # Generate features
                    features_df = self.feature_generator.generate_selected(required_features)
                    
                    if features_df.empty or len(features_df) == 0:
                        continue
                    
                    # Get latest feature row for prediction
                    latest_features = features_df.iloc[-1:]
                    
                    # Ensure we have all the features the model expects
                    model_features = self.model_metadata.get('dataset', {}).get('features', [])
                    missing_features = [f for f in model_features if f not in latest_features.columns]
                    
                    if missing_features:
                        logger.debug(f"Missing features for prediction: {missing_features}")
                        continue
                    
                    # Select only the features the model was trained on
                    prediction_features = latest_features[model_features]
                    
                    # Make prediction
                    prediction = self.model.predict(prediction_features)[0]
                    
                    # Process prediction into trading signals
                    signals = self.signal_processor.process([prediction], data_buffer)
                    signals_generated += len(signals['processed_signals'])
                    
                    # Process each signal
                    for signal_info in signals['processed_signals']:
                        processed_signals += 1
                        
                        # Enhance signal with current market data
                        enhanced_signal = self._enhance_signal(signal_info, current_data)
                        
                        # Check if we can open position (risk management)
                        if self.risk_manager.can_open_position(enhanced_signal, self.position_manager):
                            
                            # Calculate position size
                            account_info = self.simulation_broker.get_account_info()
                            position_info = self.risk_manager.calculate_position_size(
                                signal_strength=enhanced_signal['confidence'],
                                account_balance=account_info['current_equity'],
                                current_price=current_data['close'],
                                direction=enhanced_signal['direction']
                            )
                            
                            # Add stop loss and take profit to signal
                            enhanced_signal['stop_loss_price'] = position_info['stop_loss_price']
                            enhanced_signal['take_profit_price'] = position_info['take_profit_price']
                            
                            # Execute trade
                            trade_result = self.simulation_broker.place_order(
                                signal=enhanced_signal,
                                current_price=current_data['close'],
                                position_size=position_info['position_size_usd'],
                                timestamp=current_data['timestamp']
                            )
                            
                            if trade_result['success']:
                                executed_trades += 1
                                self.position_manager.add_position(trade_result)
                    
                    # Update existing positions
                    self.simulation_broker.update_positions(
                        current_price=current_data['close'],
                        timestamp=current_data['timestamp']
                    )
                    
                    # Check for position closes (stop loss / take profit)
                    self._check_position_closes(current_data)
                    
                    # Update risk management
                    account_info = self.simulation_broker.get_account_info()
                    self.risk_manager.update_drawdown(account_info['current_equity'])
                    
                except Exception as e:
                    logger.debug(f"Error processing data point {iteration}: {e}")
                    continue
                
                # Progress logging
                if iteration % progress_interval == 0:
                    progress = self.data_manager.get_progress_info()
                    account_info = self.simulation_broker.get_account_info()
                    logger.info(f"Progress: {progress['progress_percentage']:.1f}% | "
                               f"Equity: ${account_info['current_equity']:.2f} | "
                               f"Trades: {executed_trades} | "
                               f"Signals: {signals_generated}")
            
            # Close all remaining positions at end of backtest
            if self.data_manager.get_latest_data_point():
                final_data = self.data_manager.get_latest_data_point()
                self.simulation_broker.close_all_positions(
                    current_price=final_data['close'],
                    timestamp=final_data['timestamp']
                )
            
            # Calculate final results
            self.backtest_results = self._calculate_results()
            
            # Save results
            self._save_results()
            
            execution_time = time.time() - self.start_time
            logger.info(f"✅ Backtest completed in {execution_time:.2f} seconds")
            logger.info(f"📊 Final Results: Return: {self.backtest_results['backtest_summary']['total_return_pct']:.2f}%, "
                       f"Trades: {self.backtest_results['trading_performance']['total_trades']}, "
                       f"Win Rate: {self.backtest_results['trading_performance']['win_rate']:.1f}%")
            
            return self.backtest_results
            
        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            raise
        finally:
            self.is_running = False
    
    def _get_required_features(self) -> List[str]:
        """Get required features from model metadata"""
        features = self.model_metadata.get('dataset', {}).get('features', [])
        
        # Filter out non-feature columns
        excluded_columns = ['target_spread_compression_binary', 'target_spread_compression_multi', 
                          'future_price_change', 'open', 'high', 'low', 'close', 'volume', 
                          'close_og', 'open_og', 'mid_price']
        
        required_features = [f for f in features if f not in excluded_columns]
        
        logger.debug(f"Required features: {required_features}")
        return required_features
    
    def _enhance_signal(self, signal_info: Dict[str, Any], current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance signal with current market data"""
        return {
            **signal_info,
            'symbol': 'GBPUSD',  # TODO: Make configurable
            'current_price': current_data['close'],
            'timestamp': current_data['timestamp'],
            'market_data': current_data
        }
    
    def _check_position_closes(self, current_data: Dict[str, Any]):
        """Check and execute position closes based on stop loss / take profit"""
        open_positions = self.simulation_broker.get_open_positions()
        
        for position in open_positions:
            close_info = self.risk_manager.should_close_position(
                position, current_data['close']
            )
            
            if close_info['should_close']:
                self.simulation_broker.close_position(
                    position_id=position['id'],
                    current_price=current_data['close'],
                    timestamp=current_data['timestamp'],
                    reason=close_info['reason']
                )
                self.position_manager.close_position(position['id'])
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        account_info = self.simulation_broker.get_account_info()
        closed_trades = self.simulation_broker.get_closed_trades()
        
        # Basic performance metrics
        results = {
            'backtest_summary': {
                'hypothesis_name': self.config.get('hypothesis_name', 'unknown'),
                'model_type': self.model_metadata.get('model_info', {}).get('model_type'),
                'initial_capital': account_info['initial_capital'],
                'final_equity': account_info['current_equity'],
                'total_return': account_info['total_return'],
                'total_return_pct': account_info['total_return_pct'],
                'execution_time_seconds': time.time() - self.start_time
            },
            
            'trading_performance': {
                'total_trades': account_info['total_trades'],
                'winning_trades': account_info['winning_trades'],
                'losing_trades': len(closed_trades) - account_info['winning_trades'],
                'win_rate': account_info['win_rate'],
                'total_fees': account_info['total_fees'],
                'max_drawdown': account_info['max_drawdown'],
                'max_drawdown_pct': account_info['max_drawdown_pct']
            },
            
            'risk_metrics': self._calculate_risk_metrics(closed_trades, account_info),
            'trade_analysis': self._analyze_trades(closed_trades),
            'data_period': self.data_manager.get_data_summary()['date_range']
        }
        
        return results
    
    def _calculate_risk_metrics(self, closed_trades: List[Dict], account_info: Dict) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics"""
        if not closed_trades:
            return {'error': 'No closed trades for risk analysis'}
        
        # Extract returns
        returns = [trade['net_pnl'] / account_info['initial_capital'] for trade in closed_trades]
        returns_array = np.array(returns)
        
        # Risk metrics
        risk_metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio(returns_array),
            'sortino_ratio': self._calculate_sortino_ratio(returns_array),
            'profit_factor': self._calculate_profit_factor(closed_trades),
            'average_win': np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0,
            'average_loss': np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0,
            'largest_win': max(returns) if returns else 0,
            'largest_loss': min(returns) if returns else 0
        }
        
        return risk_metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (focuses on downside deviation)"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    def _calculate_profit_factor(self, closed_trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        wins = [trade['net_pnl'] for trade in closed_trades if trade['net_pnl'] > 0]
        losses = [abs(trade['net_pnl']) for trade in closed_trades if trade['net_pnl'] < 0]
        
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        
        return total_wins / total_losses if total_losses > 0 else float('inf')
    
    def _analyze_trades(self, closed_trades: List[Dict]) -> Dict[str, Any]:
        """Analyze trade patterns and characteristics"""
        if not closed_trades:
            return {'error': 'No trades to analyze'}
        
        # Holding periods
        holding_periods = [trade['holding_period_minutes'] for trade in closed_trades]
        
        # P&L analysis
        pnls = [trade['net_pnl'] for trade in closed_trades]
        
        analysis = {
            'average_holding_period_minutes': np.mean(holding_periods),
            'median_holding_period_minutes': np.median(holding_periods),
            'average_pnl_per_trade': np.mean(pnls),
            'median_pnl_per_trade': np.median(pnls),
            'pnl_std': np.std(pnls),
            'consecutive_wins_max': self._calculate_max_consecutive(closed_trades, 'win'),
            'consecutive_losses_max': self._calculate_max_consecutive(closed_trades, 'loss'),
            'exit_reasons': self._analyze_exit_reasons(closed_trades)
        }
        
        return analysis
    
    def _calculate_max_consecutive(self, closed_trades: List[Dict], result_type: str) -> int:
        """Calculate maximum consecutive wins or losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in closed_trades:
            is_win = trade['net_pnl'] > 0
            
            if (result_type == 'win' and is_win) or (result_type == 'loss' and not is_win):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _analyze_exit_reasons(self, closed_trades: List[Dict]) -> Dict[str, int]:
        """Analyze reasons for trade exits"""
        exit_reasons = {}
        for trade in closed_trades:
            reason = trade.get('exit_reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return exit_reasons
    
    def _save_results(self):
        """Save backtest results to storage"""
        try:
            # Create results directory
            model_path = Path(self.model_metadata['model_info']['storage_path'])
            results_dir = model_path / 'backtesting'
            results_dir.mkdir(exist_ok=True)
            
            # Save main results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f'backtest_results_{timestamp}.json'
            
            with open(results_file, 'w') as f:
                json.dump(self.backtest_results, f, indent=2, default=str)
            
            # Save detailed trade log
            trades_file = results_dir / f'trades_log_{timestamp}.json'
            trades_data = {
                'closed_trades': self.simulation_broker.get_closed_trades(),
                'final_account_info': self.simulation_broker.get_account_info()
            }
            
            with open(trades_file, 'w') as f:
                json.dump(trades_data, f, indent=2, default=str)
            
            logger.info(f"Backtest results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def stop(self):
        """Stop the backtest"""
        self.is_running = False
        logger.info("Backtest stop requested")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current backtest progress"""
        if not self.data_manager:
            return {'error': 'Backtest not initialized'}
        
        progress_info = self.data_manager.get_progress_info()
        account_info = self.simulation_broker.get_account_info() if self.simulation_broker else {}
        
        return {
            **progress_info,
            'account_info': account_info,
            'is_running': self.is_running,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }