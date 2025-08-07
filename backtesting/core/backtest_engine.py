import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from utils.logging_utils import get_logger
from utils.hypothesis.config_loader import load_hypothesis_config

# Import existing backtesting components
from backtesting.simulation_broker import SimulationBroker
from backtesting.backtest_data_manager import BacktestDataManager

# Import strategy components (reused)
from strategy.components.risk_manager import RiskManager
from strategy.components.position_manager import PositionManager

# Import new focused components
from .config_validator import BacktestConfigValidator
from .model_handler import BacktestModelHandler
from .signal_handler import BacktestSignalHandler

# Import plugin system
from backtesting.hypothesis_plugins import HypothesisPluginRegistry

logger = get_logger()

class BacktestEngine:
    """
    Core backtesting engine that orchestrates all components.
    
    This class coordinates the backtesting process using focused, 
    single-responsibility components for maintainability and testability.
    """
    
    def __init__(self, hypothesis_config_path: str = None, model_folder: str = None, **kwargs):
        """
        Initialize backtest engine
        
        Args:
            hypothesis_config_path: Path to hypothesis configuration file
            model_folder: Optional model folder path override
        """
        self.hypothesis_config_path = hypothesis_config_path
        self.model_folder = model_folder
        
        # Core state
        self.config = None
        self.hypothesis_plugin = None
        self.is_running = False
        self.start_time = None
        self.backtest_results = {}
        
        # Component handlers
        self.config_validator = BacktestConfigValidator()
        self.model_handler = None
        self.signal_handler = None
        
        # Existing backtesting components (reused)
        self.data_manager = None
        self.simulation_broker = None
        self.risk_manager = None
        self.position_manager = None
        
        # Data processing state
        self.preprocessed_data = None
        self.current_data_index = 0
        
        # Initialize all components
        self._initialize_engine()
        
        logger.info("BacktestEngine initialized successfully")
    
    def _initialize_engine(self):
        """Initialize all engine components"""
        try:
            # Step 1: Load and validate configuration
            self._load_and_validate_configuration()
            
            # Step 2: Initialize hypothesis plugin
            self._initialize_hypothesis_plugin()
            
            # Step 3: Initialize model handler
            self._initialize_model_handler()
            
            # Step 4: Initialize signal handler
            self._initialize_signal_handler()
            
            # Step 5: Load and prepare data
            self._load_and_prepare_data()
            
            # Step 6: Initialize trading components
            self._initialize_trading_components()
            
            logger.info("All engine components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing backtest engine: {e}")
            raise
    
    def _load_and_validate_configuration(self):
        """Load and validate the hypothesis configuration"""
        # Load configuration
        if self.hypothesis_config_path:
            config_loader = load_hypothesis_config(self.hypothesis_config_path)
            self.config = config_loader.handle_hypothesis_source()
            logger.info(f"Loaded configuration from: {self.hypothesis_config_path}")
        else:
            # Create minimal configuration for model-only backtesting
            self.config = {
                'hypothesis_name': 'model_backtest',
                'backtesting': {
                    'enabled': True,
                    'initial_capital': 10000,
                    'transaction_costs': 0.0013,
                    'slippage': 0.0002
                }
            }
            logger.info("Created minimal configuration for model-only backtesting")
        
        # Validate configuration
        is_valid, errors, warnings = self.config_validator.validate(self.config)
        
        if not is_valid:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        # Override model folder if specified
        if self.model_folder:
            self.config.setdefault('backtesting', {})['model_folder'] = self.model_folder
        
        # Log configuration summary
        config_summary = self.config_validator.get_config_summary(self.config)
        logger.info(f"Configuration summary: {config_summary}")
    
    def _initialize_hypothesis_plugin(self):
        """Initialize hypothesis-specific plugin"""
        hypothesis_name = self.config.get('hypothesis_name', 'unknown')
        self.hypothesis_plugin = HypothesisPluginRegistry.get_plugin(hypothesis_name, self.config)
        
        plugin_info = self.hypothesis_plugin.get_hypothesis_info()
        logger.info(f"Using hypothesis plugin: {plugin_info['plugin_class']} for {hypothesis_name}")
    
    def _initialize_model_handler(self):
        """Initialize model handler"""
        self.model_handler = BacktestModelHandler(
            config=self.config,
            model_folder=self.model_folder
        )
        
        model_info = self.model_handler.get_model_info()
        logger.info(f"Model loaded: {model_info['model_class']} ({model_info['model_type']})")
        logger.info(f"Features required: {model_info['feature_count']}")
    
    def _initialize_signal_handler(self):
        """Initialize signal handler"""
        self.signal_handler = BacktestSignalHandler(
            config=self.config,
            model_type=self.model_handler.model_type,
            hypothesis_plugin=self.hypothesis_plugin
        )
        
        processor_info = self.signal_handler.get_processor_stats()
        exit_info = self.signal_handler.get_exit_strategy_info()
        
        logger.info(f"Signal processor: {processor_info['processor_type']}")
        logger.info(f"Exit strategy: {exit_info['exit_strategy_class']}")
    
    def _load_and_prepare_data(self):
        """Load and filter preprocessed data for backtesting"""
        backtest_config = self.config.get('backtesting', {})
        data_source = backtest_config.get('data_source')
        
        if not data_source:
            # Try to get from model metadata
            data_source_info = self.model_handler.get_data_source_info()
            data_source = data_source_info.get('dataset_path')
        
        if not data_source:
            raise ValueError("No data source specified in config or model metadata")
        
        logger.info(f"Loading preprocessed data from: {data_source}")
        
        # Load and filter data
        full_data = pd.read_csv(data_source)
        
        # Set up timestamp column
        if 'date' in full_data.columns:
            full_data['timestamp'] = pd.to_datetime(full_data['date'], utc=True)
        else:
            raise ValueError("No 'date' column found in preprocessed data")
        
        # Filter by date range
        start_date = backtest_config.get('start_date')
        end_date = backtest_config.get('end_date')
        
        if start_date:
            start_date = pd.to_datetime(start_date, utc=True)
            full_data = full_data[full_data['timestamp'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date, utc=True)
            full_data = full_data[full_data['timestamp'] <= end_date]
        
        # Sort by timestamp
        full_data = full_data.sort_values('timestamp').reset_index(drop=True)
        
        self.preprocessed_data = full_data
        self.current_data_index = 0
        
        logger.info(f"Loaded {len(self.preprocessed_data)} data points")
        logger.info(f"Date range: {self.preprocessed_data['timestamp'].min()} to {self.preprocessed_data['timestamp'].max()}")
        
        # Validate required features are present
        required_features = self.model_handler.get_required_features()
        available_features = [f for f in required_features if f in self.preprocessed_data.columns]
        missing_features = [f for f in required_features if f not in self.preprocessed_data.columns]
        
        logger.info(f"Available features: {len(available_features)}/{len(required_features)}")
        if missing_features:
            logger.warning(f"Missing features: {missing_features[:5]}...")
    
    def _initialize_trading_components(self):
        """Initialize trading-related components"""
        backtest_config = self.config.get('backtesting', {})
        
        # Data manager (existing component)
        data_source = backtest_config.get('data_source') or self.model_handler.get_data_source_info().get('dataset_path')
        self.data_manager = BacktestDataManager(
            data_path=data_source,
            start_date=backtest_config.get('start_date'),
            end_date=backtest_config.get('end_date'),
            buffer_size=backtest_config.get('buffer_size', 200)
        )
        
        # Simulation broker (existing component)
        self.simulation_broker = SimulationBroker(
            initial_capital=backtest_config.get('initial_capital', 10000),
            transaction_cost_pct=backtest_config.get('transaction_costs', 0.0013),
            slippage_pct=backtest_config.get('slippage', 0.0002)
        )
        
        # Risk manager (existing component)
        risk_config = backtest_config.get('risk_management', {})
        self.risk_manager = RiskManager(
            max_position_size=risk_config.get('max_position_size', 0.02),
            stop_loss_pct=risk_config.get('stop_loss_pct', 0.01),
            take_profit_pct=risk_config.get('take_profit_pct', 0.02),
            max_drawdown=risk_config.get('max_drawdown', 0.05),
            max_concurrent_positions=risk_config.get('max_concurrent_positions', 3)
        )
        
        # Position manager (existing component)
        position_config = backtest_config.get('position_management', {})
        self.position_manager = PositionManager(
            max_concurrent_positions=position_config.get('max_concurrent_positions', 3),
            position_tracking=position_config.get('position_tracking', 'detailed')
        )
        
        logger.info("Trading components initialized")
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the complete backtesting simulation
        
        Returns:
            Comprehensive backtest results
        """
        try:
            logger.info("🚀 Starting backtest simulation...")
            self.is_running = True
            self.start_time = time.time()
            
            # Initialize tracking
            iteration = 0
            processed_signals = 0
            executed_trades = 0
            signals_generated = 0
            
            # Configuration
            max_loops = self.config.get('backtesting', {}).get('execution', {}).get('max_loops', 10000)
            total_data_points = len(self.preprocessed_data)
            progress_interval = 1000
            
            # Main backtesting loop
            while (self.current_data_index < total_data_points and 
                   self.is_running and iteration < max_loops):
                
                iteration += 1
                
                # Get current data point
                current_row = self.preprocessed_data.iloc[self.current_data_index]
                current_data = self._extract_market_data(current_row)
                
                # Skip invalid data
                if current_data['close'] <= 0:
                    logger.debug(f"Skipping invalid price data at index {self.current_data_index}")
                    self.current_data_index += 1
                    continue
                
                # Make predictions
                features = self.model_handler.prepare_features_for_prediction(current_row)
                if features is None:
                    self.current_data_index += 1
                    continue
                
                predictions, probabilities = self.model_handler.make_predictions(features)
                
                # Process signals
                signals = self.signal_handler.process_predictions(
                    predictions=predictions,
                    probabilities=probabilities,
                    market_data=current_data
                )
                signals_generated += len(signals['processed_signals'])
                
                # Process each signal for entry
                for signal_info in signals['processed_signals']:
                    processed_signals += 1
                    
                    # Enhance signal with market data
                    enhanced_signal = self.signal_handler.enhance_signal_with_market_data(
                        signal_info, current_data
                    )
                    
                    # Check risk management
                    if self.risk_manager.can_open_position(enhanced_signal, self.position_manager):
                        trade_result = self._execute_trade(enhanced_signal, current_data)
                        
                        if trade_result['success']:
                            executed_trades += 1
                            self._track_new_position(trade_result, enhanced_signal, current_data)
                
                # Update existing positions and check exits
                self._update_and_check_positions(current_data, predictions, probabilities)
                
                # Move to next data point
                self.current_data_index += 1
                
                # Progress logging
                if iteration % progress_interval == 0:
                    self._log_progress(iteration, max_loops, executed_trades, signals_generated)
            
            # Finalize backtest
            self._finalize_backtest()
            
            # Calculate and return results
            self.backtest_results = self._calculate_comprehensive_results()
            self._save_results()
            
            execution_time = time.time() - self.start_time
            logger.info(f"✅ Backtest completed in {execution_time:.2f} seconds")
            self._log_final_summary()
            
            return self.backtest_results
            
        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            raise
        finally:
            self.is_running = False
    
    def _extract_market_data(self, row: pd.Series) -> Dict[str, Any]:
        """Extract market data from data row"""
        return {
            'timestamp': row['date'],
            'open': row.get('open', row.get('close_og', 0)),
            'high': row.get('high', row.get('close_og', 0)),
            'low': row.get('low', row.get('close_og', 0)),
            'close': row.get('close_og', 0),
            'volume': row.get('volume', 0),
            # Include all other features for hypothesis customization
            **{col: row.get(col, 0) for col in row.index 
               if col not in ['date', 'open', 'high', 'low', 'close_og', 'volume']}
        }
    
    def _execute_trade(self, signal: Dict[str, Any], current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade through the simulation broker"""
        # Calculate position size
        account_info = self.simulation_broker.get_account_info()
        position_info = self.risk_manager.calculate_position_size(
            signal_strength=signal['confidence'],
            account_balance=account_info['current_equity'],
            current_price=current_data['close'],
            direction=signal['direction']
        )
        
        # Add stop loss and take profit to signal
        signal['stop_loss_price'] = position_info['stop_loss_price']
        signal['take_profit_price'] = position_info['take_profit_price']
        
        # Execute through simulation broker
        return self.simulation_broker.place_order(
            signal=signal,
            current_price=current_data['close'],
            position_size=position_info['position_size_usd'],
            timestamp=current_data['timestamp']
        )
    
    def _track_new_position(self, trade_result: Dict[str, Any], signal: Dict[str, Any], 
                          current_data: Dict[str, Any]):
        """Track a newly opened position"""
        if trade_result['success']:
            position_id = trade_result['position_id']
            
            # Track entry signal
            self.signal_handler.track_entry_signal(
                position_id, current_data, signal['confidence']
            )
            
            # Store entry features for hypothesis-specific exits
            position = self.simulation_broker.get_position(position_id)
            if position:
                position['entry_features'] = self.hypothesis_plugin.customize_features(current_data)
                position['entry_time'] = pd.to_datetime(current_data['timestamp'], utc=True)
    
    def _update_and_check_positions(self, current_data: Dict[str, Any], 
                                  predictions: np.ndarray, probabilities: np.ndarray):
        """Update existing positions and check for exits"""
        # Update positions with current market data
        self.simulation_broker.update_positions(
            current_price=current_data['close'],
            timestamp=current_data['timestamp']
        )
        
        # Check for position closes
        open_positions = self.simulation_broker.get_open_positions()
        
        for position in open_positions:
            # Check risk management rules (stop loss / take profit)
            close_info = self.risk_manager.should_close_position(
                position, current_data['close']
            )
            
            # Check hypothesis-specific exit strategy
            if not close_info['should_close']:
                should_exit, exit_reason = self.signal_handler.should_exit_position(
                    position, current_data, predictions, probabilities
                )
                if should_exit:
                    close_info['should_close'] = True
                    close_info['reason'] = exit_reason
            
            # Close position if needed
            if close_info['should_close']:
                self.simulation_broker.close_position(
                    position_id=position['id'],
                    current_price=current_data['close'],
                    timestamp=current_data['timestamp'],
                    reason=close_info['reason']
                )
                
                # Clean up tracking
                self.signal_handler.cleanup_closed_position(position['id'])
        
        # Update risk management
        account_info = self.simulation_broker.get_account_info()
        self.risk_manager.update_drawdown(account_info['current_equity'])
    
    def _finalize_backtest(self):
        """Finalize backtest by closing remaining positions"""
        if self.current_data_index > 0:
            final_row = self.preprocessed_data.iloc[self.current_data_index - 1]
            self.simulation_broker.close_all_positions(
                current_price=final_row['close'],
                timestamp=final_row['date']
            )
    
    def _calculate_comprehensive_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        account_info = self.simulation_broker.get_account_info()
        closed_trades = self.simulation_broker.get_closed_trades()
        
        # Import performance analyzer for advanced metrics
        from backtesting.performance_analyzer import BacktestPerformanceAnalyzer
        
        # Create analyzer
        basic_results = {
            'backtest_summary': {
                'hypothesis_name': self.config.get('hypothesis_name', 'unknown'),
                'model_type': self.model_handler.model_metadata.get('model_info', {}).get('model_type'),
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
            'data_period': self.data_manager.get_data_summary()['date_range']
        }
        
        # Add component information
        basic_results['component_info'] = {
            'model_info': self.model_handler.get_model_info(),
            'signal_processor_stats': self.signal_handler.get_processor_stats(),
            'exit_strategy_info': self.signal_handler.get_exit_strategy_info(),
            'signal_tracker_info': self.signal_handler.get_signal_tracker_info(),
            'hypothesis_plugin': self.hypothesis_plugin.get_hypothesis_info()
        }
        
        # Use performance analyzer for advanced metrics
        try:
            analyzer = BacktestPerformanceAnalyzer(results_data=basic_results)
            analyzer.trades_data = {
                'closed_trades': closed_trades,
                'final_account_info': account_info
            }
            
            # Calculate basic risk metrics manually
            risk_metrics = self._calculate_basic_risk_metrics(closed_trades, account_info)
            basic_results['risk_metrics'] = risk_metrics
            
            # Get key insights from analyzer
            insights = analyzer.get_key_insights()
            basic_results['key_insights'] = insights
            
        except Exception as e:
            logger.warning(f"Could not calculate advanced metrics: {e}")
        
        return basic_results
    
    def _calculate_basic_risk_metrics(self, closed_trades: list, account_info: dict) -> dict:
        """Calculate basic risk metrics for backtesting results"""
        try:
            if not closed_trades:
                return {
                    'sharpe_ratio': 'N/A',
                    'profit_factor': 'N/A',
                    'max_consecutive_losses': 0,
                    'max_consecutive_wins': 0
                }
            
            # Extract trade results
            trade_pnls = [trade.get('pnl', 0) for trade in closed_trades]
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl <= 0]
            
            # Calculate profit factor
            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Calculate basic Sharpe ratio (simplified)
            if len(trade_pnls) > 1:
                returns_std = np.std(trade_pnls)
                avg_return = np.mean(trade_pnls)
                sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate consecutive wins/losses
            consecutive_wins = consecutive_losses = 0
            max_consecutive_wins = max_consecutive_losses = 0
            
            for pnl in trade_pnls:
                if pnl > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            return {
                'sharpe_ratio': round(sharpe_ratio, 3) if isinstance(sharpe_ratio, (int, float)) else sharpe_ratio,
                'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 'inf',
                'max_consecutive_losses': max_consecutive_losses,
                'max_consecutive_wins': max_consecutive_wins,
                'total_winning_trades': len(winning_trades),
                'total_losing_trades': len(losing_trades),
                'avg_winning_trade': round(np.mean(winning_trades), 2) if winning_trades else 0,
                'avg_losing_trade': round(np.mean(losing_trades), 2) if losing_trades else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'sharpe_ratio': 'Error',
                'profit_factor': 'Error',
                'max_consecutive_losses': 0,
                'max_consecutive_wins': 0
            }
    
    def _save_results(self):
        """Save backtest results to storage"""
        try:
            # Create results directory
            model_path = Path(self.model_handler.model_metadata['model_info']['storage_path'])
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
    
    def _log_progress(self, iteration: int, max_loops: int, executed_trades: int, signals_generated: int):
        """Log progress information"""
        progress_pct = (self.current_data_index / len(self.preprocessed_data)) * 100
        account_info = self.simulation_broker.get_account_info()
        
        # Get processor-specific stats
        processor_stats = self.signal_handler.get_processor_stats()
        filter_stats_str = ""
        
        if 'signals_last_hour' in processor_stats:
            filter_stats_str = f" | Filter: {processor_stats['signals_last_hour']}/hr"
        elif 'signals_last_minute' in processor_stats:
            filter_stats_str = f" | Smart: {processor_stats['signals_last_minute']}/min"
        elif 'signals_last_second' in processor_stats:
            filter_stats_str = f" | Ultra: {processor_stats['signals_last_second']}/sec"
        
        logger.info(f"Progress: {progress_pct:.1f}% | "
                   f"Iteration: {iteration}/{max_loops} | "
                   f"Data: {self.current_data_index}/{len(self.preprocessed_data)} | "
                   f"Equity: ${account_info['current_equity']:.2f} | "
                   f"Trades: {executed_trades} | "
                   f"Signals: {signals_generated}{filter_stats_str}")
    
    def _log_final_summary(self):
        """Log final summary of backtest results"""
        summary = self.backtest_results['backtest_summary']
        performance = self.backtest_results['trading_performance']
        
        logger.info(f"📊 Final Results:")
        logger.info(f"  - Total Return: {summary['total_return_pct']:.2f}%")
        logger.info(f"  - Total Trades: {performance['total_trades']}")
        logger.info(f"  - Win Rate: {performance['win_rate']:.1f}%")
        logger.info(f"  - Max Drawdown: {performance['max_drawdown_pct']:.2f}%")
        
        if 'risk_metrics' in self.backtest_results:
            risk = self.backtest_results['risk_metrics']
            logger.info(f"  - Sharpe Ratio: {risk.get('sharpe_ratio', 'N/A')}")
            logger.info(f"  - Profit Factor: {risk.get('profit_factor', 'N/A')}")
    
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