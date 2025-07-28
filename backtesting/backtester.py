import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time
from datetime import datetime

from utils.logging_utils import get_logger
from utils.hypothesis.config_loader import load_hypothesis_config
import pickle
from utils.model.model_metadata_loader import load_model_metadata
from utils.model.model_type_detector import ModelTypeDetector, PredictionProcessor, get_model_type
from data.features.enhanced_generation import ModelAwareFeatureGenerator
from strategy.hypothesis_specific.spread_compression_strategy import SpreadCompressionSignalTracker

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
        self.model_type = None
        self.prediction_processor = None
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
        
        # Preprocessed data for backtesting
        self.preprocessed_data = None
        self.current_data_index = 0
        
        # Load configuration and initialize components
        self._load_configuration()
        self._load_model()
        self._detect_model_type()
        self._load_preprocessed_data()
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
                # Extract model folder from backtesting config (consistent with strategy approach)
                backtesting_config = self.config.get('backtesting', {})
                self.model_folder = backtesting_config.get('model_folder')
                logger.debug(f"Model Folder from backtesting config: {self.model_folder}")
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
            # Determine model path (consistent with strategy approach)
            if self.model_folder:
                model_path = self.model_folder
            else:
                # Try to get model folder from config sections
                backtesting_config = self.config.get('backtesting', {})
                model_training_config = self.config.get('model_training', {})
                strategy_config = self.config.get('strategy', {})
                
                # Check different config locations for model folder
                model_path = (
                    backtesting_config.get('model_folder') or
                    model_training_config.get('model_folder') or
                    strategy_config.get('model_folder')
                )
                
                if not model_path:
                    # Use existing utility to find most recent model for the hypothesis
                    from utils.model.model_metadata_loader import get_latest_model
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
            logger.info(f"Model class: {model_type}")
            
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
    
    def _detect_model_type(self):
        """Detect and set up model type and prediction processor"""
        try:
            # Try multiple detection methods
            model_metadata_path = Path(self.model_metadata['model_info']['storage_path']) / 'model_metadata.yml'
            
            self.model_type = get_model_type(
                model=self.model,
                metadata_path=model_metadata_path,
                config=self.config
            )
            
            if self.model_type is None:
                # Fallback to model object detection
                detector = ModelTypeDetector()
                self.model_type = detector.detect_from_model(self.model)
            
            # Initialize prediction processor
            self.prediction_processor = PredictionProcessor(self.model_type)
            
            logger.info(f"Model type detected: {self.model_type.value}")
            
        except Exception as e:
            logger.error(f"Error detecting model type: {e}")
            # Fallback to regression
            from utils.model.model_type_detector import ModelType
            self.model_type = ModelType.REGRESSION
            self.prediction_processor = PredictionProcessor(self.model_type)
            logger.warning("Defaulting to regression model type")
    
    def _load_preprocessed_data(self):
        """Load and filter preprocessed data for backtesting"""
        try:
            # Get data source from config
            backtest_config = self.config.get('backtesting', {})
            data_source = backtest_config.get('data_source')
            
            if not data_source:
                # Try to get from model metadata
                data_source = self.model_metadata.get('dataset', {}).get('dataset_path')
            
            if not data_source:
                raise ValueError("No data source specified in config or model metadata")
            
            logger.info(f"Loading preprocessed data from: {data_source}")
            
            # Load full dataset
            full_data = pd.read_csv(data_source)
            
            # Use 'date' column as timestamp and rename for consistency
            if 'date' in full_data.columns:
                full_data['timestamp'] = pd.to_datetime(full_data['date'], utc=True)  # Make timezone-aware to match start/end dates
            else:
                raise ValueError("No 'date' column found in preprocessed data")
            
            # Filter by date range
            start_date = backtest_config.get('start_date')
            end_date = backtest_config.get('end_date')
            
            if start_date:
                start_date = pd.to_datetime(start_date, utc=True)  # Make timezone-aware
                full_data = full_data[full_data['timestamp'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date, utc=True)  # Make timezone-aware
                full_data = full_data[full_data['timestamp'] <= end_date]
            
            # Sort by timestamp
            full_data = full_data.sort_values('timestamp').reset_index(drop=True)
            
            self.preprocessed_data = full_data
            self.current_data_index = 0
            
            logger.info(f"Loaded {len(self.preprocessed_data)} preprocessed data points")
            logger.info(f"Date range: {self.preprocessed_data['timestamp'].min()} to {self.preprocessed_data['timestamp'].max()}")
            
            # Log available features
            model_features = self.model_metadata.get('dataset', {}).get('features', [])
            available_features = [f for f in model_features if f in self.preprocessed_data.columns]
            missing_features = [f for f in model_features if f not in self.preprocessed_data.columns]
            
            logger.info(f"Available features: {len(available_features)}/{len(model_features)}")
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            raise
    
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
            signal_config = self.config.get('backtesting', {}).get('signal_processing', {})
            self.signal_processor = SignalProcessor(
                confidence_threshold=signal_config.get('confidence_threshold', 0.6),
                signal_filters=signal_config.get('signal_filters', []),
                model_type=self.model_type
            )
            
            risk_config = self.config.get('backtesting', {}).get('risk_management', {})
            self.risk_manager = RiskManager(
                max_position_size=risk_config.get('max_position_size', 0.02),
                stop_loss_pct=risk_config.get('stop_loss_pct', 0.01),
                take_profit_pct=risk_config.get('take_profit_pct', 0.02),
                max_drawdown=risk_config.get('max_drawdown', 0.05),
                max_concurrent_positions=risk_config.get('max_concurrent_positions', 3)
            )
            
            position_config = self.config.get('backtesting', {}).get('position_management', {})
            self.position_manager = PositionManager(
                max_concurrent_positions=position_config.get('max_concurrent_positions', 3),
                position_tracking=position_config.get('position_tracking', 'detailed')
            )
            
            # Initialize intelligent exit strategy for spread compression
            hypothesis_name = self.config.get('hypothesis_name', '')
            if 'spread_compression' in hypothesis_name.lower():
                signal_tracking_config = self.config.get('backtesting', {}).get('signal_tracking', {})
                self.signal_tracker = SpreadCompressionSignalTracker(signal_tracking_config)
                
                # Exit strategy configuration
                exit_config = self.config.get('backtesting', {}).get('exit_strategy', {})
                self.confidence_exit_threshold = exit_config.get('confidence_exit_threshold', 0.30)
                self.signal_decay_threshold = exit_config.get('signal_strength_decay', 0.25)
                self.volatility_exit_multiplier = exit_config.get('volatility_exit_multiplier', 1.5)
                self.minimum_signal_life = exit_config.get('minimum_signal_life', 3)  # minutes
                
                logger.info(f"Initialized intelligent exit strategy for spread compression:")
                logger.info(f"  - Confidence exit threshold: {self.confidence_exit_threshold}")
                logger.info(f"  - Signal decay threshold: {self.signal_decay_threshold}")
            else:
                self.signal_tracker = None
            
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
            
            # Main backtesting loop using preprocessed data
            iteration = 0
            max_loops = self.config.get('backtesting', {}).get('execution', {}).get('max_loops', 10000)
            total_data_points = len(self.preprocessed_data)
            
            while self.current_data_index < total_data_points and self.is_running and iteration < max_loops:
                iteration += 1
                
                # Get current data point from preprocessed data
                current_row = self.preprocessed_data.iloc[self.current_data_index]
                
                # Extract market data (OHLCV) - use 'date' as timestamp
                # Use original price columns (_og) since main columns are normalized
                current_data = {
                    'timestamp': current_row['date'],  # Use 'date' column
                    'open': current_row.get('open', current_row.get('close_og', 0)),
                    'high': current_row.get('high', current_row.get('close_og', 0)),
                    'low': current_row.get('low', current_row.get('close_og', 0)),
                    'close': current_row.get('close_og', 0),  # Use actual price, not normalized
                    'volume': current_row.get('volume', 0)
                }
                
                # Skip if we have invalid price data
                if current_data['close'] <= 0:
                    logger.debug(f"Skipping invalid price data at index {self.current_data_index}")
                    self.current_data_index += 1
                    continue
                
                # Use preprocessed features directly (no feature generation needed!)
                try:
                    # Get exact features the model expects (from model.feature_names_in_)
                    if hasattr(self.model, 'feature_names_in_'):
                        prediction_features_list = list(self.model.feature_names_in_)
                    else:
                        # Fallback: use features from metadata, but exclude only the binary target
                        model_features = self.model_metadata.get('dataset', {}).get('features', [])
                        target_column = self.model_metadata.get('dataset', {}).get('target_column', '')
                        prediction_features_list = [f for f in model_features if f != target_column]
                    
                    # Check for missing features in current data
                    missing_features = [f for f in prediction_features_list if f not in current_row.index]
                    if missing_features:
                        logger.debug(f"Missing features for prediction: {missing_features[:5]}...")  # Show first 5
                        self.current_data_index += 1
                        continue
                    
                    # Extract features for this data point in the exact order expected by model
                    prediction_features = current_row[prediction_features_list].to_frame().T
                    
                    # Process predictions using the prediction processor
                    prediction_results = self.prediction_processor.process_predictions(
                        self.model, prediction_features
                    )
                    
                    predictions = prediction_results['predictions']
                    probabilities = prediction_results.get('positive_probabilities')
                    
                    # Process prediction into trading signals (no data_buffer needed for preprocessed data)
                    signals = self.signal_processor.process(
                        predictions, 
                        market_data=None,  # No market data buffer needed
                        model=self.model,
                        prediction_probabilities=probabilities
                    )
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
                                
                                # Track entry features for intelligent exits (spread compression specific)
                                if self.signal_tracker:
                                    position_id = trade_result['position_id']
                                    entry_features = {
                                        'compression_ratio_micro': current_data.get('compression_ratio_micro', 0),
                                        'volume_ratio': current_data.get('volume_ratio', 0),
                                        'signal_strength_v2': current_data.get('signal_strength_v2', 0),
                                        'compression_consistency': current_data.get('compression_consistency', 0)
                                    }
                                    
                                    # Store entry features in position (via simulation broker)
                                    position = self.simulation_broker.get_position(position_id)
                                    if position:
                                        position['entry_features'] = entry_features
                                        position['entry_time'] = pd.to_datetime(current_data['timestamp'], utc=True)
                                    
                                    # Track with signal tracker
                                    confidence = probabilities[0] if probabilities is not None and len(probabilities) > 0 else 0.8
                                    self.signal_tracker.track_entry_signal(position_id, current_data, confidence)
                                
                                # Position tracking handled by SimulationBroker
                                logger.debug(f"Trade executed successfully: {trade_result['position_id']}")
                    
                    # Update existing positions
                    self.simulation_broker.update_positions(
                        current_price=current_data['close'],
                        timestamp=current_data['timestamp']
                    )
                    
                    # Check for position closes (stop loss / take profit / model signal)
                    self._check_position_closes(current_data, predictions, probabilities)
                    
                    # Update risk management
                    account_info = self.simulation_broker.get_account_info()
                    self.risk_manager.update_drawdown(account_info['current_equity'])
                    
                    # Move to next data point
                    self.current_data_index += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing data point {iteration}: {e}")
                    self.current_data_index += 1
                    continue
                
                # Progress logging
                if iteration % progress_interval == 0:
                    progress_pct = (self.current_data_index / total_data_points) * 100
                    account_info = self.simulation_broker.get_account_info()
                    logger.info(f"Progress: {progress_pct:.1f}% | "
                               f"Iteration: {iteration}/{max_loops} | "
                               f"Data: {self.current_data_index}/{total_data_points} | "
                               f"Equity: ${account_info['current_equity']:.2f} | "
                               f"Trades: {executed_trades} | "
                               f"Signals: {signals_generated}")
            
            # Close all remaining positions at end of backtest
            if self.current_data_index > 0:
                final_row = self.preprocessed_data.iloc[self.current_data_index - 1]
                self.simulation_broker.close_all_positions(
                    current_price=final_row['close'],
                    timestamp=final_row['date']
                )
            
            # Calculate final results
            self.backtest_results = self._calculate_results()
            
            # Save results
            self._save_results()
            
            execution_time = time.time() - self.start_time
            logger.info(f"✅ Backtest completed in {execution_time:.2f} seconds")
            logger.info(f"📊 Final Results: Iterations: {iteration}/{max_loops} | "
                       f"Return: {self.backtest_results['backtest_summary']['total_return_pct']:.2f}%, "
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
        
        # Get target column from model metadata  
        target_column = self.model_metadata.get('dataset', {}).get('target_column', '')
        
        # Filter out only the target column - all other columns from model metadata are features
        required_features = [f for f in features if f != target_column]
        
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
    
    def _check_position_closes(self, current_data: Dict[str, Any], predictions: np.ndarray = None, probabilities: np.ndarray = None):
        """Check and execute position closes based on stop loss / take profit / intelligent model signal"""
        open_positions = self.simulation_broker.get_open_positions()
        
        for position in open_positions:
            # Check risk management rules (stop loss / take profit)
            close_info = self.risk_manager.should_close_position(
                position, current_data['close']
            )
            
            # Check for intelligent exit strategy (spread compression specific)
            if not close_info['should_close'] and predictions is not None and len(predictions) > 0:
                latest_prediction = predictions[0]  # Most recent prediction
                latest_probability = probabilities[0] if probabilities is not None else None
                
                # Use intelligent exit strategy if available
                hypothesis_name = self.config.get('hypothesis_name', '')
                if 'spread_compression' in hypothesis_name.lower():
                    should_exit, exit_reason = self._check_intelligent_exit(
                        position, current_data, latest_prediction, latest_probability
                    )
                    if should_exit:
                        close_info['should_close'] = True
                        close_info['reason'] = exit_reason
                else:
                    # Fallback to original model exit logic for other hypotheses
                    if latest_prediction == 0 and (latest_probability is None or (1 - latest_probability) > 0.7):
                        close_info['should_close'] = True
                        close_info['reason'] = 'model_exit_signal'
            
            if close_info['should_close']:
                self.simulation_broker.close_position(
                    position_id=position['id'],
                    current_price=current_data['close'],
                    timestamp=current_data['timestamp'],
                    reason=close_info['reason']
                )
                # Position tracking handled by SimulationBroker
                logger.debug(f"Position closed: {position['id']} - {close_info['reason']}")
    
    def _check_intelligent_exit(self, position: Dict, current_data: Dict, prediction: int, probability: float) -> Tuple[bool, str]:
        """
        Intelligent exit logic specific to spread compression patterns.
        
        Returns:
            Tuple of (should_exit: bool, exit_reason: str)
        """
        if not self.signal_tracker:
            # Fallback to original logic if no signal tracker
            if prediction == 0 and (probability is None or (1 - probability) > 0.7):
                return True, 'model_exit_signal'
            return False, None
        
        position_id = position.get('id', 'unknown')
        current_confidence = probability if prediction == 1 else (1 - probability) if probability else 0
        
        # Update signal tracking
        self.signal_tracker.update_signal_strength(position_id, current_data, current_confidence)
        
        # Check minimum signal life (not a hard constraint, just a filter)
        entry_time = position.get('entry_time')
        if entry_time:
            from datetime import datetime
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            position_age_minutes = (pd.Timestamp.now(tz='UTC') - entry_time).total_seconds() / 60
            if position_age_minutes < self.minimum_signal_life:
                # Only exit on very strong contrary signals during minimum life period
                if prediction == 0 and current_confidence > 0.9:
                    return True, "strong_contrary_signal"
                return False, None
        
        # 1. Confidence-based exit: Exit when model predicts compression is over (class 0) with sufficient confidence
        if prediction == 0 and current_confidence > self.confidence_exit_threshold:
            logger.debug(f"Confidence exit triggered: {current_confidence:.4f} > {self.confidence_exit_threshold}")
            return True, "confidence_drop"
        
        # 2. Signal strength decay: Exit when signal deteriorates significantly
        signal_decay = self.signal_tracker.calculate_signal_decay(position_id)
        if signal_decay > self.signal_decay_threshold:
            logger.debug(f"Signal decay exit triggered: {signal_decay:.4f} > {self.signal_decay_threshold}")
            return True, "signal_decay"
        
        # 3. Feature-based exit: Exit when core compression features reverse
        if self._check_compression_reversal(position, current_data):
            return True, "compression_reversal"
        
        return False, None
    
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