import logging
import os
import pickle
import sys
from abc import ABC, abstractmethod
from utils.model.model_metadata_loader import load_model_metadata, get_latest_model
import config.config as config
import utils.shared_utils as shared_utils
from strategy.components.signal_processor import SignalProcessor
from strategy.components.risk_manager import RiskManager
from strategy.components.position_manager import PositionManager
from strategy.components.order_manager import OrderManager

class BaseStrategy(ABC):
    """
    Abstract base class for all hypothesis-specific strategies.
    Contains shared functionality while allowing for hypothesis-specific implementations.
    """
    
    def __init__(self, model_directory: str = None, save_results: bool = True, 
                 model_type: str = None, hypothesis_config: str = None):
        """
        Initialize the base strategy with common setup.
        """
        logging.info(f"---------- Initializing {self.__class__.__name__} ----------")
        
        # Common model loading logic
        if model_directory is not None:
            self.model_directory = model_directory
        else:
            self.model_directory = get_latest_model(models_directory=config.TRAINED_MODEL_STORAGE, model_type=model_type)
        
        try:
            # Load metadata and model
            self._load_model_and_metadata()
            
            # Load strategy configuration
            self.strategy_config = self._load_strategy_config(hypothesis_config)
            
            # Setup strategy directory
            self.save_results = save_results
            self.strategy_path = os.path.join(self.model_directory, 'strategy')  # TODO: consider also using the config/config.py?
            if not os.path.exists(self.strategy_path):
                os.makedirs(self.strategy_path)
            
            # Initialize strategy components
            self._initialize_components()
                
        except Exception as e:
            logging.error(f"Unable to initialize {self.__class__.__name__}: {e}")
            sys.exit()
    
    def _load_model_and_metadata(self):  # TODO: Also same as in evaluation/base_evaluation.py? refactor to a util method?
        """Common model and metadata loading logic"""
        loader = load_model_metadata(self.model_directory)
        
        # Load model
        model_type = loader.get_model_type()
        model_file = os.path.join(self.model_directory, f"{model_type}.pkl")
        
        with open(model_file, 'rb') as file:
            self.model = pickle.load(file)
        
        # Get target column and hypothesis type
        self.target_col = loader.get_target_column()
        try:
            self.hypothesis_type = loader.get_hypothesis_type()
        except AttributeError:
            logging.warning("get_hypothesis_type() method not found in model metadata, setting to 'unknown'")
            self.hypothesis_type = 'unknown'
    
    def _load_strategy_config(self, hypothesis_config: str = None):
        """Load strategy configuration from hypothesis config or defaults"""
        if hypothesis_config and os.path.exists(hypothesis_config):
            configuration = shared_utils.handle_hypothesis_source(hypothesis_config)
            return configuration.get('strategy', {})
        return {}
    
    def _initialize_components(self):
        """Initialize strategy components with configuration"""
        # Get component configurations
        signal_config = self.strategy_config.get('signal_processing', {})
        risk_config = self.strategy_config.get('risk_management', {})
        position_config = self.strategy_config.get('position_management', {})
        order_config = self.strategy_config.get('order_management', {})
        
        # Initialize components
        self.signal_processor = SignalProcessor(**signal_config)
        self.risk_manager = RiskManager(**risk_config)
        self.position_manager = PositionManager(**position_config)
        self.order_manager = OrderManager(**order_config)
    
    @abstractmethod
    def generate_signals(self, market_data):
        """Generate trading signals from market data - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def execute_strategy(self):
        """Execute the complete strategy logic - must be implemented by subclasses"""
        pass
    
    def run(self):
        """Main orchestrator - calls hypothesis-specific methods"""
        logging.info(f"Starting strategy execution for {self.hypothesis_type}")
        
        # Execute strategy
        results = self.execute_strategy()
        
        # Save results if enabled
        if self.save_results:
            self.save_strategy_results(results)
        
        return results
    
    def save_strategy_results(self, results: dict):
        """Common strategy results saving logic with error handling"""
        if self.save_results:
            import json
            try:
                results_file = os.path.join(self.strategy_path, 'strategy_results.json')
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4, default=str)
                logging.info(f"Strategy results saved to: {results_file}")
            except Exception as e:
                logging.error(f"Unable to save strategy results: {e}")
                try:
                    with open('strategy_results.json', 'w') as f:
                        json.dump(results, f, indent=4, default=str)
                    logging.info(f"Strategy results saved to: strategy_results.json")
                except Exception as e2:
                    logging.error(f"Failed to save strategy results entirely: {e2}")