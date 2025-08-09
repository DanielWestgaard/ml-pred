import os
import yaml

from utils.model.model_metadata_loader import load_model_metadata
import utils.hypothesis.config_loader as config_loader
from strategy.strategy_factory import StrategyFactory
from utils.logging_utils import get_logger

logger = get_logger()

class StrategyOrchestrator:
    """
    Main orchestrator for strategy execution. 
    Determines which hypothesis-specific strategy to use.
    """
    
    def __init__(self, model_directory: str = None, save_results: bool = True, 
                 model_type: str = None, hypothesis_config: str = None):
        """
        Initialize strategy orchestrator
        
        Args:
            model_directory: Path to model directory
            save_results: Whether to save strategy results
            model_type: Model type (for finding latest model)
            hypothesis_config: Path to hypothesis config file (optional)
        """
        self.model_directory = model_directory
        self.save_results = save_results
        self.model_type = model_type
        self.hypothesis_config = hypothesis_config
        
        # Determine hypothesis type
        self.hypothesis_type = self._determine_hypothesis_type()
        
        # Create appropriate strategy
        self.strategy = StrategyFactory.create_strategy(
            hypothesis_type=self.hypothesis_type,
            model_directory=model_directory or config_loader.load_hypothesis_config(hypothesis_config).handle_hypothesis_source().get("strat_model_folder"),
            save_results=save_results,
            model_type=model_type,
            hypothesis_config=hypothesis_config
        )
    
    def _determine_hypothesis_type(self):  # TODO: Might be made into a utils method to avoid duplication with evaluation/evaluation.py and multi_data_sources/ds_orchestrator.py ?
        """
        Determine hypothesis type from config or model metadata
        """
        # Option 1: From hypothesis config file
        if self.hypothesis_config and os.path.exists(self.hypothesis_config):
            configuration = config_loader.load_hypothesis_config(self.hypothesis_config).handle_hypothesis_source()
            return configuration.get('hypothesis_name')
        
        # Option 2: From model metadata
        elif self.model_directory:
            try:
                loader = load_model_metadata(self.model_directory)
                return loader.get_hypothesis_type()
            except:
                pass
        
        # Option 3: Default fallback
        logger.warning("Could not determine hypothesis type, defaulting to 'session_based_tod_anomalies'")
        return 'session_based_tod_anomalies'
    
    def run(self):
        """Run the appropriate strategy"""
        logger.info(f"Running strategy for hypothesis type: {self.hypothesis_type}")
        return self.strategy.run()
    
    def stop(self):
        """Stop the strategy gracefully"""
        if hasattr(self.strategy, 'stop') and callable(getattr(self.strategy, 'stop')):
            logger.info("Stopping strategy...")
            self.strategy.stop()
        elif hasattr(self.strategy, '_cleanup_resources') and callable(getattr(self.strategy, '_cleanup_resources')):
            logger.info("Cleaning up strategy resources...")
            self.strategy._cleanup_resources()
        else:
            logger.warning("Strategy does not have a stop() or _cleanup_resources() method")