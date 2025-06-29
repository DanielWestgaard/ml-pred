import logging
import yaml
from utils.model_utils import load_model_metadata
# from .evaluation_factory import EvaluationFactory

class ModelEvaluation:
    """
    Main orchestrator for model evaluation. 
    Determines which hypothesis-specific evaluator to use.
    """
    
    def __init__(self, model_directory: str = None, save_evaluations: bool = True, 
                 model_type: str = None, hypothesis_config: str = None):
        """
        Initialize evaluation orchestrator
        
        Args:
            model_directory: Path to model directory
            save_evaluations: Whether to save results
            model_type: Model type (for finding latest model)
            hypothesis_config: Path to hypothesis config file (optional)
        """
        self.model_directory = model_directory
        self.save_evaluations = save_evaluations
        self.model_type = model_type
        self.hypothesis_config = hypothesis_config
        
        # Determine hypothesis type
        self.hypothesis_type = self._determine_hypothesis_type()
        
        # Create appropriate evaluator
        self.evaluator = EvaluationFactory.create_evaluator(
            hypothesis_type=self.hypothesis_type,
            model_directory=model_directory,
            save_evaluations=save_evaluations,
            model_type=model_type
        )
    
    def _determine_hypothesis_type(self):
        """
        Determine hypothesis type from config or model metadata
        """
        # Option 1: From hypothesis config file
        if self.hypothesis_config and os.path.exists(self.hypothesis_config):
            with open(self.hypothesis_config, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('hypothesis_type')
        
        # Option 2: From model metadata
        if self.model_directory:
            try:
                loader = load_model_metadata(self.model_directory)
                return loader.get_hypothesis_type()  # You'll need to add this method
            except:
                pass
        
        # Option 3: Default fallback
        logging.warning("Could not determine hypothesis type, defaulting to 'session_volume_anomaly'")
        return 'session_volume_anomaly'
    
    def run(self):
        """Run the appropriate evaluation"""
        logging.info(f"Running evaluation for hypothesis type: {self.hypothesis_type}")
        return self.evaluator.run()
