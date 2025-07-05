import logging
import os
import pickle
import sys
from abc import ABC, abstractmethod
from utils.model.model_utils import load_model_metadata, get_latest_model
from model.model_training import ModelTraining
import config.config as config


class BaseEvaluation(ABC):
    """
    Abstract base class for all hypothesis-specific evaluations.
    Contains shared functionality while allowing for hypothesis-specific implementations.
    """
    
    def __init__(self, model_directory: str = None, save_evaluations: bool = True, model_type: str = None):
        """
        Initialize the base evaluator with common setup.
        """
        logging.info(f"---------- Initializing {self.__class__.__name__} ----------")
        
        # Suppress matplotlib's debug font messages
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        # Common model loading logic
        if model_directory is not None:
            self.model_directory = model_directory
        else:
            self.model_directory = get_latest_model(models_directory=config.TRAINED_MODEL_STORAGE, model_type=model_type)
        
        try:
            # Load metadata and model
            self._load_model_and_data()
            # Setup evaluation directory
            self.save_evaluations = save_evaluations
            self.evaluation_path = os.path.join(self.model_directory, 'evaluation')
            if not os.path.exists(self.evaluation_path):
                os.makedirs(self.evaluation_path)
                
        except Exception as e:
            logging.error(f"Unable to initialize {self.__class__.__name__}: {e}")
            sys.exit()
    
    def _load_model_and_data(self):
        """Common model and data loading logic"""
        loader = load_model_metadata(self.model_directory)
        dataset = loader.get_dataset_path()
        
        if dataset is None or dataset == 'unknown' or not os.path.exists(dataset):
            raise ValueError(f"Invalid dataset path, '{dataset}'")
        
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
        
        # Get train/test splits
        model_training = ModelTraining(dataset=dataset, model_type=model_type, target_col=self.target_col)
        self.y_train, self.y_test, self.X_train, self.X_test = model_training.temporal_train_test_split(return_datasets=True)
    
    @abstractmethod
    def evaluate_model(self):
        """Hypothesis-specific evaluation logic - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def plot_results(self, save_plot: bool = True):
        """Hypothesis-specific plotting logic - must be implemented by subclasses"""
        pass
    
    def run(self):
        """Main orchestrator - calls hypothesis-specific methods"""
        self.evaluate_model()
        self.plot_results(save_plot=self.save_evaluations)
    
    def save_metrics(self, metrics: dict, filename: str):
        """Common metrics saving logic with error handling"""
        if self.save_evaluations:
            import json
            try:
                metrics_file = os.path.join(self.evaluation_path, filename)
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=4)
                logging.info(f"Metrics saved to: {metrics_file}")
            except Exception as e:
                logging.error(f"Unable to save metrics under Models folder. Falling back to current dir: {e}")
                try:
                    with open(filename, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    logging.info(f"Metrics saved to: {filename}")
                except Exception as e2:
                    logging.error(f"Failed to save metrics entirely: {e2}")