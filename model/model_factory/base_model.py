from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """XGBoost Model"""
    @abstractmethod
    def __init__(self, dataset:pd.DataFrame, target_col:str = 'close', hypothesis:bool = False):
        """Initialize the Model."""
        pass
    
    @abstractmethod
    def train(self):
        """Train the Model."""
        pass
    
    @abstractmethod
    def save_model_and_info(self):
        """Save trained model and information (see model documentation for details)."""
        pass
    
    # @abstractmethod
    # def load_trained_model(self, path_to_model):
    #     """Load a pre-trained model for usage."""
    #     pass