from abc import ABC, abstractmethod


class BaseModel(ABC):
    """XGBoost Model"""
    @abstractmethod
    def __init__(self):
        """Initialize the Model."""
        pass
    
    @abstractmethod
    def train(self):
        """Train the Model."""
        pass
    
    # @abstractmethod
    # def load_trained_model(self, path_to_model):
    #     """Load a pre-trained model for usage."""
    #     pass