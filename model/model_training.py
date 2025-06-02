import logging

from utils import data_utils


class ModelTraining():
    """Train a specific ML Model."""
    def __init__(self, dataset, model_type:str):
        """
        Initialize the class by providing specifics.
        
        Args:
            - dataset: Path to the Engineered (ohlcv) dataset to train the model on.
            - model_type: Type of model you want to train. Eg. 'xgboost', 'decision_tree', 'random_forest', etc.
        """
        try:
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            self.model_type = model_type
        except Exception as e:
            logging.error(f"Error initializing Model Training: {str(e)}")
            raise
    
    def train(self):
        """
        Train initialized model type. Successfully training the model will store in according to
        the config.py-file.
        
        Returns:
            - If successful, will return the path of stored model, else False.
        """
        pass