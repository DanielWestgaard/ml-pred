import logging
import pandas as pd
# ML Model
from xgboost import XGBClassifier
# Read data
from sklearn.datasets import load_iris
from sktime.split import temporal_train_test_split

from model.model_factory.base_model import BaseModel

class XgboostModel(BaseModel):
    """
    XGBoost Model:
    
    XGBoost (eXtreme Gradient Boosting) is a powerful, open-source machine learning library that implements
    gradient boosting algorithms. It's known for its speed, efficiency, and ability to handle large datasets.
    XGBoost is widely used for both classification and regression tasks.
    """
    def __init__(self, dataset:pd.DataFrame):
        """Initialize XGBoost."""
        logging.debug(f"----- Initializing XGBoost Model -----")
        
        self.df = dataset
    
    def train(self):
        """Train the XGBoost model."""
        logging.info(f"---------- Starting to train XGBoost Model! ----------")