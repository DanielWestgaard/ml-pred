import logging
import sys
import pandas as pd
# ML Model
from xgboost import XGBRegressor
# Read data
from sklearn.datasets import load_iris

from model.model_factory.base_model import BaseModel
import utils.model_utils as model_utils


class XgboostModel(BaseModel):
    """
    XGBoost Model:
    
    XGBoost (eXtreme Gradient Boosting) is a powerful, open-source machine learning library that implements
    gradient boosting algorithms. It's known for its speed, efficiency, and ability to handle large datasets.
    XGBoost is widely used for both classification and regression tasks.
    """
    def __init__(self, dataset:pd.DataFrame, X_train:pd.DataFrame, y_train:pd.DataFrame):
        """
        Initialize XGBoost.
        
        Args:
            - dataset: The (processed/engineered) data the model is to be trained (and tested) on. Must be pd.DataFrame.
            - target_col: Target column. Eg. 'close' or hypothesis column. Default is 'close'.
            - TODO: hypothesis: Specify wether to target hypothesis column. 
                - If True; Code will use the column that has prefix 'target_' (for target_column, y), as per 
                implementation of target_hypothesis.py. As hypothesis is calculated and created with target_
                first in the name.
                - If False (default); Code will use 'target_col' as target (y).
        
        Returns:
            - If successfull returns trained model, else None.
        """
        logging.debug(f"----- Initializing XGBoost Model -----")
        
        self.df = dataset
        self.X_train = X_train
        self.y_train = y_train
    
    def train(self):
        """
        Train the XGBoost model.
        Splitting dataset in train/test 80/20.
        """
        logging.info(f"---------- Starting to train XGBoost Model! ----------")

        # Create model instance
        xgb = XGBRegressor(n_estimators=100,                # More trees: number of trees in the ensemble, often increased until no further improvements are seen
                           max_depth=12,                    # Deeper trees: maximum depth of each tree, often values are between 1 and 10
                           learning_rate=0.001,
                           objective='reg:squarederror',    # 'Perfect' for price prediction
                           subsample=0.8,                   # Random sampling to prevent overfitting
                           colsample_bytree=0.8,            # Feature sampling
                           reg_alpha=0.1,                   # L1 regularization
                           reg_lambda=1.0,                  # L2 regularization
                           random_state=42,
                           enable_categorical=True)         # handling categorical features
        
        # Fit (train) model
        model = xgb.fit(self.X_train, self.y_train)
        
        logging.info(f"Successfully trained XGBRegressor Model!")
        return model
    
    def save_model_and_info(self):
        """Save trained model and information (see model documentation for details)."""
        pass