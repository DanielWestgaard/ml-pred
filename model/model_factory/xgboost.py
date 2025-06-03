import logging
import sys
import pandas as pd
# ML Model
from xgboost import XGBRegressor
# Read data
from sklearn.datasets import load_iris
from sktime.split import temporal_train_test_split  # https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.split.temporal_train_test_split.html

from model.model_factory.base_model import BaseModel


class XgboostModel(BaseModel):
    """
    XGBoost Model:
    
    XGBoost (eXtreme Gradient Boosting) is a powerful, open-source machine learning library that implements
    gradient boosting algorithms. It's known for its speed, efficiency, and ability to handle large datasets.
    XGBoost is widely used for both classification and regression tasks.
    """
    def __init__(self, dataset:pd.DataFrame, target_col:str = 'close', hypothesis:bool = False):
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
        
        # Initialize train/test targets
        try:
            # Features
            self.X = self.df.drop(target_col, axis=1)
            logging.debug(f"Successfully set features (X), with {len(self.X.columns)} columns/features: {self.X.columns.to_list()} ...")
            # Target
            self.y = self.df[target_col]
            logging.debug(f"and target (y), '{target_col}', {self.y.size} values.")
        except Exception as e:
            logging.error(f"Unable to set train/test targets: {e}")
    
    def train(self):
        """
        Train the XGBoost model.
        Splitting dataset in train/test 80/20.
        """
        logging.info(f"---------- Starting to train XGBoost Model! ----------")
        
        y_train, y_test, X_train, X_test = temporal_train_test_split(y=self.y, X=self.X, test_size=0.2, train_size=0.8)  # sktime's returns values in a different order than sklearn's train_test_split

        # Create model instance
        bst = XGBRegressor(n_estimators=2, max_depth=2, learning_rate=1, objective='reg:squarederror')
        
        # Fit (train) model
        bst.fit(X_train, y_train)
        
        # Make predictions
        preds = bst.predict(X_test)
        print(preds)
