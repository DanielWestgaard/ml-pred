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
            # Need to change potential columns having dtype object to category, for model to 'accept'
            self.handle_categories()
            
            # Features
            self.X = self.df.drop(target_col, axis=1)
            logging.debug(f"Successfully set Features (X), with {len(self.X.columns)} columns/features ...")
            # Target
            self.y = self.df[target_col]
            logging.debug(f"and Target (y), '{target_col}', {self.y.size} values.")
        except Exception as e:
            logging.error(f"Unable to set train/test targets: {e}")
    
    def train(self):
        """
        Train the XGBoost model.
        Splitting dataset in train/test 80/20.
        """
        logging.info(f"---------- Starting to train XGBoost Model! ----------")

        self._ensure_correct_xy_format()

        y_train, y_test, X_train, X_test = temporal_train_test_split(y=self.y, X=self.X, test_size=0.2, train_size=0.8)  # sktime's returns values in a different order than sklearn's train_test_split

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
        xgb.fit(X_train, y_train)
        
        # Make predictions
        preds = xgb.predict(X_test)
        print(preds)

    def handle_categories(self):
        """
        Handle columns of object type, and make them categorical type.
        The reason for this need, is that xgboost don't accept columns of object.
        """
        # list for keeping track of changed columns
        column_object_list = []
        
        # Loop through df to find columns of object type, and replace with categorical.
        for col in self.df.columns:
            if self.df[col].dtype == object:
                try:
                    self.df[col] = self.df[col].astype('category')
                    column_object_list.append(col)
                    logging.debug(f"Successfully changed {col} dtype from object to category!")
                except Exception as e:
                    logging.error(f"Will drop {col}: Unable to change dtype of {col} from object to categorical: {e}")
                    self.df = self.df.drop(col, axis=1)
    
    
    def _ensure_correct_xy_format(self):
        """Temporary class used to check if the formats (dtype) is valid or not."""
        from sktime.datatypes import check_raise

        # Check your current y format
        try:
            check_raise(self.y, mtype='pd.Series')
            logging.debug("✓ y is in correct format")
        except Exception as e:
            logging.ERROR(f"✗ Error with y: {e}")

        # Check your current X format  
        try:
            check_raise(self.X, mtype='pd.DataFrame')
            logging.debug("✓ X is in correct format")
        except Exception as e:
            logging.error(f"✗ Error with X: {e}")

        # Let's also inspect your index
        logging.debug(f"Index type: {type(self.y.index)}")
        logging.debug(f"Index dtype: {self.y.index.dtype}")
        logging.debug(f"Index sample: {self.y.index[:5]}")
        logging.debug(f"Index frequency: {self.y.index.freq}")
    