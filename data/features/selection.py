import logging
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb

from data.processing.base_processor import BaseProcessor
from utils import data_utils
from utils import model_utils


class FeatureSelector(BaseProcessor):
    def __init__(self, data):
        """Class for Selecting the most important features."""
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(data)
    
    def run(self):
        """Run the feature selection process."""

        self.xgb_regressor()
        
        return self.df
    
    def xgb_regressor(self, target_col = 'close', threshold = 0.01):
        """Select features using XGB Regressor."""
        X_processed = model_utils.preprocess_features_for_xgboost(self.df, target_col=target_col, enable_categorical=False)
        y = self.df[target_col]  # close
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=6,
            tree_method='hist',  # Required for categorical support
        )
        model.fit(X_train, y_train)

        # Get importance
        importance = model.feature_importances_
        importance_df = pd.DataFrame(columns=['feature', 'score'])
        # Summarize feature importance
        for i, v in enumerate(importance):
            new_row = pd.DataFrame([{'feature': i, 'score': f"{v:.5f}"}])
            importance_df = pd.concat([importance_df, new_row], ignore_index=True)

        print(importance_df)
            
        # Get indices of features above threshold
        important_indices = [i for i, v in enumerate(importance) if v > threshold]
        
        # Select only important features
        self.df = self.df.iloc[:, important_indices]
        logging.debug(f"Selected {len(self.df.columns)} features above threshold {threshold}: {self.df.columns}")
        
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()
        
        logging.info(f"Model Score: {model.score(X_processed, y)}")