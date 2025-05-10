import numpy as np
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

        self.testing_selection()
        print(len(self.df))
        return self.df
    
    def testing_selection(self, target_col = 'close'):
        """Temporary testing method for exploring different solutions."""
        X_processed = model_utils.preprocess_features_for_xgboost(self.df, target_col=target_col, enable_categorical=True)
        # X = self.df.drop(columns=[target_col])  # Everything but close
        y = self.df[target_col]  # close
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=6,
            tree_method='hist',  # Required for categorical support
            enable_categorical=True  # Enable native categorical support
        )
        model.fit(X_train, y_train)
        
        # Method 2: Traditional encoding approach (works with all XGBoost versions)
        # X_processed = preprocess_features_for_xgboost(df, target_col=target_column, enable_categorical=False)

        # # Train with traditional approach
        # model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
        # model.fit(X_processed.drop(target_column, axis=1), X_processed[target_column])

        # Get importance
        importance = model.feature_importances_

        # Summarize feature importance
        for i, v in enumerate(importance):
            print(f'Feature: {i}, Score: {v:.5f}')
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()
        
        print(f"Model Score: {model.score(X, y)}")