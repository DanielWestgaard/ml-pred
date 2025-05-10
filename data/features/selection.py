import numpy as np
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb

from data.processing.base_processor import BaseProcessor
from utils import data_utils


class FeatureSelector(BaseProcessor):
    def __init__(self, data):
        """Class for Selecting the most important features."""
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(data)
    
    def run(self):
        """Run the feature selection process."""
        X, y = self._prepare_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)

        # Get importance
        importance = model.feature_importances_

        # Summarize feature importance
        for i, v in enumerate(importance):
            print(f'Feature: {i}, Score: {v:.5f}')
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()
        
        print(f"Model Score: {model.score(X_test, y_test)}")
        
        return self.df
    
    def _prepare_training_data(self, lookback=30):
        """Simple method to get x- and y-train."""
        X, y = [], []
        for i in range(len(self.df) - lookback):
            X.append(self.df.iloc[i:i+lookback]['close'].values)
            y.append(self.df.iloc[i+lookback]['close'])
        return np.array(X), np.array(y)