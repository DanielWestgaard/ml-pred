import logging
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, RFE, RFECV
import xgboost as xgb
from sklearn.svm import SVR

from data.processing.base_processor import BaseProcessor
from utils import data_utils
from utils import model_utils


class FeatureSelector(BaseProcessor):
    def __init__(self, data, target_col:str = 'close'):
        """
        Class for Selecting the most important features. Focusing on Correlation-based Feature Selection, 
        Tree-based Feature Importance and Recursive Feature Elimination.
        """
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(data)
        self.target_col = target_col
        
        # Getting X- and y-train for selection process/-es
        self.X = model_utils.preprocess_features_for_xgboost(df=self.df, enable_categorical=True)  # self.original_df.drop([target_col], axis=1)
        self.y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    
    def run(self):
        """
        Run the feature selection process.
        
        - First use correlation analysis to remove redundant features
        - Then apply tree-based importance to get an initial feature ranking
        - Finally use RFE with your actual model to fine-tune the selection
        """
        print()
        print()
        # self.cfs()
        # self.xgb_regressor()
        self.rfe()
        
        return self.df
    
    def cfs(self, k:int = 15):
        """
        Correlation-based Feature Selection: selects subsets of features that are 
        highly correlated with the target variable but have low correlation with each other.
        Link: https://medium.com/@sariq16/correlation-based-feature-selection-in-a-data-science-project-3ca08d2af5c6
        
        Params:
            k: Number of features the method will select - select the top k features.
        """
        try: 
            # Calculate correlation matrix
            corr_matrix = self.df.corr()
            corr_with_target = corr_matrix[self.target_col]
            # Sort correlation values in descending order and select top k features
            top_k = corr_with_target.abs().sort_values(ascending=False)[:k].index
            selected_features = self.df[top_k]
            
            # Correlation between selected k features
            selected_corr_matrix = selected_features.corr()
        except Exception as e:
            logging.error(f"Unable to perform Correlation-based Feature Selection: {e}")
    
    def xgb_regressor(self, threshold = 0.01):
        """Select features using XGB Regressor."""
        try:
            model = xgb.XGBRegressor(
                n_estimators=100, 
                max_depth=6,
                tree_method='hist',  # Required for categorical support
                enable_categorical=True
            )
            model.fit(self.X_train, self.y_train)
            
            # Get importance
            importance = model.feature_importances_
            importance_df = pd.DataFrame(columns=['feature', 'score'])
            # Summarize feature importance
            for i, v in enumerate(importance):  # just for educational purposes - can be removed
                new_row = pd.DataFrame([{'feature': i, 'score': f"{v:.5f}"}])
                importance_df = pd.concat([importance_df, new_row], ignore_index=True)
            
            # Get indices of features above threshold
            important_indices = [i for i, v in enumerate(importance) if v > threshold]
                    
            # Select only important features
            self.df = self.df.iloc[:, important_indices]
            logging.debug(f"Selected {len(self.df.columns)} features above threshold {threshold}: {self.df.columns}")
                    
            plt.bar([x for x in range(len(importance))], importance)
            plt.show()
                    
            # Change this line from self.X_processed to self.X_test
            logging.info(f"Model Score: {model.score(self.X_test, self.y_test)}")
        except Exception as e:
            logging.error(f"Unable to use XGB Regression for feature selection: {e}")
        
    def rfe(self):
        """
        Recursive Feature Elimination (Cross Validation): Fits to a model and removes the weakest featues
        until the specified number of features  is reached.
        """
        try:
            estimator = SVR(kernel="linear")
            # Time-based cross validation to avoid lookahed bias
            cv = TimeSeriesSplit(n_splits=5)
            # RFECV to help determine the optimal number of features rather than fixed feature set
            selector = RFECV(estimator, step=1, cv=cv, scoring='neg_mean_squared_error')
            selector = selector.fit(self.X, self.y)
            # print(selector.support_)
            # print(selector.ranking_)
            
            # Storing the selected features
            selected_features = self.X.columns[selector.support_]
            return selected_features
        except Exception as e:
            logging.error(f"Unable to perform Recursive Feature Elimination: {e}")