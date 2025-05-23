import logging
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, RFE, RFECV
import xgboost as xgb
from sklearn.svm import SVR
import datetime

from data.processing.base_processor import BaseProcessor
from utils import data_utils
from utils import model_utils
import config.config as config


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
        
        # Time for saving the files
        self.current_time = datetime.datetime.now()
        self.storage_path = os.path.join(config.FE_SEL_BASE_DIR, str(self.current_time))
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
    
    def run(self, methods=['cfs', 'xgb', 'rfe'], k_features=5):
        """
        Run the feature selection process.
        
        - First use correlation analysis to remove redundant features
        - Then apply tree-based importance to get an initial feature ranking
        - Finally use RFE with your actual model to fine-tune the selection
        """
        logging.info("---------- Starting Feature Selection ----------")
        
        selected_features = {}
        
        if 'cfs' in methods:
            selected_features['cfs'] = self.cfs(k=k_features)
        
        if 'xgb' in methods:
            selected_features['xgb'] = self.xgb_regressor(threshold=0.01)
        
        if 'rfe' in methods:
            selected_features['rfe'] = self.rfe(n_features_to_select=k_features)
        
        # Combine methods - features selected by at least 2 methods
        if len(methods) > 1:
            feature_counts = {}
            logging.info(f"selected features: {selected_features.items()}")
            for method, features in selected_features.items():
                for feature in features:
                    if feature != self.target_col:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            final_features = [f for f, count in feature_counts.items() 
                            if count >= len(methods)//2 + 1]
            final_features.append(self.target_col)
        else:
            final_features = selected_features[methods[0]]
        
        logging.info(f"Feature selection concluded with the following features: {final_features}.")
        
        # Create final dataframe with selected features
        self.selected_features = final_features
        return self.df[final_features]
    
    def cfs(self, k:int = 15, corr_threshold=0.7):
        """
        Correlation-based Feature Selection: selects subsets of features that are 
        highly correlated with the target variable but have low correlation with each other.
        Link: https://medium.com/@sariq16/correlation-based-feature-selection-in-a-data-science-project-3ca08d2af5c6
        
        Params:
            k: Number of features the method will select - select the top k features.
        """
        try:            
            # Get only numeric columns for correlation analysis
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            
            # Log columns that are being dropped
            non_numeric_cols = [col for col in self.df.columns if col not in numeric_cols]
            if non_numeric_cols:
                logging.warning(f"Dropping non-numeric columns for CFS analysis: {non_numeric_cols}")
            
            # Make sure target column is included in numeric_cols
            if self.target_col not in numeric_cols:
                logging.error(f"Target column '{self.target_col}' is not numeric. Cannot compute correlations.")
                # Return a default list containing just the target column to avoid None
                return [self.target_col]
            
            # Calculate correlations using only numeric columns
            corr_matrix = self.df[numeric_cols].corr()
            corr_with_target = corr_matrix[self.target_col].drop(self.target_col)
            
            try: 
                self._plot_fs_analysis_cfs(numeric_cols=numeric_cols)
            except Exception as e:
                logging.error(f"Unable to plot or save plot of CFS Analysis: {e}")
            
            # Sort and get top k candidates
            candidates = corr_with_target.abs().sort_values(ascending=False)[:k].index.tolist()
            
            # Remove highly correlated features from candidates
            selected = []
            duplicate_threshold = 0.95  # Features with >95% correlation to target are considered duplicates
            
            for feature in candidates:
                to_remove = False
                
                # Check if feature is essentially a duplicate of the target
                target_corr = abs(corr_matrix.loc[feature, self.target_col])
                if target_corr > duplicate_threshold:
                    logging.debug(f"Excluding {feature} - essentially duplicate of target (correlation: {target_corr:.3f})")
                    to_remove = True
                
                # Check correlation with other selected features
                if not to_remove:
                    for selected_feature in selected:
                        if abs(corr_matrix.loc[feature, selected_feature]) > corr_threshold:
                            to_remove = True
                            break
                
                if not to_remove:
                    selected.append(feature)
            
            # Make sure to return at least the target column to avoid empty list
            if not selected:
                logging.warning("No features selected. Returning just the target column.")
                return [self.target_col]
                
            return selected + [self.target_col]  # Return feature names
        except Exception as e:
            logging.error(f"Unable to perform Correlation-based Feature Selection: {e}")
            # Return just the target column as a fallback to avoid None
            return [self.target_col]
    
    def _plot_fs_analysis_cfs(self, top_k=20, corr_threshold=0.7, numeric_cols=None):
        """Two-part visualization for feature selection"""
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        # 1. Plot correlations with target
        corr_with_target = self.df[numeric_cols].corr()[self.target_col].drop(self.target_col)
        top_features = corr_with_target.abs().sort_values(ascending=False).head(top_k).index
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot correlations with target
        corr_with_target.loc[top_features].sort_values().plot(
            kind='barh', ax=ax1, cmap='coolwarm')
        ax1.set_title(f'Top {top_k} Features Correlated with Target')
        ax1.set_xlabel('Correlation')
        
        # Plot correlation matrix between top features
        corr_matrix = self.df[top_features].corr()
        sns.heatmap(
            corr_matrix,
            ax=ax2,
            annot=True,
            cmap="coolwarm",
            vmin=-1, vmax=1,
            center=0,
            fmt=".2f"
        )
        ax2.set_title('Correlation Matrix Among Top Features')
        
        plt.tight_layout()
        plt.savefig(f'{self.storage_path}/cfs_feature_correlation.png')
    
    def xgb_regressor(self, threshold = 0.01):
        """Select features using XGB Regressor."""
        try:
            model = xgb.XGBRegressor(
                n_estimators=100, 
                max_depth=6,
                tree_method='hist',
                enable_categorical=True
            )
            model.fit(self.X_train, self.y_train)
            
            # Create importance DataFrame with actual feature names
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Filter by threshold
            selected_features = feature_importance[feature_importance['importance'] > threshold]['feature'].tolist()
            
            try:
                # Plot importance (consider saving the plot)
                plt.figure(figsize=(10, 6))
                plt.bar(feature_importance['feature'], feature_importance['importance'])
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(f'{self.storage_path}/xgb_feature_importance.png')
            except Exception as e:
                logging.error(f"Unable to plot/save plot of XGB Regressor: {e}")
            
            logging.debug(f"Selected {len(selected_features)} features above threshold {threshold}: {selected_features}")
            return selected_features
        except Exception as e:
            logging.error(f"Unable to use XGB Regression for feature selection: {e}")
            return None
        
    def rfe(self, n_features_to_select=None, step=2):
        """
        Recursive Feature Elimination (Cross Validation): Fits to a model and removes the weakest featues
        until the specified number of features is reached.
        """
        try:
            # For intraday data, gradient boosting might perform better than (simple) SVR
            estimator = xgb.XGBRegressor(n_estimators=100, max_depth=4)
            
            # Time-based cross validation to avoid lookahead bias
            cv = TimeSeriesSplit(n_splits=5)
            
            selector = RFECV(
                estimator, 
                step=step,  # Remove more features per iteration for speed
                cv=cv, 
                scoring='neg_mean_squared_error',
                min_features_to_select=n_features_to_select if n_features_to_select else 5
            )
            
            selector = selector.fit(self.X, self.y)
            
            # Get selected features with importance ranking
            selected_features = self.X.columns[selector.support_].tolist()
            
            try:
                # Plot the number of features vs CV score
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), 
                        selector.cv_results_['mean_test_score'])
                plt.xlabel('Number of features')
                plt.ylabel('Mean test score (neg MSE)')
                plt.savefig(f'{self.storage_path}/rfe_cv_scores.png')
            except Exception as e:
                logging.error(f"Unable to plot/save plot of RFE (-CV): {e}")
            
            logging.info(f"Optimal number of features: {selector.n_features_}")
            return selected_features
        except Exception as e:
            logging.error(f"Unable to perform Recursive Feature Elimination: {e}")
            return None