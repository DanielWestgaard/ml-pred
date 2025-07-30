import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from model.model_factory.base_model import BaseModel
from utils.logging_utils import get_logger

logger = get_logger()


class RandomForestModel(BaseModel):
    """
    Random Forest Classifier Model:
    
    Random Forest is an ensemble learning method that operates by constructing multiple decision trees
    and outputting the class that is the mode of the classes (classification) of the individual trees.
    
    Ideal for:
    - Imbalanced classification tasks (like Tier 2+ spread compression signals)
    - Complex feature interactions without assumptions about relationships
    - Robust handling of outliers in financial time series data
    - Feature importance analysis for market microstructure patterns
    """
    
    def __init__(self, dataset: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame, model_params: dict = None):
        """
        Initialize Random Forest Classifier optimized for spread compression patterns.
        
        Args:
            dataset: The processed/engineered data for training and testing (pd.DataFrame)
            X_train: Training features (pd.DataFrame)
            y_train: Training targets - should be binary classification labels (pd.DataFrame/Series)
            model_params: Optional model parameters from configuration (dict)
            
        Returns:
            Trained Random Forest model optimized for imbalanced financial data
        """
        logger.debug(f"----- Initializing Random Forest Classifier Model -----")
        
        self.df = dataset
        self.X_train = X_train
        self.y_train = y_train
        self.model_params = model_params or {}
        
        # Validate that target is binary classification
        unique_targets = np.unique(y_train)
        if len(unique_targets) > 2:
            logger.warning(f"Random Forest Classifier expects binary classification, but found {len(unique_targets)} unique targets: {unique_targets}")
        
        logger.debug(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
        logger.debug(f"Target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
    def train(self):
        """
        Train the Random Forest Classifier with parameters optimized for spread compression patterns.
        
        Configuration optimized for:
        - Imbalanced data (0.18% positive class for Tier 2+ signals)
        - Complex feature interactions in market microstructure
        - Robust performance on 1-minute financial data
        """
        logger.info(f"---------- Starting to train Random Forest Classifier Model! ----------")
        
        # Default parameters optimized for spread compression patterns
        default_params = {
            # Core ensemble parameters
            'n_estimators': 300,              # More trees for rare signal detection
            'max_depth': 10,                  # Deeper trees for complex market patterns
            'min_samples_split': 5,           # Lower threshold for rare Tier 2+ patterns
            'min_samples_leaf': 2,            # Sensitive to minority class patterns
            
            # Imbalanced data handling
            'class_weight': 'balanced',       # Critical for 0.18% positive class ratio
            'bootstrap': True,                # Enable bootstrap sampling
            'oob_score': True,               # Out-of-bag evaluation for validation
            
            # Randomization and performance
            'random_state': 42,              # Reproducible results
            'n_jobs': -1,                    # Use all available CPU cores
            'warm_start': False,             # Train from scratch each time
            
            # Feature sampling for diversity
            'max_features': 'sqrt',          # sqrt(n_features) for classification
            'max_samples': None,             # Use all samples for each tree
        }
        
        # Merge default parameters with configuration parameters (config overrides defaults)
        params = {**default_params, **self.model_params}
        logger.info(f"Default parameters: {default_params}")
        logger.info(f"Config parameters: {self.model_params}")
        logger.info(f"Final merged parameters: {params}")
        
        logger.debug(f"Random Forest parameters: {params}")
        
        # Log class distribution for monitoring
        class_counts = pd.Series(self.y_train).value_counts()
        logger.info(f"Training on imbalanced data - Class distribution: {class_counts.to_dict()}")
        
        if len(class_counts) == 2:
            imbalance_ratio = class_counts.min() / class_counts.max()
            logger.info(f"Imbalance ratio: {imbalance_ratio:.4f} ({imbalance_ratio*100:.2f}%)")
        elif len(class_counts) == 3:
            minority_class_pct = (class_counts[1] + class_counts[2]) / class_counts.sum() * 100
            logger.info(f"Minority classes (BUY+SELL): {minority_class_pct:.1f}% of total data")
        
        # Check if SMOTE oversampling should be applied
        X_train_final = self.X_train
        y_train_final = self.y_train
        
        # Apply SMOTE if extreme imbalance detected (minority classes < 10%)
        if len(class_counts) >= 2 and 'use_smote' in self.model_params and self.model_params['use_smote']:
            try:
                from imblearn.over_sampling import SMOTE
                logger.info("Applying SMOTE oversampling for extreme class imbalance...")
                
                # Configure SMOTE
                smote_strategy = self.model_params.get('smote_strategy', 'auto')  # Can be 'minority', 'not majority', 'all', or dict
                smote_k_neighbors = min(5, class_counts.min() - 1) if class_counts.min() > 1 else 1
                
                smote = SMOTE(
                    sampling_strategy=smote_strategy,
                    k_neighbors=smote_k_neighbors,
                    random_state=42
                )
                
                X_train_final, y_train_final = smote.fit_resample(self.X_train, self.y_train)
                
                # Log new distribution
                new_class_counts = pd.Series(y_train_final).value_counts()
                logger.info(f"After SMOTE - Class distribution: {new_class_counts.to_dict()}")
                logger.info(f"Training samples increased from {len(self.y_train)} to {len(y_train_final)}")
                
            except ImportError:
                logger.warning("SMOTE oversampling requested but imblearn not available. Install with: pip install imbalanced-learn")
            except Exception as e:
                logger.warning(f"SMOTE oversampling failed: {e}. Continuing with original data.")
        
        # Create and train model
        try:
            rf_classifier = RandomForestClassifier(**params)
            
            # Fit the model
            logger.debug("Fitting Random Forest Classifier...")
            model = rf_classifier.fit(X_train_final, y_train_final)
            
            # Log training results
            logger.info(f"Successfully trained Random Forest Classifier!")
            logger.info(f"Model contains {model.n_estimators} trees")
            
            if hasattr(model, 'oob_score_'):
                logger.info(f"Out-of-bag score: {model.oob_score_:.4f}")
            
            # Feature importance analysis
            self._log_feature_importance(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to train Random Forest Classifier: {str(e)}")
            raise
    
    def _log_feature_importance(self, model):
        """
        Log feature importance for spread compression analysis.
        
        Args:
            model: Trained Random Forest model
        """
        try:
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 8 Most Important Features for Spread Compression:")
            for _, row in feature_importance.head(8).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Validate key spread compression features
            key_features = ['compression_consistency', 'volume_ratio', 'signal_strength_v2', 'compression_ratio_micro']
            logger.debug("Key spread compression feature importance:")
            for feature in key_features:
                if feature in feature_importance['feature'].values:
                    importance = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
                    logger.debug(f"  {feature}: {importance:.4f}")
                    
        except Exception as e:
            logger.warning(f"Could not log feature importance: {str(e)}")
    
    def save_model_and_info(self):
        """
        Save trained model and metadata following project conventions.
        
        Note: This method signature matches the BaseModel interface.
        The actual saving is handled by ModelTraining.save_model() 
        to maintain consistency with the project's storage architecture.
        """
        logger.debug("Random Forest model saving handled by ModelTraining.save_model()")
        pass