import importlib
import logging
import os
import sys
import pandas as pd
# Train-test
from sktime.split import temporal_train_test_split  # https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.split.temporal_train_test_split.html
# Evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from utils import data_utils, model_utils
import config.config as config


class ModelTraining():
    """Train a specific ML Model."""
    def __init__(self, dataset, model_type:str, target_col:str = 'close', hypothesis:str = None):
        """
        Initialize the class by providing specifics.
        
        Args:
            - dataset: Path to the Engineered (ohlcv) dataset to train the model on.
            - model_type: Type of model you want to train. Eg. 'xgboost', 'decision_tree', 'random_forest', etc.
            - TODO: hypothesis: Provide path to hypothesis' config.yml to target its column.
        """
        try:
            logging.info(f"---------- Initiating Model Training ----------")
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            self.model_type = model_type
            
            self.target_col = target_col
            
            # Most likely reading directly from the csv file, which has no index.
            if (self.df.index.name is not 'date'):
                # DataFrame has no index (name 'date'), so setting it and dropping the duplicate
                self.df.index = pd.to_datetime(self.df['date'])  # setting index, but ensure its in datetime format
                self.df = self.df.drop('date', axis=1)
                logging.debug(f"Successfully made index to {self.df.index.name}")
        except Exception as e:
            logging.error(f"Error initializing Model Training: {str(e)}")
            raise
    
    def run(self):
        """
        Train initialized model type. Successfully training the model will store in according to
        the config.py-file.
        
        Returns:
            - If successful, will return the path of stored model, else False.
        """
        logging.info(f"---------- Starting to run training of {self.model_type} Model ... ----------")
        
        self.model_module, self.module_path = self._get_implemented_models()
        
        # Using imported module
        # Current implementation: if your model files have a class like XGBoostModel, DecisionTreeModel, etc.
        try:
            # Assuming each model file has a class that follows a naming pattern
            class_name = f"{self.model_type.title().replace('_', '')}Model"
            model_class = getattr(self.model_module, class_name)
            
            # Split data into train and test
            self.temporal_train_test_split()
            
            # Instantiate and train the model
            model_instance = model_class(dataset=self.df, X_train=self.X_train, y_train=self.y_train)
            self.model = model_instance.train()

            return self.model
            
        except AttributeError as e:
            logging.error(f"Could not find expected class in {self.module_path}: {str(e)}")
            return False
        
    def _get_implemented_models(self):
        """
        Method to get implemented models by reading files in model_factory/ and matching
        it with provided model_type.
        
        If it finds a match and successfully imports the module (model file), returns the
        model_module and module_path.
        Unsuccessfull will exit.
        """
        # Find file matching provided model_type
        model_factory_dir = os.path.join(config.BASE_DIR, 'model', 'model_factory')
        model_implementation_list = os.listdir(model_factory_dir)
        model_implementation_list.remove('__pycache__')  # No need to list this as it's not a model type
        
        found_model = False  # var for wether we find the model file or not
        model_module = None  # var for potential model import
        
        # Loop through model list and find model file based on provided model type
        for model_file in model_implementation_list:
            if self.model_type in model_file and model_file.endswith('.py'):
                found_model = True
                
                # Dynamically import the module
                module_name = model_file[:-3]  # Remove .py extension
                module_path = f"model.model_factory.{module_name}"
                
                try:
                    model_module = importlib.import_module(module_path)
                    logging.debug(f"Successfully imported {module_path}")
                    return model_module, module_path
                except ImportError as e:
                    logging.error(f"Failed to import {module_path}: {str(e)}")
                    sys.exit()
        
        if not found_model or model_module is None:
            logging.error("Model Trainer could not find an implementation of wanted model type. " +
                          f"You asked for '{self.model_type}', but these are the current " +
                          f"implementations: {model_implementation_list}.")
            logging.debug("^ Remember 'model_type' must match exact name of file, eg. 'model_type == model_type.py'.")
            sys.exit()
    
    def temporal_train_test_split(self):
        """
        'Complete' way of creating X and y, and getting their respecitve *_train and *_test for model training and testing.
        Train test split is done using sktime.split.temporal_train_test_split() (meaning not random as it's important for 
        avoiding lookahead bias).
        """
        # Initialize train/test targets
        try:
            # Need to change potential columns having dtype object to category, for model to 'accept'
            self.df = model_utils.handle_categories(df = self.df)
            
            # Features (Input variables)
            self.X = self.df.drop(self.target_col, axis=1)
            logging.debug(f"Successfully set Features (X), with {len(self.X.columns)} columns/features ...")
            # Target (Output variables)
            self.y = self.df[self.target_col]
            logging.debug(f"and Target (y), '{self.target_col}', {self.y.size} values.")
        except Exception as e:
            logging.error(f"Unable to set train/test targets: {e}")
            
        model_utils.ensure_correct_xy_format(X = self.X, y = self.y)
        
        self.y_train, self.y_test, self.X_train, self.X_test = temporal_train_test_split(y=self.y, X=self.X, test_size=0.2, train_size=0.8)  # sktime's returns values in a different order than sklearn's train_test_split
        
    def evaluate_model(self):
        """Evaluate trained model. Further down the line, this could be so much more complex being its own file/component."""
        logging.info(f"----- Evaluate Model -----")
        self.y_pred = self.model.predict(self.X_test)
        # print(self.y_pred.size)
        print('F1 score:', f1_score(self.y_test, self.y_pred,
                            average="weighted"))