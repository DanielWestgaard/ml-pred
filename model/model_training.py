import datetime
import importlib
import os
import sys
import numpy as np
import pandas as pd
import pickle
import psutil
import yaml  # stroring
import time
import platform

# Train-test
from sktime.split import temporal_train_test_split  # https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.split.temporal_train_test_split.html

from utils import data_utils, shared_utils
import config.config as config
from utils.model import model_utils
import utils.hypothesis.config_loader as config_loader
from utils.logging_utils import get_logger

logger = get_logger()


class ModelTraining():
    """Train a specific ML Model."""
    def __init__(self, dataset:str = None, model_type:str = None, target_col:str = 'close', hypothesis_source:str = None):
        """
        Initialize the class by providing specifics.
        
        Args:
            - dataset: Path to the Engineered (ohlcv) dataset to train the model on. Is 'optional' as it might be provided in hypothesis_source
            - model_type: Type of model you want to train. Eg. 'xgboost', 'decision_tree', 'random_forest', etc. Is 'optional' as it might be provided in hypothesis_source
            - TODO: hypothesis: Provide path to hypothesis' config.yml to target its column. This currently only works close target.
        """
        try:
            logger.info(f"---------- Initiating Model Training ----------")
            
            self.hypothesis_source = hypothesis_source
            # Hypothesis
            if self.hypothesis_source is not None:
                logger.info(f"Model Training was initialized with Hypothesis configuration!")
                
                # Loading in all configurations in the provided config.yml
                configurations = config_loader.load_hypothesis_config(hypothesis_source).handle_hypothesis_source()
                
                self.model_type = configurations.get("model_training_type") or configurations.get("model_type")
                
                try:
                    dataset = configurations.get("processed_data")
                    
                    if dataset is None:
                        raw_dataset = configurations.get("data_pipeline_raw_source")                        
                        
                        if raw_dataset is None:
                            raw_dataset = configurations.get("raw_data_source")

                        if raw_dataset is not None:
                            dataset = self._get_processed_path(raw_dataset)
                        else:
                            raise ValueError("No raw data source found")
                except:  # Failed means user either forgot to provide datapath, og have ran data pipeline first, meaning we must find that one
                    try:
                        raw_dataset = configurations.get("data_pipeline_raw_source")
                        
                        if raw_dataset is None:
                            raw_dataset = configurations.get("raw_data_source")
                            
                        if raw_dataset is not None:
                            dataset = self._get_processed_path(raw_dataset)
                        else:
                            raise ValueError("No raw data source found")
                    except Exception as e:
                        logger.error(f"Was unable to find neither processed datapath nor raw datapath. Make sure you provide at least one of them: {e}")
                        logger.debug(f"Congigurations found in provided config.yml path: {configurations}")
                        sys.exit()
                
                # This is used (temporarily), in the very end, for storing the dataset path used for model training: 'getattr(self, 'dataset_path', 'unknown')'
                self.dataset_path = dataset
                
                # Load dataset based on format
                self.df, self.original_df = data_utils.check_and_return_df(dataset)
                        
                # Assuming all target, hypothesis, columns starts with ´target_'
                for col in self.df.columns:
                    if 'target_' in col:
                        self.target_col = col  # TODO: Logic to find target column
                        logger.debug(f"Found hypothesis target : '{self.target_col}'")
            
            # Not Hypothesis flow: User must provide dataset and model_type
            else:
                logger.info(f"Model Training initialized without hypothesis configuration.")
                
                if dataset is None or model_type is None:
                    logger.error(f"Model Training must be initialized with path to processed dataset and model_type! You provided dataset '{dataset}' and model_type '{model_type}'.")
                    sys.exit()
                    
                self.dataset_path = dataset
                self.model_type = model_type
                self.target_col = target_col
            
                # Load dataset based on format
                self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            # Reading directly from a .csv-file means df will have no index.
            if (self.df.index.name != 'date'):
                # DataFrame has no index (name 'date'), so setting it and dropping the duplicate
                self.df.index = pd.to_datetime(self.df['date'])  # setting index, but ensure its in datetime format
                self.df = self.df.drop('date', axis=1)
                logger.debug(f"Successfully made index to {self.df.index.name}")
            
        except Exception as e:
            logger.error(f"Error initializing Model Training: {str(e)}")
            raise
    
    def run(self):
        """
        Train initialized model type. Successfully training the model will store in according to
        the config.py-file.
        
        Returns:
            - If successful, will return the path of stored model, else False.
        """
        logger.info(f"---------- Starting to run training of {self.model_type} Model ... ----------")
        
        self.model_module, self.module_path = self._get_implemented_models()
        
        # Using imported module
        # Current implementation: if your model files have a class like XGBoostModel, DecisionTreeModel, etc.
        try:
            # Assuming each model file has a class that follows a naming pattern
            class_name = f"{self.model_type.title().replace('_', '')}Model"
            model_class = getattr(self.model_module, class_name)
            
            # Split data into train and test
            self.temporal_train_test_split()
            
            # Instantiate model
            model_instance = model_class(dataset=self.df, X_train=self.X_train, y_train=self.y_train)
            # Actually train the model
            self.training_start_time = time.time()  # not really accurate starting it here, but short term solution
            self.model = model_instance.train()
            self.model.type = self.model_type
            
            self.save_model()
            
            return self.model
            
        except AttributeError as e:
            logger.error(f"Could not find expected class in {self.module_path}: {str(e)}")
            return False
    
    def _get_processed_path(self, raw_path: str):
        """
        Convert a raw file path to its processed equivalent.
        Currently, the approach from raw to processed data (in terms of naming standards) is that
        I use the exact same name for processed as raw, but switch out the first name 'raw' with 'processed'
        (same with dir). Meaning we can use this same approach here.
        
        Args:
            raw_path (str): Path to the raw file
            
        Returns:
            str: Path to the processed file
        """
        # Replace /raw/ with /processed/ in the directory structure
        processed_path = raw_path.replace("/raw/", "/processed/")
        
        # Replace raw_ prefix with processed_ in the filename
        processed_path = processed_path.replace("raw_", "processed_")
        
        return processed_path
        
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
        model_implementation_list.remove('__pycache__') if '__pycache__' in model_implementation_list else None # No need to list this as it's not a model type
        model_implementation_list.remove('base_model.py')
        
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
                    logger.debug(f"Successfully imported {module_path}")
                    return model_module, module_path
                except ImportError as e:
                    logger.error(f"Failed to import {module_path}: {str(e)}")
                    sys.exit()
        
        if not found_model or model_module is None:
            logger.error("Model Trainer could not find an implementation of wanted model type. " +
                          f"You asked for '{self.model_type}', but these are the current " +
                          f"implementations: {model_implementation_list}.")
            logger.debug("^ Remember 'model_type' must match exact name of file, eg. 'model_type == model_type.py'.")
            sys.exit()
    
    def temporal_train_test_split(self, return_datasets:bool = False):
        """
        'Complete' way of creating X and y, and getting their respecitve *_train and *_test for model training and testing.
        Train test split is done using sktime.split.temporal_train_test_split() (meaning not random as it's important for 
        avoiding lookahead bias).
        
        Args:
            - retur: In case you want to return the train and test sets for outside use. True will return them.
        """
        # Initialize train/test targets
        try:
            # Need to change potential columns having dtype object to category, for model to 'accept'
            self.df = model_utils.handle_categories(df = self.df)
            
            # Features (Input variables)
            self.X = self.df.drop(self.target_col, axis=1)
            logger.debug(f"Successfully set Features (X), with {len(self.X.columns)} columns/features ...")
            # Target (Output variables)
            self.y = self.df[self.target_col]
            logger.debug(f"and Target (y), '{self.target_col}', {self.y.size} values.")
        except Exception as e:
            logger.error(f"Unable to set train/test targets: {e}")
            
        model_utils.ensure_correct_xy_format(X = self.X, y = self.y)
        
        self.y_train, self.y_test, self.X_train, self.X_test = temporal_train_test_split(y=self.y, X=self.X, test_size=0.2, train_size=0.8)  # sktime's returns values in a different order than sklearn's train_test_split
        
        if return_datasets:
            return self.y_train, self.y_test, self.X_train, self.X_test
        
    def save_model(self, filename: str = None):
        """Save trained model and metadata under timestamped folder."""
        # Track training time (you'd start this timer at beginning of training)
        # training_start_time = getattr(self, 'training_start_time', time.time())
        training_duration = time.time() - self.training_start_time
        
        current_time = datetime.datetime.now().isoformat(sep='_', timespec='seconds')
        
        # Directory and folder model and info will be saved under
        directory = os.path.join(config.TRAINED_MODEL_STORAGE, current_time)
        
        # Store the model directory for later use
        self.model_directory = directory
        
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            logger.debug(f"Creating directory now.")
            os.makedirs(directory)
        
        try:
            # Save the model itself as a pickle file
            model_pkl_file = os.path.join(directory, f"{self.model_type}.pkl")
            with open(model_pkl_file, 'wb') as file:  
                pickle.dump(self.model, file)
            logger.info(f"Successfully saved trained model: {model_pkl_file}")
            
            # Save model metadata
            self._save_model_info(directory, training_duration)
            
        except Exception as e:
            logger.error(f"Unable to save trained model: {e}")
            raise

    def _save_model_info(self, directory: str, training_duration: float):
        """Save model metadata to YAML file."""
        
        # Collect system information
        def get_system_info():
            return {
                'cpu': platform.processor() or platform.machine(),
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'python_version': platform.python_version(),
                'os': f"{platform.system()} {platform.release()}"
            }
        
        # Convert any value to a YAML-safe native Python type
        def make_yaml_safe(obj):
            """Convert numpy/pandas types to native Python types for YAML serialization."""
            import numpy as np
            import pandas as pd
            
            if obj is None:
                return None
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {key: make_yaml_safe(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_yaml_safe(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif pd.isna(obj):
                return None
            else:
                return str(obj)
        
        # Get model hyperparameters if available
        def get_model_params():
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
                return make_yaml_safe(params)
            elif hasattr(self.model, '__dict__'):
                # Filter out non-parameter attributes
                params = {}
                for key, value in self.model.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        params[key] = make_yaml_safe(value)
                return params
            else:
                return {}
        
        # Create metadata dictionary
        metadata = {
            'hypothesis': {
                'name': _hypothesis(self)
            },
            
            'model_info': {
                'model_type': self.model_type,
                'model_class': self.model.__class__.__name__,
                'storage_path': directory,
                'created_at': datetime.datetime.now().isoformat()
            },
            
            'training': {
                'duration_seconds': round(training_duration, 2),
                'duration_formatted': str(datetime.timedelta(seconds=int(training_duration))),
                'completed_at': datetime.datetime.now().isoformat(),
                'train_samples': int(len(self.X_train)),
                'test_samples': int(len(self.X_test))
            },
            
            'dataset': {
                'dataset_path': getattr(self, 'dataset_path', 'unknown'),  # Needs 'self.dataset_path = dataset' 
                'target_column': self.target_col,
                'feature_count': int(len(self.X_train.columns)),
                'features': self.X_train.columns.tolist(),
                'total_samples': int(len(self.df)),
                'date_range': {
                    'start': str(self.df.index.min()),
                    'end': str(self.df.index.max())
                },
                'missing_values': make_yaml_safe(self.X_train.isnull().sum().sum())
            },
            
            'model_parameters': get_model_params(),
            
            'system_info': get_system_info(),
            
            'data_split': {
                'train_size': int(len(self.X_train)),
                'test_size': int(len(self.X_test)),
                'split_ratio': f"{len(self.X_train)}/{len(self.X_test)}"
            }
        }
        
        # Save to YAML file with spacing between sections
        metadata_file = os.path.join(directory, 'model_metadata.yml')
        
        # Generate YAML string
        yaml_content = yaml.dump(metadata, default_flow_style=False, sort_keys=False, indent=2)
        
        # Add blank lines between top-level sections
        sections = ['training:', 'dataset:', 'model_parameters:', 'system_info:', 'data_split:']
        for section in sections:
            yaml_content = yaml_content.replace(f'\n{section}', f'\n\n{section}')
        
        # Write formatted YAML
        with open(metadata_file, 'w') as file:
            file.write(yaml_content)
        
        logger.info(f"Successfully saved model metadata: {metadata_file}")

def _hypothesis(self):
    """Simple method to check if this model is trained on a hypothesis or not, and returns json ready answer."""
    # If model is trained on hypothesis source: return json of the hypothesis_name
    if self.hypothesis_source is not None:
        return f"{config_loader.load_hypothesis_config(self.hypothesis_source).handle_hypothesis_source().get('hypothesis_name')}"
    else:  # If not, return none
        return f'{None}'