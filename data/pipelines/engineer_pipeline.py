import logging
import os
import pandas as pd

from utils import data_utils
from utils import shared_utils
import config.config as config
from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from data.features.generation import FeatureGenerator
from data.features.transformation import FeatureTransformer
from data.features.selection import FeatureSelector


class EngineeringPipeline():
    def __init__(self, raw_dataset : str, output_path : str = None):
        """
        Pipeline and orchestrator for processing and engineering raw datasets
        into data ready to be used for training.
        Currently only a very simple way of saving processed file: adding a _processed at the end of original name
        
        Args:
            raw_dataset: path to the raw dataset (csv format).
        """
        self.output_path = output_path
        self.raw_data_path = raw_dataset

        # Loading data
        self.df, self.original_df = data_utils.check_and_return_df(raw_dataset)
        logging.debug("Engineering Pipeline - Successfully loaded data!")
        
        # Initializing processors
        self.cleaner = DataCleaner(raw_dataset=self.df)
    
    def run(self):
        """
        Run the pipeline.
        
        returns:
            pd.DataFrame: Engineered data ready to use for training.
        """
        
        # Clean the data
        self.df = self.cleaner.run()
        
        # Validate
        self.validator = DataValidator(self.df)
        self.validator_results = self.validator.run()
        self.df = self.validator_results["validated_data"]
        # Logging results
        data_utils.check_validation(self.validator_results["is_valid"], self.validator_results["issues"])
        
        # Feature Generation
        self.feature_generator = FeatureGenerator(self.df)
        self.df = self.feature_generator.run()
        
        # Transformation, scaling, handle missing values, etc. of newly generated features
        self.transformation = FeatureTransformer(self.df)
        self.df = self.transformation.run()
        
        # Feature Selection
        self.selector = FeatureSelector(self.df)
        self.df = self.selector.run()
        
        # Save Engineered data
        try:
            data_utils.save_processed_file(filepath=self.raw_data_path, processed_data=self.df)
        except Exception as e:
            logging.error(f"Unable to save file 'dynamically', will store as {os.getcwd()}, : {e}")
            self.df.to_csv('testing.csv', index=True)
        
        # FIX: Return the processed DataFrame
        return self.df