import inspect
import logging
import os
import sys
import pandas as pd

from utils import data_utils
from utils import shared_utils
import config.config as config
from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from data.features.generation import FeatureGenerator
from data.features.transformation import FeatureTransformer
from data.features.selection import FeatureSelector
from data.features.target_hypothesis import TargetHypothesis


class EngineeringPipeline():
    def __init__(self, raw_dataset:str, output_path:str = None, save_cleaned_data:bool = False, hypothesis_source:str = None):
        """
        Pipeline and orchestrator for processing and engineering raw datasets
        into data ready to be used for training.
        Currently only a very simple way of saving processed file: adding a _processed at the end of original name
        Providing the argument selected_features will make Feature Generator only generate those selected features
        (of they exist). This also makes the pipeline skip the Feature Selection step entirely.
        
        Args:
            - raw_dataset: path to the raw dataset (csv format).
            - output_path: potential output path for processed data (csv format).
            - hypothesis_source: For running pipeline with hypothesis - The path of the hypothesis config.yml-file.
            xxx selected_features: Feature Generator only generates those features, and pipeline skips Selesction.
        """
        self.output_path = output_path
        self.raw_data_path = raw_dataset

        # Loading data
        self.df, self.original_df = data_utils.check_and_return_df(raw_dataset)
        logging.debug("Engineering Pipeline - Successfully loaded data!")
        
        if hypothesis_source is not None:
            configurations = shared_utils.handle_hypothesis_source(hypothesis_source)
            self.hypothesis_name = configurations.get('hypothesis_name')
            self.selected_features = configurations.get('features')
        else:
            self.selected_features = None
        
        # Potentially save cleaned dataset
        self.save_cleaned_data = save_cleaned_data
        
        # Initializing processors
        self.cleaner = DataCleaner(raw_dataset=self.df, save_cleaned_data=save_cleaned_data)
    
    def run(self):
        """
        Run the pipeline.
        
        returns:
            pd.DataFrame: Engineered data ready to use for training.
        """
        logging.info("---------- Running Data Pipeline ----------")
        
        # Clean the data
        self.df = self.cleaner.run()
        
        # Validate
        self.validator = DataValidator(self.df)
        self.validator_results = self.validator.run()
        self.df = self.validator_results["validated_data"]
        # Logging results
        data_utils.check_validation(self.validator_results["is_valid"], self.validator_results["issues"])
        
        # Feature Generation
        if self.selected_features is not None:
            logging.debug(f"User provided specific features to be generated: {self.selected_features}")
            self.skip_selection = True
            self.feature_generator = FeatureGenerator(self.df, selected_features=self.selected_features)
            self.df = self.feature_generator.run()
            
            # Also change target to the hypothesis
            self.hypo = TargetHypothesis(self.df, self.hypothesis_name or "session_based_tod_anomalies_X")
            self.df = self.hypo.run()
        else:
            logging.debug("User did not provide any specific features to be generated - generating all features.")
            self.skip_selection = False
            self.feature_generator = FeatureGenerator(self.df)
            self.df = self.feature_generator.run()
        
        # Transformation, scaling, handle missing values, etc. of newly generated features
        self.transformation = FeatureTransformer(self.df)
        self.df = self.transformation.run()
        
        # Feature Selection
        if self.skip_selection:
            logging.info("As selected features were provided, pipeline will skip Feature Selection.")
        else:
            self.selector = FeatureSelector(self.df)
            self.df = self.selector.run()
        
        logging.info("---------- Data Pipeline finished all 'data steps'! ----------")
        
        # Save Engineered data
        try:
            data_utils.save_processed_file(filepath=self.raw_data_path, processed_data=self.df)
        except Exception as e:
            logging.error(f"Unable to save file 'dynamically', will store as {os.getcwd()}, : {e}")
            self.df.to_csv('testing.csv', index=True)
        
        logging.info("---------- Data Pipeline Finished ----------")
        
        return self.df