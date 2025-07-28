import inspect
import os
import sys
import pandas as pd

from utils import data_utils
from utils import shared_utils
import config.config as config
from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from data.features.enhanced_generation import ModelAwareFeatureGenerator
from data.features.transformation import FeatureTransformer
from data.features.selection import FeatureSelector
from data.features.target_hypothesis import TargetHypothesis
import utils.hypothesis.config_loader as config_loader
from utils.logging_utils import get_logger

logger = get_logger()


class EngineeringPipeline():
    def __init__(self, raw_dataset:str = None, output_path:str = None, save_cleaned_data:bool = False, hypothesis_source:str = None, analyze_feature_importance:bool = True):
        """
        Pipeline and orchestrator for processing and engineering raw datasets
        into data ready to be used for training.
        Currently only a very simple way of saving processed file: adding a _processed at the end of original name
        Providing the argument selected_features will make Feature Generator only generate those selected features
        (of they exist). This also makes the pipeline skip the Feature Selection step entirely.
        
        Args:
            - raw_dataset: path to the raw dataset (csv format). Is now currently optional as when you use hypothesis_source,
              the path to data is provided there and thet trumphs provided raw_dataset argument.
            - output_path: potential output path for processed data (csv format).
            - hypothesis_source: For running pipeline with hypothesis - The path of the hypothesis config.yml-file.
            - analyze_feature_importance: If True, runs feature importance analysis after feature generation.
        """
        self.output_path = output_path
        self.analyze_feature_importance = analyze_feature_importance
        
        if hypothesis_source is not None:
            configurations = config_loader.load_hypothesis_config(hypothesis_source).handle_hypothesis_source()
            self.hypothesis_name = configurations.get('hypothesis_name')
            raw_dataset = configurations.get('data_pipeline_raw_source')
            self.selected_features = configurations.get('features')
            self.spread_focused = configurations.get('spread_focused', False)
        else:
            self.hypothesis_name = None
            self.selected_features = None
            self.spread_focused = False
        
        self.raw_data_path = raw_dataset
        # Loading data
        self.df, self.original_df = data_utils.check_and_return_df(raw_dataset)
        logger.debug("Engineering Pipeline - Successfully loaded data!")
        
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
        logger.info("---------- Running Data Pipeline ----------")
        
        # Clean the data
        self.df = self.cleaner.run()
        
        # Validate
        self.validator = DataValidator(self.df)
        self.validator_results = self.validator.run()
        self.df = self.validator_results["validated_data"]
        # logger results
        data_utils.check_validation(self.validator_results["is_valid"], self.validator_results["issues"])
        
        # Feature Generation
        self.feature_generator = ModelAwareFeatureGenerator(self.df)
        
        if self.selected_features is not None:
            logger.debug(f"User provided specific features to be generated: {self.selected_features}")
            self.skip_selection = True
            
            # Generate specific features using list format
            if isinstance(self.selected_features, list):
                logger.info(f"Generating specific features: {self.selected_features}")
                self.df = self.feature_generator.generate_selected(self.selected_features)
            else:
                logger.warning("selected_features must be a list of feature names, generating all features")
                self.df = self.feature_generator.generate_all()
            
            # Also change target to the hypothesis
            self.hypo = TargetHypothesis(self.df, self.hypothesis_name or "session_based_tod_anomalies_X")
            self.df = self.hypo.run()
            
            # Feature importance analysis (after feature generation but before transformation)
            if self.analyze_feature_importance:
                self._run_feature_importance_analysis()
        else:
            logger.debug("User did not provide any specific features to be generated - generating all features.")
            self.skip_selection = True
            logger.debug(f"Also, Skip Selection is (currently set to ${self.skip_selection})")
            self.df = self.feature_generator.generate_all()
            
            # Feature importance analysis (after feature generation but before transformation)
            if self.analyze_feature_importance:
                self._run_feature_importance_analysis()
        
        # Transformation, scaling, handle missing values, etc. of newly generated features
        self.transformation = FeatureTransformer(self.df)
        self.df = self.transformation.run()
        
        # Feature Selection
        if self.skip_selection:
            logger.info("As selected features were provided, pipeline will skip Feature Selection.")
        else:
            self.selector = FeatureSelector(self.df)
            self.df = self.selector.run()
        
        logger.info("---------- Data Pipeline finished all 'data steps'! ----------")
        
        # Save Engineered data
        try:
            data_utils.save_processed_file(filepath=self.raw_data_path, processed_data=self.df)
        except Exception as e:
            logger.error(f"Unable to save file 'dynamically', will store as {os.getcwd()}, : {e}")
            self.df.to_csv('testing.csv', index=True)
        
        logger.info("---------- Data Pipeline Finished ----------")
        
        return self.df
    
    def _run_feature_importance_analysis(self):
        """Run feature importance analysis at optimal point in pipeline"""
        try:
            from data.features.feature_importance import FeatureImportanceAnalyzer
            
            logger.info("Running feature importance analysis...")
            
            # Determine target column
            target_col = None
            if hasattr(self, 'hypothesis_name') and self.hypothesis_name:
                # Look for hypothesis-specific target
                target_candidates = [col for col in self.df.columns if col.startswith('target_')]
                if target_candidates:
                    target_col = target_candidates[0]
            
            # Initialize analyzer
            analyzer = FeatureImportanceAnalyzer(self.df, target_col)
            
            # Run analysis and save results
            results = analyzer.analyze_and_save()
            
            # Log top features
            consensus = results['consensus']
            logger.info("Feature Importance Analysis Results:")
            logger.info("Top 10 features by consensus ranking:")
            for i, (feature, score) in enumerate(consensus.head(10).items(), 1):
                logger.info(f"  {i:2d}. {feature:<25} {score:.4f}")
            
            logger.info(f"Full results saved to: {results['output_directory']}")
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            logger.info("Pipeline will continue without feature importance analysis")