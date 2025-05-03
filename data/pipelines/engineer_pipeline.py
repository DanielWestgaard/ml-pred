import pandas as pd


class EngineeringPipeline():
    def __init__(self, raw_dataset):
        """
        Pipeline and orchestrator for processing and engineering raw datasets
        into data ready to be used for training.
        
        Args:
            raw_dataset: path to the raw dataset (csv format).
        """
        self.data = pd.read_csv(raw_dataset)
        
        
    def run():
        """
        Run the pipeline.
        """
        
        # Clean the data
        
        # Validate 
        
        # Feature Generation
        
        # Feature Selection
        
        # Save Engineered data
        