import pandas as pd

from data.processing.base_processor import BaseProcessor


class DataCleaner(BaseProcessor):
    def __init__(self, raw_dataset):
        """
        Class for cleaning market data.
        
        Args:
            raw_dataset: path to the raw dataset (csv format). Can be either path to csv or already loaded DataFrame.
        """
        self.data = pd.read_csv(raw_dataset)
        
    def run():
        """
        Clean the data.
        
        returns:
            pd.DataFrame: Cleaned dataset.
        """