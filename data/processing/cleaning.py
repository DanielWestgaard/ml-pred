import logging
import pandas as pd

from data.processing.base_processor import BaseProcessor
import utils.data_utils as data_utils


class DataCleaner(BaseProcessor):
    def __init__(self, raw_dataset):
        """
        Class for cleaning market data.
        
        Args:
            raw_dataset: path to the raw dataset (csv format). Can be either path to csv or already loaded DataFrame.
        """
        
        # Load dataset based on format
        self.data = data_utils.check_and_return_df(raw_dataset)
        
    def run():
        """
        Clean the data.
        
        returns:
            pd.DataFrame: Cleaned dataset.
        """