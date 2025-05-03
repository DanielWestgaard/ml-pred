import pandas as pd

class DataCleaner():
    def __init__(self, raw_dataset):
        """
        Class for cleaning market data.
        
        Args:
            raw_dataset: path to the raw dataset (csv format).
        """
        self.data = pd.read_csv(raw_dataset)