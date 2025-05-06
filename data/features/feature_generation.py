from utils import data_utils


class FeatureGenerator():
    def __init__(self, dataset):
        """
        Class focusing on generating features with predictable power,
        that might help a ML model learn and predict better.
        
        Args:
            dataset: The cleaned (and validated) data, containing (index) date/datetime, and the columns OHLCV.
        
        returns:
            dataset with all generated features 
        """
    
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(dataset)
        
    def run():
        """Run feature generation on provided dataset."""
        pass