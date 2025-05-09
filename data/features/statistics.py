from utils import data_utils


class FeatureStatistics():
    def __init__(self, data):
        """Statistics and visualizations related to feature generation and selection."""
        self.df, self.og_df = data_utils.check_and_return_df(data)
    
    def generated_features(self, basic_cols = ['date', 'open', 'high', 'low', 'close', 'volume']):
        """
        Function for counting generated features, assuming that's whats in the dataset.
        
        Parameters:
            basic_cols: list of the columns that are not features, but basic columns. Default is date, OHLCV
            
        Returns:
            n_features: Number of features, excluding basic_cols
            features_list: List of all features.
        """
        columns = self.df.columns
        
        features_list = [col for col in columns if col not in basic_cols]
        
        return len(features_list), features_list