import pandas as pd

from data.processing.base_processor import BaseProcessor


# Note: not tested yet, but moved from fe_ge
class DataNormalizer(BaseProcessor):
    def __init__(self):
        """Class for normalizing features."""
    
    def _normalize_features(self, columns=None, window=20, exclude=None):  #['open', 'high', 'low', 'close']
        """
        Z-score normalize selected features in the DataFrame.
        
        Parameters:
        -----------
        columns : list or None
            List of columns to normalize. If None, normalizes all numeric columns.
        window : int
            The rolling window to use for calculating mean and standard deviation.
        exclude : list or None
            List of columns to exclude from normalization.
            
        Returns:
        --------
        None. Adds new columns to self.df with '_zscore' suffix.
        """
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = self.df.select_dtypes(include='number').columns.tolist()
        
        # Exclude specified columns
        if exclude is not None:
            columns = [col for col in columns if col not in exclude]
        
        # Skip columns that already have a z-score version
        columns = [col for col in columns if f"{col}_zscore" not in self.df.columns]
        
        # Apply z-score normalization to each column
        for col in columns:
            # Calculate rolling mean and standard deviation
            rolling_mean = self.df[col].rolling(window=window).mean()
            rolling_std = self.df[col].rolling(window=window).std()
            
            # Calculate z-score with division by zero handling
            z_score = pd.Series(float('nan'), index=self.df.index)
            mask = rolling_std != 0
            
            # Only calculate where std is not zero
            z_score[mask] = (self.df[col][mask] - rolling_mean[mask]) / rolling_std[mask]
            
            # For cases where std is zero, set z-score to 0
            z_score[~mask & ~rolling_mean.isna()] = 0
            
            # Add to dataframe
            self.df[f'{col}_zscore'] = z_score