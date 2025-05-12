import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data.processing.base_processor import BaseProcessor
from utils import data_utils


class DataTransformer(BaseProcessor):
    def __init__(self, data):
        """Class for transforming- (handling nan's, etc.) and normalizing features."""
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(data)
    
    def run(self, preserve_original=False, window=20):
        """
        Run feature normalization.
        
        Args:
            preserve_original: If True, keep original features and add normalized ones with suffix
            window: Size of rolling window for normalization statistics
        """
        # TODO: Handle missing feature values
        self.handle_missing_values()
        
        
        # Normalization
        # First identify categorical columns to exclude from normalization
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Also identify binary/regime columns (that might be numeric but shouldn't be normalized)
        regime_indicators = [col for col in self.df.columns if 
                            any(x in col for x in ['regime', 'is_', 'in_value_area', 'state'])]
        exclude_from_normalization = categorical_cols + regime_indicators
        
        # Group 1: Price data (OHLC) - Z-score normalize
        price_cols = ['open', 'high', 'low', 'close']
        price_cols_present = [col for col in price_cols if col in self.df.columns 
                            and col not in exclude_from_normalization]
        
        # Group 2: Volume data - Log transform then Z-score
        volume_cols = ['volume'] + [col for col in self.df.columns 
                                if 'volume' in col.lower() and 'relative' not in col.lower()]
        volume_cols = [col for col in volume_cols if col in self.df.columns 
                    and col not in exclude_from_normalization]
        
        # Group 3: Unbounded features - Z-score
        z_score_features = [col for col in self.df.columns if 
                        any(x in col for x in ['roc_', 'macd', 'vwap', 'log_return', 'distance', 'vol_', 
                                            'atr', 'bb_width', 'obv', 'cci', 'adx'])]
        z_score_features = [col for col in z_score_features if col not in exclude_from_normalization]
        
        
        # Group 4: Bounded indicators - Min-Max
        minmax_features = [col for col in self.df.columns if 
                        any(x in col for x in ['rsi', 'stoch', 'mfi', '_sin_', '_cos_'])]
        minmax_features = [col for col in minmax_features if col not in exclude_from_normalization]

        
        # Log transform volume first
        for col in volume_cols:
            if preserve_original:
                self.df[f"{col}_log"] = np.log1p(self.df[col])  # log(1+x) to handle zeros
            else:
                self.df[col] = np.log1p(self.df[col])
        
        # Apply z-score to price, transformed volume, and other unbounded features
        all_zscore_cols = price_cols_present + volume_cols + z_score_features
        self.z_score(columns=all_zscore_cols, window=window, 
                    preserve_original=preserve_original)
        
        # Apply min-max to bounded features using rolling window
        if minmax_features:
            self.rolling_minmax(columns=minmax_features, window=window,
                              preserve_original=preserve_original)
        
        return self.df
    
    def handle_missing_values (self):
        """Method for handling missing values in features."""
        pass
    
    def z_score(self, columns, window=20, preserve_original=False):
        """
        Z-score normalize selected features using rolling window.
        
        Parameters:
        -----------
        columns : list
            List of columns to normalize
        window : int
            The rolling window to use for calculating mean and standard deviation
        preserve_original : bool
            If True, keeps original columns and adds new ones with _norm suffix
        """
        for col in columns:
            if col not in self.df.columns:
                continue
            
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
            
            # Forward fill NaN values from initial window
            z_score = z_score.fillna(method='bfill')
            
            # Add to dataframe
            if preserve_original:
                self.df[f'{col}_norm'] = z_score
            else:
                if col == 'close':  # preserve original closing value (for placing trades)
                    self.df['close_og'] = self.df['close']
                self.df[col] = z_score
    
    def rolling_minmax(self, columns, window=20, preserve_original=False):
        """
        Apply min-max scaling using rolling window to avoid lookahead bias.
        
        Parameters:
        -----------
        columns : list
            List of columns to normalize
        window : int
            The rolling window to use for calculating min and max
        preserve_original : bool
            If True, keeps original columns and adds new ones with _norm suffix
        """
        for col in columns:
            if col not in self.df.columns:
                continue
                
            # Calculate rolling min and max
            rolling_min = self.df[col].rolling(window=window).min()
            rolling_max = self.df[col].rolling(window=window).max()
            
            # Calculate min-max scaled values with division by zero handling
            scaled = pd.Series(float('nan'), index=self.df.index)
            mask = (rolling_max - rolling_min) != 0
            
            # Only calculate where range is not zero
            scaled[mask] = (self.df[col][mask] - rolling_min[mask]) / (rolling_max[mask] - rolling_min[mask])
            
            # For cases where range is zero, set to 0.5 (middle of range)
            scaled[~mask & ~rolling_min.isna()] = 0.5
            
            # Forward fill NaN values from initial window
            scaled = scaled.fillna(method='bfill')
            
            # Add to dataframe
            if preserve_original:
                self.df[f'{col}_norm'] = scaled
            else:
                self.df[col] = scaled