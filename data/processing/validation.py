from data.processing.base_processor import BaseProcessor
from utils import data_utils

class DataValidator(BaseProcessor):
    def __init__(self, clean_dataset):
        """
        Class for validating market data. Preferably after the data has been cleaned.
        
        Args:
            clean_dataset: path to the raw dataset (csv format). Can be either path to csv or already loaded DataFrame.
        """
        self.validation_issues = []
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(clean_dataset)
        
    def run(self):
        """Run data validation checks and returns validated results."""
        
        self._validate_ohlc_relationships()
        self._validate_timestamps()
        self._validate_volume()
        self._validate_price_movement()
        self._validate_data_completeness()
        
        return {
            'is_valid': len(self.validation_issues) == 0,
            'issues': self.validation_issues,
            'validated_data': self.df
        }
    
    def _validate_ohlc_relationships(self):
        """Verify proper relationships between OHLC prices. Similar to the one in 
        Cleaning, but this is to validate that work"""
        # High should be the highest value
        high_issues = self.df[self.df['high'] < self.df[['open', 'close']].max(axis=1)]
        if not high_issues.empty:
            self.validation_issues.append({
                'type': 'OHLC Relationship',
                'description': f'Found {len(high_issues)} records where high < max(open, close)',
                'affected_rows': high_issues.index.tolist()
            })
            
        # Low should be the lowest value 
        low_issues = self.df[self.df['low'] > self.df[['open', 'close']].min(axis=1)]
        if not low_issues.empty:
            self.validation_issues.append({
                'type': 'OHLC Relationship',
                'description': f'Found {len(low_issues)} records where low > min(open, close)',
                'affected_rows': low_issues.index.tolist()
            })
            
        # Check for negative or zero prices
        for col in ['open', 'high', 'low', 'close']:
            invalid_prices = self.df[self.df[col] <= 0]
            if not invalid_prices.empty:
                self.validation_issues.append({
                    'type': 'Invalid Prices',
                    'description': f'Found {len(invalid_prices)} records with {col} <= 0',
                    'affected_rows': invalid_prices.index.tolist()
                })
    
    def _validate_timestamps(self):
        """Validate time series integrity and consistency."""
        # Check for duplicate timestamps
        duplicates = self.df.index.duplicated()
        if any(duplicates):
            self.validation_issues.append({
                'type': 'Duplicate Timestamps',
                'description': f'Found {duplicates.sum()} duplicate timestamps',
                'affected_rows': self.df.index[duplicates].tolist()
            })
            
        # Check for gaps in time series
        if len(self.df) > 1:
            # Determine expected frequency
            time_diffs = self.df.index.to_series().diff().dropna()
            common_diff = time_diffs.mode().iloc[0]
            
            # Identify gaps larger than 1.5x expected frequency
            gaps = time_diffs[time_diffs > common_diff * 1.5]
            if not gaps.empty:
                self.validation_issues.append({
                    'type': 'Time Series Gaps',
                    'description': f'Found {len(gaps)} gaps in time series',
                    'details': {str(idx): str(gap) for idx, gap in gaps.items()}
                })
                
        # Check for future timestamps
        import datetime
        future_data = self.df[self.df.index > datetime.datetime.now(self.df.index.tzinfo)]
        if not future_data.empty:
            self.validation_issues.append({
                'type': 'Future Timestamps',
                'description': f'Found {len(future_data)} records with future timestamps',
                'affected_rows': future_data.index.tolist()
            })
        
    def _validate_volume(self):
        """Validate volume data."""
        if 'volume' in self.df.columns:
            # Check for negative volume
            negative_volume = self.df[self.df['volume'] < 0]
            if not negative_volume.empty:
                self.validation_issues.append({
                    'type': 'Invalid Volume',
                    'description': f'Found {len(negative_volume)} records with negative volume',
                    'affected_rows': negative_volume.index.tolist()
                })
                
            # Check for suspiciously high volume (outliers)
            import numpy as np
            log_volume = np.log1p(self.df['volume'])
            mean, std = log_volume.mean(), log_volume.std()
            high_volume = self.df[log_volume > mean + 5*std]  # 5-sigma events
            
            if not high_volume.empty:
                self.validation_issues.append({
                    'type': 'Suspicious Volume',
                    'description': f'Found {len(high_volume)} records with abnormally high volume',
                    'affected_rows': high_volume.index.tolist()
                })
        
    def _validate_price_movement(self):
        """Validate price movements for extreme or suspicious patterns."""
        # Calculate returns
        self.df['return'] = self.df['close'].pct_change()
        
        # Check for extreme returns (potential data errors)
        extreme_returns = self.df[abs(self.df['return']) > 0.20]  # 20% move
        if not extreme_returns.empty:
            self.validation_issues.append({
                'type': 'Extreme Price Movement',
                'description': f'Found {len(extreme_returns)} records with >20% price moves',
                'affected_rows': extreme_returns.index.tolist()
            })
            
        # Remove temporary column
        self.df = self.df.drop('return', axis=1)
        
    def _validate_data_completeness(self):
        """Validate data completeness and required fields."""
        # Check for minimum required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            self.validation_issues.append({
                'type': 'Missing Required Columns',
                'description': f'Missing columns: {", ".join(missing_cols)}'
            })
            
        # Check for minimum data points required for analysis
        if len(self.df) < 30:  # Example threshold
            self.validation_issues.append({
                'type': 'Insufficient Data',
                'description': f'Only {len(self.df)} data points available, minimum 30 required'
            })
    

class FeatureValidator(BaseProcessor):
    def __init__(self, data):
        """Class for validating newly generated features."""
        self.validation_issues = []
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(data)
        
    def run(self):
        """Run validation of data and features after generation."""
        pass