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
        """Verify proper relationships between OHLC prices."""
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
        """..."""
        
    def _validate_volume(self):
        """..."""
        
    def _validate_price_movement(self):
        """..."""
        
    def _validate_data_completeness(self):
        """..."""
    