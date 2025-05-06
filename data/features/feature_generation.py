from utils import data_utils


class FeatureGenerator():
    def __init__(self, dataset):
        """
        Class focusing on generating features with helpful prediction powers,
        that might help a ML model learn and predict better.
        
        Args:
            dataset: The cleaned (and validated) data, containing (index) date/datetime, and the columns OHLCV.
        
        returns:
            dataset with all generated features 
        """
    
        # Load dataset based on format
        self.df, self.original_df = data_utils.check_and_return_df(dataset)
        
    def run(self):
        """Run feature generation on provided dataset."""
        
        self._moving_averages()
        self._price_momentum()
        self._volatility_indicators
        
        #print(self.df)
        
        return self.df
    
    # =============================================================================
    # Section: Price Action Features
    # =============================================================================
    
    def _moving_averages(self):
        """Generating/calculating both simple and exponential moving averages over different time periods."""
        num_periods = [5, 10, 20, 50, 200]
        
        for period in num_periods:
            self.df[f"sma_{period}"] = self.df["close"].rolling(window=period).mean()
            self.df[f"ema_{period}"] = self.df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
        
    def _price_momentum(self):
        """Rate of change (ROC) over various lookback periods."""
        self.df["rdc_close"] = self.df["close"].pct_change()  # Am I using this right?
    
    def _volatility_indicators(self):
        """Average True Range (ATR), Bollinger Band width."""
        pass