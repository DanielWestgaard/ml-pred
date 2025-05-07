import numpy as np

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
        
        # Price Action Features
        self._moving_averages()
        self._rate_of_change()
        self._average_true_range()  # wrong implementation?
        self._bollinger_bands()  # bad implementation?
        self._support_resistance()  # not sure how to do, yet
        # Volume-Based Features
        self._volume_moving_averages()
        self._volume_rate_of_change()
        self._on_balance_volume()
        # Technical Indicators
        # self._relative_strength_index()  # unsure of impmlementation - also now working
        
        print(self.df)
        
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
        
    def _rate_of_change(self):
        """Price momentum: Rate of change (ROC) over various lookback periods."""
        self.df["roc_close"] = self.df["close"].pct_change()  # Am I using this right?
    
    def _average_true_range(self, period_atr:int = 14):
        # ATR
        self.df["high_low"] = self.df["high"] - self.df["low"]                          # Current High — Current Low
        self.df["high_prev_close"] = abs(self.df["high"] - self.df["close"].shift(1))   # |Current High — Previous Close|
        self.df["low_prev_close"] = abs(self.df["low"] - self.df["close"].shift(1))     # |Current Low — Previous Close|
        self.df["true_range"] = self.df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
        
        self.df["atr"] = self.df["true_range"].rolling(window=period_atr).mean()
        # Dropping used columns
        self.df = self.df.drop(columns=["high_low", "high_prev_close", "low_prev_close", "true_range"], axis=1)

    def _bollinger_bands(self, period_bb:int = 20):
        # Bollinger Bands
        # middle band is sma
        self.df["std"] = self.df["close"].rolling(window=period_bb).std()  # rolling standard deviation
        num_std_dev = 2
        self.df['upper_band'] = self.df[f'sma_{period_bb}'] + (num_std_dev * self.df['std'])
        self.df['lower_band'] = self.df[f'sma_{period_bb}'] - (num_std_dev * self.df['std'])
        
    def _support_resistance(self):
        """Distance from recent high/lows."""
        pass
    
        
    # =============================================================================
    # Section: Volume-Based Features
    # =============================================================================
    def _volume_moving_averages(self):
        """Identify unusual volume spikes."""
        num_periods = [7, 15, 30, 60]
        
        for period in num_periods:
            self.df[f'vma_{period}'] = self.df['volume'].rolling(window=period).mean()
        
    def _volume_rate_of_change(self):
        """How rapidly volume is increasing/decreasing."""
        self.df["vroc"] = self.df["volume"].pct_change()  # Am I using this right?
        
    def _on_balance_volume(self):
        """Cumulative indicator that relates volume to price changes. Could also be placed under Technical Indicators."""
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df['close'][i] > self.df['close'][i-1]:
                obv.append(float(obv[-1] + self.df['volume'][i]))
            elif self.df['close'][i] < self.df['close'][i-1]:
                obv.append(float(obv[-1] - self.df['volume'][i]))
            else:
                obv.append(float(obv[-1]))
        self.df["obv"] = obv
        
    
    # =============================================================================
    # Section: Technical Indicators
    # =============================================================================
    def _relative_strength_index(self):
        """Measures momentum and overbought/oversold conditions."""
        
        def rsi_calc(self, prices, n=14):
                deltas = np.diff(prices)
                seed = deltas[:n+1]
                up = seed[seed>=0].sum()/n
                down = -seed[seed<0].sum()/n
                rs = up/down
                rsi = np.zeros_like(prices)
                rsi[:n] = 100. - 100./(1.+rs)

                for i in range(n, len(prices)):
                    delta = deltas[i-1]  # cause the diff is 1 shorter

                    if delta>0:
                        upval = delta
                        downval = 0.
                    else:
                        upval = 0.
                        downval = -delta

                    up = (up*(n-1) + upval)/n
                    down = (down*(n-1) + downval)/n

                    rs = up/down
                    rsi[i] = 100. - 100./(1.+rs)

                return rsi
        
        number = [6, 14, 20]
        for nr in number:
            self.df[f"rsi_{nr}"] = rsi_calc(self.df['close'].values, nr)
    
    # =============================================================================
    # Section: Time-based Features
    # =============================================================================
    
    
    # =============================================================================
    # Section: Market Regime Features
    # =============================================================================
    
    
    # =============================================================================
    # Section: Feature Transformations
    # =============================================================================
        