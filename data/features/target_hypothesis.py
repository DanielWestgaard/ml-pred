import inspect
import numpy as np

from data.processing.base_processor import BaseProcessor
from utils import data_utils
from utils import shared_utils
from utils.logging_utils import get_logger

logger = get_logger()

class TargetHypothesis(BaseProcessor):
    """Change target of dataset according to the hypothesis."""
    def __init__(self, dataset, hypothesis:str):
        """
        Initialize hypothesis to change target column.
        
        Args:
            dataset: Processed data with specific features generated.
            hypothesis: The hypothesis you intend to test. Must match the name of the function you want to target!
        """
        try:
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            self.hypothesis = hypothesis
        except Exception as e:
            logger.error(f"Unable to initialize Target Hypothesis: {e}")
    
    def run(self):
        """
        Change target.
        
        returns:
            dataset: with added new target.
        """
        logger.info("---------- Running Hypothesis Targeting ----------")
        
        # Making a list of existing methods in this class
        functions = shared_utils.list_of_functions_in_class(path_to_class='data.features.target_hypothesis.TargetHypothesis')
        
        try:
            successfull = False
            for hypotehesis in functions:
                if hypotehesis == self.hypothesis:  # The one hypothesis user wants to target
                    successfull = shared_utils.safely_execute(self, hypotehesis)
            if not successfull:
                logger.error(f"Could not find {self.hypothesis} in the list of functions: {functions}. Make sure your name matches the function you want to target!")
        except Exception as e:
            logger.error(f"Error trying to find and execute hypothesis method: {e}")
        
        return self.df
    
    def session_based_tod_anomalies(self):
        """
        This target captures when future volatility is abnormally high relative to that specific session's
        typical behavior - directly testing the hypothesis that sessions behave differently.
        """
        try:
            if 'atr' in self.df.columns and 'primary_session' in self.df.columns:
                # Calculate session-specific ATR norms using primary_session feature
                self.df['session_atr_mean'] = self.df.groupby('primary_session')['atr'].transform(
                    lambda x: x.rolling(window=100, min_periods=20).mean()
                )
                self.df['session_atr_std'] = self.df.groupby('primary_session')['atr'].transform(
                    lambda x: x.rolling(window=100, min_periods=20).std()
                )
                
                # Future volatility (next 30 minutes, assuming 1-min bars)
                self.df['future_atr'] = self.df['atr'].shift(-30).rolling(window=30).mean()
                # Target: Future volatility > session mean + 1.5 * session std
                self.df['target_session_vol_anomaly'] = (
                    (self.df['future_atr'] > self.df['session_atr_mean'] + 1.5 * self.df['session_atr_std'])
                ).astype(int)
                
                logger.debug(f"Target column, target_session_vol_anomaly, for hypothesis session_based_tod_anomalies created.")
            else:
                logger.error(f"Target hypothesis column is not possible to calculate based on missing needed features: 'atr' and 'primary_session'")
        except Exception as e:
            logger.error(f"Unable to calculate the target hypothesis column: {e}")
    
    def spread_compression_patterns(self):
        """
        Target for spread compression pattern hypothesis.
        
        This targets moments when spread compression leads to significant price movements,
        particularly during London session trading hours when market liquidity changes.
        """
        try:
            # Required features for this hypothesis
            required_features = [
                'compression_ratio', 'spread_volatility_ewm', 'compression_consistency',
                'volume_ratio', 'london_session', 'signal_strength_compression'
            ]
            
            missing_features = [f for f in required_features if f not in self.df.columns]
            if missing_features:
                logger.error(f"Missing required features for spread_compression_patterns: {missing_features}")
                return
            
            # Future price movement (next 15 periods ahead for 5m data = 75 minutes)
            self.df['future_price_change'] = (
                self.df['close'].shift(-15) - self.df['close']
            ) / self.df['close']
            
            # Define significant movement threshold (0.15% for GBPUSD)
            movement_threshold = 0.0015
            
            # Target conditions:
            # 1. Strong compression ratio (> 0.1 = 10% compression)
            # 2. Consistent compression (> 0.5)
            # 3. Volume confirmation (> 1.2 = 20% above average)
            # 4. During London session hours (bonus weight)
            # 5. Results in significant price movement
            
            compression_signal = (
                (self.df['compression_ratio'] > 0.1) &
                (self.df['compression_consistency'] > 0.5) &
                (self.df['volume_ratio'] > 1.2)
            )
            
            # Enhanced signal during London session
            london_enhanced_signal = (
                compression_signal & 
                (self.df['london_session'] == 1)
            )
            
            # Future significant movement
            significant_movement = (
                abs(self.df['future_price_change']) > movement_threshold
            )
            
            # Target: Compression pattern that leads to significant movement
            # Higher weight for London session occurrences
            self.df['target_spread_compression'] = np.where(
                london_enhanced_signal & significant_movement, 2,  # Strong signal during London
                np.where(
                    compression_signal & significant_movement, 1,  # Regular signal
                    0  # No signal
                )
            ).astype(int)
            
            # Create binary version for simpler classification
            self.df['target_spread_compression_binary'] = (
                self.df['target_spread_compression'] > 0
            ).astype(int)
            
            logger.info(f"Target columns for spread_compression_patterns hypothesis created.")
            logger.info(f"Target distribution: {self.df['target_spread_compression'].value_counts().to_dict()}")
            
        except Exception as e:
            logger.error(f"Unable to calculate target for spread_compression_patterns hypothesis: {e}")
    
    # IMPORTANT FOR FURTHER DEVELOPMENT: Remember to add the prefix 'target_*' for the target column to the hypothesis.
    # This is because the normalization step after generation and this step will not normalize columns that starts with 'target_'.