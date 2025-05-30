import logging
import inspect

from data.processing.base_processor import BaseProcessor
from utils import data_utils


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
            logging.error(f"Unable to initialize Target Hypothesis: {e}")
    
    def run(self):
        """
        Change target.
        
        returns:
            dataset: with added new target.
        """
        logging.info("---------- Running Hypothesis Targeting ----------")
        
        # Making a list of existing methods in this class
        functions = []
        never_mind = ['__init__', 'run', 'safely_execute']
        for method in inspect.getmembers(TargetHypothesis, predicate=inspect.isfunction):
            if method[0] not in never_mind:  # Won't add the 'basic' funtions, only the hypothesis ones
                functions.append(method[0])
        
        try:
            successfull = False
            for hypotehesis in functions:
                if hypotehesis == self.hypothesis:  # The one hypothesis user wants to target
                    successfull = self.safely_execute(hypotehesis)
            if not successfull:
                logging.error(f"Could not find {self.hypothesis} in the list of functions: {functions}. Make sure your name matches the function you want to target!")
        except Exception as e:
            logging.error(f"Error trying to find and execute hypothesis method: {e}")
        
        return self.df
    
    def safely_execute(self, function_name):  # Similar to the one in features/generation.py
        """Safely execute a hypothesis method with proper error handling."""
        try:
            method = getattr(self, function_name)
            method()
            
            logging.debug(f"Hypothesis {function_name} successfully changed target.")
            return True
        except Exception as e:
            logging.error(f"Error running {function_name} method: {e}")
            return False
    
    def session_based_tod_anomalies(self):
        """
        This target captures when future volatility is abnormally high relative to that specific session's
        typical behavior - directly testing the hypothesis that sessions behave differently.
        """
        # Calculate session-specific volatility norms
        self.df['session_vol_mean'] = self.df.groupby('fx_trading_sessions')['true_range'].transform(
            lambda x: x.rolling(window=100, min_periods=20).mean()
        )
        self.df['session_vol_std'] = self.df.groupby('fx_trading_sessions')['true_range'].transform(
            lambda x: x.rolling(window=100, min_periods=20).std()
        )
        # Future volatility (next 30 minutes assuming 1-min bars)
        self.df['future_volatility'] = self.df['true_range'].shift(-30).rolling(window=30).mean()
        
        self.df['target_vol_anomaly'] = (
            (self.df['future_volatility'] > self.df['session_vol_mean'] + 1.5 * self.df['session_vol_std'])
        ).astype(int)
        logging.debug(f"Target column for hypothesis session_based_tod_anomalies: {self.df['target_vol_anomaly']}")
        