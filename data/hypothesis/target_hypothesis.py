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
        """..."""
        pass
        