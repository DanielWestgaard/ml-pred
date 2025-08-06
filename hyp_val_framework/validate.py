from utils.data_utils import check_and_return_df
from utils.logging_utils import get_logger

logger = get_logger()

class HypothesisValidator:
    def __init__(self, dataset):
        """
        Base Hypothesis Validator class.
        
        Args:
            dataset: Dataset you want to test your hypothesis on. Can be either path to dataset or actual df
        """
        # Get the data
        self.df, self.og_df = check_and_return_df(dataset)
        # Clean & validate data
        # ...
        # Split into train/test
        # ...
        # 
    
    def run(self):
        """Run the hypothesis validator on you hypothesis."""