from data.processing.cleaning import DataCleaner
from data.processing.validation import DataValidator
from utils import data_utils
from utils.data_utils import check_and_return_df
from utils.performance_utils import memory_monitor, PerformanceContext

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
        self.cleaner = DataCleaner(raw_dataset=self.df, save_cleaned_data=False)
        
        # Split into train/test
        # ...
        # 
    
    def run(self):
        """Run the hypothesis validator on you hypothesis."""
        # Clean
        with PerformanceContext("Data Cleaning"):
            self.df = self.cleaner.run()
            memory_monitor.safe_cleanup()  # Cleanup after cleaning
        
        # Validate
        with PerformanceContext("Data Validation"):
            self.validator = DataValidator(self.df)
            self.validator_results = self.validator.run()
            self.df = self.validator_results["validated_data"]
            # logger results
            data_utils.check_validation(self.validator_results["is_valid"], self.validator_results["issues"])
            memory_monitor.safe_cleanup()  # Cleanup after validation
        