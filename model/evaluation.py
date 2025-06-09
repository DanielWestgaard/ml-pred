import logging
import sys
# Evaluation
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

from model.model_training import ModelTraining 


class ModelEvaluation():
    """
    Different approaches of evaluating a Trained Models abilities. Also with a main orchestrator, 
    run(), for running all methods.
    """
    
    def __init__(self, dataset, trained_model):
        """Initialize the Evaluator."""
        logging.info(f"---------- Initializing Model Ealuation ----------")
        
        # Getting the training and test sets model was trained on to evaluate properly
        model_training = ModelTraining(dataset=dataset, model_type=trained_model.type)
        self.y_train, self.y_test, self.X_train, self.X_test = model_training.temporal_train_test_split(retur=True)
        
        # Model
        try:
            self.model = trained_model
            logging.debug(f"Have initialized model: {self.model}")
        except Exception as e:
            logging.error(f"Unable to get trained model: {self.model}")
        
    def run(self):
        """Main orchestrator for running all implemented evaluations."""
        self.evaluate_regression_model()
        self.plot_regression_model_prediction()
    
    def evaluate_regression_model(self):
        """
        Evaluate trained model. Further down the line, this could be so much more complex being its own file/component.
        Implemented methods: Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, Directional Accuracy
        
        Remember we cannot calculate accuracy for a regression model.
            "The skill or performance of a regression model must be reported as an error in those predictions."
            - https://machinelearningmastery.com/regression-metrics-for-machine-learning/
        """
        logging.info(f"----- Evaluate Model -----")
        self.y_pred = self.model.predict(self.X_test)
        
        # Mean Squared Error
        mse = mean_squared_error(self.y_test, self.y_pred)  # perfect mse value is 0.0, which means that all predictions matched the expected values exactly
        logging.info(f"Mean Squared Error: {mse:.4%}")
        
        # Root Mean Squared Error
        rmse = root_mean_squared_error(self.y_test, self.y_pred)
        logging.info(f"Root Mean Squared Error: {rmse:.4%}")  # perfect RMSE value is 0.0, which means that all predictions matched the expected values exactly
        
        # Mean Absolute Error
        mae = mean_absolute_error(self.y_test, self.y_pred)
        logging.info(f"Mean Absolute Error: {mae:.4%}")  # perfect mae value is 0.0, which means that all predictions matched the expected values exactly
        
        # Financial metrics
        directional_accuracy = np.mean(
            np.sign(self.y_pred - self.y_test.shift(1)) == np.sign(self.y_test - self.y_test.shift(1))
        )
        logging.info(f"Directional Accuracy: {directional_accuracy:.2%}")
    
    def plot_regression_model_prediction(self, save_plot:bool = True):
        """
        Plot actual test data vs. (trained models) prediction.
        Make sure the test dataset provided in init of class is the same one you trained the model on.
        
        This method reuses the train test data from model_training.py (done in this class' init),
        and makes model predict here before plotting.
        """
        logging.info(f"----- Plot Regression Models Prediction-----")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        try:
            test_dates = self.y_test.index
            logging.debug(f"date in y_test was index, which is correct. Using this for plotting.")
        except Exception as e:
            try:  # Date might not be index and just set as normal colum - weird, but try
                test_dates = self.y_test['date']
                logging.debug(f"date in y_test was normal date column. Should be index, but ok.")
            except Exception as e:
                logging.error(f"Unable to find/use date in the provided dataset: {e}")
                sys.exit()
        
        # df should have columns: date, open, high, low, close, volume
        # predictions should be array/series with same length as test data
        plt.figure(figsize=(15, 8))
        plt.plot(test_dates, self.y_test,
                 label=f"Actual '{self.y_test.name}' Price (Test Data)",
                 linewidth=2, alpha=0.8)
        plt.plot(test_dates, y_pred, label=f"Predicted '{self.y_test.name}'", linewidth=2, alpha=0.8)
        plt.xlabel('Datetime')
        plt.ylabel('Price/values')
        plt.title(f'{self.model.type} Model Price Prediction')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        if save_plot:
            # TODO: Save location in "models folder"
            plt.savefig("test.png")