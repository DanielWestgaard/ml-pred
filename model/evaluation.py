import logging
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
    
    def plot_regression_model_prediction(self):
        """
        Plot trained models prediction with dataset.
        Will have model predict first. I'm sure there is a smarter way of doing this, but works for now.
        """
        logging.info(f"----- Plot Regression Models Prediction-----")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()
    