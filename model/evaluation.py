import logging
import os
import pickle
import sys
import pandas as pd
# Evaluation
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

from model.model_training import ModelTraining
# from utils.model_utils import ModelMetadataLoader
from utils.model_utils import load_model_metadata, find_models_by_type, get_latest_model
import config.config as config


class ModelEvaluation():
    """
    Different approaches of evaluating a Trained Models abilities. Also with a main orchestrator, 
    run(), for running all methods.
    """
    
    def __init__(self, model_directory:str = None, save_evaluations:bool = True, model_type:str = None):
        """
        Initialize the Evaluator.
        If both trained_model and dataset are None, we assume the evaluation will be done on a 'pre-trained'
        model, hence we need to extract the model and data from the models metadatafile.
        
        Args:
            - model_directory: Path to the model directory to evaluate
            - save_evaluations: Whether to save evaluation results
            - model_type: Only for use of a newly trained model. This is used to find the newest trained model of this type.
        """
        logging.info(f"---------- Initializing Model Evaluation ----------")
        
        # Suppress matplotlib's debug font messages
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        
        # User can now both provide the path to model they want to evaluate, and not - meaning we get the newest for them
        if model_directory is not None:  # Model path is provided
            self.model_directory = model_directory
        else:  # Model path is not provided
            self.model_directory = get_latest_model(models_directory = config.TRAINED_MODEL_STORAGE, model_type = model_type)
            
        try:
            # Load the metadata
            loader = load_model_metadata(self.model_directory)
            # Get data path
            dataset = loader.get_dataset_path()
            
            # Check if dataset path is valid
            if dataset is None or dataset == 'unknown' or not os.path.exists(dataset):
                logging.error(f"Dataset path from metadata is invalid: {dataset}")
                logging.info("This might be due to a race condition with fresh model evaluation.")
                logging.info("Trying to find the dataset from the model directory...")
                
                # Fallback: try to find the dataset from the model type and recent files
                # This is a workaround for when metadata hasn't been fully written
                raise ValueError("Invalid dataset path, will retry with fallback method")
            
            # Get model file path
            model_type = loader.get_model_type()
            model_file = os.path.join(self.model_directory, f"{model_type}.pkl")
            
            # Load the actual model
            with open(model_file, 'rb') as file:
                self.model = pickle.load(file)
            
            # Dynamically using target column used for training the model (important for different models trained on different hypotheses)
            self.target_col = loader.get_target_column()
            
            # Getting the training and test sets model was trained on to evaluate properly
            model_training = ModelTraining(dataset=dataset, model_type=model_type, target_col = self.target_col)
            self.y_train, self.y_test, self.X_train, self.X_test = model_training.temporal_train_test_split(retur=True)
            
            self.save_evaluations = save_evaluations
            # Storing results under the models directory and evaluation. Create directory if it doesn't exist.
            self.evaluation_path = os.path.join(self.model_directory, 'evaluation')
            if not os.path.exists(self.evaluation_path):
                logging.debug(f"Creating directory now.")
                os.makedirs(self.evaluation_path)
            
            logging.debug(f"Have initialized model: {self.model}")
            
        except Exception as e:
            logging.error(f"Unable to initialize Model Evaluation: {e}")
            logging.info("This might be due to timing issues with fresh model evaluation.")
            logging.info("Try running the evaluation separately after a moment:")
            logging.info(f"python main.py -evo '{self.model_directory}'")
            sys.exit()
        
    def run(self):
        """Main orchestrator for running all implemented evaluations."""
        self.evaluate_regression_model()
        self.plot_regression_model_prediction(save_plot=self.save_evaluations)
    
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
        
            # Convert to pandas Series if not already for consistent handling
        if not isinstance(self.y_test, pd.Series):
            y_test = pd.Series(self.y_test)
            y_pred = pd.Series(self.y_pred, index=y_test.index)
        else:
            y_test = self.y_test
            y_pred = pd.Series(self.y_pred, index=y_test.index)
        
        # === Traditional Regression Metrics ===
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = root_mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        
        # === Financial-Specific Metrics ===
        
        # Directional Accuracy
        # Predicting direction of price movement
        actual_direction = np.sign(y_test.diff())
        pred_direction = np.sign(y_pred.diff())
        # Remove NaN values from the comparison (first element will be NaN due to diff)
        valid_mask = ~(np.isnan(actual_direction) | np.isnan(pred_direction))
        directional_accuracy = np.mean(actual_direction[valid_mask] == pred_direction[valid_mask])
    
        
        # Hit Rate (percentage of profitable predictions)
        # Assuming we're predicting returns
        hit_rate = np.mean((self.y_pred > 0) == (self.y_test > 0))
        
        # Prediction-based Returns
        # Simple strategy: go long when prediction is positive
        strategy_returns = np.where(self.y_pred > 0, self.y_test, 0)
        cumulative_strategy_return = np.sum(strategy_returns)
        
        # Sharpe Ratio (risk-adjusted return)
        if len(strategy_returns) > 1 and np.std(strategy_returns) != 0:
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar Ratio (return/max_drawdown)
        calmar_ratio = cumulative_strategy_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win/Loss Ratio
        profitable_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        
        if len(losing_trades) > 0:
            avg_win = np.mean(profitable_trades) if len(profitable_trades) > 0 else 0
            avg_loss = np.mean(np.abs(losing_trades))
            win_loss_ratio = avg_win / avg_loss
        else:
            win_loss_ratio = float('inf') if len(profitable_trades) > 0 else 0
        
        # 8. Information Coefficient (IC)
        # Correlation between predictions and actual returns
        ic = np.corrcoef(self.y_pred, self.y_test)[0, 1] if len(self.y_pred) > 1 else 0
        
        # === Logging Results ===
        logging.info(f"=== Regression Metrics ===")
        logging.info(f"Mean Squared Error: {mse:.6f}")
        logging.info(f"Root Mean Squared Error: {rmse:.6f}")
        logging.info(f"Mean Absolute Error: {mae:.6f}")
        
        logging.info(f"=== Trading Performance Metrics ===")
        logging.info(f"Directional Accuracy: {directional_accuracy:.2%}")
        logging.info(f"Hit Rate: {hit_rate:.2%}")
        logging.info(f"Information Coefficient: {ic:.4f}")
        
        logging.info(f"=== Risk-Adjusted Metrics ===")
        logging.info(f"Total Strategy Return: {cumulative_strategy_return:.4f}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        logging.info(f"Maximum Drawdown: {max_drawdown:.4f}")
        logging.info(f"Calmar Ratio: {calmar_ratio:.4f}")
        logging.info(f"Win/Loss Ratio: {win_loss_ratio:.4f}")
        
        # Store metrics for potential saving (convert numpy types to native Python types for JSON compatibility)
        self.evaluation_metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'directional_accuracy': float(directional_accuracy),
            'hit_rate': float(hit_rate),
            'information_coefficient': float(ic) if not np.isnan(ic) else None,
            'total_return': float(cumulative_strategy_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio) if not np.isinf(calmar_ratio) else None,
            'win_loss_ratio': float(win_loss_ratio) if not np.isinf(win_loss_ratio) else None
        }
        
        # Save metrics to file (similar to plot saving)
        if self.save_evaluations:
            import json
            try:
                metrics_file = os.path.join(self.evaluation_path, 'evaluation_metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump(self.evaluation_metrics, f, indent=4)
                logging.info(f"Evaluation metrics saved to: {metrics_file}")
            except Exception as e:
                logging.error(f"Unable to save metrics under Models folder, '{self.model_directory}'. Falling back to current dir: {e}")
                try:
                    with open("evaluation_metrics.json", 'w') as f:
                        json.dump(self.evaluation_metrics, f, indent=4)
                    logging.info("Evaluation metrics saved to: evaluation_metrics.json")
                except Exception as e2:
                    logging.error(f"Failed to save metrics entirely: {e2}")
        
        logging.info(self.evaluation_metrics)
        
        return self.evaluation_metrics
    
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
        
        # Get dates for plotting
        try:
            test_dates = self.y_test.index
        except:
            test_dates = range(len(self.y_test))
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.model.type} Model Comprehensive Evaluation', fontsize=16)
        
        # 1. Actual vs Predicted
        axes[0, 0].plot(test_dates, self.y_test, label='Actual', alpha=0.8)
        axes[0, 0].plot(test_dates, y_pred, label='Predicted', alpha=0.8)
        axes[0, 0].set_title('Actual vs Predicted Values')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Residuals
        residuals = self.y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        
        # 3. Cumulative Returns
        strategy_returns = np.where(y_pred > 0, self.y_test, 0)
        cumulative_returns = np.cumsum(strategy_returns)
        axes[1, 0].plot(test_dates, cumulative_returns)
        axes[1, 0].set_title('Cumulative Strategy Returns')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Prediction Distribution
        axes[1, 1].hist(y_pred, bins=30, alpha=0.7, label='Predictions')
        axes[1, 1].hist(self.y_test, bins=30, alpha=0.7, label='Actual')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plot:
            try:
                plt.savefig(f"{self.evaluation_path}/enhanced_evaluation_plot.png", dpi=300, bbox_inches='tight')
            except Exception as e:
                logging.error(f"Unable to save plot: {e}")
                plt.savefig("enhanced_evaluation_plot.png", dpi=300, bbox_inches='tight')
        
        plt.show()