import logging
import os
from datetime import time
import sys
from .base_evaluation import BaseEvaluation
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

class SessionVolumeAnomalyEvaluation(BaseEvaluation):
    """
    Evaluation specific to session-based volume anomaly prediction hypothesis
    """
    
    def evaluate_model(self):
        """Evaluate session volume anomaly model"""
        """
        Specialized evaluation for session-based volume anomaly prediction
        """
        print("HELLOOOOO")
        logging.info(f"----- Session-Based Volume Anomaly Evaluation -----")
        
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Convert to pandas Series for easier handling
        if not isinstance(self.y_test, pd.Series):
            y_test = pd.Series(self.y_test)
            y_pred = pd.Series(self.y_pred, index=y_test.index)
        else:
            y_test = self.y_test
            y_pred = pd.Series(self.y_pred, index=y_test.index)
        
        # === Basic Regression Metrics ===
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # === Anomaly Detection Specific Metrics ===
        
        # 1. Anomaly Detection Accuracy (treating as binary classification)
        # Define anomaly threshold (you might want to make this configurable)
        anomaly_threshold = y_test.quantile(0.8)  # Top 20% as anomalies
        
        actual_anomalies = y_test > anomaly_threshold
        predicted_anomalies = y_pred > anomaly_threshold
        
        # Binary classification metrics
        true_positives = np.sum(actual_anomalies & predicted_anomalies)
        false_positives = np.sum(~actual_anomalies & predicted_anomalies)
        true_negatives = np.sum(~actual_anomalies & ~predicted_anomalies)
        false_negatives = np.sum(actual_anomalies & ~predicted_anomalies)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 2. Anomaly Magnitude Accuracy (how well do we predict the size of anomalies)
        anomaly_mask = actual_anomalies
        if np.sum(anomaly_mask) > 0:
            anomaly_mae = mean_absolute_error(y_test[anomaly_mask], y_pred[anomaly_mask])
            anomaly_correlation = np.corrcoef(y_test[anomaly_mask], y_pred[anomaly_mask])[0, 1]
        else:
            anomaly_mae = float('nan')
            anomaly_correlation = float('nan')
        
        # === Session-Specific Analysis ===
        
        # Define trading sessions (you might want to adjust these times)
        sessions = {
            'Asian': (time(0, 0), time(9, 0)),      # 00:00-09:00 UTC
            'European': (time(7, 0), time(16, 0)),   # 07:00-16:00 UTC  
            'US': (time(13, 0), time(22, 0)),        # 13:00-22:00 UTC
            'Overlap_EU_US': (time(13, 0), time(16, 0))  # EU-US overlap
        }
        
        session_performance = {}
        
        # Create datetime index if it doesn't exist
        if hasattr(y_test.index, 'time'):
            test_times = y_test.index.time
        else:
            # If no datetime index, create dummy session analysis
            logging.warning("No datetime index found, skipping session-specific analysis")
            test_times = None
        
        if test_times is not None:
            for session_name, (start_time, end_time) in sessions.items():
                if start_time <= end_time:
                    session_mask = (test_times >= start_time) & (test_times <= end_time)
                else:  # Handle overnight sessions
                    session_mask = (test_times >= start_time) | (test_times <= end_time)
                
                if np.sum(session_mask) > 0:
                    session_y_test = y_test[session_mask]
                    session_y_pred = y_pred[session_mask]
                    
                    session_mae = mean_absolute_error(session_y_test, session_y_pred)
                    session_corr = np.corrcoef(session_y_test, session_y_pred)[0, 1] if len(session_y_test) > 1 else 0
                    
                    # Session-specific anomaly detection
                    session_anomaly_threshold = session_y_test.quantile(0.8)
                    session_actual_anomalies = session_y_test > session_anomaly_threshold
                    session_predicted_anomalies = session_y_pred > session_anomaly_threshold
                    
                    session_precision = np.sum(session_actual_anomalies & session_predicted_anomalies) / np.sum(session_predicted_anomalies) if np.sum(session_predicted_anomalies) > 0 else 0
                    
                    session_performance[session_name] = {
                        'sample_count': int(np.sum(session_mask)),
                        'mae': float(session_mae),
                        'correlation': float(session_corr) if not np.isnan(session_corr) else None,
                        'anomaly_precision': float(session_precision)
                    }
        
        # === Timing-Based Performance ===
        
        # Performance by hour of day (if datetime index available)
        hourly_performance = {}
        if test_times is not None and hasattr(y_test.index, 'hour'):
            for hour in range(24):
                hour_mask = y_test.index.hour == hour
                if np.sum(hour_mask) > 0:
                    hour_mae = mean_absolute_error(y_test[hour_mask], y_pred[hour_mask])
                    hourly_performance[hour] = {
                        'sample_count': int(np.sum(hour_mask)),
                        'mae': float(hour_mae)
                    }
        
        # === Trading Signal Quality ===
        
        # Generate trading signals based on anomaly predictions
        # Signal: 1 if predicted anomaly above threshold, 0 otherwise
        signal_threshold = y_pred.quantile(0.8)  # Top 20% predictions as signals
        trading_signals = (y_pred > signal_threshold).astype(int)
        
        # Calculate signal returns (assuming we trade on the anomaly)
        # This is a simplified approach - you'd need your actual trading logic
        signal_returns = trading_signals * y_test  # Simplified: return = signal * actual anomaly
        
        total_signal_return = np.sum(signal_returns)
        hit_rate = np.mean(signal_returns > 0) if np.sum(trading_signals) > 0 else 0
        
        # Average return per signal
        avg_return_per_signal = np.mean(signal_returns[trading_signals == 1]) if np.sum(trading_signals) > 0 else 0
        
        # === Store Results ===
        
        self.evaluation_metrics = {
            # Basic regression metrics
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            
            # Anomaly detection metrics
            'anomaly_precision': float(precision),
            'anomaly_recall': float(recall),
            'anomaly_f1_score': float(f1_score),
            'anomaly_mae': float(anomaly_mae) if not np.isnan(anomaly_mae) else None,
            'anomaly_correlation': float(anomaly_correlation) if not np.isnan(anomaly_correlation) else None,
            
            # Trading performance
            'total_signal_return': float(total_signal_return),
            'hit_rate': float(hit_rate),
            'avg_return_per_signal': float(avg_return_per_signal),
            'num_signals': int(np.sum(trading_signals)),
            
            # Session-specific performance
            'session_performance': session_performance,
            'hourly_performance': hourly_performance
        }
        
        # === Logging Results ===
        logging.info(f"=== Basic Regression Metrics ===")
        logging.info(f"Mean Absolute Error: {mae:.6f}")
        logging.info(f"Root Mean Squared Error: {rmse:.6f}")
        
        logging.info(f"=== Anomaly Detection Performance ===")
        logging.info(f"Anomaly Precision: {precision:.2%}")
        logging.info(f"Anomaly Recall: {recall:.2%}")
        logging.info(f"Anomaly F1-Score: {f1_score:.3f}")
        
        logging.info(f"=== Trading Signal Performance ===")
        logging.info(f"Number of Signals Generated: {np.sum(trading_signals)}")
        logging.info(f"Hit Rate: {hit_rate:.2%}")
        logging.info(f"Average Return per Signal: {avg_return_per_signal:.6f}")
        logging.info(f"Total Signal Return: {total_signal_return:.6f}")
        
        if session_performance:
            logging.info(f"=== Session-Specific Performance ===")
            for session, metrics in session_performance.items():
                logging.info(f"{session}: MAE={metrics['mae']:.6f}, Samples={metrics['sample_count']}, Precision={metrics['anomaly_precision']:.2%}")
        
        # Save metrics with same error handling pattern
        if self.save_evaluations:
            import json
            try:
                metrics_file = os.path.join(self.evaluation_path, 'session_anomaly_metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump(self.evaluation_metrics, f, indent=4)
                logging.info(f"Session anomaly metrics saved to: {metrics_file}")
            except Exception as e:
                logging.error(f"Unable to save metrics under Models folder, '{self.model_directory}'. Falling back to current dir: {e}")
                try:
                    with open("session_anomaly_metrics.json", 'w') as f:
                        json.dump(self.evaluation_metrics, f, indent=4)
                    logging.info("Session anomaly metrics saved to: session_anomaly_metrics.json")
                except Exception as e2:
                    logging.error(f"Failed to save metrics entirely: {e2}")
        
        return self.evaluation_metrics
    
    def plot_results(self, save_plot: bool = True):
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

        if save_plot:
            # Save location under models folder
            try:
                plt.savefig(f"{self.evaluation_path}/plot.png")
            except Exception as e:
                logging.error(f"Unable to save plot under Models folder, '{self.model_directory}'. Falling back to your current dir : {e}")
                plt.savefig("plot.png")
        
        plt.show()
        