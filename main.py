# When things are getting more complete, this file needs to be cleaned up as it will be the main file for running all other "systems".
import argparse
import json
import sys
import yaml

from config import config
import utils.logging_utils as log_utils
from broker.capital_com.capitalcom import CapitalCom
from data.providers.capital_com import ProviderCapitalCom
from data.providers.alpha_vantage import ProviderAlphaVantage
import utils.alpha_vantage_utils as alpha_utils
from data.pipelines.engineer_pipeline import EngineeringPipeline
import utils.shared_utils as shared_utils
from data.plot.plot_data import PlotData
from model.model_training import ModelTraining
from model.evaluation import ModelEvaluation
from utils.model_utils import get_latest_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train hybrid trading strategy models with uncertainty quantification')
    
    # Full Broker functionality
    parser.add_argument('--broker-func', action='store_true', default=False, help='Test all broker functionality')
    
    # Provider usage
    parser.add_argument('--fetch-data-capcom', action='store_true', default=False, help='Fetch historical data using Capital.com API.')
    parser.add_argument('--fetch-data-alpha', action='store_true', default=False, help='Fetch historical data using Alpha Vantage API.')
    
    # Plotting
    parser.add_argument('--plot-data', '-pd', default=False, help='Plot dataset containing candlestick data.')
    
    # Data Pipeline
    parser.add_argument('--engineer-data', '-eg', default=False, help='Run data engineering pipeline to engineer raw datasets (csv-format).')
    
    # Hypothesis 
    parser.add_argument('--hypothesis', '-hyp', default=False, help='Run data engineering pipeline with hypoethesis: Provide path to the config.yml file.')
    
    # Model Training
    parser.add_argument('--model-training', '-ml', default=False, help='Train ML Model on Engineered Data.')
    parser.add_argument('--model-type', '-mt', default=False, help='Type of model to train.')
    
    # Model Evaluation
    parser.add_argument('--evaluate-fresh', '-evf', action='store_true', default=False, help='Evaluate Trained model.')
    parser.add_argument('--evaluate-old', '-evo', default=False, help='Evaluate Trained ML Model using this dataset.')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    log_utils._is_configured = False
    logging = log_utils.setup_logging(name="engineering", type="data", log_to_file=False, log_level=config.DEBUG_LOG_LEVEL)
    
    if args.engineer_data:  # python main.py --engineer-data 'storage/data/capital_com/raw/raw_GBPUSD_m_20250101_20250523.csv' --hypothesis "hypothesis/1_session_based_tod_anomalies/config.yml"
        save_cleaned_data = True
        if args.hypothesis:
            logging.info("Trying to run data engineering with hypothesis config file.")
            
            data_pipeline = EngineeringPipeline(raw_dataset=args.engineer_data or "storage/data/capital_com/raw/raw_GBPUSD_m_20250501_20250501.csv", 
                                                save_cleaned_data=save_cleaned_data, hypothesis_source=args.hypothesis)
        else:
            data_pipeline = EngineeringPipeline(raw_dataset=args.engineer_data or "storage/data/capital_com/raw/raw_GBPUSD_m_20250501_20250501.csv",
                                                save_cleaned_data = save_cleaned_data)
        
        data_pipeline.run()

    if args.hypothesis:
        logging.info(f"User using pure hypothesis flow!")
        save_cleaned_data = True
        
        # Getting all the configurations
        conf = shared_utils.handle_hypothesis_source(args.hypothesis)
        conf_keys = conf.keys()
        
        # Data Pipeline
        if 'raw_data_source' in conf_keys:  # Found raw data path config, meaning user intends to run Data Pipeline?
            data_pipeline = EngineeringPipeline(save_cleaned_data=save_cleaned_data, hypothesis_source=args.hypothesis)
            data_pipeline.run()
        
        # Model Training
        if 'model_type' in conf_keys:  # Found Model Type in config, meaning user wants to use Model Training
            # Considering argument 'sending': 'Cheap' solution compared to Data Pipeline?
            # As Data Pipeline also has the capability to read the config, but here we send them. Though this might also be a cleaner solution?
            model_training = ModelTraining(dataset = conf.get('processed_data'), model_type = conf.get('model_type'))
            model = model_training.run()
            
            # Store the model directory for potential evaluation
            if hasattr(model_training, 'model_directory'):
                trained_model_dir = model_training.model_directory
            else:
                # Get the latest model directory that was just created
                trained_model_dir = get_latest_model(models_directory = config.TRAINED_MODEL_STORAGE,
                                                model_type = conf.get('model_type'))
    
    if args.plot_data:
        plot_data = PlotData(dataset=args.plot_data, save_plot=False)
        plot_data.run()
    
    if args.fetch_data_alpha:
        provider = ProviderAlphaVantage()
        provider.fetch_and_save_historical_data(symbol="NVDA", timeframe="1min",
                                                  month="2025-04", print_answer=False, store_answer=True)
    
    if args.fetch_data_capcom:
        provider = ProviderCapitalCom()
        provider.fetch_and_save_historical_data(symbol="GBPUSD", timeframe="MINUTE",
                                                  from_date="2018-01-01T00:00:00", to_date="2023-05-23T01:00:00",
                                                  print_answer=False)
        provider.end_session()
    
    if args.broker_func:
        broker = CapitalCom()
        broker.start_session()
        # broker.session_details(print_answer=True)
        # broker.switch_active_account(print_answer=False)
        # broker.get_account_capital()
        # broker.list_all_accounts(print_answer=True)
        # broker.get_historical_data(epic="GBPUSD", resolution="MINUTE",
        #                            max=1000,
        #                            from_date="2025-04-10T12:00:00", to_date="2025-04-10T13:10:00",
        #                            print_answer=True)
        # broker.fetch_and_save_historical_prices(epic="GBPUSD", resolution="MINUTE",
        #                                         from_date="2025-04-15T00:00:00", to_date="2025-05-01T01:00:00",
        #                                         print_answer=False)
        
        # broker.place_market_order(symbol="GBPUSD", direction="BUY", size="100", stop_level="1.32", profit_level="1.34")
        # broker.all_positions()
        # broker.modify_position(stop_level="1.25", profit_level="1.29")
        # broker.close_all_orders(print_answer=True)
        # broker.all_positions()
        
        # broker.sub_live_market_data(symbol="GBPUSD", timeframe="MINUTE")
        
        broker.end_session()
    
    if args.model_training:
        model_training = ModelTraining(dataset=args.model_training, model_type=args.model_type or 'xgboost')
        model = model_training.run()
        
        # Store the model directory for potential evaluation
        if hasattr(model_training, 'model_directory'):
            trained_model_dir = model_training.model_directory
        else:
            # Get the latest model directory that was just created
            trained_model_dir = get_latest_model(models_directory=config.TRAINED_MODEL_STORAGE, 
                                               model_type=args.model_type or 'xgboost')
    
    if args.evaluate_fresh or args.evaluate_old:
        if args.evaluate_old:
            # Evaluate specific model directory
            evaluation = ModelEvaluation(model_directory=args.evaluate_old)
        elif args.evaluate_fresh and trained_model_dir:
            # Evaluate the model that was just trained
            logging.info(f"Evaluating freshly trained model at: {trained_model_dir}")
            evaluation = ModelEvaluation(model_directory=trained_model_dir)
        else:
            # Find the latest model of the specified type
            logging.info(f"Finding latest {args.model_type or 'xgboost'} model for evaluation...")
            evaluation = ModelEvaluation(model_type=args.model_type or 'xgboost')
        
        evaluation.run()

if __name__=="__main__":
    main()