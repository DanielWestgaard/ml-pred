# When things are getting more complete, this file needs to be cleaned up as it will be the main file for running all other "systems".
import argparse
import json
import signal
import sys
import time
import yaml
import os

# Import config early but defer other imports that might initialize logging
from config import config

# Defer logger initialization until after dynamic logging setup
logger = None

def _initialize_dynamic_logging():
    """Initialize logging with dynamic categorization based on command line args."""
    global logger
    
    # Parse args early just for logging setup
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--hypothesis', '-hyp', help='Hypothesis config file path')
    parser.add_argument('--strategy', '-strat', help='Strategy config file path')
    args, _ = parser.parse_known_args()
    
    hypothesis_config_path = args.hypothesis or args.strategy
    
    if hypothesis_config_path and os.path.exists(hypothesis_config_path):
        # Import logging utilities
        import utils.logging_utils as log_utils
        
        # Setup logging with dynamic categorization
        log_utils.setup_logging(
            name="logging", 
            type="mixed", 
            hypothesis_config_path=hypothesis_config_path,
            log_level=config.DEBUG_LOG_LEVEL
        )
        logger = log_utils.get_logger()
    else:
        # Fallback to default logging
        import utils.logging_utils as log_utils
        log_utils.setup_logging(
            name="logging",
            type="mixed", 
            log_level=config.DEBUG_LOG_LEVEL
        )
        logger = log_utils.get_logger()

# Initialize logging before any other imports
_initialize_dynamic_logging()

# Now import everything else
from broker.capital_com.capitalcom import CapitalCom
from data.providers.capital_com import ProviderCapitalCom
from data.providers.alpha_vantage import ProviderAlphaVantage
import utils.alpha_vantage_utils as alpha_utils
from data.pipelines.engineer_pipeline import EngineeringPipeline
import utils.shared_utils as shared_utils
from data.plot.plot_data import PlotData
from model.model_training import ModelTraining
from evaluation.evaluation import ModelEvaluation
from strategy.strategy import StrategyOrchestrator
from utils.model.model_metadata_loader import get_latest_model
import utils.hypothesis.config_loader as config_loader
import utils.logging_utils as log_utils
from utils.logging_utils import get_logger
from hyp_val_framework.validate_hypothesis import HypothesisValidator

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
    
    # Strategy Execution
    parser.add_argument('--strategy', '-strat', default=False, help='Run strategy with hypothesis config')
    
    # Feature importance analysis
    parser.add_argument('--feature-importance', '-fi', default=False, help='Analyze feature importance of processed dataset (path to CSV file).')
    parser.add_argument('--fi-target', default=None, help='Target column for feature importance analysis (auto-detected if not provided).')
    parser.add_argument('--fi-in-pipeline', action='store_true', default=False, help='Enable feature importance analysis within data pipeline.')
    
    # Walk-forward validation - TODO: Not used? Remove?
    parser.add_argument('--walk-forward', '-wf', default=False, help='Run walk-forward validation with hypothesis config file.')
    parser.add_argument('--wf-training-months', type=int, default=6, help='Initial training period in months for walk-forward validation.')
    parser.add_argument('--wf-validation-months', type=int, default=1, help='Validation period length in months.')
    parser.add_argument('--wf-step-months', type=int, default=1, help='Step size between validation periods in months.')

    parser.add_argument('--validate-hypothesis', '-vh', default=False, help='Run validation on your new hypothesis.')
    parser.add_argument('--validate-hypothesis', '-vh', default=False, help='')

    return parser.parse_args()


def main():
    global logger
    args = parse_arguments()
    
    # Logger is already initialized by _initialize_dynamic_logging()
    # Just get the logger instance
    logger = get_logger()
    
    if args.engineer_data:  # python main.py --engineer-data 'storage/data/capital_com/GBPUSD/raw/1m/2025-01-01_2025-05-23.csv' --hypothesis "hypothesis/1_session_based_tod_anomalies/config.yml"
        save_cleaned_data = True
        if args.hypothesis:
            logger.info("Trying to run data engineering with hypothesis config file.")
            
            # Use new path structure or fallback to legacy for compatibility
            default_raw_path = args.engineer_data or "storage/data/capital_com/GBPUSD/raw/1m/2025-05-01_2025-05-01.csv"
            data_pipeline = EngineeringPipeline(raw_dataset=default_raw_path, 
                                                save_cleaned_data=save_cleaned_data, 
                                                hypothesis_source=args.hypothesis,
                                                analyze_feature_importance=args.fi_in_pipeline)
        else:
            default_raw_path = args.engineer_data or "storage/data/capital_com/GBPUSD/raw/1m/2025-05-01_2025-05-01.csv"
            data_pipeline = EngineeringPipeline(raw_dataset=default_raw_path,
                                                save_cleaned_data=save_cleaned_data,
                                                analyze_feature_importance=args.fi_in_pipeline)
        
        data_pipeline.run()

    if args.hypothesis:
        logger.info(f"User using pure hypothesis flow!")
        save_cleaned_data = True
        # 'Keys' for running hypothesis
        raw_data_source:str = 'raw_data_source'
        # ...
        strategy:str = 'strategy'
        backtesting:str = 'backtesting'
        
        # Getting all the configurations
        conf = config_loader.load_hypothesis_config(args.hypothesis).handle_hypothesis_source()
        conf_keys = conf.keys()
        
        if 'data_pipeline_raw_source' in conf_keys:  # Found raw data path config, meaning user intends to run Data Pipeline?
            data_pipeline = EngineeringPipeline(save_cleaned_data=save_cleaned_data, hypothesis_source=args.hypothesis, analyze_feature_importance=True)
            data_pipeline.run()
        
        # Model Training
        if 'model_training_type' in conf_keys:  # Found Model Type in config, meaning user wants to use Model Training
            model_training = ModelTraining(hypothesis_source = args.hypothesis)
            model = model_training.run()
            
            # Store the model directory for potential evaluation
            if hasattr(model_training, 'model_directory'):
                trained_model_dir = model_training.model_directory
            else:
                # Get the latest model directory that was just created
                trained_model_dir = get_latest_model(models_directory = config.TRAINED_MODEL_STORAGE,
                                                model_type = conf.get('model_type'))
        
        if 'evaluate_model' in conf_keys:
            eval_type = conf.get('evaluation_type')
            
            evaluation = ModelEvaluation(hypothesis_config = args.hypothesis)
            
            # if eval_type in 'stored':
            #     # Evaluate specific model directory
            #     evaluation = ModelEvaluation(model_directory = conf.get('model_folder'))
            # elif eval_type in 'new' and trained_model_dir:
            #     time.sleep(1)  # Temporary, just to not get any race conditions
            #     # Evaluate the model that was just trained
            #     logger.info(f"Evaluating freshly trained model at: {trained_model_dir}")
            #     evaluation = ModelEvaluation(model_directory = trained_model_dir)
            # else:
            #     # Find the latest model of the specified type
            #     logger.info(f"Finding latest {args.model_type or 'xgboost'} model for evaluation...")
            #     evaluation = ModelEvaluation(model_type=args.model_type or 'xgboost')
            evaluation.run()
        
        if strategy in conf_keys:
            _strategy_flow(args.hypothesis)
        
        if backtesting in conf_keys:
            _backtesting_flow(args.hypothesis)
    
    if args.plot_data:
        plot_data = PlotData(dataset=args.plot_data, save_plot=False)
        plot_data.run()
    
    if args.fetch_data_alpha:
        provider = ProviderAlphaVantage()
        provider.fetch_and_save_historical_data(symbol="NVDA", timeframe="1min",
                                                  month="2025-04", print_answer=False, store_answer=True)
    
    if args.fetch_data_capcom:
        provider = ProviderCapitalCom()
        # Enable spread data collection for spread-focused analysis 
        # (use include_spreads=False for backward compatibility with existing pipelines)
        provider.fetch_and_save_historical_data(symbol="GBPUSD", timeframe="MINUTE",
                                                  from_date="2025-07-02T00:00:00", to_date="2025-07-31T01:00:00",
                                                  print_answer=False, include_spreads=True)
        provider.end_session()
    
    if args.broker_func:
        broker = CapitalCom()
        broker.start_session()
        broker.session_details(print_answer=True)
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
        
        broker.sub_live_market_data(symbol="BTCUSD", timeframe="MINUTE")
        
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
            logger.info(f"Evaluating freshly trained model at: {trained_model_dir}")
            evaluation = ModelEvaluation(model_directory=trained_model_dir)
        else:
            # Find the latest model of the specified type
            logger.info(f"Finding latest {args.model_type or 'xgboost'} model for evaluation...")
            evaluation = ModelEvaluation(model_type=args.model_type or 'xgboost')
        
        evaluation.run()
    
    if args.strategy:
        logger.info(f"Starting strategy execution with config: {args.strategy}")
        strategy = StrategyOrchestrator(hypothesis_config=args.strategy)
        strategy_results = strategy.run()
        logger.info(f"Strategy execution completed: {strategy_results.get('status', 'unknown')}")
    
    # Feature importance analysis
    if args.feature_importance:
        logger.info(f"Starting feature importance analysis on: {args.feature_importance}")
        from data.features.feature_importance import analyze_processed_data
        results = analyze_processed_data(args.feature_importance, args.fi_target)
        logger.info(f"Feature importance analysis completed. Results saved to: {results['output_directory']}")
        
        # Show top 10 consensus features in console
        consensus = results['consensus']
        print("\n" + "="*60)
        print("TOP 10 CONSENSUS FEATURES:")
        print("="*60)
        for i, (feature, score) in enumerate(consensus.head(10).items(), 1):
            print(f"{i:2d}. {feature:<30} {score:.4f}")
        print("="*60)
    
    # Walk-forward validation - TODO: Remove?
    if args.walk_forward:
        logger.info(f"Starting walk-forward validation with config: {args.walk_forward}")
        from model.walk_forward_validator import WalkForwardValidator, create_walk_forward_config
        
        # Create validation configuration
        validation_config = create_walk_forward_config(
            initial_training_months=args.wf_training_months,
            validation_period_months=args.wf_validation_months,
            step_forward_months=args.wf_step_months
        )
        
        validator = WalkForwardValidator(
            hypothesis_config_path=args.walk_forward,
            validation_config=validation_config
        )
        
        results = validator.run_validation()
        
        # Display summary results
        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*80)
        print(f"Total Validation Periods: {results['validation_summary']['total_periods']}")
        print(f"Mean Return: {results['performance_metrics']['mean_return']:.4f}")
        print(f"Mean Win Rate: {results['performance_metrics']['mean_win_rate']:.2%}")
        print(f"Consistency Ratio: {results['consistency_metrics']['consistency_ratio']:.2%}")
        print(f"Results saved to: {validator.validation_directory}")
        print("="*80)

    if args.validate_hypothesis:
        logger.info("Validate hypothesis is wanted.")
        hyp_val = HypothesisValidator(dataset = args.validate_hypothesis)
        hyp_val.run()

def _strategy_flow(hypothesis_config: str):
    """
    Run strategy flow with hypothesis with proper cleanup on interruption.
    
    Args:
        - hypothesis_config: Path to the hypothesis configuration file.
    """
    strategy = None
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("Received interrupt signal. Cleaning up strategy...")
        if strategy:
            try:
                # Stop the strategy gracefully
                strategy.stop()
                logger.info("Strategy stopped gracefully")
            except Exception as e:
                logger.error(f"Error during strategy cleanup: {e}")
        
        # Force exit to stop any lingering threads (like ping timers)
        import os
        os._exit(0)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        strategy = StrategyOrchestrator(hypothesis_config = hypothesis_config)
        strategy.run()
    except KeyboardInterrupt:
        logger.info("Strategy execution interrupted by user")
        if strategy:
            try:
                strategy.stop()
            except Exception as e:
                logger.error(f"Error during strategy cleanup: {e}")
    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        if strategy:
            try:
                strategy.stop()
            except Exception as e:
                logger.error(f"Error during strategy cleanup: {e}")
        raise

def _backtesting_flow(hypothesis_config: str):
    from backtesting.backtester import BacktestOrchestrator

    # Test the trained model
    backtester = BacktestOrchestrator(
        hypothesis_config_path = hypothesis_config  # "hypothesis/spread_compression_patterns/config_complete.yml"
    )

    results = backtester.run_backtest()

if __name__=="__main__":
    main()