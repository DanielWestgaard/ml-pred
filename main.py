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
from multi_data_sources.ds_orchestrator import DataSourceOrchestrator
from broker.alpaca.alpaca import Alpaca

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train hybrid trading strategy models with uncertainty quantification')
    
    # Full Broker functionality
    parser.add_argument('--broker-func', action='store_true', default=False, help='Test all broker functionality')
    parser.add_argument('--alpaca-api', '-alp', default=False, help='Alpaca API')
    
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
    # parser.add_argument('--validate-hypothesis-specific', '-vhs', default=False, help='')

    # Multi-data sources
    parser.add_argument('--multi-data', '-md', default=False, help='Run multi-data sources orchestrator with hypothesis config file.')

    return parser.parse_args()


def main():
    global logger
    args = parse_arguments()
    
    # Logger is already initialized by _initialize_dynamic_logging()
    # Just get the logger instance
    logger = get_logger()
    
    if args.alpaca_api:
        # alpaca = Alpaca(print_responses = True)
        
        # alpaca.get_positions()
        # alpaca.get_orders()
        # alpaca.place_market_order(symbol="ATLO", side="buy", quantity=1)
        # alpaca.get_positions()
        # time.sleep(10)
        # alpaca.close_all_positions()
        # alpaca.fetch_and_save_historical_data(symbol="BDTX", timeframe="5min", start_date="2016-01-01T00:00:00", end_date="2025-08-22T00:00:00")
        
        timeframes = ["1h", "5m", "15m"]
        symbols = ["ISSC", "CCLD", "PLX", "LFVN"]
        
        for symbol in symbols:
            for timeframe in timeframes:
                alpaca = Alpaca(print_responses = False)
                alpaca.fetch_and_save_historical_data(symbol=symbol, timeframe=timeframe, start_date="2015-01-01T00:00:00", end_date="2025-08-22T00:00:00")
                
                logger.info("Sleeping for 3 seconds until next one.")
                time.sleep(3)

    
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
        
        # Check for multi-data sources first
        if 'multi_data' in conf_keys:
            logger.info("Multi-data sources configuration detected. Running data source orchestrator...")
            multi_data_orchestrator = DataSourceOrchestrator(args.hypothesis)
            multi_data_results = multi_data_orchestrator.run()
            
            # Log summary of results
            summary = multi_data_results.get('summary', {})
            logger.info(f"Multi-data orchestrator completed: {summary.get('successful_sources', 0)}/{summary.get('total_sources', 0)} sources successful")
            
            if summary.get('failed_sources', 0) > 0:
                failed_sources = summary.get('failed_source_names', [])
                logger.warning(f"Failed data sources: {failed_sources}")
        
        # Check for combination testing
        if 'combination_testing' in conf_keys:
            logger.info("Combination testing configuration detected. Running systematic testing...")
            from multi_data_sources.combination_tester import CombinationTestOrchestrator
            
            tester = CombinationTestOrchestrator(args.hypothesis)
            test_results = tester.run()
            
            # Log summary of results
            summary = test_results.get('summary', {})
            logger.info(f"Combination testing completed: {summary.get('total_combinations', 0)} combinations tested, "
                       f"{summary.get('valid_combinations', 0)} valid (execution time: {summary.get('execution_time_seconds', 0)}s)")
            
            # Highlight top combinations
            analysis = test_results.get('analysis', {})
            top_combinations = analysis.get('top_combinations', [])[:3]
            
            if top_combinations:
                logger.info("🏆 Top combination results:")
                for i, combo in enumerate(top_combinations, 1):
                    logger.info(f"  #{i}: {combo.get('combination_name', 'unknown')} "
                               f"(score: {combo.get('score', 0):.4f}, correlation: {combo.get('correlation', 0):.4f})")
            
            # Check for institutional blind spots identified
            blind_spots = analysis.get('institutional_blind_spots', 0)
            if blind_spots > 0:
                logger.info(f"🎯 Identified {blind_spots} potential institutional blind spot combinations")
        
        if 'data_pipeline_raw_source' in conf_keys:  # Found raw data path config, meaning user intends to run Data Pipeline?
            # Check if multi-data sources are configured
            if 'multi_data' in conf_keys and conf.get('multi_data'):
                logger.info("🚀 Multi-data sources detected, using Enhanced Multi-Data Pipeline...")
                from data.pipelines.multi_data_engineer_pipeline import MultiDataEngineeringPipeline
                data_pipeline = MultiDataEngineeringPipeline(
                    save_cleaned_data=save_cleaned_data, 
                    hypothesis_source=args.hypothesis, 
                    analyze_feature_importance=True
                )
            else:
                logger.info("📊 Using standard Engineering Pipeline...")
                data_pipeline = EngineeringPipeline(
                    save_cleaned_data=save_cleaned_data, 
                    hypothesis_source=args.hypothesis, 
                    analyze_feature_importance=True
                )
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
        # provider = ProviderCapitalCom()
        # Enable spread data collection for spread-focused analysis 
        # (use include_spreads=False for backward compatibility with existing pipelines)
        
        # provider.fetch_and_save_historical_data(symbol="TLRY", timeframe="DAY",
        #                                        from_date="2020-01-02T00:00:00", to_date="2025-07-31T01:00:00",
        #                                        print_answer=False, include_spreads=False)
        # provider.end_session()
        
        timeframes = ["HOUR", "MINUTE_5", "MINUTE_15"]
        symbols = ["ISSC", "CCLD", "PLX"]
        
        for symbol in symbols:
            for timeframe in timeframes:
                provider = ProviderCapitalCom()
                provider.fetch_and_save_historical_data(symbol=symbol, timeframe=timeframe,
                                            from_date="2020-01-01T00:00:00", to_date="2025-08-22T01:00:00",
                                            print_answer=False, include_spreads=False)
                provider.end_session()
                logger.info("Sleeping for 5 seconds until next one.")
                time.sleep(5)
            
    
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
        
        validator = HypothesisValidator(args.validate_hypothesis)
        
        # Example 2: Quick hypothesis testing
        # results = validator.quick_test_volume_hypothesis(volume_threshold_pct=90)
        
        # Example 3: Advanced validation
        # from hyp_val_framework.examples.spread_expansion_validator import SpreadExpansionValidator
        
        # validator = HypothesisValidator(args.validate_hypothesis)
        # advanced_results = validator.run_comprehensive_validation(SpreadExpansionValidator)
        
        # Example 4: Custom hypothesis
        def my_signal_logic(data, params):
            return data['volume'] > params['threshold']
        
        def my_parameter_logic(estimation_data):
            return {'threshold': estimation_data['volume'].quantile(0.9)}
        
        custom_results = validator.quick_test_custom_hypothesis(
            "My Volume Hypothesis",
            "Test if high volume predicts returns",
            my_signal_logic,
            my_parameter_logic
        )

    # Multi-data sources standalone execution
    if args.multi_data:
        logger.info(f"Starting multi-data sources orchestrator with config: {args.multi_data}")
        try:
            multi_data_orchestrator = DataSourceOrchestrator(args.multi_data)
            results = multi_data_orchestrator.run()
            
            # Display results summary
            summary = results.get('summary', {})
            config_info = results.get('config', {})
            
            print("\n" + "="*60)
            print("MULTI-DATA SOURCES EXECUTION SUMMARY")
            print("="*60)
            print(f"Symbol: {config_info.get('symbol', 'N/A')}")
            print(f"Purpose: {config_info.get('purpose', 'N/A')}")
            print(f"Date Range: {config_info.get('from_date', 'N/A')} to {config_info.get('to_date', 'N/A')}")
            print(f"Total Sources: {summary.get('total_sources', 0)}")
            print(f"Successful: {summary.get('successful_sources', 0)}")
            print(f"Failed: {summary.get('failed_sources', 0)}")
            
            if summary.get('successful_source_names'):
                print(f"Successful Sources: {', '.join(summary.get('successful_source_names', []))}")
            
            if summary.get('failed_source_names'):
                print(f"Failed Sources: {', '.join(summary.get('failed_source_names', []))}")
            
            print(f"Execution Time: {results.get('timestamp', 'N/A')}")
            print("="*60)
            
            logger.info("Multi-data sources orchestrator completed successfully")
            
        except Exception as e:
            logger.error(f"Multi-data sources orchestrator failed: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")

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
    """
    Run backtesting flow with the modernized plugin-based system
    
    Args:
        hypothesis_config: Path to the hypothesis configuration file
    """
    from backtesting.backtester import BacktestOrchestrator
    
    try:
        logger.info("🚀 Starting backtesting flow with modernized system...")
        logger.info(f"Using hypothesis config: {hypothesis_config}")
        
        # Initialize the modern backtesting orchestrator
        backtester = BacktestOrchestrator(
            hypothesis_config_path=hypothesis_config
        )
        
        # Validate configuration before running
        validation = backtester.validate_configuration()
        if not validation.get('is_valid', False):
            logger.error(f"❌ Backtesting configuration validation failed: {validation.get('error')}")
            return None
        
        # Log component information
        component_info = backtester.get_component_info()
        logger.info(f"Using model: {component_info.get('model_info', {}).get('model_class', 'Unknown')}")
        logger.info(f"Hypothesis plugin: {component_info.get('hypothesis_plugin', {}).get('plugin_class', 'Unknown')}")
        logger.info(f"Exit strategy: {component_info.get('exit_strategy_info', {}).get('exit_strategy_class', 'Unknown')}")
        
        # Run the backtest
        results = backtester.run_backtest()
        
        # The BacktestEngine already logs comprehensive results, 
        # so we just log a simple completion message here
        summary = backtester.get_results_summary()
        logger.info(f"✅ Backtesting completed: {summary.get('hypothesis', 'Unknown')} - Return: {summary.get('total_return_pct', 0):.2f}% ({summary.get('total_trades', 0)} trades)")
        return results
        
    except Exception as e:
        logger.error(f"❌ Backtesting flow failed: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return None

if __name__=="__main__":
    main()