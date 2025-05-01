import argparse

from config import config
import utils.logging_utils as log_utils
from broker.capital_com.capitalcom import CapitalCom


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train hybrid trading strategy models with uncertainty quantification')
    
    parser.add_argument('--broker-func', action='store_true', default=True, help='Test all broker functionality')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    log_utils._is_configured = False
    logger = log_utils.setup_logging(name="blaaa", type="training", log_to_file=False, log_level=config.DEFAULT_LOG_LEVEL)
    
    
    if args.broker_func:
        broker = CapitalCom()
        broker.start_session()
        # broker.session_details(print_answer=True)
        broker.switch_active_account(print_answer=False)
        # broker.get_account_capital()
        # broker.list_all_accounts(print_answer=True)
        # broker.get_historical_data(epic="GBPUSD", resolution="MINUTE",
        #                            max=1000,
        #                            from_date="2025-04-10T12:00:00", to_date="2025-04-10T13:10:00",
        #                            print_answer=True)
        # broker.fetch_and_save_historical_prices(epic="GBPUSD", resolution="MINUTE_5",
        #                                         from_date="2024-10-01T00:00:00", to_date="2025-01-01T01:00:00",
        #                                         print_answer=False)
        
        # broker.place_market_order(symbol="GBPUSD", direction="BUY", size="100", stop_level="1.32", profit_level="1.34")
        # broker.all_positions()
        # broker.modify_position(stop_level="1.25", profit_level="1.29")
        # broker.close_all_orders(print_answer=True)
        # broker.all_positions()
        
        broker.sub_live_market_data(symbol="GBPUSD", timeframe="MINUTE")
        
        broker.end_session()

if __name__=="__main__":
    main()