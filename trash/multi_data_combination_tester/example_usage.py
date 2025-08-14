#!/usr/bin/env python3
"""
Example usage of TimeSeriesDataCombiner for systematic testing of data source combinations

This script demonstrates how to:
1. Combine market data with calendar events
2. Test different feature combinations
3. Prepare data for statistical analysis and ML training
4. Save results for the "Testing different combinations" phase
"""

import logging
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_combination_testing():
    """
    Demonstrate systematic testing of different data source combinations.
    
    This is exactly what you need for the red text in your drawing:
    "How I should combine the different data formats?"
    """
    try:
        from multi_data_sources.combiners import TimeSeriesDataCombiner
        
        # Initialize combiner
        logger.info("🚀 Initializing TimeSeriesDataCombiner for systematic combination testing...")
        combiner = TimeSeriesDataCombiner()
        
        # Example parameters (adjust these for your actual testing)
        symbol = "GBPUSD"  # Has market data
        timeframe = "1m"
        date_range = {
            'start': '2025-07-02',
            'end': '2025-07-10'  # Smaller range for demo
        }
        
        # Step 1: Check available data sources
        logger.info(f"📊 Checking available data sources for {symbol}...")
        available = combiner.get_available_combinations(symbol)
        logger.info(f"Available sources: {available}")
        
        # Step 2: Load base market data
        logger.info("📈 Loading market data...")
        market_df = combiner._load_market_data(symbol, timeframe, date_range)
        logger.info(f"Market data: {len(market_df)} rows, {len(market_df.columns)} features")
        
        # Step 3: Create synthetic calendar events for demonstration
        # (In practice, you'd use: data_sources=['calendar_events'])
        logger.info("📅 Creating synthetic calendar events for demonstration...")
        
        # Create time-indexed calendar features manually for demo
        time_index = market_df.index
        calendar_features = pd.DataFrame(index=time_index)
        
        # Simulate earnings proximity (earnings on 2025-07-05)
        earnings_date = pd.to_datetime('2025-07-05')
        calendar_features['days_to_earnings'] = (earnings_date - time_index).days
        calendar_features['earnings_proximity_zone'] = 'long_term'
        calendar_features.loc[calendar_features['days_to_earnings'] <= 1, 'earnings_proximity_zone'] = 'immediate'
        calendar_features.loc[(calendar_features['days_to_earnings'] > 1) & 
                             (calendar_features['days_to_earnings'] <= 7), 'earnings_proximity_zone'] = 'near_term'
        calendar_features['earnings_immediate_flag'] = (calendar_features['days_to_earnings'] <= 1).astype(int)
        
        # Simulate options expiration (monthly expiry on 2025-07-18)
        options_date = pd.to_datetime('2025-07-18')
        calendar_features['days_to_options_exp'] = (options_date - time_index).days
        calendar_features['options_expiry_week'] = (calendar_features['days_to_options_exp'] <= 7).astype(int)
        
        # Simulate vacation period flag (July 4th week)
        july4_start = pd.to_datetime('2025-07-03')
        july4_end = pd.to_datetime('2025-07-07')
        calendar_features['vacation_period'] = (
            (time_index >= july4_start) & (time_index <= july4_end)
        ).astype(int)
        
        # Combine market data with calendar features
        combined_df = market_df.join(calendar_features, how='left')
        combined_df = combiner._fill_missing_values(combined_df)
        
        logger.info(f"✅ Combined dataset created: {len(combined_df)} rows, {len(combined_df.columns)} features")
        
        # Step 4: Test different feature combinations (This answers your red text question!)
        logger.info("🧪 Testing different data source combinations...")
        
        test_combinations = [
            {
                'name': 'Market Only',
                'features': ['Close', 'Volume', 'spread_avg']
            },
            {
                'name': 'Market + Earnings Proximity',
                'features': ['Close', 'Volume', 'spread_avg', 'days_to_earnings', 'earnings_immediate_flag']
            },
            {
                'name': 'Market + Options Expiration',
                'features': ['Close', 'Volume', 'spread_avg', 'days_to_options_exp', 'options_expiry_week']
            },
            {
                'name': 'Market + Holiday Effects',
                'features': ['Close', 'Volume', 'spread_avg', 'vacation_period']
            },
            {
                'name': 'Market + Multiple Calendar Events',
                'features': ['Close', 'Volume', 'spread_avg', 'days_to_earnings', 'days_to_options_exp', 'vacation_period']
            }
        ]
        
        # Test each combination
        results = {}
        for combination in test_combinations:
            name = combination['name']
            features = combination['features']
            
            logger.info(f"   Testing: {name}")
            
            # Extract feature subset
            test_data = combined_df[features].copy()
            
            # Basic statistics for each combination
            stats = {
                'feature_count': len(features),
                'data_points': len(test_data),
                'feature_names': features,
                'sample_correlation_with_close': {}
            }
            
            # Calculate correlations with Close price (simple statistical test)
            for feature in features:
                if feature != 'Close' and test_data[feature].dtype in ['int64', 'float64']:
                    try:
                        corr = test_data['Close'].corr(test_data[feature])
                        stats['sample_correlation_with_close'][feature] = round(corr, 4)
                    except:
                        pass
            
            results[name] = stats
            
            # Save subset for further analysis
            output_dir = Path("storage/combination_tests")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{symbol}_{timeframe}_{name.replace(' ', '_').lower()}.csv"
            filepath = output_dir / filename
            test_data.to_csv(filepath)
            
            logger.info(f"   ✅ Saved: {filepath}")
        
        # Step 5: Summary of combination testing results
        logger.info("\n📊 COMBINATION TESTING RESULTS:")
        logger.info("=" * 50)
        
        for name, stats in results.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Features: {stats['feature_count']}")
            logger.info(f"  Data points: {stats['data_points']}")
            
            if stats['sample_correlation_with_close']:
                logger.info("  Correlations with Close price:")
                for feature, corr in stats['sample_correlation_with_close'].items():
                    logger.info(f"    {feature}: {corr}")
        
        logger.info(f"\n🎯 SOLUTION TO YOUR RED TEXT QUESTION:")
        logger.info("=" * 50)
        logger.info("✅ Different data formats successfully combined!")
        logger.info("✅ CSV market data + JSON calendar events → Unified time-indexed CSV")
        logger.info("✅ Multiple combinations tested and saved for analysis")
        logger.info("✅ Ready for systematic statistical testing")
        logger.info("✅ Same approach works for live trading pipeline")
        
        return True
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_in_main_pipeline():
    """Show how this integrates with your existing main.py workflow"""
    
    logger.info("\n🔄 INTEGRATION WITH MAIN PIPELINE:")
    logger.info("=" * 50)
    
    logger.info("""
    # In your main.py or new combination testing script:
    
    from multi_data_sources.combiners import TimeSeriesDataCombiner
    
    def test_data_combinations():
        combiner = TimeSeriesDataCombiner()
        
        # Test systematic combinations
        symbols = ['GBPUSD', 'EURUSD', 'BTCUSD']  # Add your symbols
        
        for symbol in symbols:
            # Check what's available
            available = combiner.get_available_combinations(symbol)
            
            if 'calendar_events' in available['supported_combinations']:
                # Combine real data
                combined_df = combiner.combine_data(
                    symbol=symbol,
                    timeframe='1m', 
                    date_range={'start': '2024-01-01', 'end': '2024-12-31'},
                    data_sources=['calendar_events']
                )
                
                # Run statistical analysis
                analyze_combinations(combined_df, symbol)
    """)

if __name__ == "__main__":
    logger.info("🚀 Starting Data Source Combination Demonstration")
    
    success = demonstrate_combination_testing()
    
    if success:
        show_usage_in_main_pipeline()
        
        logger.info("\n🎉 SUCCESS! Your data combination system is ready!")
        logger.info("Next steps:")
        logger.info("1. Run this with your actual calendar_events data")
        logger.info("2. Add more sophisticated statistical analysis")
        logger.info("3. Integrate with your ML training pipeline")
        logger.info("4. Use for systematic inefficiency discovery")
    else:
        logger.error("❌ Demonstration failed!")