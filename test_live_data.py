#!/usr/bin/env python3
"""
Test script for live data integration with session anomaly strategy.
This demonstrates the new live data capabilities in dry-run mode.
"""

import logging
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.live.live_data_manager import LiveDataManager
import utils.logging_utils as log_utils
import config.config as config


def test_live_data_manager():
    """Test the live data manager functionality"""
    print("🚀 Testing Live Data Manager")
    print("=" * 50)
    
    # Initialize live data manager
    print("1. Initializing LiveDataManager...")
    live_data = LiveDataManager(
        symbol="GBPUSD",
        timeframe="MINUTE", 
        buffer_size=100,
        feature_window=50,
        selected_features=None  # Use all features
    )
    
    # Add a simple callback to monitor data updates
    def data_callback(latest_features):
        print(f"📊 New data processed - Features shape: {latest_features.shape}")
        print(f"   Latest timestamp: {latest_features.index[-1]}")
        print(f"   Sample features: {list(latest_features.columns[:5])}")
    
    live_data.add_data_callback(data_callback)
    
    try:
        # Start streaming in dry-run mode (simulation)
        print("2. Starting data streaming (simulation mode)...")
        if live_data.start_streaming(dry_run=True):
            print("✅ Data streaming started successfully")
        else:
            print("❌ Failed to start data streaming")
            return
        
        # Monitor the system for a while
        print("3. Monitoring live data for 30 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 30:
            status = live_data.get_buffer_status()
            
            print(f"\r📈 Buffer: {status['buffer_size']}/{status['max_buffer_size']} "
                  f"({status['buffer_utilization']:.1%}) | "
                  f"Ready: {'✅' if live_data.is_ready_for_trading() else '⏳'} | "
                  f"Streaming: {'✅' if status['is_streaming'] else '❌'}", end="")
            
            # Test getting latest features when ready
            if live_data.is_ready_for_trading():
                latest = live_data.get_latest_features()
                if latest is not None and len(latest) > 0:
                    # Show some sample feature values
                    if hasattr(latest, 'columns') and len(latest.columns) > 0:
                        sample_col = latest.columns[0]
                        sample_val = latest[sample_col].iloc[0]
                        print(f" | Sample: {sample_col}={sample_val:.4f}")
            
            time.sleep(1)
        
        print("\n4. Testing data retrieval...")
        
        # Test getting processed data
        processed_data = live_data.get_processed_data(n_periods=10)
        if processed_data is not None:
            print(f"✅ Retrieved processed data: {processed_data.shape}")
            print(f"   Columns: {len(processed_data.columns)} features")
            print(f"   Time range: {processed_data.index[0]} to {processed_data.index[-1]}")
        else:
            print("❌ No processed data available")
        
        # Test getting latest features
        latest_features = live_data.get_latest_features()
        if latest_features is not None:
            print(f"✅ Retrieved latest features: {latest_features.shape}")
        else:
            print("❌ No latest features available")
            
    finally:
        print("\n5. Stopping data streaming...")
        live_data.stop_streaming()
        print("✅ Live data manager stopped")


def test_strategy_integration():
    """Test the strategy integration with live data"""
    print("\n🎯 Testing Strategy Integration")
    print("=" * 50)
    
    try:
        from strategy.strategy import StrategyOrchestrator
        
        print("1. Initializing strategy with live data integration...")
        strategy = StrategyOrchestrator(
            hypothesis_config="hypothesis/1_session_based_tod_anomalies/config.yml"
        )
        
        print("2. Running strategy for a short test...")
        results = strategy.run()
        
        print("3. Strategy Results:")
        print(f"   Status: {results.get('status', 'unknown')}")
        print(f"   Signals Generated: {results.get('signals_generated', 0)}")
        print(f"   Errors: {len(results.get('errors', []))}")
        
        if results.get('errors'):
            print("   Error Details:")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"   - {error}")
        
        if results.get('session_analysis'):
            print("   Session Analysis:")
            for session, stats in results['session_analysis'].items():
                print(f"   - {session}: {stats}")
                
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    # Setup logging
    log_utils._is_configured = False
    logging = log_utils.setup_logging(
        name="live_data_test", 
        type="test", 
        log_to_file=False, 
        log_level=config.DEBUG_LOG_LEVEL
    )
    
    print("🧪 Live Data Integration Test Suite")
    print("=" * 60)
    print("This test demonstrates the new live data capabilities")
    print("in simulation mode (no real broker connection required).")
    print("=" * 60)
    
    try:
        # Test 1: Live Data Manager
        test_live_data_manager()
        
        # Test 2: Strategy Integration
        test_strategy_integration()
        
        print("\n🎉 Test Suite Completed")
        print("=" * 60)
        print("The live data integration is working! Next steps:")
        print("1. Set dry_run=false in config for live trading")
        print("2. Ensure Capital.com credentials are configured")
        print("3. Test with smaller position sizes first")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()