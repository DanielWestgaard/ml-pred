#!/usr/bin/env python3
"""
WebSocket Integration Test Script

This script tests the new Capital.com WebSocket integration with the session anomaly strategy.
It demonstrates real-time data streaming, buffer management, and strategy execution.
"""

import logging
import time
import sys
import signal
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.live.websocket_data_manager import WebSocketDataManager
from data.live.live_data_manager import LiveDataManager
from strategy.hypothesis_specific.session_anomaly_strategy import SessionAnomalyStrategy
import utils.shared_utils as shared_utils

# Global variables for cleanup
websocket_manager = None
live_data_manager = None
strategy = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Interrupt received, cleaning up...")
    cleanup_resources()
    sys.exit(0)

def cleanup_resources():
    """Cleanup all resources"""
    global websocket_manager, live_data_manager, strategy
    
    if strategy:
        try:
            strategy._cleanup_resources()
        except Exception as e:
            logging.error(f"Error cleaning up strategy: {e}")
    
    if live_data_manager:
        try:
            live_data_manager.stop_streaming()
        except Exception as e:
            logging.error(f"Error stopping live data manager: {e}")
    
    if websocket_manager:
        try:
            websocket_manager.stop_streaming()
        except Exception as e:
            logging.error(f"Error stopping WebSocket manager: {e}")

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def test_websocket_data_manager():
    """Test WebSocket data manager standalone"""
    global websocket_manager
    
    print("\n🔌 Testing WebSocket Data Manager")
    print("="*50)
    
    try:
        # Initialize WebSocket manager
        websocket_manager = WebSocketDataManager(
            symbols=["GBPUSD"],
            timeframe="MINUTE",
            buffer_size=100
        )
        
        # Add data callback
        def data_callback(symbol, data):
            print(f"📊 WebSocket Data - {symbol}: Close={data.get('close'):.5f}, "
                  f"Volume={data.get('volume', 0)}, Source={data.get('source')}")
        
        websocket_manager.add_data_callback(data_callback)
        
        # Start streaming (dry run mode first)
        print("🚀 Starting WebSocket streaming...")
        if websocket_manager.start_streaming(jumpstart_buffer=True):
            print("✅ WebSocket streaming started successfully")
            
            # Monitor for 30 seconds
            start_time = time.time()
            while time.time() - start_time < 30:
                status = websocket_manager.get_connection_status()
                print(f"\n📈 Status: Connected={status['connected']}, "
                      f"Messages={status['messages_received']}, "
                      f"Active={status['active_connections']}")
                
                # Show latest prices
                latest = websocket_manager.get_latest_price("GBPUSD")
                if latest:
                    print(f"💰 Latest GBPUSD: {latest.get('close', 'N/A')}")
                
                time.sleep(5)
            
            print("✅ WebSocket test completed successfully")
            return True
        else:
            print("❌ Failed to start WebSocket streaming")
            return False
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        logging.exception("WebSocket test error")
        return False
    finally:
        if websocket_manager:
            websocket_manager.stop_streaming()
            websocket_manager = None

def test_live_data_manager_with_websocket():
    """Test Live Data Manager with WebSocket integration"""
    global live_data_manager
    
    print("\n🌊 Testing Live Data Manager with WebSocket")
    print("="*50)
    
    try:
        # Initialize live data manager with WebSocket enabled
        live_data_manager = LiveDataManager(
            symbol="GBPUSD",
            timeframe="MINUTE",
            buffer_size=200,
            feature_window=35,
            use_websocket=True
        )
        
        # Add data callback
        def data_callback(latest_features):
            if latest_features is not None and not latest_features.empty:
                print(f"🔮 Features generated: {len(latest_features.columns)} features, "
                      f"Latest close: {latest_features.get('close_og', [0]).iloc[-1] if 'close_og' in latest_features.columns else 'N/A'}")
        
        live_data_manager.add_data_callback(data_callback)
        
        # Start streaming in dry run mode (simulation)
        print("🚀 Starting live data streaming with WebSocket...")
        if live_data_manager.start_streaming(dry_run=False, jumpstart=True):
            print("✅ Live data streaming started successfully")
            
            # Wait for buffer to fill
            print("⏳ Waiting for data buffer to fill...")
            max_wait = 60
            start_wait = time.time()
            
            while not live_data_manager.is_ready_for_trading():
                if time.time() - start_wait > max_wait:
                    print("⚠️ Timeout waiting for live data manager to be ready")
                    break
                
                status = live_data_manager.get_buffer_status()
                print(f"📊 Buffer: {status['buffer_size']}/{status['max_buffer_size']} "
                      f"({status['buffer_utilization']:.1%}), "
                      f"WebSocket: {status.get('websocket_connected', False)}")
                time.sleep(5)
            
            if live_data_manager.is_ready_for_trading():
                print("✅ Live data manager ready for trading!")
                
                # Test feature generation
                features = live_data_manager.get_latest_features()
                if features is not None:
                    print(f"🎯 Generated {len(features.columns)} features successfully")
                    print(f"📅 Data shape: {features.shape}")
                else:
                    print("⚠️ No features generated yet")
                
                # Monitor for a bit
                print("🔍 Monitoring live data for 30 seconds...")
                monitor_start = time.time()
                while time.time() - monitor_start < 30:
                    status = live_data_manager.get_buffer_status()
                    print(f"📈 Live monitoring - Buffer: {status['buffer_size']}, "
                          f"WebSocket: {status.get('websocket_connected', False)}")
                    time.sleep(10)
                
                print("✅ Live data manager test completed successfully")
                return True
            else:
                print("❌ Live data manager failed to become ready")
                return False
        else:
            print("❌ Failed to start live data streaming")
            return False
            
    except Exception as e:
        print(f"❌ Live data manager test failed: {e}")
        logging.exception("Live data manager test error")
        return False
    finally:
        if live_data_manager:
            live_data_manager.stop_streaming()
            live_data_manager = None

def test_strategy_with_websocket():
    """Test full strategy execution with WebSocket"""
    global strategy
    
    print("\n🎯 Testing Strategy with WebSocket Integration")
    print("="*50)
    
    try:
        # Load strategy configuration
        config_path = "hypothesis/1_session_based_tod_anomalies/config.yml"
        strategy_config = shared_utils.load_config(config_path)
        
        # Ensure WebSocket is enabled and dry_run is false for real WebSocket test
        strategy_config['strategy']['data']['use_websocket'] = True
        strategy_config['strategy']['data']['dry_run'] = False  # Use real WebSocket
        strategy_config['strategy']['execution']['max_loops'] = 3  # Short test
        strategy_config['strategy']['execution']['loop_interval'] = 10  # 10 second intervals
        
        print(f"📋 Strategy config loaded: {strategy_config['strategy']['hypothesis_name']}")
        print(f"🔌 WebSocket enabled: {strategy_config['strategy']['data']['use_websocket']}")
        print(f"🏃 Dry run mode: {strategy_config['strategy']['data']['dry_run']}")
        
        # Initialize strategy
        strategy = SessionAnomalyStrategy(**strategy_config['strategy'])
        
        print("🚀 Starting strategy execution with WebSocket...")
        
        # Execute strategy
        results = strategy.execute_strategy()
        
        print("\n📊 Strategy Execution Results:")
        print(f"✅ Status: {results.get('status')}")
        print(f"📈 Signals generated: {results.get('signals_generated', 0)}")
        print(f"📊 Positions opened: {results.get('positions_opened', 0)}")
        print(f"📉 Positions closed: {results.get('positions_closed', 0)}")
        print(f"🔄 Total loops: {results.get('total_loops', 0)}")
        
        if results.get('errors'):
            print(f"⚠️ Errors: {results['errors']}")
        
        if results.get('session_analysis'):
            print(f"📅 Session analysis: {results['session_analysis']}")
        
        # Check if WebSocket was used
        if strategy.live_data_manager and hasattr(strategy.live_data_manager, 'websocket_manager'):
            ws_status = strategy.live_data_manager.websocket_manager.get_connection_status() if strategy.live_data_manager.websocket_manager else None
            if ws_status:
                print(f"🔌 WebSocket stats: {ws_status['messages_received']} messages received")
        
        success = results.get('status') in ['completed', 'running']
        print(f"\n{'✅ Strategy test completed successfully' if success else '❌ Strategy test failed'}")
        return success
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        logging.exception("Strategy test error")
        return False
    finally:
        if strategy:
            strategy._cleanup_resources()
            strategy = None

def main():
    """Run all WebSocket integration tests"""
    print("🚀 Capital.com WebSocket Integration Test Suite")
    print("=" * 60)
    print(f"📅 Test started at: {datetime.now()}")
    
    # Setup
    setup_logging()
    signal.signal(signal.SIGINT, signal_handler)
    
    test_results = {}
    
    try:
        # Test 1: WebSocket Data Manager
        print("\n🔍 Test 1: WebSocket Data Manager")
        test_results['websocket_manager'] = test_websocket_data_manager()
        
        # Test 2: Live Data Manager with WebSocket
        print("\n🔍 Test 2: Live Data Manager with WebSocket")
        test_results['live_data_manager'] = test_live_data_manager_with_websocket()
        
        # Test 3: Full Strategy with WebSocket
        print("\n🔍 Test 3: Strategy with WebSocket Integration")
        test_results['strategy_execution'] = test_strategy_with_websocket()
        
        # Summary
        print("\n📋 Test Results Summary")
        print("=" * 40)
        
        all_passed = True
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
            if not result:
                all_passed = False
        
        print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        if all_passed:
            print("\n🎉 WebSocket integration is working perfectly!")
            print("🔥 Your session anomaly strategy is now ready for live Capital.com data!")
        else:
            print("\n🔧 Some tests failed. Check the logs above for details.")
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        logging.exception("Test suite error")
    finally:
        print(f"\n📅 Test completed at: {datetime.now()}")
        cleanup_resources()

if __name__ == "__main__":
    main()