"""
Test script for risk management optimizer integration
"""

from risk_management_optimizer import RiskManagementOptimizer
import yaml

def test_integration():
    """Test the risk management optimizer integration"""
    
    print("🔧 Testing Risk Management Optimizer Integration")
    print("=" * 60)
    
    # Initialize optimizer
    config_path = "hypothesis/spread_compression_patterns/config_backtest_v3_quick.yml"
    
    try:
        optimizer = RiskManagementOptimizer(config_path)
        print(f"✅ Successfully loaded base config: {config_path}")
        
        # Test parameter generation
        quick_params = optimizer.generate_test_parameter_grid('quick')
        print(f"✅ Generated {len(quick_params)} quick test combinations")
        
        # Test config creation
        test_params = quick_params[0]  # Test with first combination
        test_config = optimizer.create_test_config(optimizer.base_config, test_params)
        print(f"✅ Successfully created test config with params: {test_params}")
        
        # Show what the modified config looks like
        print("\n📄 Sample Test Configuration Changes:")
        print("Original vs Modified:")
        
        original_conf_thresh = optimizer.base_config['backtesting']['signal_processing']['confidence_threshold']
        modified_conf_thresh = test_config['backtesting']['signal_processing']['confidence_threshold']
        print(f"  confidence_threshold: {original_conf_thresh} → {modified_conf_thresh}")
        
        original_stop_loss = optimizer.base_config['backtesting']['risk_management']['stop_loss_pct']
        modified_stop_loss = test_config['backtesting']['risk_management']['stop_loss_pct']
        print(f"  stop_loss_pct: {original_stop_loss} → {modified_stop_loss}")
        
        original_take_profit = optimizer.base_config['backtesting']['risk_management']['take_profit_pct']
        modified_take_profit = test_config['backtesting']['risk_management']['take_profit_pct']
        print(f"  take_profit_pct: {original_take_profit} → {modified_take_profit}")
        
        # Test single backtest (will create temp file and try to run)
        print(f"\n🧪 Testing single backtest execution...")
        print("⚠️  This will attempt to run: python main.py --hyp <temp_config>")
        print("   If this fails, it means the integration is working but backtesting has issues")
        
        # Uncomment to test actual backtest execution
        # result = optimizer._run_single_backtest(test_config, test_params)
        # if result:
        #     print(f"✅ Backtest successful: {result}")
        # else:
        #     print("❌ Backtest failed (check logs for details)")
        
        print("\n🎯 Integration Test Summary:")
        print("✅ Config loading: PASS")
        print("✅ Parameter generation: PASS")
        print("✅ Config modification: PASS")
        print("⏸️  Backtest execution: SKIPPED (uncomment to test)")
        
        print(f"\n🚀 Ready to run optimization!")
        print("Run with: python risk_management_optimizer.py config_backtest_v3_quick.yml --quick-test")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()