"""
Test single optimization run to verify fixes work
"""

from risk_management_optimizer import RiskManagementOptimizer

def test_single_run():
    """Test a single optimization run with fixed parsing"""
    
    print("🧪 Testing Single Optimization Run")
    print("=" * 40)
    
    # Initialize optimizer
    config_path = "hypothesis/spread_compression_patterns/config_backtest_v3_quick.yml"
    optimizer = RiskManagementOptimizer(config_path)
    
    # Test with very low confidence threshold (based on diagnostic)
    test_params = {
        'confidence_threshold': 0.05,     # Much lower than 0.485
        'stop_loss_pct': 0.1,            # Tighter than 1.2%
        'take_profit_pct': 0.2,          # Conservative
        'cooldown_minutes': 5,           # Faster
        'max_signals_per_hour': 12       # More signals allowed
    }
    
    print(f"🔧 Test Parameters: {test_params}")
    
    # Create test config
    test_config = optimizer.create_test_config(optimizer.base_config, test_params)
    
    # Run single backtest
    print(f"\n🚀 Running single backtest...")
    result = optimizer._run_single_backtest(test_config, test_params)
    
    if result:
        print(f"\n✅ SUCCESS! Parsed Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # Check if we got more trades with lower confidence
        trades = result.get('total_trades', 0)
        if trades > 2:
            print(f"\n🎯 IMPROVEMENT: {trades} trades (vs 2 with high confidence threshold)")
        else:
            print(f"\n⚠️  Still only {trades} trades - model confidence issue persists")
            
    else:
        print(f"\n❌ FAILED: Could not parse backtest results")
        print(f"Check logs for details")

if __name__ == "__main__":
    test_single_run()