"""
Debug script to verify config parameter mapping is working
"""

import yaml
import tempfile
from risk_management_optimizer import RiskManagementOptimizer

def debug_config_mapping():
    """Debug the config parameter mapping"""
    
    print("🔍 Debugging Config Parameter Mapping")
    print("=" * 50)
    
    # Load optimizer
    config_path = "hypothesis/spread_compression_patterns/config_backtest_v3_quick.yml"
    optimizer = RiskManagementOptimizer(config_path)
    
    # Test parameters
    test_params = {
        'confidence_threshold': 0.45,      # Original: 0.485
        'stop_loss_pct': 0.05,            # Original: 0.012
        'take_profit_pct': 0.15,          # Original: 0.035
        'cooldown_minutes': 15,           # Original: 10
        'max_signals_per_hour': 6,        # Original: 8
    }
    
    print("📋 Test Parameters:")
    for param, value in test_params.items():
        print(f"  {param}: {value}")
    
    # Create test config
    test_config = optimizer.create_test_config(optimizer.base_config, test_params)
    
    print("\n🔧 Config Changes:")
    print("Original → Modified")
    
    # Check signal processing changes
    original_sp = optimizer.base_config['backtesting']['signal_processing']
    modified_sp = test_config['backtesting']['signal_processing']
    
    print(f"confidence_threshold: {original_sp['confidence_threshold']} → {modified_sp['confidence_threshold']}")
    print(f"cooldown_minutes: {original_sp['cooldown_minutes']} → {modified_sp['cooldown_minutes']}")
    print(f"max_signals_per_hour: {original_sp['max_signals_per_hour']} → {modified_sp['max_signals_per_hour']}")
    
    # Check risk management changes
    original_rm = optimizer.base_config['backtesting']['risk_management'] 
    modified_rm = test_config['backtesting']['risk_management']
    
    print(f"stop_loss_pct: {original_rm['stop_loss_pct']} → {modified_rm['stop_loss_pct']}")
    print(f"take_profit_pct: {original_rm['take_profit_pct']} → {modified_rm['take_profit_pct']}")
    
    # Save test config to file and verify
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False)
        temp_path = f.name
    
    print(f"\n📄 Saved test config to: {temp_path}")
    
    # Read it back and verify
    with open(temp_path, 'r') as f:
        saved_config = yaml.safe_load(f)
    
    print("\n✅ Verification - Config file contents:")
    sp_saved = saved_config['backtesting']['signal_processing']
    rm_saved = saved_config['backtesting']['risk_management']
    
    print(f"  confidence_threshold: {sp_saved['confidence_threshold']}")
    print(f"  cooldown_minutes: {sp_saved['cooldown_minutes']}")
    print(f"  max_signals_per_hour: {sp_saved['max_signals_per_hour']}")
    print(f"  stop_loss_pct: {rm_saved['stop_loss_pct']}")
    print(f"  take_profit_pct: {rm_saved['take_profit_pct']}")
    
    # Verify the changes actually took effect
    changes_correct = (
        sp_saved['confidence_threshold'] == 0.45 and
        sp_saved['cooldown_minutes'] == 15 and
        sp_saved['max_signals_per_hour'] == 6 and
        rm_saved['stop_loss_pct'] == 0.05 and
        rm_saved['take_profit_pct'] == 0.15
    )
    
    if changes_correct:
        print("\n🎯 SUCCESS: All parameter changes applied correctly!")
    else:
        print("\n❌ FAILURE: Parameter changes did not apply correctly!")
        print("This explains why all backtests produce identical results.")
    
    # Clean up
    import os
    try:
        os.unlink(temp_path)
    except:
        pass

if __name__ == "__main__":
    debug_config_mapping()