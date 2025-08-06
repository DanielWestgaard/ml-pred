#!/usr/bin/env python3
"""
Test script for the modernized backtesting system

This script validates that all components of the new plugin-based
backtesting system work correctly together.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all new components can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test core components
        from backtesting.core import BacktestEngine, BacktestModelHandler, BacktestSignalHandler, BacktestConfigValidator
        print("✅ Core components imported successfully")
        
        # Test hypothesis plugins
        from backtesting.hypothesis_plugins import HypothesisPluginRegistry, ExitStrategyFactory, SignalTrackerFactory
        print("✅ Plugin system imported successfully")
        
        # Test specific plugins
        from backtesting.hypothesis_plugins.types import SpreadCompressionPlugin
        print("✅ Specific hypothesis plugins imported successfully")
        
        # Test main orchestrator
        from backtesting.backtester import BacktestOrchestrator
        print("✅ Main BacktestOrchestrator imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_plugin_registry():
    """Test the hypothesis plugin registry"""
    print("\n🧪 Testing plugin registry...")
    
    try:
        from backtesting.hypothesis_plugins import HypothesisPluginRegistry
        
        # Test plugin availability
        available_plugins = HypothesisPluginRegistry.get_available_plugins()
        print(f"✅ Available plugins: {available_plugins}")
        
        # Test getting a plugin
        test_config = {
            'hypothesis_name': 'spread_compression_test',
            'backtesting': {
                'enabled': True,
                'initial_capital': 10000
            }
        }
        
        plugin = HypothesisPluginRegistry.get_plugin('spread_compression', test_config)
        plugin_info = plugin.get_hypothesis_info()
        print(f"✅ Plugin created: {plugin_info['plugin_class']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Plugin registry test failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation"""
    print("\n🧪 Testing configuration validation...")
    
    try:
        from backtesting.core import BacktestConfigValidator
        
        validator = BacktestConfigValidator()
        
        # Test valid configuration
        valid_config = {
            'hypothesis_name': 'test_hypothesis',
            'backtesting': {
                'enabled': True,
                'initial_capital': 10000,
                'transaction_costs': 0.001,
                'slippage': 0.0001,
                'risk_management': {
                    'max_position_size': 0.02,
                    'stop_loss_pct': 0.01,
                    'take_profit_pct': 0.02
                },
                'signal_processing': {
                    'confidence_threshold': 0.6,
                    'processor_type': 'basic'
                }
            }
        }
        
        is_valid, errors, warnings = validator.validate(valid_config)
        
        if is_valid:
            print("✅ Configuration validation passed")
            if warnings:
                print(f"⚠️  Warnings: {len(warnings)} warning(s)")
        else:
            print(f"❌ Configuration validation failed: {errors}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_factory_patterns():
    """Test factory patterns for exit strategies and signal trackers"""
    print("\n🧪 Testing factory patterns...")
    
    try:
        from backtesting.hypothesis_plugins import ExitStrategyFactory, SignalTrackerFactory
        
        # Test exit strategy factory
        exit_config = {'confidence_exit_threshold': 0.3}
        exit_strategy = ExitStrategyFactory.create('spread_compression', exit_config)
        print(f"✅ Exit strategy created: {exit_strategy.__class__.__name__}")
        
        # Test signal tracker factory
        tracker_config = {'track_compression_features': True}
        signal_tracker = SignalTrackerFactory.create('spread_compression', tracker_config)
        print(f"✅ Signal tracker created: {signal_tracker.__class__.__name__}")
        
        # Test available options
        available_strategies = ExitStrategyFactory.get_available_strategies()
        available_trackers = SignalTrackerFactory.get_available_trackers()
        
        print(f"✅ Available exit strategies: {available_strategies}")
        print(f"✅ Available signal trackers: {available_trackers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory pattern test failed: {e}")
        return False

def test_orchestrator_initialization():
    """Test that the orchestrator can be initialized"""
    print("\n🧪 Testing orchestrator initialization...")
    
    try:
        from backtesting.backtester import BacktestOrchestrator
        
        # Test initialization without actually running (no config file needed)
        orchestrator = BacktestOrchestrator()
        print("✅ BacktestOrchestrator initialized successfully")
        
        # Test validation method exists
        if hasattr(orchestrator, 'validate_configuration'):
            print("✅ Configuration validation method available")
        
        if hasattr(orchestrator, 'get_component_info'):
            print("✅ Component info method available")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator initialization test failed: {e}")
        return False

def test_architecture_overview():
    """Display an overview of the new architecture"""
    print("\n📋 MODERNIZED BACKTESTING ARCHITECTURE OVERVIEW")
    print("=" * 60)
    
    # Show directory structure
    print("\n📁 NEW DIRECTORY STRUCTURE:")
    structure = """
backtesting/
├── backtester.py                    # Main orchestrator (MODERNIZED)
├── core/                            # Focused core components
│   ├── backtest_engine.py          # Main coordination engine
│   ├── model_handler.py            # Model loading & prediction
│   ├── signal_handler.py           # Signal processing coordination
│   └── config_validator.py         # Configuration validation
├── hypothesis_plugins/              # Plugin system
│   ├── base_plugin.py              # Abstract base class
│   ├── exit_strategies.py          # Exit strategy implementations
│   ├── signal_trackers.py          # Signal tracking implementations
│   ├── plugin_registry.py          # Plugin management
│   └── types/                      # Hypothesis-specific plugins
│       └── spread_compression_plugin.py
└── [existing components reused]     # simulation_broker.py, etc.

trash/backtesting_deprecated/        # Old monolithic code
└── backtester.py                   # Original orchestrator (DEPRECATED)
"""
    print(structure)
    
    print("\n🔌 PLUGIN SYSTEM BENEFITS:")
    benefits = """
✅ Modular: Each hypothesis gets its own plugin
✅ Extensible: Easy to add new trading strategies  
✅ Maintainable: Focused, single-responsibility classes
✅ Testable: Components can be tested independently
✅ Dynamic: Exit strategies and signal tracking are configurable
✅ Compatible: Leverages existing feature generation system
"""
    print(benefits)
    
    return True

def run_all_tests():
    """Run all tests and report results"""
    print("🚀 TESTING MODERNIZED BACKTESTING SYSTEM")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Plugin Registry", test_plugin_registry),
        ("Configuration Validation", test_configuration_validation),
        ("Factory Patterns", test_factory_patterns),
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Architecture Overview", test_architecture_overview)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The modernized backtesting system is working correctly.")
        print("\nNext steps:")
        print("1. Try running a backtest with: python backtester.py <config.yml>")
        print("2. Create custom hypothesis plugins in backtesting/hypothesis_plugins/types/")
        print("3. Explore advanced signal processing options in your configs")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)