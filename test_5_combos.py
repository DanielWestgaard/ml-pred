"""
Test 5 parameter combinations to verify backtesting integration works
"""

from enhanced_risk_optimizer import EnhancedRiskOptimizer
import time

def test_5_combinations():
    """Test just 5 combinations to verify everything works"""
    
    print("🔬 Testing 5 Parameter Combinations")
    print("=" * 50)
    
    config_path = "hypothesis/spread_compression_patterns/config_backtest_v3_ultra_risk_optimized.yml"
    optimizer = EnhancedRiskOptimizer(config_path)
    
    # Generate and select first 5 combinations
    all_combinations = optimizer.generate_breakthrough_grid('quick')
    test_combinations = all_combinations[:5]
    
    print(f"Testing these 5 parameter combinations:")
    for i, params in enumerate(test_combinations):
        print(f"  {i+1}. {params}")
    
    print(f"\n⏱️  Starting 5-combination test (should take ~5-10 minutes)...")
    start_time = time.time()
    
    # Run the test
    results = optimizer.run_optimization_batch(test_combinations)
    
    end_time = time.time()
    print(f"⏱️  Test completed in {end_time - start_time:.1f} seconds")
    
    # Analyze results
    analysis = optimizer.analyze_results()
    
    print(f"\n📊 Results Summary:")
    print(f"Total tested: {analysis.get('total_tested', 0)}")
    print(f"Successful: {analysis.get('successful_count', 0)}")
    
    if analysis.get('successful_count', 0) > 0:
        print("✅ SUCCESS! Backtesting integration works!")
        
        if analysis.get('best_win_rate'):
            best = analysis['best_win_rate']
            print(f"🏆 Best result: {best['value']:.1f}% win rate")
            print(f"🏆 Best params: {best['parameters']}")
    else:
        print("❌ No successful results - there may be backtesting issues")
        
        # Show what went wrong
        failed_count = 0
        for key, result_data in results.items():
            if result_data.get('error'):
                print(f"❌ Failed: {result_data['error']}")
                failed_count += 1
        
        print(f"Failed combinations: {failed_count}/{len(test_combinations)}")
    
    return analysis.get('successful_count', 0) > 0

if __name__ == "__main__":
    success = test_5_combinations()
    
    if success:
        print("\n🎉 5-combination test PASSED!")
        print("✅ Ready to run full optimization")
        print("\nTo run full optimization:")
        print("python enhanced_risk_optimizer.py config_backtest_v3_ultra_risk_optimized.yml --test-type focused")
    else:
        print("\n❌ 5-combination test FAILED!")
        print("Fix backtesting issues before running full optimization")