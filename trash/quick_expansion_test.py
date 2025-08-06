from spread_compression_analyzer import SpreadCompressionAnalyzer
import pandas as pd

# Quick test of just expansion patterns
file_path = "storage/data/capital_com/GBPUSD/raw/1m/2024-06-01_2025-05-23.csv"

print("Testing Spread Expansion Patterns...")
analyzer = SpreadCompressionAnalyzer(file_path)

# Test only expansion patterns
expansion_results = analyzer.test_predictive_power(
    percentile_thresholds=[5, 10, 15, 20], 
    test_expansion=True
)

print(f"\n=== EXPANSION RESULTS SUMMARY ===")
if len(expansion_results) > 0:
    # Show significant results
    significant = expansion_results[expansion_results['statistically_significant'] == True]
    
    if len(significant) > 0:
        print(f"✅ Found {len(significant)} statistically significant patterns:")
        for _, row in significant.iterrows():
            expansion_pct = 100 - row['percentile_threshold']
            print(f"• {expansion_pct}th percentile expansion, {row['forward_minutes']}min forward:")
            print(f"  Mean return: {row['mean_return_pattern']:.6f} vs {row['mean_return_all']:.6f}")
            print(f"  Events: {row['pattern_events']}, P-value: {row['p_value']:.4f}")
            print(f"  Direction: {row['positive_direction_pct']:.1f}% positive\n")
        
        # Save results
        expansion_results.to_csv('expansion_results.csv', index=False)
        print("💾 Saved expansion results to: expansion_results.csv")
    else:
        print("❌ No statistically significant expansion patterns found")
else:
    print("❌ No expansion results generated")