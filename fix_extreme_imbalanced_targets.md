# Quick Intro
Read `README.md` for and overview of the project, and `hypothesis/spread_compression_patterns/README.md` for information about the hypothesis I'm trying to make work.

I've really been struggling to make my model profitable (backtesting showed consistently "0% win rate and always negative returns, with extremly low confidence on BUY/SELL"). After many fixes, we think the next core issue is extreme class imbalance: 269.5:1 ratio
```
HOLD (Class 0): 605,783 samples (99.22%)
BUY (Class 1):   2,248 samples (0.37%)
SELL (Class 2):  2,524 samples (0.41%)
```

# 🎯 ROOT CAUSE ANALYSIS:

The fundamental issue isn't the thresholds (in the configs) - it's the target generation philosophy. We're using:

❌ CURRENT FLAWED APPROACH:

"Generate signals when market conditions are perfect"
- Uses arbitrary momentum thresholds (0.0005%)
- Requires multiple confirmations (short + medium momentum + volume)
- Results in ultra-rare signals (0.37% BUY)

✅ ROOT CAUSE SOLUTION:

"Generate balanced targets, then validate quality"

Instead of trying to engineer "perfect" signals upfront, we should:

🛠️ ROOT CAUSE FIX STRATEGY:

1. TARGET-FIRST APPROACH

- Generate balanced targets (5-10% each for BUY/SELL)
- Use percentile-based thresholds rather than absolute values
- Let the model learn what combinations work

2. QUALITY VALIDATION SECOND

- After generating balanced targets, validate their predictive power
- Iterate on features that discriminate signal quality
- Use cross-validation to ensure targets predict future moves

3. EVIDENCE-BASED THRESHOLDS

- Analyze actual forex microstructure data to set realistic thresholds
- 0.0005% moves might be noise in 1-minute FX data
- Use statistical analysis of typical price movements

💡 CONCRETE ROOT CAUSE SOLUTION:

Replace hard-coded thresholds with data-driven percentile approach:

# Instead of: price_momentum > 0.0005
# Use: price_momentum > np.percentile(price_momentum, 85)  # Top 15%

# Instead of: volume_momentum > 0.1  
# Use: volume_momentum > np.percentile(volume_momentum, 80)  # Top 20%

This ensures:
- Consistent signal rates across different market regimes
- Adaptive thresholds based on actual data distribution
- Balanced targets by design (15% BUY + 15% SELL + 70% HOLD)
