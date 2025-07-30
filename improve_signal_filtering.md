## Quick Intro
I've really been struggling to make my model profitable. However, after now being confident I'

- No lookahead bias in features
- Features are "optimized" and picked out the best feature of both worlds ("hybrid")
- Improved the hypothsesis target, now using v3
- "Fixed"/reduced chances of data leakage with Walk-Forward Validation
- Tested a bunch of different feature combinations, and parameters in both model training and backtesting

Then, we figured out that the last (major) issue was major class imbalance: Eg. 
- **Class 0 (No signal)**: 1,231,402 samples (93.9%)
- **Class 1 (Basic signal)**: 78,392 samples (6.0%) 
- **Class 2+ (Strong signals)**: 2,065 samples (0.15%)

This led to model learning and only predicting HOLD ("no signal"), and also very low confidence on positive predictions: 0% win rate and always negative returns (varying from -9% to -0,5%).

I tried to change and specify `model_training.parameters.class_weight: {0: 1, 1: 40, 2: 40}` instead of using the `balanced` on that is standard, but his didn't help either.

However, after we implemented SMOTE into the model training and "changed the distribution of classes", that changed a lot! Model still isn't profitable, yet, but we're finally seeing positive win rates! 

The next obstacle however, is that model is "too confident": it now predicts "0.4948" almost  constantly (on sell, a bit lower on buy - always around "0.3875") instead of the previous "0.0000".
Increasing the `confidence_threshold` won't help as this leads to 0 trades, instead of thousands when threshold is low enough for trades to "come through".

Hence, the solution is to attack this at root cause: The signal processor is currrently not good enough. We need intelligent signal filtering.

Here's how proper signal filtering should behave for a trading system:

1. Temporal Filtering (Anti-Spam)

❌ Current: Signal every minute → 1440 trades/day
✅ Ideal: Max 1 signal every 15-30 minutes → 32-96 trades/day

2. Directional Change Detection

❌ Current: SELL, SELL, SELL, SELL (redundant)
✅ Ideal: SELL → wait → BUY (only on direction change)

3. Confidence Delta Filtering

❌ Current: Accept any 0.4948 signal
✅ Ideal: Only trade when confidence jumps significantly
    - Previous: 0.3000 → Current: 0.4948 = ✅ Trade (big jump)
    - Previous: 0.4900 → Current: 0.4948 = ❌ Skip (minor change)

4. Multi-Feature Confluence

❌ Current: Trade on model confidence alone
✅ Ideal: Require 2-3 confirmations:
    - Model confidence > threshold
    - Volume spike > normal
    - Spread compression detected
    - Price momentum aligned

5. Market Regime Awareness

❌ Current: Trade in any market condition
✅ Ideal: Only trade during favorable conditions:
    - High volatility periods
    - Active trading sessions (London/NY)
    - Avoid major news events

6. Position-Aware Filtering

❌ Current: Open unlimited same-direction positions
✅ Ideal: Smart position management:
    - Max 1-2 positions in same direction
    - Don't add to losers
    - Scale into winners

🔧 Practical Implementation Strategy

The most effective approach would be layered filtering:

Layer 1: Temporal (Essential)

- Minimum 15-minute gap between signals
- Or: Only 1 signal per hour per direction

Layer 2: Confidence Delta (High Impact)

- Only trade when confidence increases by >0.05 from recent average
- Track rolling 10-minute confidence average

Layer 3: Feature Confluence (Quality)

- Require 2+ supporting indicators beyond model confidence
- Example: Model + Volume spike + Spread compression

Layer 4: Market Timing (Optimization)

- Prefer high-activity trading sessions
- Avoid low-liquidity periods

💡 Immediate Quick Fix

Before implementing full filtering, we could add a simple cooldown mechanism:

signal_processing:
confidence_threshold: 0.49    # Just below 0.4948
cooldown_minutes: 30          # Wait 30min between signals
max_signals_per_hour: 2       # Hard limit

Which filtering approach interests you most?
1. Temporal spacing (quickest to implement)
2. Confidence delta detection (highest quality improvement)
3. Multi-feature confluence (most sophisticated)
4. All of the above in phases

The goal is to transform "1000+ low-value signals" into "10-20 high-value signals" per day.
