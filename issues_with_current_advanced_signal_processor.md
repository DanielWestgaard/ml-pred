Based on my analysis, I've identified the core issues with your Advanced Signal Processor. The
problem is it's over-filtering with extremely restrictive parameters that work against each
other:

# Key Issues in AdvancedSignalProcessor:

## 1. Excessive Temporal Cooldown (Lines 435-439)

- 30-minute default cooldown between same-direction signals
- Your model generates valid signals every few minutes, but this blocks 99% of them
- 10-minute cooldown in your config is still too restrictive for a model producing good
predictions

## 2. Confidence Delta Filter Too Strict (Lines 441-465)

- Requires each new signal to have significantly higher confidence than recent average
- With your model outputting consistent ~0.44-0.48 confidence, this filter rejects most
signals
- min_confidence_delta: 0.0 in your config helps, but the logic still compares unnecessarily

## 3. Hourly Signal Limit Too Low (Lines 467-476)

- Max 8 signals per hour default (2 in some configs)
- For a model with 66% high-confidence predictions, this is extremely restrictive
- Creates artificial scarcity when your model should be actively trading

## 4. Compounding Filter Effects

- All 4 filter layers must pass simultaneously
- Even if one filter has 50% pass rate, four filters = 6.25% final pass rate
- Your 408k predictions → 2 trades = 0.0005% pass rate

# Solutions to Improve Signal Processor:

## ❌ Option 1: Calibrated Advanced Processor

Modify the existing AdvancedSignalProcessor with balanced parameters:

# In your config:
```yml
signal_processing:
use_advanced_filtering: true
confidence_threshold: 0.40          # Lower from 0.485
cooldown_minutes: 2                 # Much lower from 10
min_confidence_delta: 0.0           # Keep at 0 for flat confidence models
max_signals_per_hour: 30            # Increase from 8
confidence_window_minutes: 5        # Shorter window
```
> [!NOTE]
> This was tested out with `hypothesis/spread_compression_patterns/config_backtest_v3_quick.yml`, but showed no improvements in results: `Final Results: Iterations: 3000/3000 | Return: -0.01%, Trades: 2, Win Rate: 0.0%`. Logfile `logs/backtesting/logging_31072025_103424.log`.

## Option 2: Smart Hybrid Processor

Create a new processor that applies intelligent filtering without over-restriction:

### Key improvements:
- Dynamic cooldown: 30 seconds for direction changes, 2 minutes for same direction
- Confidence percentile filtering: Only filter bottom 20% of predictions
- Volume-based limits: Scale signal limits based on market activity
- Quality over quantity: Prioritize directional changes and confidence spikes

### ✅ SmartHybridProcessor Features:

1. Dynamic Confidence Thresholds: Uses percentile filtering instead of fixed thresholds
2. Intelligent Temporal Filtering:
- 30s cooldown for direction changes (prioritized)
- 2min cooldown for same-direction signals
3. Per-Minute Rate Limiting: 3 signals/minute (180/hour) vs old 8/hour
4. Quality over Quantity: Filters bottom 20% of predictions dynamically

### Test the New Processor:

Run this command to test:
`python main.py --hyp hypothesis/spread_compression_patterns config_backtest_v3_smart.yml`

#### 📊 Expected Results:

- Old AdvancedSignalProcessor: 2 trades (1,217 blocked by temporal cooldown)
- New SmartHybridProcessor: 50-200 trades (balanced filtering)


## Option 3: Adaptive Signal Processor

Implement learning-based filtering that adjusts parameters based on recent performance:
- Track win rates for different confidence ranges
- Adjust thresholds automatically based on market conditions
- Learn optimal cooldown periods from historical data

---

The main issue is your **Advanced Signal Processor has compounding over-restrictive filters**:

1. 30min cooldown between same-direction signals (blocks 99%+ of valid signals)
2. Confidence delta requirement that rejects consistent model outputs
3. Hourly limits of 2-8 signals when your model produces thousands
4. All filters must pass simultaneously = 0.0005% final pass rate

