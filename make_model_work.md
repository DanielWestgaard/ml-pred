## Quick Intro
Read `README.md` for and overview of the project, and `hypothesis/spread_compression_patterns/README.md` for information about the hypothesis I'm trying to make work.
I've really been struggling to make my model profitable (backtesting showed consistently "0% win rate and always negative returns, with extremly low confidence on BUY/SELL"). However, after now being confident I've solved some key, major issues, we're finally getting somewhere.

## Previous "fixes" which helped results
- "No" lookahead bias in features (`data/features/enhanced_generation.py`)
- Features are "optimized", has predictive powers and picked out the best feature of both worlds ("hybrid")
- Not using OHLCV in training, only `close_og` (un-normalized original `close` prices) as this is needed for placing orders
- Improved the hypothsesis target, now using v3 (`data/features/target_hypothesis.py`) and proven to be working
- "Fixed"/reduced chances of data leakage with Walk-Forward Validation in model training (Random Forest) (`model/model_training.py` and `model/walk_forward_validator.py`)
- Tested a bunch of different feature combinations and parameters in both model training (`model/`) and backtesting (`backtesting/`)

## Current obstacle - Extreme Class Imbalance
Then, we figured out that the last (major) issue was extreme class imbalance: Eg. 
- **Class 0 (No signal)**: 1,231,402 samples (93.9%)
- **Class 1 (Basic signal)**: 78,392 samples (6.0%) 
- **Class 2+ (Strong signals)**: 2,065 samples (0.15%)

This led to model learning and only predicting HOLD ("no signal"), and also very low confidence on positive predictions: still 0% win rate and always negative returns (varying from -9% to -0,5%). Eg. from this logfile `logs/mixed/logging_30072025_134550.log`

I tried to change and specify `model_training.parameters.class_weight: {0: 1, 1: 40, 2: 40}` instead of using the `balanced` (standard/default), but his didn't help either.

However, after we implemented SMOTE into the model training (`model/model_factory/random_forest.py`) and "changed the distribution of classes", that changed a lot! Model still isn't profitable, **yet**, but we're finally seeing positive win rates!

### How to solve
The next obstacle however, is that model is "too confident": it now predicts "0.4948" almost  constantly (on sell, a bit lower on buy - always around "0.3875") instead of the previous "0.0000".
Increasing the `confidence_threshold` won't help as this leads to 0 trades, instead of thousands when threshold is low enough for trades to "come through".

Then, I implemented a new `Advanced Signal Processor` (intelligent signal filtering) (`strategy/components/signal_processor.py`). This helped somewhat in handling the excessive trades (when confidence is configured to low enough so trades come through), but with a "broken model" the results were still not much.<br>
Though, implementing this gave us:
- **Dramatic signal reduction**: 1440→2-8 signals/day  
- **Quality filtering**: Temporal + directional change logic working
- **Model issue discovered**: Problem is model outputting only 2-3 constant confidence values (0.4948, 0.3875), not the signal processor

### Usage
```yaml
backtesting:
  signal_processing:
    use_advanced_filtering: true
    confidence_threshold: 0.485
    min_confidence_delta: 0.0      # 0.0 for flat confidence models
    cooldown_minutes: 10           # 10min between same-direction signals  
    max_signals_per_hour: 8        # Max 8 signals per hour
    confidence_window_minutes: 15  # Rolling confidence window
```

### Key Insight
**The signal processor works perfectly.** The core issue is the model needs retraining - it's overfitted and only outputs ~3 constant confidence values instead of a proper confidence distribution.<br>
Both new models show:
- Proper confidence distribution ✅
- All 3 classes predicted ✅
- Advanced signal filtering working ✅
- But 0% win rate ❌

**Possible causes**:
1. ✅ Target/labels wrong - Model learning opposite of what it should
    Not this, as you can read in `actions_mmw.md`
2. Feature quality - Features don't predict spread compression well
3. Market regime - March 2025 data doesn't match training period patterns
4. ✅ Risk management - Stop losses too tight, take profits too wide
    Currently, not this, as you can read in `actions_mmw.md`