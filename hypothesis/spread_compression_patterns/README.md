## Hypothesis description
**✅ VALIDATED: Use spread compression to predict price movements 15-75 minutes before they occur**.

**Core Discovery**: When bid-ask spreads suddenly compress, it predicts significant price movements with **1.83x movement ratio** on 1-minute GBPUSD data. This represents a genuine market microstructure inefficiency - spread compression signals market maker expectations of increased activity.

**Key Validation (July 2025)**:
- **Compression signals**: 1.29x movement prediction (85,548 events)
- **Volume signals**: 1.52x movement prediction (strongest individual predictor)  
- **Combined multi-feature signals**: 1.83x movement prediction (4,673 premium events)
- **Critical insight**: 1-minute data essential (5-minute smooths out compression patterns)

The broker Capital.com provides real bid-ask OHLC data enabling precise spread compression detection.


## **Validated Performance Targets (GBPUSD 1m):**

- **Signal Frequency**: 17.1 signals per day (premium quality: ~15.7 daily)
- **Movement Prediction**: 1.83x ratio for multi-feature combinations
- **Tier 2+ Quality Signals**: 721 events with 0.000823 average movement (3.8x baseline)
- **Optimal Trading Hours**: London session (proven effective in validation)
- **Data Requirement**: 1-minute data critical for capturing compression patterns


## **Optimized Feature Set (8 Core Features)**
After validation testing and redundancy elimination, the optimal feature set for model training:

```python
# === PRIMARY COMPRESSION SIGNALS (2 features) ===
'compression_ratio_micro'      # Most sensitive (30min baseline, 3min recent)
'compression_consistency'      # Strongest predictor (-0.4169 correlation)

# === ENHANCED SIGNAL STRENGTH (1 feature) ===  
'signal_strength_v2'          # Optimized weights: volume(2.0x) + compression(1.5x) + acceleration(1.2x)

# === VOLUME CONFIRMATION SUITE (3 features) ===
'volume_ratio'                 # 1.40x movement ratio (strongest individual)
'volume_acceleration'          # Volume momentum detection (5-period rate of change)
'volume_breakout'              # Z-score > 2 breakout detection

# === CONTEXT FEATURES (2 features) ===
'london_session'               # Session timing (proven effective)
'price_momentum_5'             # 5-period price momentum
```

**Key Optimizations Made**:
- ✅ **Eliminated redundancy**: Removed `compression_ratio_short` (0.905 correlation with micro)
- ✅ **Multi-timeframe compression**: Micro timeframe (30min/3min) provides highest sensitivity
- ✅ **Enhanced volume detection**: Added momentum and breakout detection beyond simple ratios
- ✅ **Optimized signal weights**: Based on validation performance (volume 2.0x, compression 1.5x)
- ✅ **Perfect data quality**: 100% completeness on 427,741 1-minute data points


## **Model Training Strategy**

**Target Focus**: Train on **Tier 2+ signals** (721 quality events) for premium signal detection.

**Validated Target Distribution**:
- **Tier 0**: 402,311 events (no signal)
- **Tier 1**: 24,709 events (basic compression + movement) 
- **Tier 2**: 626 events (+ consistency) - **Highest quality (0.000842 avg movement)**
- **Tier 3+4**: 95 events (+ volume + London session)

**Training Approach**:
- **Focus on Tier 2+ signals**: Best movement prediction and trading quality
- **Time-series cross-validation**: Respect temporal order of 1-minute data
- **Random Forest**: Handles non-linear relationships and feature interactions
- **Class balancing**: Address imbalance while preserving signal quality
- **Walk-forward validation**: Test on out-of-sample future data

**Success Metrics**:
- **Precision on Tier 2+**: Minimize false signals (transaction costs)
- **Movement prediction**: Maintain 1.8x+ movement ratio
- **Signal frequency**: Target 15-20 quality signals per day
- **London session performance**: Focus on optimal trading hours
