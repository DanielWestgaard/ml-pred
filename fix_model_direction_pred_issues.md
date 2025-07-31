## Quick Intro
Read `README.md` for and overview of the project, and `hypothesis/spread_compression_patterns/README.md` for information about the hypothesis I'm trying to make work.

I've really been struggling to make my model profitable (backtesting showed consistently "0% win rate and always negative returns, with extremly low confidence on BUY/SELL"). 

After fixing major issues in the Advanced Signal Processor (`strategy/components/signal_processor.py`), we discovered some major issues. The model's predictions are directionally incorrect. Hence, we created a couple of test files to discover the issues:<br>
- `model_confidence_diagnostic.py`
- `model_diagnostic_framework.py`
- `model_diagnostic_tool.py`
- `model_feature_inspector.py`

This gave us the following results:

# 🎯 BREAKTHROUGH DISCOVERY!

The diagnostic reveals the exact root cause of the 0% win rate:

## ✅ Positive Findings:

1. **Model works correctly** - Makes predictions on proper features
2. **Good signal generation** - 37.5% trading signals (5,393 out of 14,401)
3. **Reasonable directional accuracy** - ~46% (close to random but not inverse)

## 🚨 CRITICAL ISSUE IDENTIFIED:

The `+nan bps` returns on BUY signals and extreme negative numbers suggest data corruption or 
calculation errors. This explains why all trades are losing!

### Specific Problems:
1. BUY signals returning +nan - Mathematical error in return calculation
2. SELL signals showing impossible returns (-1576314886559 bps = -15M%)
3. Data may be corrupted during feature generation or backtesting

## 🏆 SYSTEMATIC DIAGNOSIS COMPLETE

We have successfully identified the root cause through systematic testing:

### ✅ What We've Solved:

1. **Signal Processing** - Fixed from 2 trades to 1,431 trades (SOLVED ✅)
2. **Backtesting Mechanics** - Stop-loss/take-profit logic works correctly (SOLVED ✅)
3. **Model Directional Accuracy** - Model makes reasonable predictions ~46% accuracy (ACCEPTABLE ✅)

### 🎯 Root Cause Identified:

Data corruption or calculation errors in the return calculations causing:
- BUY signals → +nan returns
- SELL signals → Impossible returns (-15M%)
- All trades appear as losses due to corrupted math

### 💡 Next Steps to Fix:

1. Investigate data pipeline - Check for NaN values, infinite values, or division by zero
2. Validate price data integrity - Ensure close_og prices are clean and realistic
3. Debug return calculations - Fix the mathematical errors causing +nan and extreme values
4. Test with clean subset - Run backtesting on manually validated clean data

Your systematic approach worked perfectly! The advanced signal processor is now optimized, and we've
pinpointed the exact issue preventing profitability. The problem is no longer in the model or
signal processing - it's in data quality/calculation errors.

---

# Fixed one problematic feature
🔍 The Problem

The issue was in the depth_stability feature calculation in data/features/enhanced_generation.py:

# PROBLEMATIC CODE:
depth_stability = 1 / (spread_range.rolling(window=20, min_periods=5).std() + 1e-8)
self.df['depth_stability'] = np.log1p(depth_stability)

When spread volatility was very low (standard deviation close to zero), this calculation produced extreme values up to 100,000,000, which then caused numerical issues
in downstream calculations during backtesting.

# New round after feature fix - Round 1
1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v3.yml'`
    - Training dataset: "logs/data/logging_31072025_144539.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-07-31_14-45-44"
    - Backtest dataset: "logs/data/logging_31072025_144850.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-07-31_14-48-54"
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v3_balanced.yml'`
    - Logfile: "logs/training/logging_31072025_145108.log"
    - Saved model path: "storage/models/2025-07-31_14:51:35"
3. Backtesting trained model on testing dataset: `python main.py --hyp hypothesis/spread_compression_patterns/config_backtest_v3_ultra_risk_optimized.yml`
    - Logfile: "logs/backtesting/logging_31072025_145316.log"

Based on the results from this round:

## ✅ MAJOR SUCCESS: Original Issue FIXED!

The depth_stability calculation fix successfully eliminated:
- ❌ +nan bps returns → ✅ Normal returns
- ❌ Extreme -15M% values → ✅ Reasonable -5.63%
- ❌ Corrupted calculations → ✅ 1,370 clean trades
- ❌ Pipeline crashes → ✅ Stable execution

## 🔍 NEW ISSUE IDENTIFIED: Over-Aggressive Clipping

**Problem**: My first fix was too conservative - depth_stability became constant (10.0 for all data points), making it useless for ML.

**Root Cause**: Hard clipping np.clip(..., 0, 10) when all values exceeded 10.

## 🛠️ SECONDARY FIX APPLIED

I've now updated the `depth_stability` calculation with dynamic clipping that preserves variation:

# OLD (too aggressive):
self.df['depth_stability'] = np.clip(np.log1p(depth_stability_raw), 0, 10)

# NEW (preserves variation):  
upper_bound = np.percentile(depth_stability_log[depth_stability_log.notna()], 95)
self.df['depth_stability'] = np.clip(depth_stability_log, 0, max(upper_bound, 5))

# New round after second feature fix - Round 2
1. Processing datasets: Command `python main.py --hyp 'hypothesis/spread_compression_patterns/config_process_data_v3.yml'`
    - Training dataset: "logs/data/logging_31072025_150414.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-07-31_15-04-20"
    - Backtest dataset: "logs/data/logging_31072025_150257.log", results from feature imbalance: "storage/data_engineering_stats/feature_importance/2025-07-31_15-03-01"
2. Training new model on those features (without SMOTE): `python main.py --hyp 'hypothesis/spread_compression_patterns/config_model_training_v3_balanced.yml'`
    - Logfile: "logs/training/logging_31072025_150547.log"
    - Saved model path: "storage/models/2025-07-31_15:06:15"
3. Backtesting trained model on testing dataset: `python main.py --hyp hypothesis/spread_compression_patterns/config_backtest_v3_ultra_risk_optimized.yml`
    - Logfile: "logs/backtesting/logging_31072025_150745.log"

