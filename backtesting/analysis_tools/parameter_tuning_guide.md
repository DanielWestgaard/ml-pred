# Parameter Tuning Guide

## 🎯 **QUICK START GUIDE**

The framework is ready for extended testing and parameter optimization. Here's how to use it:

### **1. Quick Parameter Testing**
```bash
# Run quick parameter comparison (4 configurations)
python tools/quick_parameter_tuning.py

# Test single conservative configuration
python main.py -hyp 'hypothesis/spread_compression_patterns/config_test_conservative.yml'
```

### **2. Extended Testing Framework**
```bash
# Run comprehensive testing (27 parameter combinations across 3 periods)
python tools/extended_testing_framework.py

# Analyze results
python tools/extended_testing_framework.py analyze
```

---

## 📊 **PARAMETER TESTING CONFIGURATIONS**

### **Quick Testing (4 configs)**
1. **Baseline** - Current settings (confidence: 0.30, decay: 0.25, life: 3min)
2. **Conservative** - Higher thresholds (confidence: 0.35, decay: 0.30, life: 5min)  
3. **Aggressive** - Lower thresholds (confidence: 0.25, decay: 0.20, life: 2min)
4. **Balanced** - Middle ground (confidence: 0.28, decay: 0.23, life: 4min)

### **Extended Testing (27 configs)**
- **Confidence thresholds:** 0.25, 0.30, 0.35
- **Signal decay thresholds:** 0.20, 0.25, 0.30  
- **Minimum signal life:** 2, 3, 5 minutes
- **Test periods:** Top 3 optimal periods from analysis

---

## 🔧 **MANUAL PARAMETER TUNING**

### **Create Custom Configuration**
1. Copy `config_backtest_improved_simple.yml`
2. Modify these parameters in the `exit_strategy` section:

```yaml
exit_strategy:
  confidence_exit_threshold: 0.XX     # Test range: 0.25-0.35
  signal_strength_decay: 0.XX         # Test range: 0.20-0.30  
  minimum_signal_life: X              # Test range: 2-5 minutes
```

### **Run Single Test**
```bash
python main.py -hyp 'path/to/your/config.yml'
```

---

## 📈 **PARAMETER MEANINGS & OPTIMIZATION TIPS**

### **1. Confidence Exit Threshold (0.25-0.35)**
- **What it does:** Exit when model confidence for opposite direction exceeds this threshold
- **Lower values (0.25):** More aggressive exits, faster response to changing conditions
- **Higher values (0.35):** More conservative exits, fewer false exits
- **Sweet spot:** Usually around 0.28-0.32 for spread compression

### **2. Signal Strength Decay (0.20-0.30)**
- **What it does:** Exit when signal strength drops by this percentage from peak
- **Lower values (0.20):** Exit on smaller deterioration, more sensitive
- **Higher values (0.30):** Allow more signal deterioration before exiting
- **Sweet spot:** Usually around 0.23-0.27 for spread compression

### **3. Minimum Signal Life (2-5 minutes)**
- **What it does:** Protect positions from immediate exit for this duration
- **Lower values (2min):** Allow faster exits, more responsive
- **Higher values (5min):** Force longer holding periods, reduce noise
- **Sweet spot:** Usually around 3-4 minutes for spread compression

---

## 🎯 **OPTIMIZATION WORKFLOW**

### **Phase 1: Quick Validation**
1. Run `config_test_conservative.yml` to verify system works
2. Compare with baseline `config_backtest_improved_simple.yml`
3. Look for improvements in:
   - Total return
   - Exit reason distribution
   - Trade frequency

### **Phase 2: Parameter Sweep**
1. Run `quick_parameter_tuning.py` for rapid comparison
2. Identify best-performing configuration family
3. Note patterns (conservative vs aggressive performance)

### **Phase 3: Extended Testing**
1. Run `extended_testing_framework.py` for comprehensive analysis
2. Test across multiple periods to validate robustness
3. Analyze parameter sensitivity

### **Phase 4: Fine-Tuning**
1. Create custom configs around best-performing parameters
2. Test intermediate values (e.g., if 0.25 and 0.30 work well, try 0.27, 0.28)
3. Validate on out-of-sample periods

---

## 📊 **RESULTS ANALYSIS**

### **Key Metrics to Track**
- **Total Return:** Primary performance metric
- **Total Trades:** Trading frequency (avoid over-trading)
- **Exit Distribution:** Confidence exits vs signal decay vs volatility
- **Win Rate:** Percentage of profitable trades
- **Average Holding Time:** Position duration

### **Interpretation Guide**
- **High confidence exits (>60%):** Consider lowering confidence threshold
- **High signal decay exits (>30%):** Consider adjusting decay threshold
- **Very short holding times (<2min):** Consider increasing minimum signal life
- **Very long holding times (>15min):** Consider more aggressive parameters

---

## 🚀 **NEXT STEPS AFTER PARAMETER TUNING**

### **1. Live Paper Trading**
- Deploy optimized parameters with real-time data
- Monitor performance vs backtesting results
- Collect market condition data for further optimization

### **2. Multi-Period Validation**
- Test optimized parameters on different market conditions
- Validate robustness across various volatility regimes
- Test on longer time horizons (weeks/months)

### **3. Advanced Optimizations**
- Dynamic parameter adjustment based on market conditions
- Ensemble approach with multiple parameter sets
- Machine learning-based parameter optimization

---

## 📋 **EXAMPLE USAGE**

```bash
# 1. Quick validation test
python main.py -hyp 'hypothesis/spread_compression_patterns/config_test_conservative.yml'

# 2. Run parameter comparison
python tools/quick_parameter_tuning.py

# 3. Extended testing (if time permits)
python tools/extended_testing_framework.py

# 4. Analyze extended results
python tools/extended_testing_framework.py analyze
```

The framework is designed to be:
- **Modular:** Test individual parameters or combinations
- **Scalable:** From quick tests to comprehensive analysis
- **Analytical:** Automatic results analysis and optimization recommendations
- **Production-ready:** Generate optimized configurations for live trading