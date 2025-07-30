# 🧪 Feature Combination Testing - Context & Plan

## **Current Situation & Problem**

### **Background**
We're optimizing the feature set for `spread_compression_patterns_v2` hypothesis. The goal is to find the optimal combination of traditional and enhanced features that provides the best trading performance.

### **Performance History**
1. **Old Model (Mixed Features, 25 features)**: -3.58% return, 364 trades, 0% win rate
2. **New Model (Mixed with Dependencies, 23 features)**: -6.49% return, 745 trades, 1.48% win rate  
3. **Enhanced Only Model (11 features)**: -9.57% return, 1157 trades, 0% win rate

### **Root Cause Analysis**
- **Enhanced features alone** lack market context and fundamentals
- **Traditional features alone** miss sophisticated pattern detection
- **Uncontrolled feature sets** (with dependencies) dilute signal quality
- **Need hybrid approach** combining best of both worlds

---

## **Feature Analysis Results**

### **Top Performing Features by Category**

#### **Traditional Features (Consensus Score > 0.4)**
1. `depth_stability` (0.967) - TOP predictor across all metrics
2. `volume_ratio` (0.912) - Strong volume confirmation  
3. `institutional_footprint` (0.878) - Institutional activity detection
4. `cumulative_order_flow` (0.637) - Order flow analysis
5. `compression_prediction_strength` (0.423) - Compression signals
6. `signal_conviction` (0.410) - Composite signal strength

#### **Enhanced Features (Latest Model)**
1. `microstructure_momentum_fusion` (0.999) - Best enhanced feature
2. `top_predictor_interaction` (0.905) - Combined top predictors
3. `volume_profile_enhanced` (0.600) - Enhanced volume analysis
4. `ultra_momentum_suite` (0.488) - High-frequency momentum

#### **Context Features**
- `london_session` (0.175) - FX session timing
- `price_level_support` (0.244) - Support/resistance levels

---

## **🎯 Manual Feature Combination Testing Plan**

### **Priority Testing Queue**

#### **Test 1: Top 3 Traditional (Foundation)**
```yaml
selected_features: ['depth_stability', 'volume_ratio', 'institutional_footprint']
```
- **Rationale**: Highest consensus scores (0.967, 0.912, 0.878)
- **Hypothesis**: Traditional fundamentals provide stable base performance

#### **Test 2: Top 3 Enhanced (Pattern Detection)**  
```yaml
selected_features: ['microstructure_momentum_fusion', 'top_predictor_interaction', 'volume_profile_enhanced']
```
- **Rationale**: Best enhanced features (0.999, 0.905, 0.600)
- **Hypothesis**: Enhanced patterns alone might be sufficient with proper cleanup

#### **Test 3: Hybrid Top 5**
```yaml
selected_features: ['microstructure_momentum_fusion', 'depth_stability', 'volume_ratio', 'institutional_footprint', 'top_predictor_interaction']
```
- **Rationale**: Mix of best from both categories
- **Hypothesis**: Optimal balance of pattern detection + market fundamentals

#### **Test 4: Volume-Focused**
```yaml
selected_features: ['volume_ratio', 'volume_profile_enhanced', 'institutional_footprint', 'cumulative_order_flow']
```
- **Rationale**: Volume-based signals tend to be reliable in FX
- **Hypothesis**: Volume patterns are core to spread compression detection

#### **Test 5: Signal-Rich**
```yaml
selected_features: ['compression_prediction_strength', 'signal_conviction', 'microstructure_momentum_fusion', 'depth_stability', 'london_session']
```
- **Rationale**: Focus on composite signals + context
- **Hypothesis**: Pre-computed signals + timing provide edge

### **Additional Test Ideas (If Time Permits)**
- **Minimal Set**: `['depth_stability', 'microstructure_momentum_fusion']` (top 2 overall)
- **Order Flow Focus**: `['cumulative_order_flow', 'intrabar_pressure', 'institutional_footprint']`
- **Depth Focus**: `['depth_stability', 'depth_variability', 'liquidity_stress']`

---

## **Testing Methodology**

### **Step-by-Step Process**
1. **Update Config**: Modify `hypothesis/spread_compression_patterns/config_process_data_v2.yml` with test features
2. **Process Data**: Run `python main.py --hyp 'hypothesis/.../config_process_data_v2.yml'`
3. **Train Model**: Could be Automatic (included in data processing pipeline), but to be sure, run `python main.py --hyp 'hypothesis/.../config_model_training.yml'` (switch out `model_training.processed_data`) with the processed data.
4. **Extract Model Path**: Look for `storage/models/YYYY-MM-DD_HH:MM:SS` in output
5. **Update Backtest Config**: Set `model_folder` in `config_backtest.yml`
6. **Run Backtest**: Run `python main.py --hyp 'hypothesis/.../config_backtest.yml'`
7. **Record Results**: Extract performance metrics from final output
> [!NOTE]
> To ensure not backtesting on trained data, I've been using `storage/data/capital_com/GBPUSD/raw/1m/2024-01-01_2025-03-01.csv` for training and `storage/data/capital_com/GBPUSD/raw/1m/2025-03-02_2025-07-01.csv` (in processed version) to ensure model is backtested on unseed data. If you store them "as is"; This means both of those files needs to be processed with the same features/config, but the `2024-01-01_2025-03-01` file is the one to train the model and `2025-03-02_2025-07-01` file is the one to backtest.
> [!IMPORTANT]
> To repeat the note above, It is vital: Make sure all processing of datasets are not missing features. If you, potentially, are only getting missing features in backtesting, make sure you have: processed (training file:)  "storage/data/capital_com/GBPUSD/raw/1m/2024-01-01_2025-03-01.csv" AND THEN, right after, process the test ("storage/data/capital_com/GBPUSD/raw/1m/2025-03-02_2025-07-01.csv") file with the same features/configuration-file. Or else the model will be trained on one set of features, but the backtest will be on some old processed dataset with potentially completely different features. Also, this will ensure when the model is backtested, it is on unseen data - so we can actually measure its and the features' performance.

### **Key Metrics to Track**
- **Return %**: Total return percentage
- **Max Drawdown %**: Maximum drawdown percentage  
- **Win Rate %**: Percentage of winning trades
- **Total Trades**: Number of trades executed
- **Sharpe Ratio**: Risk-adjusted return measure
- **Profit Factor**: Ratio of gross profit to gross loss

### **Performance Scoring Formula**
```python
performance_score = (
    total_return_pct * 0.4 +           # 40% weight on returns
    (1 - max_drawdown_pct) * 0.3 +     # 30% weight on drawdown (inverted)
    win_rate * 100 * 0.2 +             # 20% weight on win rate  
    profit_factor * 0.1                # 10% weight on profit factor
)
```

---

## **Current Status**

### **✅ Completed**
- Feature importance analysis across all models
- Performance comparison of existing models
- Root cause analysis of performance degradation
- Feature cleanup fix (removes all dependency features properly)
- Testing framework design
- Priority test combinations identified

### **🔄 In Progress**
- Manual testing of feature combinations
- Performance data collection
- Results analysis

### **📋 Next Steps**
1. Execute the 5 priority test combinations
2. Collect and compare performance metrics
3. Identify the optimal feature set
4. Update production config with best combination
5. Validate with final backtest

---

## **Files & Configs**

### **Key Files**
- **Data Processing**: `hypothesis/spread_compression_patterns/config_process_data_v2.yml`
- **Backtesting**: `hypothesis/spread_compression_patterns/config_backtest.yml` 
- **Feature Analysis**: `storage/data_engineering_stats/feature_importance/2025-07-29_23-29-35/`
- **Testing Script**: `feature_combination_tester.py` (automated, needs fixes)

### **Current Config Status**
- Raw data source: `storage/data/capital_com/GBPUSD/raw/1m/2024-01-01_2025-03-01.csv`
- Target: `target_spread_compression_v2_directional`
- Feature cleanup: ✅ Fixed (removes all temporary dependencies)

---

## **Expected Outcomes**

### **Success Criteria**
- Return > -3.58% (beat current best)
- Drawdown < 6% (reasonable risk level)
- Win rate > 1% (some positive trades)
- Trade count: 300-600 (reasonable frequency)
- Clear feature importance rankings

### **Decision Matrix**
Once testing is complete, select features based on:
1. **Best overall performance score**
2. **Consistent performance across metrics**
3. **Reasonable trade frequency (not over-trading)**
4. **Feature interpretability and stability**

---

## **Notes & Considerations**

### **Technical Notes**
- Feature cleanup fix ensures only selected features remain (no dependencies leak)
- Target generation temporarily adds dependencies but cleans them up properly
- Session features important for FX but can add noise
- Enhanced features require careful combination with traditional ones

### **Risk Considerations**  
- Small test period (May 20-30) limits statistical significance
- Over-optimization risk with too many tests
- Feature instability across different market regimes
- Transaction costs impact (currently set to 0.0005)

### **Market Context**
- FX market behavior changes across sessions
- Spread compression patterns are regime-dependent  
- Volume patterns crucial in institutional markets
- Depth stability indicates liquidity conditions

---

*Created: 2025-07-29 23:49*
*Context: Feature optimization for spread_compression_patterns_v2 hypothesis*
*Status: Ready for manual testing execution*