# 🎯 Spread Compression Model Training Plan

## 🏆 **VALIDATED FOUNDATION**
- **8 optimized features** (redundancy eliminated)
- **3.13x movement prediction** on multi-feature combinations (Updated!)
- **749 Tier 2+ quality signals** with 0.000858 average movement (Updated!)
- **427,741 1-minute data points** with feature completeness verified

---

## 📋 **TRAINING PIPELINE ARCHITECTURE**

### **Phase 1: Data Preparation**
```python
# Target: Focus on Tier 2+ signals (quality over quantity)
target_strategy = "tier2_plus"  # 749 events vs 26,639 total signals

# Features: Optimized 8-feature set
features = [
    'compression_ratio_micro',    # Primary compression
    'compression_consistency',    # Strongest predictor  
    'signal_strength_v2',         # Enhanced signal
    'volume_ratio',               # Volume confirmation
    'volume_acceleration',        # Volume momentum  
    'volume_breakout',           # Breakout detection
    'london_session',            # Session timing
    'price_momentum_5'           # Price context
]

# Time range: 2024-10-01 to 2025-07-25 (9+ months of 1-minute data)
training_period = "2024-10-01_2025-07-25"
```

### **Phase 2: Model Architecture**
```python
# Primary model: Random Forest
model_type = "RandomForestClassifier"
parameters = {
    'n_estimators': 200,           # Sufficient trees for stability
    'max_depth': 8,                # Prevent overfitting on rare signals
    'min_samples_split': 10,       # Conservative splitting
    'min_samples_leaf': 5,         # Robust leaf nodes
    'class_weight': 'balanced',    # Handle Tier 2+ imbalance
    'random_state': 42
}

# Alternative models to test:
backup_models = [
    'GradientBoostingClassifier',  # Good for imbalanced data
    'XGBClassifier',               # Fast and effective
    'LogisticRegression'           # Interpretable baseline
]
```

### **Phase 3: Cross-Validation Strategy**
```python
# Time-series aware validation (critical for 1-minute data)
validation_strategy = "TimeSeriesSplit"
cv_parameters = {
    'n_splits': 5,                 # 5-fold time series CV
    'test_size': 0.2,              # 20% for testing each fold
    'gap': 1440,                   # 1-day gap between train/test
    'max_train_size': 100000       # Limit training window
}

# Walk-forward validation for final testing
walk_forward = {
    'train_months': 6,             # 6 months training
    'test_months': 1,              # 1 month testing
    'step_size': 30                # Monthly steps
}
```

---

## 🎯 **TRAINING OBJECTIVES**

### **Primary Goals**
1. **Precision on Tier 2+ signals**: Minimize false positives (transaction costs)
2. **Movement prediction accuracy**: Maintain 3.0x+ movement ratio (Updated baseline!)
3. **Signal frequency**: Optimize Tier 2+ signal detection (749 premium events)
4. **London session optimization**: Peak performance during optimal hours

### **Success Metrics**
```python
evaluation_metrics = {
    'precision_tier2': 0.65,       # 65% precision on Tier 2+ signals
    'recall_tier2': 0.40,          # 40% recall (find quality signals)
    'movement_ratio': 3.0,         # Maintain superior predictive power (Updated!)
    'daily_signals': (15, 20),     # Target signal frequency
    'sharpe_ratio': 1.5            # Risk-adjusted returns
}
```

---

## 🔧 **IMPLEMENTATION STRATEGY**

### **Step 1: Feature Engineering Pipeline**
- ✅ **Load optimized feature set** (already implemented)
- ✅ **Generate Tier 2+ targets** (already implemented)
- ✅ **Feature scaling/normalization** for model stability (already implemented in `data/features/transformation.py`?)
- 🔄 **Time-aware train/test splits**

### **Step 2: Model Training Framework**
```python
# Training pipeline structure
class SpreadCompressionTrainer:
    def __init__(self, features, target_tier="tier2_plus"):
        self.features = features
        self.target_tier = target_tier
        
    def prepare_data(self, df):
        # Focus on Tier 2+ signals
        # Handle class imbalance
        # Create time-aware splits
        
    def train_model(self, X_train, y_train):
        # Random Forest with optimized parameters
        # Cross-validation for hyperparameter tuning
        # Feature importance analysis
        
    def validate_performance(self, X_test, y_test):
        # Precision/recall on Tier 2+ signals
        # Movement prediction validation
        # Signal frequency analysis
        
    def backtest_strategy(self, predictions):
        # Realistic transaction costs (1.3 pips GBPUSD)
        # Signal timing and execution
        # Risk-adjusted returns
```
> [!NOTE]
> This is meant to serve as an "overall, theoretical" structure. We will follow the current structure and pipelines already outlines in this project with configuration files under the hypothesis (`hypothesis/`).

### **Step 3: Hyperparameter Optimization**
```python
# Grid search for optimal parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [3, 5, 8],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Focus on Tier 2+ precision and movement prediction
scoring = 'precision'  # Custom scoring for Tier 2+ signals
```

---

## 📊 **EXPECTED OUTCOMES**

### **Model Performance Targets**
- **Tier 2+ Precision**: 60-70% (minimize false signals)
- **Movement Prediction**: 3.0x+ ratio maintained (Updated baseline!)
- **Signal Quality**: Focus on 749 Tier 2+ premium events
- **Feature Importance**: compression_consistency and volume_ratio top features

### **Trading Strategy Results**
- **Win Rate**: 55-65% on Tier 2+ signals
- **Risk/Reward**: 2:1 minimum (stop loss vs take profit)
- **Sharpe Ratio**: 1.5+ (risk-adjusted returns)
- **Maximum Drawdown**: <15%

---

## 🚀 **NEXT STEPS TO IMPLEMENT**

### **Immediate (Phase 1)**
1. **Create model training script** (with optimal 8-feature set) following the model training structure (`model/`)
2. **Implement time-series cross-validation** for 1-minute data
3. **Set up Tier 2+ target generation** and class balancing
4. **Build feature scaling pipeline** for model stability

### **Medium Term (Phase 2)**
1. **Train Random Forest models** with hyperparameter optimization
2. **Validate on walk-forward basis** with realistic constraints
3. **Implement backtesting framework** with transaction costs
4. **Compare multiple model architectures**

### **Long Term (Phase 3)**
1. **Live trading integration** with real-time feature generation
2. **Model retraining pipeline** for concept drift handling
3. ✅ **Risk management integration** with position sizing (reuse/alredy implemented in `strategy/` and `strategy/components`)
4. ✅ **Performance monitoring** and signal quality tracking (reuse/alredy implemented in `strategy/` and `strategy/components`)

---

## 💡 **KEY INSIGHTS FOR SUCCESS**

1. **Quality over Quantity**: Focus on Tier 2+ signals (749 events) rather than all signals
2. **Time-Series Awareness**: Use proper CV for 1-minute temporal data
3. **Feature Optimization**: 8 core features eliminate redundancy while maintaining power
4. **Realistic Testing**: Include transaction costs and execution delays
5. **London Session Focus**: Optimize for peak trading hours
6. **Movement Prediction**: Maintain 3.0x+ ratio for highly profitable trading (Updated!)

**Your spread compression strategy is validated and ready for model training! 🎯**