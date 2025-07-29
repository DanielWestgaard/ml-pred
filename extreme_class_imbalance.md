# 🎯 ROOT CAUSE FOUND: Extreme Class Imbalance

### Target Distribution:
- Class 0 (No signal): 1,231,402 (93.9%)
- Class 1 (Basic signal): 78,392 (6.0%)
- Class 2+ (Strong signals): 2,065 (0.15%)

### Binary Target:
- Negative: 1,231,402 (93.9%)
- Positive: 80,457 (6.1%)

**This is severe class imbalance! The model learns to predict "no signal" almost always, leading to very low confidence on positive predictions.**

## 💡 Solutions:

## ~~Option 1: Adjust Model Training for Imbalanced Data~~

- Use class weights in Random Forest: class_weight='balanced'
- Lower confidence threshold from 0.8 to 0.3-0.5

> [!NOTE]
> 👆 Already in use balanced class weight, and lower threshold only made performance worse. 

## Option 2: Improve Target Quality

- Increase target sensitivity to generate more positive examples
- Focus on higher-tier targets only (remove weak Tier 1 signals)

### More detailed plan
#### Phase 1: Tier 2+ Focus System
1. Eliminate Tier 1 signals - They're adding noise without prediction value
2. Enhance Tier 2+ detection - Use more sensitive thresholds for quality signals
3. Create gradient targeting - Smooth transition from "no signal" to "strong signal"

#### Phase 2: Feature-Driven Enhancement
4. Leverage top predictors - depth_stability (0.399 correlation) and institutional_footprint as primary filters
5. Multi-timeframe validation - Ensure signals work across different time windows
6. Volume confirmation - Strengthen volume-based signal validation



## Option 3: Rebalance Dataset

- Undersample negative class or oversample positive class during training

---

The 0% win rate isn't due to data leakage anymore - it's due to the **model being too conservative due to extreme class imbalance**. Even with perfect features, the model refuses to predict positive signals.

Status: Option 1 has been tested and made things worse. On to Option 2.
