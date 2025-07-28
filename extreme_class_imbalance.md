 🎯 ROOT CAUSE FOUND: Extreme Class Imbalance

Target Distribution:
- Class 0 (No signal): 1,231,402 (93.9%)
- Class 1 (Basic signal): 78,392 (6.0%)
- Class 2+ (Strong signals): 2,065 (0.15%)

Binary Target:
- Negative: 1,231,402 (93.9%)
- Positive: 80,457 (6.1%)

This is severe class imbalance! The model learns to predict "no signal" almost always, leading to very low confidence on
positive predictions.

💡 Solutions:

Option 1: Adjust Model Training for Imbalanced Data

- Use class weights in Random Forest: class_weight='balanced'
- Lower confidence threshold from 0.8 to 0.3-0.5

Option 2: Improve Target Quality

- Increase target sensitivity to generate more positive examples
- Focus on higher-tier targets only (remove weak Tier 1 signals)

Option 3: Rebalance Dataset

- Undersample negative class or oversample positive class during training

The 0% win rate isn't due to data leakage anymore - it's due to the model being too conservative due to extreme class
imbalance. Even with perfect features, the model refuses to predict positive signals.

Recommendation: Try Option 1 first - adjust the confidence threshold to 0.3 and see if the model starts making
predictions.
