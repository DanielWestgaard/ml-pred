# Quick Intro
Read `README.md` for and overview of the project, and `hypothesis/spread_compression_patterns/README.md` for information about the hypothesis I'm trying to make work.

I've really been struggling to make my model profitable (backtesting showed consistently "0% win rate and always negative returns, with extremly low confidence on BUY/SELL"). 

This files predecessor is `fix_model_direction_pred_issues.md`

#  🎯 SYSTEMATIC 0% WIN RATE INVESTIGATION PLAN

The data corruption is fixed, but 0% win rate suggests fundamental model-target misalignment. Here's our systematic approach:

## Phase 1: Target Label Analysis 🔍

1. Examine target generation logic - How are target_spread_compression_v3_long_score labels created?
2. Check label distribution - Are we getting balanced BUY/SELL/HOLD signals?
3. Validate forward-looking accuracy - Do target labels actually predict future price movements?

## Phase 2: Feature-Target Relationship 📊

4. Correlation analysis - Which features actually correlate with successful trades?
5. Baseline model test - Train simple model on just volume_ratio (strongest feature)
6. Feature predictive power - Test individual features against actual price outcomes

## Phase 3: Model Behavior Diagnosis 🤖

7. Prediction confidence analysis - Are predictions too confident/conservative?
8. Signal threshold optimization - Is 0.25 confidence threshold appropriate?
9. Temporal validation - Do predictions work on different time periods?

**Priority**: Start with target label analysis since that's most likely to reveal the root cause.
