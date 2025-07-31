# Quick Intro
Read `README.md` for and overview of the project, and `hypothesis/spread_compression_patterns/README.md` for information about the hypothesis I'm trying to make work.

I've really been struggling to make my model profitable (backtesting showed consistently "0% win rate and always negative returns, with extremly low confidence on BUY/SELL"). After many fixes, some of them being two critical issues:
1. Extreme class imbalance: 269.5:1 ratio (HOLD:SELL:BUY)
2. Bug in signal processor: When the model predicts Class 0 (HOLD), the signal processor deliberately sets confidence to 0.0 instead of using the actual model confidence.

We saw results on models improve quite a bit after these. However, there's still some things that needs ironing out. The core and main one now being the **exit strategy**, look at this backtest logfile for example: "logs/backtesting/logging_31072025_194734.log"

## Analysis from the logfile and last fix
✅ HOLD SIGNALS PROPERLY FILTERED:

- Prediction 0: value=0, confidence=0.0000, threshold=0.01
- Signal rejected: confidence 0.0000 < threshold 0.01 ✅
- HOLD signals are now correctly filtered out!

✅ BUY/SELL SIGNALS WORKING:

- Prediction 0: value=2, confidence=0.4362 → SELL signal ✅
- Prediction 0: value=1, confidence=0.3671 → BUY signal ✅
- Prediction 0: value=1, confidence=0.4994 → BUY signal ✅

⚠️ REMAINING ISSUE: Exit Strategy Problem

The trades are still being closed immediately due to confidence_drop because the exit
strategy uses a different confidence calculation than the entry.

The core issues are now FIXED:
1. ✅ Class imbalance: Fixed with percentile-based targets
2. ✅ Signal processor confidence bug: Fixed
3. ✅ HOLD signals creating unwanted trades: Fixed

The remaining 0% win rate is due to the exit strategy configuration - it's closing all
trades immediately due to confidence drops. This is a strategy tuning issue, not a
fundamental model problem.

Your systematic approach worked perfectly! The model is now:
- ✅ Generating proper BUY/SELL signals with real confidence
- ✅ Filtering out HOLD signals correctly
- ✅ Ready for strategy optimization

The 0% win rate is now a strategy parameter tuning issue, not a fundamental model failure!
🏆

