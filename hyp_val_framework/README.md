# Overview
Read README.md for a better understanding of this repo.<br>
Seeing as this repo has as a main goal of testing out (and potentially implementing) many different hypotheses, everything (especially core components) must be dynamic and easily reusable.<br>
A "general hypothesis validation framework" would be the very first step of a hypothesis to every single new hypothesis. Instead of manually writing individual test scripts, and potentially implement a hypothesis into this project (features, processing, training, backtesting), I need a generalized testing script/framework that's used on new hypotheses to see if they actually hold any predictive power on future price movements.

## Quick thoughts 
> Just to get started, even though some hypotheses might need some preparation work with features and whatnot (which could be classes/files that implements/inherits from main one), there are still a few points that that all hypotheses have in common.
- Needs to be an actual **inefficiency**
- **Repeatable** occurrences
- **Predict future price movements**, no matter how long, short, timeframe, symbol/ticker, etc.
- **Simple**: Don't overcomplicate it with many features. A hypothesis should be "lightweight"
- **Testable**: Should be able to test and verify it
- **Concrete** "hypothesis statement"

### Suggestions
- Even though we don't engineer the data, we could still reuse some of the components from `data/`. Eg. [Cleaning](data/processing/cleaning.py) and [Validation](data/processing/validation.py)