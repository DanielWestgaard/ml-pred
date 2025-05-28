# About hypothesis/
This folder is meant to document testing of different hypothesis'. Meaning trying to find patterns, anomalies, inefficiencies, available/repeatable liquidy, etc. in the market.<br>
The purpose of this is that beating the market is _near_ impossible (considering all factors, challenges, and actors), but maybe there are higher changes of success finding smaller oppurtunities in inefficiencies.

## Structure
```
├── 1_session_based_tod_anomalies           # Name of the hypothesis
│   ├── config.yml                          # Spesific configuration in terms of needed features, needed ML models (potentially their parameters/configurations again), etc.
│   └── README.md                           # More through description of the hypothesis
└── README.md                               # You are here - description about this folder
```

## config.yml structure
This should serve as an example template to how features are selected to new hypothesis'.
```yml
name: <name-of-the-hypothesis>

required_features:
  moving_averages: [5, 20, 50]  # Only these periods
  time_of_day: []  # All "features" from the function
  volatility_features: ['atr', 'volatility_regime'] # Only these features from the function
```