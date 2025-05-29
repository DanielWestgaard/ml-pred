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
  feature_name_1: [5, 20, 50]                                # Only these periods of that feature
  feature_name_2: []                                         # All "features" from this feature method
  feature_name_3: ['feature_value_1', 'feature_value_2']     # Only these features from the feature method
```
> [!NOTE]
> It's important to remember that the ```feature_name``` matches the way it's used in the Feature Generator. Eg. ```moving_averages```, ```relative_strength_index```. Also to note that the possibility to select the granular ```feature_values``` is something I create 'per need basis'. So if I haven't used a feature with feature_values before, its not been implemented. Either you must implement this yourself (into the existing method - should be relatively easy utilizing the check_feature_specification() in data_utils.py), or just create all the feature_values of that feature (_name).
### Example
```yml
name: 1_session_based_tod_anomalies

required_features:
  moving_averages: [5, 20, 50]
  time_of_day: []
  volatility_features: ['atr', 'volatility_regime']
```
