# About hypothesis/
This folder is meant to document testing of different hypotheses. Meaning trying to find patterns, anomalies, **inefficiencies**, available/repeatable liquidy, etc. in the market.<br>
The purpose of this is that beating the market is _near_ impossible (considering all factors, challenges, and actors), but maybe there are higher changes of success by finding smaller oppurtunities in inefficiencies.


## Structure
```
├── 1_session_based_tod_anomalies           # Name of the hypothesis
│   ├── config.yml                          # Spesific configuration in terms of needed features, needed ML models (potentially their parameters/configurations again), etc.
│   └── README.md                           # More through description of the hypothesis
└── README.md                               # You are here - description about this folder
```


# Adding a new hypothesis 
Each new (and all existing) hypotheses should follow _this_ structure.<br>
1. Create a new folder named after your hypothesis.
2. Create a `config.yml` file for configuring the hypothesis and other potential parameters, following the given format.
3. Optionally, though recommended, a `README.md` describing the hypothesis in more detail and with better formulations.
4. Add a method in the (`data/features/`)`target_hypothesis.py` that calculates the target column, representing the hypothesis (that a model will predict).

## config.yml structure - WIP
These are the different componentes/sections you can specify in the configuration-file. None of these are required (I think), but could be thought of as flags when running `main.py`, to tell the main orchestrator what components to run, and potentially specify how (features, parameters, etc.).
- **Data Pipeline**: 
- **Model Training**: 

This should serve as an example template to how features are selected to new hypotheses.
```yml
name: <name-of-the-hypothesis>

data_pipeline:
  raw_source: <source-to-raw-data>
  
  required_features:
    feature_name_1: [5, 20, 50]                                # Only these periods of that feature
    feature_name_2: []                                         # All "features" from this feature method
    feature_name_3: ['feature_value_1', 'feature_value_2']     # Only these features from the feature method

# TODO: implement functionality
model_training:
  name: <name-of-ml-model>

  parameters:                                                # Completely or somewhat empty means using 'default' hyperparameter values
    - n_estimators: 100
    - max_depth: 12
```
> [!IMPORTANT]
> The ```name``` of the hypothesis _must match_ the _function name_ in the ```data/hypothesis/target_hypothesis.py```. I plan to make it so you can specify the function-name dynamically in the Data Pipeline, but not yet.<br>
> ```feature_name``` must also _match_ it's _function name_ in the Feature Generator. Eg. ```moving_averages```, ```relative_strength_index```. 

> [!NOTE]
> The possibility to select the granular ```feature_values``` (of a ```feature_name```) is something I create 'per need basis'. So if I haven't used a feature with feature_values in a hypothesis before, it has not been implemented. Either you must implement this yourself (into the existing method - should be relatively easy utilizing the ```utils/data_utils/check_feature_specification()``` and some if-statements before each calculation), or just create all the ```feature_values``` of that ```feature``` (-_name).


### Example
```yml
name: session_based_tod_anomalies

required_features:
  moving_averages: [5, 20, 50]
  time_of_day: []
  volatility_features: ['atr', 'volatility_regime']
```
