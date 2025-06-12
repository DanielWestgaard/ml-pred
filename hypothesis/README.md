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

data_pipeline:  # TODO: Now have to change logic for reading required features and possibility for providing data source
  raw_source: <source-to-raw-data>
  
  required_features:
    feature_name_1: [5, 20, 50]                                # Only these periods of that feature
    feature_name_2: []                                         # All "features" from this feature method
    feature_name_3: ['feature_value_1', 'feature_value_2']     # Only these features from the feature method

# TODO: implement all of this functionality
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

## How to use
You can use the "hypothesis flow" in two, main, different ways:
1. The preferred way: Configure `config.yml` with all the components you need/want to run. Eg. If you want the hypothesis to be used in (only) the Data Pipeline, then you "configure" that in there. If you want to run only Model Training on a processed dataset (or first Data Pipeline and then MT), you configure that in the `config.yml`.
    - And then you can use `main.py` with the -hyp flag (might change flag name in the future) to run "all the components in the configuration file" (as long as you've followed the structure for the yml.):
        ```bash
        python main.py -hyp 'hypothesis/1_session_based_tod_anomalies/config.yml'
        ```
    - Or implement it yourself, as the different components should be able to (in the future) to dynamically read the contents of the configuration file. Eg.:
        ```python
        import utils.shared_utils as shared utils
        from data.pipelines.engineer_pipeline import EngineeringPipeline

        # Getting all the configurations
        conf = shared_utils.handle_hypothesis_source(args.hypothesis)
        conf_keys = conf.keys()
        
        if 'raw_data_source' in conf_keys:  # Found raw data path config, meaning user intends to run Data Pipeline?
            data_pipeline = EngineeringPipeline(save_cleaned_data=save_cleaned_data, hypothesis_source=args.hypothesis)
            data_pipeline.run()
        
        # TODO: Same logic for Model Training
        # ...
        ```

> [!NOTE]
> The whole point is for it to be a centralized way to configure what you want to do, and not remember all the different flags you can run with `main.py`.

2. Or using flags (however, I can imagine this might get removed/deprecated later), using the first component you want to use followed by the -hyp flag again (and the path to the config-file):
    - For Data Pipeline:
        ```bash
        python main.py -eg 'storage/data/capital_com/raw/raw_GBPUSD_m5_20240101_20250101.csv' -hyp 'hypothesis/1_session_based_tod_anomalies/config.yml'
        ```