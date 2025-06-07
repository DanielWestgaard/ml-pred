# About model/ 🤖
This folder focuses on training ML Models. <br>

## Structure
```
├── model_training.py           # Main file to train different models
├── evaluation.py               # TODO: Evaluate trained model
│
├── model_factory/              # Different ML Models that main file can use to train
│   ├── base_model.py           # Interface for all model type implementations
│   ├── xgboost.py              # XGBoost Regressor
│   └── ...
│
└── ...
```


# Approach
Main file is `model_training.py`. Use this to train a model on your processed data.<br>
It is created to "dynamically" run the implementation of your `model_type` parameter. Eg. `xgboost` will use the XGBRegressor class to train your model.<br>
> [!NOTE] 
> This is done in `ModelTraining.run()`, by listing python-files under `model/model_factory/` (`_get_implemented_models()`) and tries to import the class like we would normally do (`model_module = importlib.import_module(module_path)`).

> [!IMPORTANT]
> It's important that the `model_type` matches the name of the file. Eg. `xgboost` will match `xgboost.py`.

## How to use
```python
# Model instance
model_training = ModelTraining(dataset=args.model_training, model_type='xgboost')
# Actually train a model
model_training.run()
# Evaluate model - currently only supporting regression models
model_training.evaluate_regression_model()
```

## Further development
As mentioned, `model_training.py` dynamically runs and imports model-file based on `model_type`.<br>
To add a new implementation of a new Model Type:
1. Name the file of the Model Type (you will be using this to call, and match, it later).
2. Use the `BaseModel` Interface as a Blueprint for the needed classes. Eg:
    ```python
    # Import
    from model.model_factory.base_model import BaseModel
    # Implement Interface
    class XgboostModel(BaseModel):
    ```
3. Pretty much good to go! `ModelTraining.run()` will split training and testing data, so make sure all model implementations have the same required parameters (as they will all be initiated, dynamically, the same): `model_instance = model_class(dataset=self.df, X_train=self.X_train, y_train=self.y_train)`
> [!NOTE]
> I plan on later enchanding the evaluation method into its own class, as evaluating different models (Classification v. Regression) is quite different.


## Training - model_training.py
...


## Evaluation
Simple class having different evaluation implementations. One 'main' `run()` to orcestrate and run all implementations, but also possible to run each individual implementation.
- `plot_model_prediction()`: Plot (trained) data with model predictions for 'manual inspection'.

### How to use
1. Evaluate **right after training** a model:
In main.py, first train the model, and then evaluate:
```bash
python main.py -ml 'storage/data/capital_com/processed/processed_GBPUSD_m_20250101_20250523.csv' -ev
```
2. Evaluate an 'old'/**stored model**:
...