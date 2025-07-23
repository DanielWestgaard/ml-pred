# About model/ 🤖
This folder focuses on training ML Models with **automatic GPU acceleration**. <br>

## Structure
```
├── model_training.py           # Main file to train different models
├── evaluation.py               # TODO: Evaluate trained model
│
├── model_factory/              # Different ML Models with GPU support
│   ├── base_model.py           # Interface for all model type implementations
│   ├── xgboost.py              # XGBoost Regressor (GPU accelerated)
│   ├── pytorch.py              # PyTorch Neural Networks (CUDA/Metal)
│   ├── tensorflow.py           # TensorFlow Models (CUDA/Metal)
│   └── ...
│
└── ...
```

## 🚀 GPU Acceleration Support
All models now include **automatic GPU acceleration** with intelligent fallback:

- **🖥️ Local Mac**: Apple Metal GPU support via MPS (PyTorch) and Metal (TensorFlow)
- **☁️ Cloud NVIDIA**: Full CUDA acceleration (XGBoost gpu_hist, PyTorch CUDA, TensorFlow GPU)
- **🔄 Auto-Fallback**: Graceful CPU fallback if GPU unavailable
- **📊 Multi-Framework**: XGBoost, PyTorch, and TensorFlow support


## Training - model_training.py
The ModelTraining class is the 'main' orchestrator. Use this to train a model on your processed data.<br>
It is created to "dynamically" run the implementation of your provided Model Type. Eg. `xgboost` will use the XGBRegressor class to train your model.<br>
> [!NOTE] 
> This is done in `ModelTraining.run()`, by listing python-files under `model/model_factory/` (`_get_implemented_models()`) and tries to import the class like we would normally do (`model_module = importlib.import_module(module_path)`).

> [!IMPORTANT]
> It's important that the `model_type` matches the name of the file. Eg. `xgboost` will match `xgboost.py`.


### Storage 💽
The trained model will, as for all kind of storage in this project, be stored under `storage/`.<br>
Currently, this is the organization for storage:
```
storage/
├── models/                     # Trained models, metadata, evaluation
│   │
│   ├── <datetimenow>/          # Eg. '2025-06-09_20:19:29/'
│   │   ├── trained_model.pkl   # Actual trained model. Eg. 'xgboost.pkl'
│   │   ├── model_metadata.yml  # Information (metadata) about model training, location, parameters, etc.
│   │   └── evaluation/         # Evaluation of model and plots of its predictions
│   │       ├── plot.pdf        # Currently, plot of test data and model prediction (done in Evaluation)
│   │       └── evaluation.pdf  # Todo: 
│   │
│   └── ...
│
└── ...
```
The **structure for metadata.yml** gets auto-generated after successfully training the model (might be changed to .json or split up later). It's done in `model_training.py` in `_save_model_info()`, and these are currently the implemented sections: ['training:', 'dataset:', 'model_parameters:', 'system_info:', 'data_split:']


### Hypothesis 'flow'
Following the projects goal of having a hypothesis to configure what processes to run, Model Training also supports this.<br>
Using the configuration structure outlined in the `hypothesis/README.md`, the Model Trainers `__init__()` will...
1. Check to see if you have a **provided path to the hypothesis config-file** using the **`hypothesis_source` parameter**.
2. Find the `model_type` (in/from configuration).
3. Then, if provided, path to `processed_data`.
    - However, for the usecase of "running Data Pipeline first", you don't necessarily know the path to the processed data. Meaning if `processed_data` fails or returns `None` (from config), I will use `raw_source` to find the path to its equivalent processed path. The current approach for this is very simple: Change directory `raw/` to `processed/` and first prefix of `raw_*` to `processed_*`.

> [!NOTE] 
> Because of this "Hypothesis approach/flow", I've made the arguments `dataset` and `model_type` "optional". However, they must be provided/set for the Model Trainer to work. So, if `hypothesis_source` is not provided and either `dataset` or `model_type` (or both) is not provided, an error will be raised and the execution will be stopped (`sys.exit()`).


### How to use 🤓
To train a Machine Learning Model, you can choose how you want to use it:
1. Configure **hypothesis** `config.yml`, following the structure outlined in `hypothesis/README.md`, and either 
    - Run `main.py` with:
        ```bash
        python main.py -hyp 'hypothesis/1_session_based_tod_anomalies/config.yml'
        ```
    - Or implement it yourself:
        ```python
        from model.model_training import ModelTraining

        model_training = ModelTraining(hypothesis_source = args.hypothesis)
        model = model_training.run()

        # Optional: Store the model directory for potential evaluation
        if hasattr(model_training, 'model_directory'):
            trained_model_dir = model_training.model_directory
        else:
            # Get the latest model directory that was just created
            trained_model_dir = get_latest_model(models_directory = config.TRAINED_MODEL_STORAGE,
                                            model_type = conf.get('model_type'))
        ```

2. Use `main.py` and run it with the `--model-training`/`-ml` flag, followed by the path to the processed dataset you intend to train the model on. Eg:
    ```bash
    python main.py -ml 'storage/data/capital_com/processed/processed_GBPUSD_m_20250101_20250523.csv'
    ```
    Optionally, to specify which model type you want to train (and not use defalut, 'xgboost'), you can also provide the `--model-type`/`-mt` flag and model type string:
    ```bash
    python main.py -ml 'storage/data/capital_com/processed/processed_GBPUSD_m_20250101_20250523.csv' --mt 'xgboost'
    ```
> [!IMPORTANT]
> It's important that the `model_type`/`-mt` matches the name of the file. Eg. `xgboost` will match `xgboost.py`.

3. Implement/use it yourself like this:
    ```python
    # Model instance
    model_training = ModelTraining(dataset=args.model_training, model_type='xgboost')  # 'model_type' must match name of implemented model file
    # Actually train a model
    model_training.run()
    # Evaluate model - currently only supporting regression models
    model_training.evaluate_regression_model()
    ```

### Further development - Implementing new model types
As mentioned, `model_training.py` dynamically runs and imports model-file based on `model_type`.<br>
To add a new implementation of a new Model Type **with GPU support**:

1. **Name the file** of the Model Type (you will be using this to call, and match it, later). Eg. 'xgboost.py' will match 'xgboost'

2. **Use the `BaseModel` Interface** as a Blueprint for the needed classes:
    ```python
    # Import
    from model.model_factory.base_model import BaseModel
    from utils.gpu_utils import get_gpu_config, log_gpu_status
    
    # Implement Interface
    class YourModel(BaseModel):
        def train(self):
            # Get GPU configuration
            gpu_config = get_gpu_config('your_framework')
            log_gpu_status()
            
            # Your GPU-accelerated training code here
            pass
    ```

3. **GPU Configuration**: Use `get_gpu_config(framework_name)` to get optimal settings for your framework
   - Supported frameworks: `'xgboost'`, `'pytorch'`, `'tensorflow'`
   - Returns device type, specific parameters, and fallback options

4. **Model Initialization**: Ensure same required parameters: `model_instance = model_class(dataset=self.df, X_train=self.X_train, y_train=self.y_train)`

### 🎯 Available Model Types

| Model Type | Framework | GPU Support | Use Case |
|------------|-----------|-------------|----------|
| `xgboost` | XGBoost | ✅ CUDA/CPU | Tree-based models, tabular data |
| `pytorch` | PyTorch | ✅ CUDA/Metal/CPU | Neural networks, deep learning |
| `tensorflow` | TensorFlow | ✅ CUDA/Metal/CPU | Neural networks, production ML |

### 🔧 GPU Configuration Examples

```python
# XGBoost GPU config (automatically applied)
{
    'device_type': 'cuda',
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor'
}

# PyTorch GPU config
{
    'device_type': 'cuda',
    'device': 'cuda',
    'device_count': 1
}

# TensorFlow GPU config  
{
    'device_type': 'cuda',
    'device': '/GPU:0',
    'memory_growth': True
}
```


## Evaluation
Simple class having different evaluation implementations. One 'main' `run()` to orcestrate and run all implementations, but also possible to run each individual implementation.
- `evaluate_regression_model()`: Evaluate a regression model.
- `plot_model_prediction()`: Plot (test) data with model predictions for _manual inspection_.

### How to use
#### Evaluate an _old_/**pre-trained**/**'stored' model**:
When a model is, successfully trained, the Model Trainer will save it with metadata. The Evaluator will use the metadata file to find the used dataset and model type.<br>
Using `main.py` provide the `-evo` flag followed by the path to the models `metadata.yml`-file.
```bash
python main.py -evo 'storage/models/2025-06-09_20:19:29/model_metadata.yml'
```
> [!NOTE]
> The `--evaluate-old`/`-evo` flag requires the path to `<SAVED INFORMATION ABOUT TRAINED MODEL>` info about the trained model.

#### Evaluate **right after training** a model (_fresh_), though be aware of possible race condition:
In main.py, first train the model, and then evaluate:
```bash
python main.py -ml 'storage/data/capital_com/processed/processed_GBPUSD_m_20250101_20250523.csv' -evf
```
> [!NOTE]
> The `--evaluate-fresh`/`-evf` flag stores true, so nothing needed after. This is meant to be used after training a model (after the `-ml` flag).

#### Using it with Hypothesis:
You can use the Hypothesis configuration file to run Evaluation with both of these usecases (though one at a time).<br>
Evaluation naturally follows the exact same flow as other components supporting Hypothesis Flow: Configure evaluation in the `config.yml`-file, following syntax provided in `hypothesis/README.md`.<br>
Although you should follow the documentation and syntax under `hypothesis/README.md`, this is roughly it:
```yml
# Can have Data Pipeline- and/or Model Training config before if needed..

evaluate_model: 
    type: <stored-or-new>                             # Optional: Keeping it simple: 'stored' means already trained model, 'new' means you just ran model training 'in this run'
    model_folder: <path-to-processed-data>            # Only provide if you are evaluating a 'stored' model. Path to model folder
```
When ready, run `python main.py -hyp 'hypothesis/1_session_based_tod_anomalies/config.yml'`