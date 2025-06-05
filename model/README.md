# About model/ 🤖
This folder focuses on training ML Models. <br>

## Structure
```
├── model_training.py           # Main file to train different models
├── model_evaluation.py         # Evaluate trained model
│
├── model_factory/              # Different ML Models that main file can use to train
│   ├── base_model.py           # Interface for all model type implementations
│   └── xgboost.py              # XGBoost Classifier
│
└── ...
```

# Approach
