# About evaluation/

## Overview 🎯
This evaluation module is designed to work with the **hypothesis-driven approach** for exploring financial market inefficiencies. Each hypothesis requires its own specialized evaluation methodology since different market theories and strategies need different metrics and analysis approaches.

## Hypothesis-Driven Evaluation Philosophy 🧑‍🚀
Following the project's core philosophy of testing specific market hypotheses:

1. **Hypothesis-Specific Metrics**: Each hypothesis (e.g., session-based volume anomalies, price prediction) requires different evaluation criteria
2. **Specialized Analysis**: Different hypotheses need different types of analysis (time-of-day effects, volatility patterns, anomaly detection)
3. **Scalable Architecture**: Easy to add new hypothesis evaluations without modifying core components

## Structure 🗂️
```
├── evaluation.py                           # Main orchestrator
├── evaluation_factory.py                   # Creating the evaluator based on hypothesis name
│   
└── hypothesis_specific/                    # Specific evaluation for selected hypotheses
    ├── base_evaluation.py                  # Interface for hypothesis-specific evaluations
    ├── session_volume_anomaly_detection.py # Session-based volume anomaly evaluation
    ├── price_prediction_evaluation.py      # Price prediction evaluation
    └── ...                                 # Additional hypothesis evaluations
```

## Usage 🤓
This evaluation component works as follows:

- **`evaluation.py`**: Main orchestrator for evaluating trained models. Uses the **hypothesis name** from `config.yml` to match the corresponding evaluation implementation.
- **`evaluation_factory.py`**: Dynamic evaluation selector based on hypothesis name, mapping to the `EVALUATORS` registry.
- **`base_evaluation.py`**: Abstract interface ensuring consistent evaluation structure across different hypotheses.

### Adding New Hypothesis Evaluations
When creating a new hypothesis:

1. Create a new evaluation file under `evaluation/hypothesis_specific/`
2. Inherit from `BaseEvaluation` class
3. Implement hypothesis-specific evaluation methods
4. Add the mapping to `EVALUATORS` in `evaluation_factory.py`
5. Ensure the hypothesis name matches exactly between config and evaluator registry

## Key Features 🔧
- **Hypothesis-Specific Metrics**: Each evaluation provides metrics relevant to its specific market hypothesis
- **Flexible Architecture**: Easy to extend with new hypothesis evaluations
- **Consistent Interface**: All evaluations follow the same base structure
- **Model Persistence**: Evaluations are saved with hypothesis tags for better organization
- **Comprehensive Analysis**: Includes both general ML metrics and hypothesis-specific insights

> [!NOTE]
> Each hypothesis evaluation is designed to test specific market inefficiencies or opportunities. The evaluation metrics and analysis methods are tailored to validate or invalidate the underlying hypothesis about market behavior.