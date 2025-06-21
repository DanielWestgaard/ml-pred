# About evaluation/

## Structure 🗂️
```
├── evaluation.py                           # Main orchestrator 
│   
└── hypothesis_specific/                    # Specific evaluation for selected hypotheses
    ├── base_hyp_eval_interface.py          # Just a simple interface for the implementations to follow
    ├── session_based_tod_anomalies.py      # based on the name of the hypothesis?
    └── ...
```

## Usage 🤓
This evaluation "component" work as follows:
- `evaluation.py`: _Main orchestrator_ for evaluating trained model. If you have provided a hypothesis `config.yml`-file, it will use the **hypothesis name** to match the implemented Hypothesis Evaluation file/class.
    - This means that for new hypotheses, you must also create a hypothesis evaluation file. Mostly because evaluation might look different depending on the different hypotheses. If not, I will implement (in the future) a way of running the default evaluation.
- `base_evaluation.py`: Simple interface for guiding creation/implemention of new hypothesis evaluations.