# About evaluation/

## Structure 🗂️
```
├── evaluation.py                           # Main orchestrator
├── evaluation_factory.py                   # Creating the evaluator based on hypothesis name
│   
└── hypothesis_specific/                    # Specific evaluation for selected hypotheses
    ├── base_evaluation.py                  # Just a simple interface for the implementations to follow
    ├── session_based_tod_anomalies.py      # based on the name of the hypothesis?
    ├── price_predicion_evaluation.py       # TODO...
    └── ...
```

## Usage 🤓
This evaluation "component" work as follows:
- `evaluation.py`: _Main orchestrator_ for evaluating trained model. If you have provided a hypothesis `config.yml`-file, it will use the **hypothesis name** to match the implemented Hypothesis Evaluation file/class.
    - This means that for new hypotheses, you must also create a hypothesis evaluation file. Mostly because evaluation might look different depending on the different hypotheses. If not, I will implement (in the future) a way of running the default evaluation.
- `evaluation_factory.py`: Super simple way of semi-dynamically running the selected evaluation type based on the name in the hypothesis config-file, matching with the `EVALUATORS` list (that maps the name to the class/file).
- `base_evaluation.py`: Simple interface for guiding creation/implemention of new hypothesis evaluations.

> [!NOTE]
> It's important to note that when your creating a new hypothesis, you will (also) need to create its equivalent evaluation file under `evaluation/hypothesis_specific`, and that the name of "this file" is mapped to the excact same way you "spell" the hypothesis_name in the `EVALUATORS`.