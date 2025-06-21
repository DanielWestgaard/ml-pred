# About evaluation/

## Structure
```
├── evaluation.py                           # General evaluation, most suited for closing price, volume, etc.
│   
├── hypothesis_targeted_evaluation/         # Specially focused on hypotheses? evaøuation.py will use one of these based on config.yml?
│   ├── base_hyp_eval_interface.py          # Just a simple interface for the implementations to follow
│   ├── session_based_tod_anomalies.py      # based on the name of the hypothesis?
│   └── ...
```