# ml-pred
The goal of this repository, is to create a complete prediction and trading system that uses machine learning to predict intraday financial market data. I will also have a complete data pipeline, backtesting system of model, and live data functionality.<br>
It's important to note that this will probably not be the very best system, as I will also use this as a place to learn and take the time I need to really create this system myself.

## Project outline (planned)
```
├── main.py                 # Main orchestrator for running all other "systems"
├── data/                   # Data engineering
├── model/                  # Model training
├── backtesting/            # Backtest trained models
├── live/                   # Deploy trained model on live market data
├── storage/                # General storage for data, models, results, etc.
├── broker/                 # Different broker integrations
├── secrets/                # Folder using secrets.txt to store sensitive vars (not saved in repo)
├── config/                 # Configuration files
├── utils/                  # Utility files
├── images/                 # Storing images and sources used in documentation
├── trash/                  # Deprecated files and code no longer needed/in use
├── README.md               # General project description
└── secrets/                # Sensitive content (must be created individually)
    └── secrets.txt         # Main secret file
```

## Setup repo
If this is the first time using the repo, follow these instructions:
1. Clone the environment (eg.): ``` git clone git@github.com:DanielWestgaard/ml-pred.git ```
2. Create a virtual environment (in root of project): ``` python3.11 -m venv venv ```
    - Recommended Python version: 3.11
3. Install dependencies: ``` pip install -r requirements.txt ```
4. Create secrets path: ``` mkdir secrets; cd secrets; touch secrets.txt; cd .. ```. Then, fill in the following (based on what you will use in the repo):
    - **For using Capital.com's [API](https://open-api.capital.com/)**: 
        1. Make sure you're following these [getting started steps](https://open-api.capital.com/#section/Getting-started).
        2. Paste in these variables in the secrets-file:
        ```
        API_KEY_CAP=
        PASSWORD_CAP=
        EMAIL=
        ```
        3. Fill in the values with from your own properties.
    - **For using [Alpha Vantage API](https://www.alphavantage.co/documentation/)**:
        1. Create your own [API token](https://www.alphavantage.co/support/#api-key).
        2. Paste in this variable in the secrets-file: ```alpha_vantage_free_api_key=```.
        3. Fill in your own API token for its value.
### Recommended, but not needed
- If you're using VSCode, I highly recommend installing the extensions
    - [Rainbow CSV](https://marketplace.visualstudio.com/items/?itemName=mechatroner.rainbow-csv): Prettier view of .csv-files.
    - [Material Icon Theme](https://marketplace.visualstudio.com/items/?itemName=PKief.material-icon-theme): Helps massively when you have many folders (icon some names).

## Nice to know
Trying to follow best practice for python, I have...
- Added readme's to each (important) folder to more granular descriptions and information about the different systems, and also trying to update them with a ```Structure``` paragraph for easier overview short descriptions.
- Using methods with prefix ```_``` before the method name to indicate a "private" method (/helper function). Eg. ```_plot_fs_analysis_cfs()```