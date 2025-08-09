# Introduction
This folder aims to _create dynamic and modular components_ that are each **responsible for a specific data type**. Wether it being "sentiment analysing SOME (Instagram, Twitter/X, etc.) or news (websites, forums, etc.)", weather reports, new laws/regulations, rumors, etc. At one point, the more and weirder data types I have at my disposal, the better. The reason is that I intend to systematically test combinations of data sources with price data, to find real inefficiencies (in a separate component/system/folder).<br>
**The goal isn't to beat institutions at their own game, but to find the games they're not playing**.


## Background
**Pure price analysis alone is mostly a dead end**, as institutions have been running the most sophisticated pattern recognition algorithms on this for decades. This project aims not to compete directly with them, as that would be a losing battle, but rather use the knowledge to our advantage and exploit small inefficiencies.<br>
Furthermore, **the markets are not made up of pure prices, but all kinds of data** (some/news sentiment, emotions/fear, weather, earnings, politics, law, etc.) which gives investors and system their reasons to buy or sell. This being, my feeling (also based on research) is that the markets cannot be fully efficient, since we live in a day and age where there are almost infinetly ways to get data and sources that can affect prices/markets.<br>
The real alpha is in the **intersections and combinations** that institutions struggle with due to organizational silos, data integartion challenges, and their focus on large-scale opportunities.


## Focus and purpose on "multi-data"
> Because of this Background, I really think that my competetive moat is an agile approach to multi-data.

1. **Data Integration Agility** - I can quickly combine (eg.):
    - Earnings surprises × social media sentiment × calendar timing
    - News sentiment × analyst revisions × option flow
    - Economic calendar × sector rotation × retail trading flows

2. **Weird Combination Testing** - Institutions often have teams that specialize in one data type. I can test bizarre combinations, like:
    - Weather data × retail stock performance × social media mentions
    - Congressional trading disclosures × small-cap momentum × earnings calendar
    - Supply chain disruption news × inventory levels × stock reactions

3. **Speed to New Data Sources** - When new alternative data becomes available, I can integrate and test it much faster than institutions with their compliance and approval processes.

**The key insight**: _Inefficiencies live in the GAPS between data types, not within pure price patterns_.

---

# Structure
```
├── README.md                       # This documentation
├── data_types_todo.md              # Light plan for sources to implement and why
│
├── ds_orchestrator.py              # An "orchestrator" to easily use singular or multiple data source components
│
└── sources/                        # All different data source implementations
    ├── ds_base.py                  # Base class all individual data source components implements
    │
    ├── calendar_events/            # Calendar/Event Data
    │   ├── README.md               # Detailed info regarding this data source implementation
    │   ├── cross_calendar.py       # main class?
    │   └── complexities/           # Individual "things" that will make up the calendar (eg. earnings, holidays, etc.) that is too "complex" for a single method
    │       ├── earnings.py         # ...
    │       └── options_expiration.py # ...
    │
    ├── .../
    │
    └── ...
```

---

# Plan


## For this component
My initial thought is that each component should have ways to get
- **Historical data**: for processing, training, analysis, testing combinations, etc.
- **Live data**: Almost to monitor features/data and get it when something changes. Similar to connecting to a Broker`s websocket and getting market prices every new candle. Some components might be able to get "new data" every minute (in pace with live market data), but others might not get new data as quickly (like waiting for new tweets or news reports) - as long as this is "reported" once new/fresh data is available.


## Future implementations and work
> Once some of the data-type/components are working, I need to start thinking/working on usage and incorporating this into my existing system.

### Storage
For storing the "historical data" of different data types, this should be done in `storage/` (continuing with a centralized storage). Each data type could follow the same strucure used for historical data, being organized by folders after (eg.) type, timeframe (?), etc.<br>

### Potential usage
1. First using `main.py` to get historical data from components and storing them in storage maybe as "raw", since they will be un-processed?

2. Testing combinations of data sources to **find actual inefficiencies** are the first real step. This could be to, initially, add a new folder/component (in root) that systematically tests a symbol's historical data (eg. GBP/USD 1 minute timeframe) together with combinations of "historical data" from different data sources, as well with statistical analysis. Once it's done, we could get a report/results on what had the most statistical potential or predictive power. This system should also be created dynamically and modular, so its easily reusable for continueing testing. Could take inspiration off of `hyp_val_framework`, that was intended to manually test out individual hypotheses (using pure price values and analysis).

2. **Data processing**: Creating a new data pipeline (eg. `data/pipelines/hybrid_data`) that can improve/expand the existing one (`data/pipelines/engineer_pipeline.py`) by also processing/engineering combinations of these data types/components (maybe even create features off of them again).<br>
The usage for this data pipaline is after a set of combinations/inefficiency are "found", and preparing the data for training a ML model.
    - Build on the current "configuration driven" flow, by adding the ability to specify/configure which data types/sources I want included.
    - I'll also need to create a new data transformation step (`data/features/transformation.py`) to translate the different data formats (eg. sentiment/text, binary, degrees, size, etc.) into a correct/suiting format for data processing. There could also be need for creating individual handling, but this can be considered when getting there.

3. **Training** and **Backtesting**: I think that these two components will continue to be the same, as the new data type combinations will be turned into faetures and data handled appropriately. 

5. **Live**: This might need some work, but not directly in the live part of the `strategy/` component. The "selected/sed data types" in training (as well as any potential features made off them) must be recreated as live data come in. There might be need for another orchestration component that will use these data parts for "live usage". Either the data type actually gets data in pace with new live price candles (in which all "columns" must be processed and delivered together), eg. a lot of tweets are coming in giving something to "report" (being data), or it doesn't (waiting for weather report), which means there might need to be a "basic value" for no updates/no changes. The logic/handling for new events and no events/changes must also be implemented as part of the processing of the data, but its the handling here that matters.
