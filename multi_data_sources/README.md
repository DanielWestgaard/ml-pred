# Introduction
This folder aims to _create dynamic and modular components_ that are **responsible for a specific data type**. Wether it being "sentiment analysing SOME (Instagram, Twitter/X, etc.) or news (websites, forums, etc.)", weather reports, new laws/regulations, rumors, etc. At one point, the more and weirder data types I have at my disposal, the better.

## Background
**Pure price analysis alone is mostly a dead end**, as institutions have been running the most sophisticated pattern recognition algorithms on this for decades. This project aims not to compete directly with them, as that would be a losing battle, but rather use the knowledge to our advantage.<br>
Furthermore, **the markets are not made up of pure prices, but all kinds of data** (some/news sentiment, emotions/fear, weather, earnings, politics, law, etc.) which gives investors and system their reason to buy or sell. This being, my feeling (also based on research) is that the markets cannot be fully efficient, since we live in a day and age where there are almost infinetly ways to get data and sources that can affect prices/markets.<br>
The real alpha is in the **intersections and combinations** that institutions struggle with due to organizational silos, data integartion challenges, and their focus on large-scale opportunities.

## Focus and purpose on "multi-data"
> Because of this Background, I really think that my competetive moat is an agile approach to multi-data.
1. **Data Integration Agility** - I can quickly combine:
    - Earnings surprises × social media sentiment × calendar timing
    - News sentiment × analyst revisions × option flow
    - Economic calendar × sector rotation × retail trading flows
2. **Weird Combination Testing** - Institutions often have teams that specialize in one data type. You can test bizarre combinations like:
    - Weather data × retail stock performance × social media mentions
    - Congressional trading disclosures × small-cap momentum × earnings calendar
    - Supply chain disruption news × inventory levels × stock reactions
3. **Speed to New Data Sources** - When new alternative data becomes available, you can integrate and test it much faster than institutions with their compliance and approval processes.

**The key insight**: Inefficiencies live in the GAPS between data types, not within pure price patterns.

# Structure
```
...
```

# Plan
My initial thought is that each component should have ways to both get "historical data" (for processing/training/analysis/combinations) and "live data", being kind of a monitoring features (similar to connecting to a Broker`s websocket and getting market prices every minute).<br>
Later, I'm thinking of creating a new data pipeline (eg. `data/pipelines/hybrid_data`) that can improve/expand the existing one (`data/pipelines/engineer_pipeline.py`) by using combinations of these data types/components, or perhaps first using `main.py` to get historical data from components here (storing them in the existing `storage/`) and then the data pipeline will use that data location.<br>
I'll also need a 