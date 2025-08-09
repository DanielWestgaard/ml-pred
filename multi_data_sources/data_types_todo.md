# List of Data Types to start implementing
> Historical price data/volume/spreads are already implemented through `broker/` (Capital.com)

This document is meant to serve as a "light plan" for what data types/sources to implement and why.

---


# Tier 1: Start Here (Highest Priority)

## 1. Calendar/Event Data + Basic Price/Volume
**Why institutions struggle**:
- Too many small, dispersed opportunities across thousands of stocks
- Infrastructure costs exceed profit potential for individual patterns
- Cross-calendar complexity (earnings + options expiration + holidays + fiscal year-ends)

**My advantages**:
- Cheap/free data sources available
- Can focus on tiny patterns across small universes
- Quick to implement and test

**Specific focus areas**:
- Earnings announcement timing interactions (Friday vs Tuesday announcements)
- Option expiration proximity effects on small caps
- Holiday/vacation period patterns in specific sectors
- Fiscal quarter-end effects combined with company size

## 2. News/Sentiment + Small-Cap Focus
**Why institutions struggle**:
- Small caps don't justify dedicated sentiment analysis resources
- Information processing lag between large and small caps
- Sentiment tools optimized for large-cap coverage

**Your advantages**:
- Can use free/cheap sentiment APIs (NewsAPI, social media APIs)
- Focus on <$2B market cap where institutions have less coverage
- Rapid implementation of new sentiment sources

**Specific focus areas**:
- Large-cap earnings impact on similar small caps (2-3 day lag)
- Sector sentiment spillovers to neglected stocks
- Social media attention vs. institutional coverage gaps


# Tier 2: Add After Tier 1 is Working
## 3. Earnings/Fundamental Data + Cross-Effects
**Why institutions struggle**:
- They focus on direct earnings plays, miss interaction effects
- Compliance restrictions on rapid strategy changes
- Too much data for humans, too complex for simple algorithms

**Your advantages**:
- Can quickly test weird combinations
- Focus on secondary and tertiary effects
- Rapid hypothesis iteration

**Specific focus areas**:
- Earnings surprise spillovers to suppliers/customers
- Guidance change timing vs. peer earnings calendar
- Analyst revision clustering effects

## 4. Options Flow + Equity Intersections
**Why institutions struggle**:
- Separate options and equity trading desks
- Focus on large-scale flow, miss small anomalies
- Regulatory separation between market making and proprietary trading

**Your advantages**:
- Can combine equity and options signals seamlessly
- Focus on small patterns institutions ignore
- Retail options flow data becoming more accessible

**Specific focus areas**:
- Unusual options activity in small caps before earnings
- Put/call ratio anomalies in specific sectors
- Volatility surface inefficiencies around events

# Tier 3: Advanced (Add Later)
## 5. Economic/Alternative Data
- Weather data for specific sectors
- Economic indicators + sector rotation
- Supply chain disruption signals

## 6. Social Media/Retail Flow Data
- Reddit sentiment
- Robinhood/retail trading flows
- Google Trends


---


# Implementation Strategy
## Phase 1: Calendar + Price Foundation
```py
data_priority = {
    'essential': ['daily_prices', 'volume', 'earnings_calendar', 'option_expiration'],
    'sources': ['yfinance', 'alpha_vantage', 'earnings_calendar_api'],
    'cost': '<$100/month',
    'complexity': 'low'
}
```

## Phase 2: Add Sentiment
```python
data_expansion = {
    'add': ['news_sentiment', 'social_media_mentions'],
    'sources': ['newsapi', 'twitter_api', 'reddit_api'],
    'cost': '+$200/month',
    'complexity': 'medium'
}
```

## Phase 3: Sophisticated Combinations
```python
advanced_data = {
    'add': ['options_flow', 'insider_trading', 'analyst_revisions'],
    'sources': ['paid_data_vendors'],
    'cost': '+$500/month',
    'complexity': 'high'
}
```

---

## Why This Sequence Works
### 1. Calendar + Price Data:
- Research showed numerous unexploited calendar anomalies
- Institutions know about them but can't profitably exploit small patterns
- My system can find the "micro" versions they miss

### 2. Sentiment + Small Caps:
- Clear institutional blind spot identified in research
- Information flow delays documented in academic literature
- Retail can move faster than institutional sentiment analysis

### 3. Build Complexity Gradually:
- Prove your system works on simple data first
- Each layer adds signal without losing the foundation
- Avoid the trap of over-complicating early on

---

## Key Success Factors:
- Start simple, add complexity incrementally
- Focus on combinations institutions struggle with organizationally
- Prioritize data types where you have structural advantages
- Test quickly, fail fast, iterate rapidly
