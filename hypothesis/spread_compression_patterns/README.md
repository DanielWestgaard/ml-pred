## Hypothesis description
**Use spread compression to predict possible volatile price action and execute trades before this happens**.<br>

When bid-ask spreads suddenly compress, market makers (brokers) are essentially saying “we expect more volume/activity soon”.
They’re willing to accept lower margins per trade because they anticipate higher turnover. 
This often happens 15-45 minutes before news/volatility events, not after.<br>
In many cases, when spreads suddenly compress, it reflects an expectation of higher volume or activity.<br>

The broker currently being used, Capital.com, provides separate bid- and ask-prices OHLC through their REST API and websocket (dynamic spread).


## **Realistic Targets for liquid pairs (GBP/USD):**

- **Signal Frequency**: 3-6 compression events per day
- **Success Rate Target**: 58-65% (profitable above 52% with 2:1 risk/reward)
- **Average Signal Strength**: Look for signals > 2.0 for best quality
- **Optimal Trading Hours**: London session (8 AM - 4 PM GMT) for maximum liquidity


## Features:
Extremly important to not have too many features as I don't want to confuse and overfit the model.<br>
Suggestions:
```python
# === CORE SPREAD FEATURES (5 features) ===

# 1. Current spread metrics
df['spread_tight'] = df['ask_low'] - df['bid_high']  # Tightest possible spread
df['spread_wide'] = df['ask_high'] - df['bid_low']   # Widest spread in period
df['spread_avg'] = df['ask_close'] - df['bid_close'] # Standard spread

# 2. Relative spread (most important!)
df['mid_price'] = (df['bid_close'] + df['ask_close']) / 2
df['spread_relative'] = df['spread_avg'] / df['mid_price']

# === COMPRESSION DETECTION (4 features) ===

# 3. Rolling baseline for compression detection
df['spread_baseline'] = df['spread_relative'].rolling(self.lookback_window).mean()
df['spread_recent'] = df['spread_relative'].rolling(3).mean()  # Last 3 minutes

# 4. THE KEY FEATURE: Compression ratio
df['compression_ratio'] = (df['spread_baseline'] - df['spread_recent']) / df['spread_baseline']

# 5. Compression consistency (less volatility = stronger signal)
df['spread_volatility'] = df['spread_relative'].rolling(5).std()
df['compression_consistency'] = 1 / (1 + df['spread_volatility'] * 1000)

# === VOLUME CONFIRMATION (3 features) ===

# 6. Volume patterns (compression should come with volume)
df['volume_ma'] = df['volume'].rolling(self.lookback_window).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma']
df['volume_recent'] = df['volume'].rolling(3).mean() / df['volume_ma']

# === MARKET CONTEXT (3 features) ===

# 7. Price momentum (compression during trends is stronger)
df['price_momentum_5'] = (df['mid_price'] - df['mid_price'].shift(5)) / df['mid_price'].shift(5)
df['price_volatility'] = df['mid_price'].pct_change().rolling(10).std()

# 8. Time-based features (spreads vary by session)
df['hour'] = df['timestamp'].dt.hour
df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)

# === SIGNAL STRENGTH (1 composite feature) ===

# 9. Combined signal strength
df['signal_strength'] = (
    df['compression_ratio'] * 2.0 +           # Weight compression highest
    df['volume_ratio'] * 1.0 +                # Volume confirmation
    df['compression_consistency'] * 1.5 +     # Consistency matters
    df['is_london_session'] * 0.3             # Session bonus
)
```


## Goal / Target:
I'm thinking of using 'target_spread_compression_binary' for these reasons:
- **Simpler learning problem**: Binary classification is easier to optimize than multi-class
- **Class balance**: You'll likely have severe class imbalance with the 0/1/2 target
- **Clear actionable signal**: Either trade or don't trade
- **Better evaluation metrics**: Easier to interpret precision/recall for binary
