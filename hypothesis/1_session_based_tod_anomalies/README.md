**Hypothesis**: Price action behaves differently across sessions (Asian, London, NY), especially around session overlaps and economic news windows.


## Features:
- Encoded time-of-day (hour, session)
- Return and volatility over last 5–15 minutes
- Day of week (some FX pairs behave differently on Fridays or Mondays)
- Economic event proximity (if you can pull a calendar)


## Goal:
- Discover if certain hours consistently yield above-average volatility or reversal patterns
- Cluster similar time windows using unsupervised ML (KMeans, t-SNE, DBSCAN)


## Configuration files 🗂️
- ✅ config_minimal.yml - Working simulation (tested & confirmed)
- 🔧 config_fixed.yml - WebSocket with real Capital.com data
- 🔧 config_simulation.yml - Pure simulation backup

### For WebSocket Testing: `config_fixed.yml`
- `data.dry_run`: false (real WebSocket data)
- `order_management.dry_run`: true (no real orders)
- `use_websocket`: true (WebSocket enabled)

### For Pure Simulation: `config_simulation.yml`
- `data.dry_run`: true (simulated data)
- `use_websocket`: false (no WebSocket)
- Faster execution for testing
