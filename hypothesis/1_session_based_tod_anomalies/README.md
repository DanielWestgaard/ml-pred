**Hypothesis**: Price action behaves differently across sessions (Asian, London, NY), especially around session overlaps and economic news windows.

## Features:
- Encoded time-of-day (hour, session)
- Return and volatility over last 5–15 minutes
- Day of week (some FX pairs behave differently on Fridays or Mondays)
- Economic event proximity (if you can pull a calendar)

## Goal:
- Discover if certain hours consistently yield above-average volatility or reversal patterns
- Cluster similar time windows using unsupervised ML (KMeans, t-SNE, DBSCAN)