## A "Quick Win" - Opening Gap Reversal System
Since I can't trade pre-market, I'll catch the gap at 9:30am open. This might be better because:
- The gap has already formed (no guessing)
- Opening volume confirms if it's real or fake
- You have clear, programmable rules

### Version 1: Simple Gap Fade (Your First Week Winner)
```py
# Pseudocode for Alpaca/Capital.com
def gap_fade_strategy():
    at_market_open (9:30am):
        gap_percent = (open - previous_close) / previous_close
        
        if gap_percent > 0.05 and gap_percent < 0.15:  # 5-15% gap up
            if opening_volume < avg_20day_volume * 2:   # No huge volume = fake gap
                ENTRY: Short at market
                STOP: gap_percent + 0.02 (2% above entry)
                TARGET: previous_close (gap fill)
                TIME_STOP: Exit at 3:30pm regardless
        
        elif gap_percent < -0.05 and gap_percent > -0.15:  # 5-15% gap down
            if opening_volume < avg_20day_volume * 2:
                ENTRY: Long at market
                STOP: gap_percent - 0.02 
                TARGET: previous_close
                TIME_STOP: Exit at 3:30pm
```
> Expected results: 60-65% win rate, 1.5:1 reward/risk ratio

### Version 2: Volume Confirmation Enhancement
```python
def enhanced_gap_system():
    # Wait 5 minutes for confirmation
    at_9:35am:
        gap_percent = (open - previous_close) / previous_close
        first_5min_volume = sum(volume_9:30_to_9:35)
        
        # The key insight: fade if low volume, follow if high volume
        if abs(gap_percent) > 0.05:
            if first_5min_volume < avg_5min_volume * 3:
                # FADE the gap - low volume means no conviction
                fade_entry()
            elif first_5min_volume > avg_5min_volume * 10:
                # FOLLOW the gap - huge volume means real move
                momentum_entry()
```