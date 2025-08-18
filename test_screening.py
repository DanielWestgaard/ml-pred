from finvizfinance.screener import Overview


def screen_microcaps():
    # Target: $50M - $500M market cap
    # Why: Too small for institutions, liquid enough to trade
    
    screener = Overview()
    
    # Set filters using correct filter names
    screener.set_filter(filters_dict={
        'Market Cap.': 'Small ($300mln to $2bln)',  # Small cap filter
        'P/E': 'Under 15',                          # P/E under 15
        'P/B': 'Under 1',                           # P/B under 1
        'Debt/Equity': 'Under 0.5',                 # Low debt
        'InsiderTransactions': 'Positive (>0%)',    # Positive insider activity
    })
    
    return screener.screener_view()

def main():
    print("Hello!")
    
    print(screen_microcaps())

if __name__=="__main__":
    main()