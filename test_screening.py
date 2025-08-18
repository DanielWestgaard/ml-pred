from finvizfinance.screener import Overview
import pandas as pd


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

def change_market_cap_format(df: pd.DataFrame, column_name: str):
    """Format market cap column to readable format (K, M, B, T)"""
    
    def format_number(num):
        """Convert large numbers to readable format"""
        if pd.isna(num):
            return num
        
        # Handle string inputs (sometimes finviz returns strings)
        if isinstance(num, str):
            # Remove any existing formatting like commas, B, M, etc.
            num = num.replace(',', '').replace('B', '').replace('M', '').replace('K', '')
            try:
                num = float(num)
            except:
                return num
        
        num = float(num)
        abs_num = abs(num)
        
        if abs_num >= 1e12:
            return f"{num/1e12:.2f}T"
        elif abs_num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif abs_num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif abs_num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.0f}"
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Check if column exists
    if column_name in df_copy.columns:
        df_copy[column_name] = df_copy[column_name].apply(format_number)
    else:
        print(f"Warning: Column '{column_name}' not found in DataFrame")
        print(f"Available columns: {list(df_copy.columns)}")
    
    return df_copy

def main():
    print("Hello!")
    
    me = screen_microcaps()
    print("Original data:")
    print(me)
    print("\n" + "="*50 + "\n")
    
    me_formatted = change_market_cap_format(me, "Market Cap")
    
    print("Formatted data:")
    print(me_formatted)

if __name__=="__main__":
    main()