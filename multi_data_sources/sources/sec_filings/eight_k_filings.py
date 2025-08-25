from multi_data_sources.sources.ds_base import DataSourceBase

class Eight_k_filings(DataSourceBase):
    """
    Fetch 8-K filings from the SEC EDGAR.
    """
    
    def __init__(self, symbol, stock = False, from_date = None, to_date = None, purpose = "historical"):
        STOCK_CIK = {
            'BDTX' : '0001701541'
        }
        
        headers = {
            'User-Agent': 'research@example.com'  # SEC requires a User-Agent
        }
        
        # Get company filing history from SEC EDGAR API
        url = f'https://data.sec.gov/submissions/CIK{str(STOCK_CIK.get(stock)).zfill(10)}.json'
        print(STOCK_CIK.get(stock))
    
    def get_historical_data(self):
        return super().get_historical_data()
    
    def subscribe_live_data(self):
        return super().subscribe_live_data()