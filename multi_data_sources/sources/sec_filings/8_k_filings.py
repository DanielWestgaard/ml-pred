from multi_data_sources.sources.ds_base import DataSourceBase

class eight_k_filings(DataSourceBase):
    """
    Fetch 8-K filings from the SEC EDGAR.
    """
    
    def get_historical_data(self):
        return super().get_historical_data()
    
    def subscribe_live_data(self):
        return super().subscribe_live_data()