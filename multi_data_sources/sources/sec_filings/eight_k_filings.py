import csv
import os
from multi_data_sources.sources.ds_base import DataSourceBase
import requests
import config.config as config
import utils.shared_utils as shared_utils


class Eight_k_filings(DataSourceBase):
    """
    Fetch 8-K filings from the SEC EDGAR.
    """
    
    def __init__(self, symbol, stock = False, from_date = None, to_date = None, purpose = "historical"):
        STOCK_CIK = {
            'BDTX' : '0001701541'
        }
        
        self.headers = {
            'User-Agent': 'research@example.com'  # SEC requires a User-Agent
        }
        
        # Get company filing history from SEC EDGAR API
        self.url = f'https://data.sec.gov/submissions/CIK{str(STOCK_CIK.get(symbol)).zfill(10)}.json'
        
        self.output_path = os.path.join(config.SEC_EDGAR_filings_storage_dir, symbol, 'eight_k')
    
    def get_historical_data(self):
        """Get historical 8-K filings. """
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            filings = data['filings']['recent']
            
            # Extract 8-K filings
            eightk_filings = []
            
            for i, form in enumerate(filings['form']):
                if form in ['8-K', '8-K/A']:  # Include amendments
                    filing_info = {
                        'filing_date': filings['filingDate'][i],
                        'form': form,
                        'accession_number': filings['accessionNumber'][i],
                        'act': filings.get('act', [None] * len(filings['form']))[i],
                        'items': filings.get('items', [None] * len(filings['form']))[i] if i < len(filings.get('items', [])) else None
                    }
                    eightk_filings.append(filing_info)
            
            # Sort by filing date
            eightk_filings.sort(key=lambda x: x['filing_date'])
            
            # Save to CSV
            shared_utils.ensure_path_exists(self.output_path)
            with open(self.output_path, 'w', newline='') as csvfile:
                fieldnames = ['filing_date', 'form', 'accession_number', 'act', 'items']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(eightk_filings)
            
            print(f"Found {len(eightk_filings)} 8-K filings")
            print(f"Date range: {eightk_filings[0]['filing_date']} to {eightk_filings[-1]['filing_date']}")
            print(f"Saved to: {self.output_path}")
            
            # Show recent filings
            print("\nMost recent 8-K filings:")
            for filing in eightk_filings[-5:]:
                items_str = filing['items'] if filing['items'] else 'N/A'
                print(f"  {filing['filing_date']}: {filing['form']} - Items: {items_str}")
            
            return eightk_filings
            
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            return []
    
    def subscribe_live_data(self):
        return super().subscribe_live_data()