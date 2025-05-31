import logging
import plotly.graph_objects as go
import pandas as pd
import sys
import mplfinance as mpf

from utils import data_utils
from data.processing.base_processor import BaseProcessor
from utils import shared_utils


class PlotData(BaseProcessor):
    """Plot simple candlestick data"""
    def __init__(self, dataset, view:str = "simple", save_plot:bool = True):
        """
        A plotting class used to view ohlcv charts.
        
        Args:
            - dataset: Dataset containing open, high, low, close, volume, and date (-time). Can be
            either path to .csv-file or pandas.DataFrame.
            - view: What kind of view you want to use. Standard is 'simple'. The different kinds:
                ['simple', '']
        """
        try:
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            self.class_path = 'data.plot.plot_data.PlotData'
            # Available plot functions 
            self.plot_functions = shared_utils.list_of_functions_in_class(path_to_class=self.class_path)
            
            # Set view
            self.view = view
        except Exception as e:
            logging.error(f"Error initializing FeatureGenerator: {str(e)}")
            raise
    
    def run(self):
        """Run plotting of data."""
        # Make sure the columns are lower case
        self.df.columns = self.df.columns.str.lower()
        
        for function in self.plot_functions:
            if self.view in function:
                shared_utils.safely_execute(self, function)
    
    def simple_go_candle_plot(self):
        """Simplistic plotting. Suitable for shorter datasets."""
        self.fig = go.Figure(data=go.Candlestick(x=self.df['date'],
                            open=self.df['open'],
                            high=self.df['high'],
                            low=self.df['low'],
                            close=self.df['close']))

        self.fig.update_layout(yaxis=dict(fixedrange=False))

        self.fig.show()
    
    def somple2_mplfinance_candle_plot(self):
        """Suitable for decent size datasets. Must be processed data."""
        # Convert date column to datetime if it's not already
        try:
            self.df['date'] = pd.to_datetime(self.df['date'])
        except Exception as e:
            logging.error(f"Unable to plot dataset because of error reading date column. Must be datetime format: {e}")
        
        # Set the date column as index
        self.df = self.df.set_index('date')
        
        mpf.plot(self.df, type='candle', style='yahoo')
        mpf.show()
