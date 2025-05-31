import logging
import plotly.graph_objects as go
import pandas as pd
import sys

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
        for function in self.plot_functions:
            if self.view in function:
                shared_utils.safely_execute(self, function)
    
    def simple_go_candle_plot(self):
        """Simplistic plotting. Suitable for shorter datasets."""
        self.fig = go.Figure(data=go.Candlestick(x=self.df['Date'],
                            open=self.df['Open'],
                            high=self.df['High'],
                            low=self.df['Low'],
                            close=self.df['Close']))
        self.fig.update_layout(
            xaxis=dict(type='category'),  # Can help with performance
        )
        self.fig.show()
    
    def simpler_xxx_candle_plot(self):
        """Simplistic plotting. Suitable for medium to large-minus datasets."""
        pass