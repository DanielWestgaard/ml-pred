import datetime
import logging
import plotly.graph_objects as go
import pandas as pd
import sys
import mplfinance as mpf

from utils import data_utils
from data.processing.base_processor import BaseProcessor
from utils import shared_utils
import config.config as config


class PlotData(BaseProcessor):
    """Plot simple candlestick data"""
    def __init__(self, dataset, view:str = "mplfinance", save_plot:bool = False, storage_path:str = config.CLEANED_DATA_BASE_DIR, output_file:str = None):
        """
        A plotting class used to view ohlcv charts.
        
        Args:
            - dataset: Dataset containing open, high, low, close, volume, and date (-time). Can be
            either path to .csv-file or pandas.DataFrame.
            - view: What kind of view you want to use. Standard is 'simple'. The different kinds:
                ['go',              # using plotly.graph_objects to plot dataset, suitable for smaller datasets
                'mplfinance']       # using mplfinance to plot dataset, suitable for larger datasets
        """
        try:
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
            
            self.class_path = 'data.plot.plot_data.PlotData'
            # Available plot functions 
            self.plot_functions = shared_utils.list_of_functions_in_class(path_to_class=self.class_path)
            
            # Set view
            self.view = view
            
            self.save_plot = save_plot
            self.plot_storage_path = storage_path
            self.output_file = output_file or f"{self.plot_storage_path}/plot_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png"
        except Exception as e:
            logging.error(f"Error initializing PlotData: {str(e)}")
            raise
    
    def run(self):
        """Run plotting of data."""
        # Make sure the columns are lower case
        self.df.columns = self.df.columns.str.lower()
        
        # Plot dataset based on 'needs'
        for function in self.plot_functions:
            if self.view in function:
                shared_utils.safely_execute(self, function)
    
    def simple_small_go_candle_plot(self):
        """Very simplistic plotting, a somewhat optimized for 'larger' datasets."""
        try:
            x = self.df['date']
        except:
            x = self.df.index
        
        self.fig = go.Figure(data=go.Candlestick(x=x,
                            open=self.df['open'],
                            high=self.df['high'],
                            low=self.df['low'],
                            close=self.df['close'],
                            hoverinfo='x+y'
                            ))

        # Optimize layout
        self.fig.update_layout(
            yaxis=dict(fixedrange=False),
            xaxis=dict(rangeslider=dict(visible=False)),  # Disable range slider
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='white'
        )
        
        # Configure for better performance
        self.fig.update_xaxes(showspikes=False)
        self.fig.update_yaxes(showspikes=False)
        
        self.fig.show(config={'displayModeBar': False})  # Hide toolbar for speed
        
        if self.save_plot:
            self.fig.write_image(self.output_file)
    
    def simple_large_mplfinance_candle_plot(self):
        """Suitable for decent size datasets. Must be processed data."""
        try:
            self.df = self.df[['date', 'open', 'high', 'low', 'close']]
        except:  # Coming from the data pipeline, df['date'] is df.index, hence directly calling 'date' will fail in such cases
            self.df['date'] = self.df.index
            self.df = self.df[['date', 'open', 'high', 'low', 'close']]
        
        # Convert date column to datetime if it's not already
        try:
            # Only plotting OHLC - in case dataset is processed (don't need to plot features)
            # self.df = self.df[['date', 'open', 'high', 'low', 'close']]
            self.df['date'] = pd.to_datetime(self.df['date'])
        except Exception as e:
            logging.error(f"Unable to plot dataset because of error reading date column. Must be datetime format: {e}")
        
        # Set the date column as index
        self.df = self.df.set_index('date')
        
        mpf.plot(self.df, type='candle', style='yahoo')
        if self.save_plot:
            mpf.plot(self.df, type='candle', style='yahoo', savefig=self.output_file)
        mpf.show()
    