import logging
import plotly.graph_objects as go
import pandas as pd

from utils import data_utils
from data.processing.base_processor import BaseProcessor


class PlotData(BaseProcessor):
    """Plot simple candlestick data"""
    def __init__(self, dataset):
        """..."""
        try:
            # Load dataset based on format
            self.df, self.original_df = data_utils.check_and_return_df(dataset)
        except Exception as e:
            logging.error(f"Error initializing FeatureGenerator: {str(e)}")
            raise
    
    def run(self):
        """Run plotting of data"""
        print("hei")