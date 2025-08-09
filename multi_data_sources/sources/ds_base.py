from abc import ABC, abstractmethod


class DataSourceBase(ABC):
    def __init__(self):
        """
        This class is meant to serve as a base for future implementations/additions of
        different data source components.
        """
        pass
    
    @abstractmethod
    def get_historical_data(self):
        """Fetch historical to store in storage/"""
        pass
        
    @abstractmethod
    def subscribe_live_data(self):
        """
        Subscribeto live date, using the source's approach to this.
        """
        pass