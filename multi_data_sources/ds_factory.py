from multi_data_sources.sources.ds_base import DataSourceBase
from multi_data_sources.sources.calendar_events.cross_calendar import CalendarEventsDataSource
from multi_data_sources.sources.sec_filings.eight_k_filings import Eight_k_filings

class DataSourceFactory:
    """
    Factory to create the appropriate data source component based on configuration
    """
    
    DATA_SOURCE_COMPONENTS = {  # Maps configuration named data source to actual class implementation
        'calendar_events': CalendarEventsDataSource,  # TODO: Implement first data source
        'sec_filings': Eight_k_filings
        # Add more data source "types" here as you develop them
        # Format: 'data_source_name_from_config': DataSourceClass,
    }
    
    @classmethod
    def create_data_source(cls, data_source_type: str, **kwargs) -> DataSourceBase:
        """
        Create the appropriate data source component based on configuration
        
        Args:
            data_source_type: The type of data source being executed (e.g., 'calendar_events')
            **kwargs: Arguments to pass to the data source constructor
        
        Returns:
            Appropriate data source instance
            
        Raises:
            ValueError: If data source type is not recognized
        """
        if data_source_type not in cls.DATA_SOURCE_COMPONENTS:
            raise ValueError(f"Unknown data source type: {data_source_type}. "
                           f"Available types: {list(cls.DATA_SOURCE_COMPONENTS.keys())}")
        
        data_source_class = cls.DATA_SOURCE_COMPONENTS[data_source_type]
        return data_source_class(**kwargs)