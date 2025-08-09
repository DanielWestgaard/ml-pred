

# Perhaps getting some inspiration from how strategy/strategy.py and strategy_factory.py is designed
class DataSourceOrchestrator:
    """
    Place to get features from 
    """
    
    def __init__(self, path_to_config:str):
        """
        This 'Orchestrator' is meant to serve as an easy place to use singel or multiple data sources implemented.
        
        Args: 
            - path_to_config: Path to configuration file specifying what data sources you want, and "from how far back".
        
        Returns:
            - bool: true if all went well, false if fail
        """
        # Read the config
        
        # Extract variables (list of data source/-s and time)
        
        # Loop through provided data source components to create appropriate instances and (late) run them one by one
        
    
    # TBC..
    
    # We continue with the existing configuration driven method (similar/reusing current one 'utils/hypothesis/config_loader.py')?
    # So that it's possible to provide a config with what data source components are wanted and timeframe
    # i intend to use main.py, so this will need to be ran with the path to it, and this would reuse that path and read it
    
    # Need to make sure that this is covered: 
        # Time alignment for different source frequencies
        # Feature combination templates for hypothesis testing
        # Data quality monitoring across sources