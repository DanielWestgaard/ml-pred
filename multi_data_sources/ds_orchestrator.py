

class DataSourceOrchestrator:
    def __init__(self):
        """
        This 'Orchestrator' is meant to serve as an easy place to use singel or multiple data sources implemented.
        ...
        """
        pass
    
    # TBC..
    
    # We continue with the existing configuration driven method (similar/reusing current one 'utils/hypothesis/config_loader.py')?
    # So that it's possible to provide a config with what data source components are wanted and timeframe
    # i intend to use main.py, so this will need to be ran with the path to it, and this would reuse that path and read it
    
    # Need to make sure that this is covered: 
        # Time alignment for different source frequencies
        # Feature combination templates for hypothesis testing
        # Data quality monitoring across sources