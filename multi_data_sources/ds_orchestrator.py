from typing import List, Dict, Any
from datetime import datetime

from multi_data_sources.ds_factory import DataSourceFactory
from multi_data_sources.sources.ds_base import DataSourceBase
import utils.hypothesis.config_loader as config_loader
from utils.logging_utils import get_logger

logger = get_logger()


class DataSourceOrchestrator:
    """
    Orchestrator for coordinating multiple data sources for multi-data analysis.
    Handles time alignment, data quality monitoring, and feature combination.
    """
    
    def __init__(self, path_to_config: str):
        """
        Initialize the data source orchestrator with configuration.
        
        Args: 
            path_to_config: Path to configuration file specifying data sources and timeframes
        """
        self.config_path = path_to_config
        self.config_data = self._load_config()
        self.multi_data_config = self._extract_multi_data_config()
        
        # Initialize data source instances
        self.data_sources: List[DataSourceBase] = []
        self.data_results: Dict[str, Any] = {}
        
        # Create data source instances based on configuration
        self._initialize_data_sources()
        
        logger.info(f"DataSourceOrchestrator initialized with {len(self.data_sources)} data sources")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file using existing config loader."""
        try:
            config_loader_instance = config_loader.load_hypothesis_config(self.config_path)
            return config_loader_instance.handle_hypothesis_source()
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _extract_multi_data_config(self) -> Dict[str, Any]:
        """Extract multi_data section from configuration."""
        multi_data_config = self.config_data.get('multi_data', {})
        
        if not multi_data_config:
            raise ValueError("No 'multi_data' section found in configuration")
        
        # Validate required fields
        required_fields = ['purpose', 'symbol', 'sources']
        missing_fields = [field for field in required_fields if field not in multi_data_config]
        if missing_fields:
            raise ValueError(f"Missing required fields in multi_data config: {missing_fields}")
        
        logger.info(f"Multi-data config loaded for {multi_data_config['symbol']} with purpose: {multi_data_config['purpose']}")
        return multi_data_config
    
    def _initialize_data_sources(self):
        """Initialize data source instances based on configuration."""
        sources = self.multi_data_config.get('sources', [])
        
        for source_name in sources:
            try:
                # Create data source instance with config parameters
                data_source = DataSourceFactory.create_data_source(
                    data_source_type=source_name,
                    symbol=self.multi_data_config.get('symbol'),
                    from_date=self.multi_data_config.get('from_date'),
                    to_date=self.multi_data_config.get('to_date'),
                    purpose=self.multi_data_config.get('purpose')
                )
                
                self.data_sources.append(data_source)
                logger.info(f"Initialized data source: {source_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize data source {source_name}: {e}")
                # Continue with other sources rather than failing entirely
                continue
    
    def run(self) -> Dict[str, Any]:
        """Run the data orchestrator and collect data from all sources."""
        logger.info(f"Running DataSourceOrchestrator with {len(self.data_sources)} data sources")
        
        purpose = self.multi_data_config.get('purpose')
        
        if purpose == 'historical':
            return self._run_historical()
        elif purpose == 'live':
            return self._run_live()
        else:
            raise ValueError(f"Unknown purpose: {purpose}. Must be 'historical' or 'live'")
    
    def _run_historical(self) -> Dict[str, Any]:
        """Run historical data collection from all sources."""
        logger.info("Running historical data collection")
        
        for i, data_source in enumerate(self.data_sources):
            source_name = self.multi_data_config['sources'][i]
            try:
                logger.info(f"Collecting historical data from {source_name}")
                historical_data = data_source.get_historical_data()
                self.data_results[source_name] = {
                    'data': historical_data,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
                logger.info(f"Successfully collected historical data from {source_name}")
                
            except Exception as e:
                logger.error(f"Failed to collect historical data from {source_name}: {e}")
                self.data_results[source_name] = {
                    'data': None,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed',
                    'error': str(e)
                }
        
        return self._prepare_results()
    
    def _run_live(self) -> Dict[str, Any]:
        """Run live data subscription from all sources."""
        logger.info("Running live data subscription")
        
        for i, data_source in enumerate(self.data_sources):
            source_name = self.multi_data_config['sources'][i]
            try:
                logger.info(f"Subscribing to live data from {source_name}")
                live_subscription = data_source.subscribe_live_data()
                self.data_results[source_name] = {
                    'subscription': live_subscription,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'subscribed'
                }
                logger.info(f"Successfully subscribed to live data from {source_name}")
                
            except Exception as e:
                logger.error(f"Failed to subscribe to live data from {source_name}: {e}")
                self.data_results[source_name] = {
                    'subscription': None,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed',
                    'error': str(e)
                }
        
        return self._prepare_results()
    
    def _prepare_results(self) -> Dict[str, Any]:
        """Prepare final results with metadata."""
        successful_sources = [name for name, result in self.data_results.items() 
                            if result['status'] in ['success', 'subscribed']]
        failed_sources = [name for name, result in self.data_results.items() 
                        if result['status'] == 'failed']
        
        results = {
            'config': self.multi_data_config,
            'sources': self.data_results,
            'summary': {
                'total_sources': len(self.data_sources),
                'successful_sources': len(successful_sources),
                'failed_sources': len(failed_sources),
                'successful_source_names': successful_sources,
                'failed_source_names': failed_sources
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"DataSourceOrchestrator completed: {len(successful_sources)}/{len(self.data_sources)} sources successful")
        return results
    
    def get_aligned_data(self, alignment_method: str = 'forward_fill') -> Dict[str, Any]:
        """Get time-aligned data from all sources (future enhancement)."""
        # TODO: Implement time alignment logic for different source frequencies
        logger.info(f"Time alignment with method: {alignment_method} - Not yet implemented")
        return self.data_results
    
    def get_feature_combinations(self) -> List[str]:
        """Get possible feature combinations from data sources (future enhancement)."""
        # TODO: Implement feature combination templates for hypothesis testing
        successful_sources = [name for name, result in self.data_results.items() 
                            if result.get('status') in ['success', 'subscribed']]
        logger.info(f"Feature combinations from sources: {successful_sources} - Not yet implemented")
        return successful_sources
    
    def validate_data_quality(self) -> Dict[str, Dict[str, Any]]:
        """Validate data quality across all sources (future enhancement)."""
        # TODO: Implement data quality monitoring across sources
        quality_report = {}
        for source_name, result in self.data_results.items():
            quality_report[source_name] = {
                'status': result.get('status', 'unknown'),
                'has_data': result.get('data') is not None,
                'timestamp': result.get('timestamp')
            }
        
        logger.info("Data quality validation - Basic implementation")
        return quality_report