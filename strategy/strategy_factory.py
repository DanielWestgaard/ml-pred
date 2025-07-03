from strategy.hypothesis_specific.session_anomaly_strategy import SessionAnomalyStrategy
from strategy.hypothesis_specific.price_prediction_strategy import PricePredictionStrategy

class StrategyFactory:
    """
    Factory to create the appropriate strategy based on hypothesis type
    """
    
    STRATEGIES = {
        'session_based_tod_anomalies': SessionAnomalyStrategy,
        'price_prediction': PricePredictionStrategy,
        # Add more hypothesis strategy types as you develop them
    }
    
    @classmethod
    def create_strategy(cls, hypothesis_type: str, **kwargs):
        """
        Create the appropriate strategy based on hypothesis type
        
        Args:
            hypothesis_type: The type of hypothesis being executed
            **kwargs: Arguments to pass to the strategy constructor
        
        Returns:
            Appropriate strategy instance
        """
        if hypothesis_type not in cls.STRATEGIES:
            raise ValueError(f"Unknown hypothesis type: {hypothesis_type}. "
                           f"Available types: {list(cls.STRATEGIES.keys())}")
        
        strategy_class = cls.STRATEGIES[hypothesis_type]
        return strategy_class(**kwargs)