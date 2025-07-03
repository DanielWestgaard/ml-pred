from evaluation.hypothesis_specific.session_volume_anomaly_detection import SessionVolumeAnomalyEvaluation
from evaluation.hypothesis_specific.price_prediction_evaluation import PricePredictionEvaluation
#from evaluation.hypothesis_specific.volatility_prediction_evaluation import VolatilityPredictionEvaluation  # Not implemented yet

class EvaluationFactory:
    """
    Factory to create the appropriate evaluator based on hypothesis type
    """
    
    EVALUATORS = {  # TODO: make this dynamic
        'session_based_tod_anomalies': SessionVolumeAnomalyEvaluation,  #session_volume_anomaly
        'price_prediction': PricePredictionEvaluation,
        #'volatility_prediction': VolatilityPredictionEvaluation,
        # Add more hypothesis types as you develop them
    }
    
    @classmethod
    def create_evaluator(cls, hypothesis_type: str, **kwargs):
        """
        Create the appropriate evaluator based on hypothesis type
        
        Args:
            hypothesis_type: The type of hypothesis being evaluated
            **kwargs: Arguments to pass to the evaluator constructor
        
        Returns:
            Appropriate evaluator instance
        """
        if hypothesis_type not in cls.EVALUATORS:
            raise ValueError(f"Unknown hypothesis type: {hypothesis_type}. "
                           f"Available types: {list(cls.EVALUATORS.keys())}")
        
        evaluator_class = cls.EVALUATORS[hypothesis_type]
        print(evaluator_class)
        return evaluator_class(**kwargs)