from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize the class."""
        pass
    
    @abstractmethod
    def run(self):
        """Run the processor."""
        pass