import sys
import pandas as pd
import numpy as np

from model.model_factory.base_model import BaseModel
import utils.model.model_utils as model_utils
from utils.gpu_utils import get_gpu_config, log_gpu_status
from utils.logging_utils import get_logger

logger = get_logger()

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Install with: pip install torch")
    TORCH_AVAILABLE = False


class PyTorchModel(BaseModel):
    """
    PyTorch Neural Network Model with automatic GPU acceleration.
    
    Supports CUDA (NVIDIA) and Metal Performance Shaders (Apple Silicon).
    Automatically falls back to CPU if GPU is not available.
    """
    
    def __init__(self, dataset: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Initialize PyTorch model with GPU detection."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this model. Install with: pip install torch")
            
        logger.debug("----- Initializing PyTorch Model -----")
        
        self.df = dataset
        self.X_train = X_train
        self.y_train = y_train
        
        # Get GPU configuration
        self.gpu_config = get_gpu_config('pytorch')
        self.device = torch.device(self.gpu_config['device'])
        
        logger.info(f"PyTorch device: {self.device}")
        
        # Neural network architecture
        self.input_size = len(X_train.columns)
        self.hidden_size = 128
        self.output_size = 1
        
    def train(self):
        """Train PyTorch neural network with GPU acceleration."""
        logger.info("---------- Starting to train PyTorch Model! ----------")
        
        log_gpu_status()
        
        # Convert data to tensors
        X_tensor = torch.FloatTensor(self.X_train.values).to(self.device)
        y_tensor = torch.FloatTensor(self.y_train.values.reshape(-1, 1)).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Define neural network
        model = NeuralNetwork(self.input_size, self.hidden_size, self.output_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        epochs = 100
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.debug(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")
        
        logger.info(f"Successfully trained PyTorch Model using {self.device}!")
        return model
    
    def save_model_and_info(self):
        """Save trained PyTorch model."""
        pass


class NeuralNetwork(nn.Module):
    """Simple feedforward neural network for regression."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out