"""
GPU Detection and Configuration Utilities

Provides automatic GPU detection, configuration, and fallback mechanisms
for multiple ML frameworks across different hardware (Mac Metal, NVIDIA CUDA, CPU).
"""

import os
import platform
import subprocess
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GPUType(Enum):
    """GPU types supported by the system."""
    CUDA = "cuda"
    METAL = "metal" 
    CPU = "cpu"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """GPU information container."""
    gpu_type: GPUType
    device_count: int
    device_name: str
    memory_gb: Optional[float]
    compute_capability: Optional[str]
    framework_support: Dict[str, bool]


class GPUDetector:
    """Detects available GPU acceleration capabilities."""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
    
    def _detect_gpu(self) -> GPUInfo:
        """Detect GPU capabilities and return configuration."""
        
        # Check for NVIDIA CUDA first (preferred for cloud)
        cuda_info = self._check_cuda()
        if cuda_info.gpu_type == GPUType.CUDA:
            return cuda_info
        
        # Check for Metal (macOS)
        if platform.system() == "Darwin":
            metal_info = self._check_metal()
            if metal_info.gpu_type == GPUType.METAL:
                return metal_info
        
        # Fallback to CPU
        return self._get_cpu_info()
    
    def _check_cuda(self) -> GPUInfo:
        """Check for NVIDIA CUDA support."""
        try:
            # Check nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_name = lines[0].split(',')[0].strip()
                memory_mb = float(lines[0].split(',')[1].strip())
                
                # Check framework support
                framework_support = {
                    'xgboost': self._test_xgboost_cuda(),
                    'pytorch': self._test_pytorch_cuda(),
                    'tensorflow': self._test_tensorflow_cuda()
                }
                
                return GPUInfo(
                    gpu_type=GPUType.CUDA,
                    device_count=len(lines),
                    device_name=gpu_name,
                    memory_gb=round(memory_mb / 1024, 2),
                    compute_capability=self._get_cuda_compute_capability(),
                    framework_support=framework_support
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
        
        return GPUInfo(GPUType.UNKNOWN, 0, "", None, None, {})
    
    def _check_metal(self) -> GPUInfo:
        """Check for macOS Metal support."""
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=10)
            
            if "Metal" in result.stdout:
                # Extract GPU info from system_profiler
                lines = result.stdout.split('\n')
                gpu_name = "Apple GPU"
                for line in lines:
                    if "Chipset Model:" in line:
                        gpu_name = line.split(':')[1].strip()
                        break
                
                framework_support = {
                    'xgboost': self._test_xgboost_metal(),
                    'pytorch': self._test_pytorch_metal(),
                    'tensorflow': self._test_tensorflow_metal()
                }
                
                return GPUInfo(
                    gpu_type=GPUType.METAL,
                    device_count=1,
                    device_name=gpu_name,
                    memory_gb=None,  # Hard to detect programmatically
                    compute_capability="Metal 3",
                    framework_support=framework_support
                )
        except Exception:
            pass
        
        return GPUInfo(GPUType.UNKNOWN, 0, "", None, None, {})
    
    def _get_cpu_info(self) -> GPUInfo:
        """Get CPU fallback information."""
        import psutil
        
        return GPUInfo(
            gpu_type=GPUType.CPU,
            device_count=psutil.cpu_count(),
            device_name=platform.processor() or platform.machine(),
            memory_gb=round(psutil.virtual_memory().total / (1024**3), 2),
            compute_capability=None,
            framework_support={
                'xgboost': True,
                'pytorch': True,
                'tensorflow': True
            }
        )
    
    def _test_xgboost_cuda(self) -> bool:
        """Test XGBoost CUDA support."""
        try:
            import xgboost as xgb
            # Try to create a GPU-enabled booster
            dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
            params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            xgb.train(params, dtrain, num_boost_round=1)
            return True
        except Exception:
            return False
    
    def _test_xgboost_metal(self) -> bool:
        """Test XGBoost Metal support (limited)."""
        try:
            import xgboost as xgb
            # XGBoost has limited Metal support, mainly CPU on Mac
            return hasattr(xgb.XGBRegressor(), 'tree_method')
        except Exception:
            return False
    
    def _test_pytorch_cuda(self) -> bool:
        """Test PyTorch CUDA support."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _test_pytorch_metal(self) -> bool:
        """Test PyTorch Metal Performance Shaders (MPS) support."""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            return False
    
    def _test_tensorflow_cuda(self) -> bool:
        """Test TensorFlow CUDA support."""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            return False
    
    def _test_tensorflow_metal(self) -> bool:
        """Test TensorFlow Metal support."""
        try:
            import tensorflow as tf
            # TensorFlow Metal plugin for Mac
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            return False
    
    def _get_cuda_compute_capability(self) -> Optional[str]:
        """Get CUDA compute capability."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def get_optimal_config(self, framework: str) -> Dict[str, Any]:
        """Get optimal configuration for the specified framework."""
        
        config = {
            'device_type': self.gpu_info.gpu_type.value,
            'framework': framework,
            'fallback_cpu': True
        }
        
        if framework == 'xgboost':
            config.update(self._get_xgboost_config())
        elif framework == 'pytorch':
            config.update(self._get_pytorch_config())
        elif framework == 'tensorflow':
            config.update(self._get_tensorflow_config())
        
        return config
    
    def _get_xgboost_config(self) -> Dict[str, Any]:
        """Get XGBoost-specific GPU configuration."""
        if self.gpu_info.gpu_type == GPUType.CUDA and self.gpu_info.framework_support.get('xgboost'):
            return {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            }
        else:
            return {
                'tree_method': 'hist',  # CPU optimized
                'n_jobs': -1  # Use all CPU cores
            }
    
    def _get_pytorch_config(self) -> Dict[str, Any]:
        """Get PyTorch-specific GPU configuration."""
        if self.gpu_info.gpu_type == GPUType.CUDA and self.gpu_info.framework_support.get('pytorch'):
            return {
                'device': 'cuda',
                'device_count': self.gpu_info.device_count
            }
        elif self.gpu_info.gpu_type == GPUType.METAL and self.gpu_info.framework_support.get('pytorch'):
            return {
                'device': 'mps',
                'device_count': 1
            }
        else:
            return {
                'device': 'cpu',
                'device_count': 1
            }
    
    def _get_tensorflow_config(self) -> Dict[str, Any]:
        """Get TensorFlow-specific GPU configuration."""
        if self.gpu_info.gpu_type == GPUType.CUDA and self.gpu_info.framework_support.get('tensorflow'):
            return {
                'device': '/GPU:0',
                'memory_growth': True  # Allocate GPU memory as needed
            }
        elif self.gpu_info.gpu_type == GPUType.METAL and self.gpu_info.framework_support.get('tensorflow'):
            return {
                'device': '/GPU:0',  # TensorFlow Metal
                'memory_growth': True
            }
        else:
            return {
                'device': '/CPU:0'
            }
    
    def log_gpu_info(self):
        """Print detailed GPU information."""
        info = self.gpu_info
        logger.info(f"\n🚀 GPU Detection Results:")
        logger.info(f"├── Type: {info.gpu_type.value.upper()}")
        logger.info(f"├── Device: {info.device_name}")
        logger.info(f"├── Count: {info.device_count}")
        if info.memory_gb:
            logger.info(f"├── Memory: {info.memory_gb} GB")
        if info.compute_capability:
            logger.info(f"├── Capability: {info.compute_capability}")
        
        logger.info(f"└── Framework Support:")
        for framework, supported in info.framework_support.items():
            status = "✅" if supported else "❌"
            logger.info(f"    ├── {framework}: {status}")


# Global GPU detector instance
_gpu_detector = None

def get_gpu_config(framework: str = 'xgboost') -> Dict[str, Any]:
    """Get GPU configuration for the specified framework."""
    global _gpu_detector
    if _gpu_detector is None:
        _gpu_detector = GPUDetector()
    
    return _gpu_detector.get_optimal_config(framework)

def get_gpu_info() -> GPUInfo:
    """Get GPU information."""
    global _gpu_detector
    if _gpu_detector is None:
        _gpu_detector = GPUDetector()
    
    return _gpu_detector.gpu_info

def print_gpu_status():
    """Print current GPU status."""
    global _gpu_detector
    if _gpu_detector is None:
        _gpu_detector = GPUDetector()
    
    _gpu_detector.log_gpu_info()