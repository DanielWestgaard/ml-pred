#!/usr/bin/env python3
"""
Cloud GPU Setup Script

Automated setup and configuration for cloud GPU training environments.
Supports AWS, GCP, and Azure with automatic dependency installation and GPU verification.
"""

import argparse
import subprocess
import sys
import os
import yaml
from pathlib import Path

def run_command(command, description=""):
    """Run shell command and handle errors."""
    print(f"🔧 {description}")
    print(f"   Running: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✅ Success")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
    else:
        print(f"   ❌ Failed: {result.stderr.strip()}")
        return False
    
    return True

def detect_cloud_provider():
    """Detect which cloud provider we're running on."""
    
    # Check for AWS metadata service
    try:
        result = subprocess.run(['curl', '-s', '--connect-timeout', '2', 
                               'http://169.254.169.254/latest/meta-data/instance-id'], 
                               capture_output=True)
        if result.returncode == 0:
            return "aws"
    except:
        pass
    
    # Check for GCP metadata service
    try:
        result = subprocess.run(['curl', '-s', '--connect-timeout', '2', 
                               'http://metadata.google.internal/computeMetadata/v1/instance/id',
                               '-H', 'Metadata-Flavor: Google'], 
                               capture_output=True)
        if result.returncode == 0:
            return "gcp"
    except:
        pass
    
    # Check for Azure metadata service
    try:
        result = subprocess.run(['curl', '-s', '--connect-timeout', '2',
                               'http://169.254.169.254/metadata/instance?api-version=2021-02-01',
                               '-H', 'Metadata: true'], 
                               capture_output=True)
        if result.returncode == 0:
            return "azure"
    except:
        pass
    
    return "unknown"

def install_gpu_dependencies(framework="all"):
    """Install GPU-enabled ML framework dependencies."""
    
    commands = []
    
    if framework in ["all", "pytorch"]:
        commands.append({
            "cmd": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
            "desc": "Installing PyTorch with CUDA support"
        })
    
    if framework in ["all", "tensorflow"]:
        commands.append({
            "cmd": "pip install tensorflow[and-cuda]",
            "desc": "Installing TensorFlow with CUDA support"
        })
    
    if framework in ["all", "xgboost"]:
        commands.append({
            "cmd": "pip install xgboost",
            "desc": "Installing XGBoost (GPU support included)"
        })
    
    # Install additional ML dependencies
    additional_deps = [
        "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", 
        "jupyter", "ipykernel", "tqdm", "wandb"
    ]
    
    commands.append({
        "cmd": f"pip install {' '.join(additional_deps)}",
        "desc": "Installing additional ML dependencies"
    })
    
    success = True
    for cmd_info in commands:
        if not run_command(cmd_info["cmd"], cmd_info["desc"]):
            success = False
    
    return success

def verify_gpu_setup():
    """Verify GPU setup and frameworks."""
    
    print("🧪 Verifying GPU setup...")
    
    # Check NVIDIA driver
    if not run_command("nvidia-smi", "Checking NVIDIA driver"):
        return False
    
    # Check CUDA
    run_command("nvcc --version", "Checking CUDA compiler (optional)")
    
    # Test PyTorch GPU
    pytorch_test = """
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name()}')
    # Test tensor operation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('PyTorch GPU test: ✅ PASSED')
else:
    print('PyTorch GPU test: ❌ FAILED')
"""
    
    run_command(f'python -c "{pytorch_test}"', "Testing PyTorch GPU support")
    
    # Test TensorFlow GPU
    tensorflow_test = """
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {len(gpus)}')
if gpus:
    for gpu in gpus:
        print(f'  - {gpu}')
    # Test operation
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    print('TensorFlow GPU test: ✅ PASSED')
else:
    print('TensorFlow GPU test: ❌ FAILED')
"""
    
    run_command(f'python -c "{tensorflow_test}"', "Testing TensorFlow GPU support")
    
    # Test XGBoost GPU
    xgboost_test = """
import xgboost as xgb
import numpy as np
print(f'XGBoost version: {xgb.__version__}')
try:
    # Create test data
    X = np.random.random((1000, 10))
    y = np.random.random(1000)
    dtrain = xgb.DMatrix(X, label=y)
    
    # Test GPU training
    params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
    model = xgb.train(params, dtrain, num_boost_round=10)
    print('XGBoost GPU test: ✅ PASSED')
except Exception as e:
    print(f'XGBoost GPU test: ❌ FAILED - {str(e)}')
"""
    
    run_command(f'python -c "{xgboost_test}"', "Testing XGBoost GPU support")
    
    return True

def setup_monitoring():
    """Setup GPU monitoring tools."""
    
    commands = [
        {
            "cmd": "pip install gpustat py3nvml",
            "desc": "Installing GPU monitoring tools"
        },
        {
            "cmd": "pip install wandb tensorboard",
            "desc": "Installing experiment tracking tools"
        }
    ]
    
    for cmd_info in commands:
        run_command(cmd_info["cmd"], cmd_info["desc"])

def create_gpu_test_script():
    """Create a comprehensive GPU test script."""
    
    test_script = """#!/usr/bin/env python3
import sys
sys.path.append('/path/to/ml-pred')  # Adjust this path

from utils.gpu_utils import print_gpu_status, get_gpu_config

if __name__ == '__main__':
    print("🚀 ML-Pred GPU Test Suite")
    print("=" * 50)
    
    # Print GPU status
    print_gpu_status()
    
    print("\\n📊 Framework Configurations:")
    print("-" * 30)
    
    frameworks = ['xgboost', 'pytorch', 'tensorflow']
    for framework in frameworks:
        try:
            config = get_gpu_config(framework)
            print(f"{framework.upper():12}: {config}")
        except Exception as e:
            print(f"{framework.upper():12}: Error - {str(e)}")
    
    print("\\n✅ Setup complete! Your GPU environment is ready for ML training.")
"""
    
    with open("test_gpu_setup.py", "w") as f:
        f.write(test_script)
    
    os.chmod("test_gpu_setup.py", 0o755)
    print("📝 Created test_gpu_setup.py script")

def main():
    parser = argparse.ArgumentParser(description="Setup cloud GPU environment for ML training")
    parser.add_argument("--framework", choices=["pytorch", "tensorflow", "xgboost", "all"], 
                       default="all", help="ML framework to install")
    parser.add_argument("--skip-verify", action="store_true", help="Skip GPU verification")
    parser.add_argument("--provider", choices=["aws", "gcp", "azure", "auto"], 
                       default="auto", help="Cloud provider")
    
    args = parser.parse_args()
    
    print("🚀 ML-Pred Cloud GPU Setup")
    print("=" * 40)
    
    # Detect cloud provider
    if args.provider == "auto":
        provider = detect_cloud_provider()
        print(f"🔍 Detected cloud provider: {provider}")
    else:
        provider = args.provider
        print(f"🔧 Using specified provider: {provider}")
    
    # System info
    run_command("uname -a", "System information")
    run_command("python --version", "Python version")
    
    # Install dependencies
    print("\\n📦 Installing dependencies...")
    if not install_gpu_dependencies(args.framework):
        print("❌ Failed to install some dependencies")
        sys.exit(1)
    
    # Setup monitoring
    print("\\n📊 Setting up monitoring...")
    setup_monitoring()
    
    # Verify setup
    if not args.skip_verify:
        print("\\n🧪 Verifying setup...")
        verify_gpu_setup()
    
    # Create test script
    create_gpu_test_script()
    
    print("\\n🎉 Cloud GPU setup complete!")
    print("\\nNext steps:")
    print("1. Run: python test_gpu_setup.py")
    print("2. Start training your models with GPU acceleration")
    print("3. Monitor GPU usage with: gpustat -i 1")

if __name__ == "__main__":
    main()