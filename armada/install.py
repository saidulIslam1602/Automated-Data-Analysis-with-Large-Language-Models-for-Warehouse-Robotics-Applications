#!/usr/bin/env python3
"""
Installation script for the anomaly detection system.

This script sets up the environment and installs required dependencies.
"""

import os
import sys
import subprocess
import argparse
import platform

def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"ERROR: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        sys.exit(1)
    
    print(f"Python version {current_version[0]}.{current_version[1]} detected. ✓")

def check_gpu():
    """Check if GPU is available for TensorFlow."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU detected: {len(gpus)} available. ✓")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            return True
        else:
            print("No GPU detected. Using CPU only.")
            return False
    except ImportError:
        print("TensorFlow not installed yet. Will check GPU after installation.")
        return False
    except Exception as e:
        print(f"Error checking GPU: {str(e)}")
        return False

def install_requirements(dev=False, gpu=False):
    """Install Python dependencies."""
    print("\nInstalling dependencies...")
    
    # Basic requirements
    requirements_file = "requirements.txt"
    
    # Try to install with pip
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("Base requirements installed. ✓")
    except subprocess.CalledProcessError:
        print("Error installing requirements.")
        sys.exit(1)
    
    # Install TensorFlow with GPU support if requested
    if gpu:
        try:
            print("Installing TensorFlow with GPU support...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "tensorflow>=2.8.0"
            ])
        except subprocess.CalledProcessError:
            print("Error installing TensorFlow with GPU support.")
            
    # Install development packages if requested
    if dev:
        try:
            print("Installing development packages...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "pytest>=7.0.1", "black>=22.1.0", "flake8>=4.0.1"
            ])
            print("Development packages installed. ✓")
        except subprocess.CalledProcessError:
            print("Error installing development packages.")

def setup_environment():
    """Set up the environment variables."""
    print("\nSetting up environment...")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("Directories created: logs/, data/, results/, models/ ✓")
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Environment variables\n")
            f.write("PYTHONPATH=.\n")
            f.write("RANDOM_SEED=42\n")
            f.write("USE_GPU=true\n")
            f.write("LOGGING_LEVEL=INFO\n")
        print(".env file created with default settings. ✓")

def install_development_tools():
    """Install and set up development tools."""
    print("\nSetting up development environment...")
    
    try:
        # Install pre-commit hooks
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pre-commit"])
        
        # Create pre-commit config if it doesn't exist
        if not os.path.exists(".pre-commit-config.yaml"):
            with open(".pre-commit-config.yaml", "w") as f:
                f.write("""repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
""")
            print("Pre-commit hooks configuration created. ✓")
        
        # Install the hooks
        subprocess.check_call(["pre-commit", "install"])
        print("Pre-commit hooks installed. ✓")
        
    except subprocess.CalledProcessError:
        print("Error setting up development tools.")

def verify_installation():
    """Verify the installation."""
    print("\nVerifying installation...")
    
    tests = [
        ("Python", "python --version"),
        ("NumPy", "python -c \"import numpy; print(numpy.__version__)\""),
        ("Pandas", "python -c \"import pandas; print(pandas.__version__)\""),
        ("Scikit-learn", "python -c \"import sklearn; print(sklearn.__version__)\""),
        ("TensorFlow", "python -c \"import tensorflow as tf; print(tf.__version__)\"")
    ]
    
    for name, command in tests:
        try:
            result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            print(f"{name}: {result.decode('utf-8').strip()} ✓")
        except subprocess.CalledProcessError:
            print(f"{name}: Not installed or error ✗")

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*80)
    print("Installation completed!")
    print("="*80)
    
    print("\nNext steps:")
    print("1. Run 'python run_benchmark.py' to evaluate the base system")
    print("2. Run 'python run_benchmark_hybrid.py' to evaluate the enhanced system")
    print("3. For visualization, run 'python streamlit_app.py'")
    
    print("\nFor developers:")
    print("- Run 'pytest' to execute tests")
    print("- Run 'black .' to format code")
    print("- Run 'flake8' to check code quality")

def main():
    """Main installation process."""
    parser = argparse.ArgumentParser(description="Install the anomaly detection system")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--gpu", action="store_true", help="Install with GPU support")
    args = parser.parse_args()
    
    print("="*80)
    print("Anomaly Detection System Installation")
    print("="*80)
    
    # Check Python version
    check_python_version()
    
    # Check for GPU
    has_gpu = check_gpu()
    
    # Override GPU flag if GPU was detected
    gpu_support = args.gpu or has_gpu
    
    # Install requirements
    install_requirements(dev=args.dev, gpu=gpu_support)
    
    # Set up environment
    setup_environment()
    
    # Install development tools if requested
    if args.dev:
        install_development_tools()
    
    # Verify installation
    verify_installation()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 