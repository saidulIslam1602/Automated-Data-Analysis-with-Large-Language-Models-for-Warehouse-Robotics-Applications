#!/usr/bin/env python
"""
Setup script for the PDF Analysis and Insight Generation system.
This script creates necessary directories and performs initial setup.
"""
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create the directory structure for the project."""
    # Define directories to create
    directories = [
        "output",
        "output/analysis",
        "output/visualizations",
        "output/temp",
        "logs",
        "models",
        "vector_db"
    ]
    
    # Create directories
    root_dir = Path(__file__).parent.absolute()
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return True

def set_environment_variables():
    """Set up environment variables."""
    # Check if .env file exists, if not create a template
    env_path = os.path.join(Path(__file__).parent.absolute(), ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("# Environment variables for PDF Analysis system\n")
            f.write("OPENAI_API_KEY=your_api_key_here\n")
            f.write("OPENAI_MODEL=gpt-3.5-turbo-1106\n")
            f.write("FINE_TUNED_MODEL=ft:gpt-3.5-turbo-0125:personal:pdf-analysis-1742275954:BCK2dWgb\n")
            f.write("USE_LOCAL_SEARCH=false\n")
            f.write("VERIFICATION_THRESHOLD=0.75\n")
            f.write("HYBRID_VERIFICATION=true\n")
        logger.info(f"Created .env template file: {env_path}")
        logger.warning("Please edit the .env file to add your OpenAI API key")
    
    # If the API key is not already set in the environment, try to get it from the .env file
    if not os.environ.get("OPENAI_API_KEY"):
        try:
            with open(env_path, 'r') as env_file:
                for line in env_file:
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=')[1].strip()
                        # Remove quotes if present
                        if api_key and api_key != "your_api_key_here" and (not api_key.startswith('"') or not api_key.endswith('"')):
                            os.environ["OPENAI_API_KEY"] = api_key
                            logger.info("Successfully loaded API key from .env file")
                        break
        except Exception as e:
            logger.error(f"Error reading API key from .env file: {e}")
    
    return True

def check_dependencies():
    """Check for required dependencies."""
    try:
        import pdfplumber
        import openai
        import chromadb
        import streamlit
        import matplotlib
        import plotly
        logger.info("All core dependencies installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main function to run the setup."""
    logger.info("Starting PDF Analysis system setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Set environment variables
    set_environment_variables()
    
    # Check dependencies
    dependencies_ok = check_dependencies()
    
    if dependencies_ok:
        logger.info("Setup completed successfully.")
        logger.info("You can now run the system with: streamlit run app.py")
    else:
        logger.warning("Setup completed with warnings. Please resolve the issues above.")
    
    return True

if __name__ == "__main__":
    main() 