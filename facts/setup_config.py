#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from src.config import config

def setup_config(api_key=None, model_id=None, verbose=True):
    """
    Set up configuration for the application.
    
    Args:
        api_key: OpenAI API key
        model_id: Fine-tuned model ID
        verbose: Whether to print status messages
    """
    # Get API key if not provided
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter your OpenAI API key: ")
    
    # Get model ID if not provided        
    if not model_id:
        model_id = input("Enter your fine-tuned model ID (leave empty for default 'gpt-3.5-turbo'): ")
        if not model_id:
            model_id = "gpt-3.5-turbo"
    
    # Set configuration
    config.set_openai_api_key(api_key)
    config.set("fine_tuned_model_id", model_id)
    config.save_config()
    
    if verbose:
        print(f"Configuration saved successfully.")
        print(f"API key: {'*' * (len(api_key) - 8) + api_key[-8:] if api_key else 'Not set'}")
        print(f"Model ID: {model_id}")
        print(f"Configuration file: {config.config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up configuration for the factuality verification system")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model-id", help="Fine-tuned model ID")
    parser.add_argument("--quiet", action="store_true", help="Suppress status messages")
    
    args = parser.parse_args()
    
    setup_config(api_key=args.api_key, model_id=args.model_id, verbose=not args.quiet) 