#!/usr/bin/env python
"""
Setup script for configuring OpenAI API key from a text file.
"""
import os
import sys
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_key_from_file(file_path):
    """
    Extract OpenAI API key from a text file.
    """
    try:
        with open(file_path, 'r') as file:
            # Read the file and get the first non-empty line
            for line in file:
                line = line.strip()
                if line:
                    # Remove any quotes or extra characters
                    key = line.strip('"\'')
                    # Basic validation - API keys are typically long strings
                    if len(key) > 30 and ('sk-' in key or key.startswith('org-')):
                        return key
                    elif line.startswith('OPENAI_API_KEY='):
                        # Handle .env format
                        key = line.split('=', 1)[1].strip().strip('"\'')
                        if len(key) > 30:
                            return key
                    else:
                        # For very short files, return first line anyway
                        if len(line) > 10:
                            return key
            
            # Check for JSON format
            file.seek(0)
            content = file.read()
            if '{' in content and '}' in content:
                try:
                    import json
                    data = json.loads(content)
                    # Check common JSON key names
                    for key_name in ['api_key', 'apiKey', 'key', 'secret', 'OPENAI_API_KEY']:
                        if key_name in data and data[key_name]:
                            key = data[key_name]
                            if isinstance(key, str) and len(key) > 30:
                                return key
                except:
                    pass
            
            # If we reach here, couldn't find a key
            return None
    except Exception as e:
        logger.error(f"Error reading key file: {e}")
        return None

def update_env_file(api_key):
    """
    Update the .env file with the OpenAI API key.
    """
    root_dir = Path(__file__).parent.absolute()
    env_path = os.path.join(root_dir, ".env")
    
    # Read existing .env file
    env_content = []
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as env_file:
                env_content = env_file.readlines()
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
            return False
    
    # Update or add the API key
    api_key_set = False
    for i, line in enumerate(env_content):
        if line.startswith('OPENAI_API_KEY='):
            env_content[i] = f'OPENAI_API_KEY={api_key}\n'
            api_key_set = True
            break
    
    if not api_key_set:
        env_content.append(f'OPENAI_API_KEY={api_key}\n')
        # Also make sure USE_LOCAL_SEARCH is set to false
        local_search_set = False
        for line in env_content:
            if line.startswith('USE_LOCAL_SEARCH='):
                local_search_set = True
                break
        
        if not local_search_set:
            env_content.append('USE_LOCAL_SEARCH=false\n')
    
    # Write the updated content back to the .env file
    try:
        with open(env_path, 'w') as env_file:
            env_file.writelines(env_content)
        return True
    except Exception as e:
        logger.error(f"Error writing to .env file: {e}")
        return False

def main():
    """
    Main function to set up the API key.
    """
    parser = argparse.ArgumentParser(description="Set up OpenAI API key from a text file")
    parser.add_argument("--file", "-f", help="Path to the file containing the OpenAI API key")
    parser.add_argument("--key", "-k", help="Directly specify the OpenAI API key")
    args = parser.parse_args()
    
    print("OpenAI API Key Setup")
    print("====================")
    
    # Check if a key was provided directly
    if args.key:
        api_key = args.key
        print("Using API key provided via command line.")
    else:
        # Get file path if specified
        file_path = args.file
        
        # If no file path provided, ask for it
        if not file_path:
            file_path = input("\nEnter the path to the file containing your OpenAI API key: ")
        
        # Exit if no path provided
        if not file_path:
            print("No file path provided. Setup canceled.")
            return False
        
        # Expand user directory if needed
        file_path = os.path.expanduser(file_path)
        
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"Error: File not found at {file_path}")
            return False
        
        print(f"Using file: {file_path}")
        
        # Read the API key from the file
        api_key = get_key_from_file(file_path)
        
        if not api_key:
            print("Could not find a valid API key in the file.")
            # Ask if user wants to enter key directly
            manual_key = input("Would you like to enter your API key manually? (y/n): ")
            if manual_key.lower() == 'y':
                api_key = input("Enter your OpenAI API key: ")
            else:
                return False
    
    # Verify we have a key
    if not api_key:
        print("No API key provided. Setup canceled.")
        return False
    
    # Mask the API key for display
    masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    print(f"API key: {masked_key}")
    
    # Ask for confirmation
    confirm = input("Do you want to use this API key? (y/n): ")
    if confirm.lower() != 'y':
        print("Setup canceled.")
        return False
    
    # Update the .env file
    if update_env_file(api_key):
        print("\nSuccessfully updated .env file with your API key.")
        
        # Set environment variable for current session
        os.environ["OPENAI_API_KEY"] = api_key
        print("API key has been set for the current session.")
        
        print("\nYou can now run the PDF analysis system with LLM capabilities:")
        print("  streamlit run pdfAnswersingAnalysis/src/app.py")
        
        return True
    else:
        print("\nFailed to update .env file. Please try again or manually edit the .env file.")
        return False

if __name__ == "__main__":
    main() 