import logging
import os


def load_secrets(desired_keys:dict, file_path="secrets/secrets.txt"):
    """
    Utility function to extract provided ("desired") keys.
    
    Input:
        desired_keys: Very important that this matches the way the key/-s are written in the "secret"-file. Eg.: 'API_KEY_CAP'
    
    Returns:
        secrets{} with key-value-pair
    """
    desired_keys = desired_keys
    secrets = {}
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if "=" in line:  # Ensure valid format
                key, value = line.split("=", 1)  # Split at the first '='
                if key in desired_keys:
                    secrets[key] = value
    
    logging.info(f"Loaded credentials. Length: {len(secrets)}")
    # logging.debug(f"Content of secrets: {secrets}")

    return secrets

def ensure_path_exists(path:str):
    """
    Check if the given path exists and create it if it doesn't.
    
    Args:
        path (str): The directory path to check/create
        
    Returns:
        bool: True if path exists or was created successfully, False otherwise
    """
    try:
        # Check if the path already exists
        if not os.path.exists(path):
            # Create all directories in the path
            os.makedirs(path)
            logging.info(f"Created directory: {path}")
        return True
    except Exception as e:
        logging.error(f"Error creating directory {path}: {e}")
        return False