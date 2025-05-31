import inspect
import logging
import os
import sys
import yaml


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

def ensure_path_exists(path : str):
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
    
def get_available_filename(directory:str, filename:str):
    """
    Check if a file exists in the given directory. If it does, add an incremental
    number to the filename until an available filename is found.
    
    Args:
        directory (str): The directory path to check
        filename (str): The original filename to check
    
    Returns:
        str: An available filename that doesn't exist in the directory
    """
    # Get the base name and extension
    base, ext = os.path.splitext(filename)
    
    # Check if the original file exists
    if not os.path.exists(os.path.join(directory, filename)):
        return filename
    
    # Try with incremental numbers
    counter = 1
    while True:
        new_filename = f"{base}-{counter}{ext}"
        if not os.path.exists(os.path.join(directory, new_filename)):
            return new_filename
        counter += 1
        
def handle_hypothesis_source(config):
    """
    Function for checking provided config.yml source for hypothesis, and returning the name and features.
    
    Args:
        config: Path for the config.yml, specifying hypothesis and selected features.
    
    returns:
        hypothesis_name, features object
    """
    # Load YAML file
    try:
        with open(config, 'r') as file:
            config_h = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Error: Config file '{config}' not found!")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)
    
    # Extract values
    hypothesis_name = config_h['hypothesis']['name']
    features = config_h['required_features']
    
    logging.info(f"Hypothesis: {hypothesis_name}")
    logging.debug(f"Features: {features}")
    
    return hypothesis_name, features

def list_of_functions_in_class(path_to_class:str, never_mind_functions:list = ['__init__', 'run', 'safely_execute']):
    """
    A simple function that gets and returns a list of existing functions based on the
    provided path.
    
    Args:
        - path_to_class: Path to the file and class of the desired functions list. Eg. 'my_package.my_module.MyClass'
        - never_min_functions: Optional. Functions you want excluded in the returned list. Default: ['__init__', 'run', 'safely_execute']
    
    Returns:
        - List of existing funtions in the provided class (excluding never_min_functions).
    """
    logging.debug(f"Starting shared utility method list_of_functions_in_class().")
    
    # List to populate with functions
    functions = []
    
    try:
        # Strip path to class and import it
        components = path_to_class.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        
        # Loop through functions in the class, and add them to the list (as long they are not part of never_mind_functions)
        try:
            for method in inspect.getmembers(mod, predicate=inspect.isfunction):
                if method[0] not in never_mind_functions:  # Won't add the 'basic' funtions, only the hypothesis ones
                    functions.append(method[0])
        except Exception as e:
            logging.error(f"Unable to add functions to list: {e}")
        
    except Exception as e:
        logging.error(f"Unable to open and import provided path_to_class, {path_to_class}. Error: {e}")
    
    return functions