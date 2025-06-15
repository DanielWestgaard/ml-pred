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
        
def handle_hypothesis_source(config_file: str) -> dict:
    """
    Function for checking provided config.yml source for hypothesis, and returning the configurations it finds.
    
    Args:
        - config_file: Path to config.yml, specifying hypothesis (-name), selected features, etc.
    
    returns:
        - list of successfully ectracted 'components'/configurations in Dict format.
            - Eg: '{'hypothesis_name': 'session_based_tod_anomalies', 'raw_data_source': 'bla/raw_data.csv', 'features': {'time_of_day': [], 'time_of_week': []}'
            - Can be accessed like this: features = shared_utils.handle_hypothesis_source.get('features') 
    """
    # Load YAML file
    try:
        with open(config_file, 'r') as file:
            config_h = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Error: Config file '{config_file}' not found!")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)
    
    # Dict to populate with successfully extracted 'values'
    dict = {}
    
    # Hypothesis name - must match with the name of its target method in target_hypothesis.py
    try:
        hypothesis_name = config_h['hypothesis']['name']
        dict.update({'hypothesis_name': hypothesis_name})
        logging.info(f"hypothesis_name: {hypothesis_name}")
    except Exception as e:
        logging.error(f"Unable to get hypothesis name : {e}")
        sys.exit()
    
    # Data Pipeline
    try:
        # Raw data
        raw_data_source = config_h['data_pipeline']['raw_source']
        dict.update({'raw_data_source': raw_data_source})
        try:
            # Provided features
            features = config_h['data_pipeline']['required_features']
            dict.update({"features": features})
            logging.debug(f"Data source: '{raw_data_source}' and required Features: {features}")
        except Exception as e:
            logging.error(f"Unable to extract Required/Selected Features for generation : {e}")
    except Exception as e:
        logging.warning(f"Unable to extract Raw Dataset, for Data Pipeline, from configurations (might not exist?) : {e}")
    
    # Model Training
    try:
        # Model Type
        model_type = config_h['model_training']['type']
        dict.update({'model_type': model_type})
        logging.info(f"model_type: {model_type}")
        
        try:
            # Potential Processed Data
            processed_data = config_h['model_training']['processed_data']
            dict.update({"processed_data": processed_data})
            logging.debug(f"processed_data: {processed_data}")
        except Exception as e:
            logging.warning(f"Unable to extract processed_data for model training, but they might also not be there : {e}")
        
        try:
            # Potential Parameters
            parameters = config_h['model_training']['parameters']
            dict.update({"parameters": parameters})
            logging.debug(f"parameters: {parameters}")
        except Exception as e:
            logging.warning(f"Unable to extract Parameters for model training, but they might also not be there : {e}")
        
    except Exception as e:
        logging.warning(f"Unable to get model_type from configuration, might not be there? : {e}")

    # Evaluation
    try:
        # Evaluate model
        evaluate_model = config_h['evaluate_model']
        dict.update({"evaluate_model": evaluate_model})
        logging.debug(f"evaluate_model: {evaluate_model}")
        try:
            evaluation_type = config_h['evaluate_model']['type']
            dict.update({"evaluation_type": evaluation_type})
            logging.debug(f"evaluation_type: {evaluation_type}")
            try:
                evaluation_model_path = config_h['evaluate_model']['model_folder']
                dict.update({"model_folder": evaluation_model_path})
                logging.debug(f"model_folder: {evaluation_model_path}")
            except:
                logging.warning(f"Unable to get evaluation model folder. Might having type 'new'?")
        except:
            logging.warning(f"Unable to get evaluation type.")
    except Exception as e:
        logging.warning(f"Unable to extract evaluation from model training, but might also not be there : {e}")
    
    return dict

def safely_execute(self:object, function_name:str):  # Similar to the one in features/generation.py
    """
    Safely execute a function method with proper error handling.
    The reason for this need is to dynamically run a function given some input or need, or looping through
    specific functions.
    
    Args:
        - self: instance of the class calling this method.
        - function_name: the name of the function you want to run.
    
    Returns:
        - True (success) or False (error) based on success.
    """
    logging.debug(f"Starting shared utility method safely_execute().")
    
    try:
        # Run the method
        method = getattr(self, function_name)
        method()
        
        logging.debug(f"Successfully ran function {function_name}().")
        return True
    except Exception as e:
        logging.error(f"Error trying to run {function_name} method: {e}")
        return False

def list_of_functions_in_class(path_to_class:str, never_mind_functions:list = ['__init__', 'run', 'safely_execute']):
    """
    A simple function that gets and returns a list of existing functions based on the
    provided path.
    Note: There is room for improvements in this implementation!
    
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
            logging.debug(f"Successfully populated list with functions in provided class! {functions}")
        except Exception as e:
            logging.error(f"Unable to add functions to list: {e}")
        
    except Exception as e:
        logging.error(f"Unable to open and import provided path_to_class, {path_to_class}. Error: {e}")
    
    return functions