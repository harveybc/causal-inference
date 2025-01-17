import json
import sys
import requests
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin

def load_config(file_path):
    """
    Load a configuration file from the specified path.

    :param file_path: Path to the configuration file.
    :return: A dictionary containing the configuration.
    """
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def get_plugin_default_params(plugin_group, plugin_name):
    """
    Retrieve the default parameters for a given plugin.

    :param plugin_group: The plugin group (e.g., 'causal_inference.preprocessing').
    :param plugin_name: The name of the plugin.
    :return: A dictionary containing the plugin's default parameters.
    """
    plugin_class, _ = load_plugin(plugin_group, plugin_name)
    plugin_instance = plugin_class()
    return plugin_instance.plugin_params

def compose_config(config):
    """
    Compose the configuration to save by filtering out default values.

    :param config: The current configuration dictionary.
    :return: A dictionary with only the non-default configuration values.
    """
    preprocessing_name = config.get('preprocessing_plugin', DEFAULT_VALUES.get('preprocessing_plugin'))
    inference_name = config.get('inference_plugin', DEFAULT_VALUES.get('inference_plugin'))
    transformation_name = config.get('transformation_plugin', DEFAULT_VALUES.get('transformation_plugin'))

    preprocessing_default_params = get_plugin_default_params('causal_inference.preprocessing', preprocessing_name)
    inference_default_params = get_plugin_default_params('causal_inference.inference', inference_name)
    transformation_default_params = get_plugin_default_params('causal_inference.transformation', transformation_name)

    config_to_save = {}
    for k, v in config.items():
        if k not in DEFAULT_VALUES or v != DEFAULT_VALUES[k]:
            if (k not in preprocessing_default_params or v != preprocessing_default_params[k]) and \
               (k not in inference_default_params or v != inference_default_params[k]) and \
               (k not in transformation_default_params or v != transformation_default_params[k]):
                config_to_save[k] = v

    print(f"Filtered config to save: {config_to_save}")
    return config_to_save

def save_config(config, path='config_out.json'):
    """
    Save the configuration to a local file.

    :param config: The configuration dictionary to save.
    :param path: Path to the output file.
    :return: A tuple containing the saved configuration and the path.
    """
    config_to_save = compose_config(config)
    with open(path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    return config, path

def save_debug_info(debug_info, path='debug_out.json'):
    """
    Save debugging information to a local file.

    :param debug_info: Debugging information to save.
    :param path: Path to the output file.
    """
    with open(path, 'w') as f:
        json.dump(debug_info, f, indent=4)

def remote_save_config(config, url, username, password):
    """
    Save the configuration to a remote endpoint.

    :param config: The configuration dictionary to save.
    :param url: The endpoint URL.
    :param username: Username for authentication.
    :param password: Password for authentication.
    :return: True if the configuration was successfully saved, False otherwise.
    """
    config_to_save = compose_config(config)
    try:
        response = requests.post(
            url,
            auth=(username, password),
            data={'json_config': json.dumps(config_to_save)}
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Failed to save remote configuration: {e}", file=sys.stderr)
        return False

def remote_load_config(url, username=None, password=None):
    """
    Load a configuration from a remote endpoint.

    :param url: The endpoint URL.
    :param username: (Optional) Username for authentication.
    :param password: (Optional) Password for authentication.
    :return: A dictionary containing the configuration, or None if an error occurs.
    """
    try:
        if username and password:
            response = requests.get(url, auth=(username, password))
        else:
            response = requests.get(url)
        response.raise_for_status()
        config = response.json()
        return config
    except requests.RequestException as e:
        print(f"Failed to load remote configuration: {e}", file=sys.stderr)
        return None

def remote_log(config, debug_info, url, username, password):
    """
    Log the configuration and debug information to a remote endpoint.

    :param config: The configuration dictionary to log.
    :param debug_info: Debugging information to log.
    :param url: The endpoint URL.
    :param username: Username for authentication.
    :param password: Password for authentication.
    :return: True if the log was successfully sent, False otherwise.
    """
    config_to_save = compose_config(config)
    try:
        data = {
            'json_config': json.dumps(config_to_save),
            'json_result': json.dumps(debug_info)
        }
        response = requests.post(
            url,
            auth=(username, password),
            data=data
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Failed to log remote information: {e}", file=sys.stderr)
        return False
