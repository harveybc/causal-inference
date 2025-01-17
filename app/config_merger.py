import sys
from typing import List, Dict, Any
from app.config import DEFAULT_VALUES, ARGUMENT_MAPPING
from app.logger import get_logger

logger = get_logger(__name__)

def process_unknown_args(unknown_args: List[str]) -> Dict[str, Any]:
    """
    Processes unknown command-line arguments into a configuration dictionary.

    Parameters
    ----------
    unknown_args : List[str]
        List of unknown command-line arguments.

    Returns
    -------
    Dict[str, Any]
        Dictionary of processed unknown arguments.
    """
    logger.debug("Starting to process unknown command-line arguments.")
    processed_args = {}
    i = 0
    while i < len(unknown_args):
        key = unknown_args[i].lstrip('-')
        value = unknown_args[i + 1] if i + 1 < len(unknown_args) else None

        # Convert short-form to long-form using the mapping
        if key in ARGUMENT_MAPPING:
            original_key = key
            key = ARGUMENT_MAPPING[key]
            logger.debug(f"Converted short-form argument '{original_key}' to long-form '{key}'.")

        processed_args[key] = value
        logger.debug(f"Processed argument: {key} = {value}")
        i += 2

    logger.debug(f"Completed processing unknown arguments: {processed_args}")
    return processed_args

def convert_type(value: Any) -> Any:
    """
    Attempts to convert a value to int or float. Returns the original value if conversion fails.

    Parameters
    ----------
    value : Any
        The value to convert.

    Returns
    -------
    Any
        The converted value or the original value if conversion fails.
    """
    logger.debug(f"Attempting to convert value: {value}")
    if value is None:
        logger.debug("Value is None; returning as is.")
        return value

    try:
        converted = int(value)
        logger.debug(f"Converted value to int: {converted}")
        return converted
    except (ValueError, TypeError):
        try:
            converted = float(value)
            logger.debug(f"Converted value to float: {converted}")
            return converted
        except (ValueError, TypeError):
            logger.debug(f"Value remains as string: {value}")
            return value

def merge_config(defaults: Dict[str, Any],
                 plugin_params: Dict[str, Any],
                 config: Dict[str, Any],
                 cli_args: Dict[str, Any],
                 unknown_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges configuration dictionaries with the following precedence:
    CLI arguments > Unknown arguments > File configuration > Plugin parameters > Default configuration.

    Parameters
    ----------
    defaults : Dict[str, Any]
        The default configuration parameters.
    plugin_params : Dict[str, Any]
        Plugin-specific configuration parameters.
    config : Dict[str, Any]
        Configuration loaded from files (local or remote).
    cli_args : Dict[str, Any]
        Configuration parameters passed via command-line arguments.
    unknown_args : Dict[str, Any]
        Additional configuration parameters not recognized by the CLI.

    Returns
    -------
    Dict[str, Any]
        The merged configuration dictionary.
    """
    logger.debug("Starting configuration merge process.")

    # Step 1: Start with default values from config.py
    merged_config = defaults.copy()
    logger.debug(f"Step 1 - Defaults: {merged_config}")

    # Step 2: Merge with plugin default parameters
    for key, value in plugin_params.items():
        logger.debug(f"Step 2 - Merging plugin_param '{key}' = {value}")
        merged_config[key] = value
    logger.debug(f"Step 2 Output: {merged_config}")

    # Step 3: Merge with file configuration
    for key, value in config.items():
        logger.debug(f"Step 3 - Merging from file config: '{key}' = {value}")
        merged_config[key] = value
    logger.debug(f"Step 3 Output: {merged_config}")

    # Step 4: Merge with CLI arguments (CLI args always override)
    for key, value in cli_args.items():
        if value is not None:
            logger.debug(f"Step 4 - Merging from CLI args: '{key}' = {value}")
            merged_config[key] = value

    # Step 5: Merge with unknown arguments
    for key, value in unknown_args.items():
        if value is not None:
            converted_value = convert_type(value)
            logger.debug(f"Step 5 - Merging from unknown args: '{key}' = {converted_value}")
            merged_config[key] = converted_value

    logger.debug(f"Final merged configuration: {merged_config}")
    return merged_config
