#plugin_loader.py
from importlib import metadata
from app.logger import get_logger  # Import the centralized logger

logger = get_logger(__name__)

def load_plugin(plugin_group: str, plugin_name: str):
    """
    Dynamically loads a plugin class from the specified entry point group.

    Parameters:
    ----------
    plugin_group : str
        The entry point group under which the plugin is registered.
    plugin_name : str
        The name of the plugin to load.

    Returns:
    -------
    tuple:
        - The plugin class.
        - A list of required parameter names for the plugin.

    Raises:
    ------
    ImportError:
        If the plugin cannot be found or loaded.
    """
    logger.debug(f"Attempting to load plugin '{plugin_name}' from group '{plugin_group}'.")

    try:
        # Retrieve all entry points for the specified group
        entry_points = metadata.entry_points()
        if hasattr(entry_points, 'select'):  # For Python >=3.10
            group_entries = entry_points.select(group=plugin_group)
        else:  # For older Python versions
            group_entries = entry_points.get(plugin_group, [])

        logger.debug(f"Found {len(group_entries)} entries in group '{plugin_group}'.")

        # Find the entry point with the specified plugin name
        entry_point = next((ep for ep in group_entries if ep.name == plugin_name), None)
        if entry_point is None:
            logger.error(f"Plugin '{plugin_name}' not found in group '{plugin_group}'.")
            raise ImportError(f"Plugin '{plugin_name}' not found in group '{plugin_group}'.")

        # Load the plugin class
        plugin_class = entry_point.load()
        logger.debug(f"Successfully loaded plugin class '{plugin_class.__name__}' from '{entry_point.module}'.")

        # Retrieve required parameters if available
        required_params = list(getattr(plugin_class, 'plugin_params', {}).keys())
        logger.debug(f"Plugin '{plugin_name}' has parameters: {required_params}")

        return plugin_class, required_params

    except ImportError as ie:
        logger.exception(f"ImportError while loading plugin '{plugin_name}' from group '{plugin_group}': {ie}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error while loading plugin '{plugin_name}' from group '{plugin_group}': {e}")
        raise

def load_preprocessing_plugin(plugin_name: str):
    """
    Loads a preprocessing plugin by name.

    Parameters:
    ----------
    plugin_name : str
        The name of the preprocessing plugin to load.

    Returns:
    -------
    tuple:
        - The preprocessing plugin class.
        - A list of required parameter names for the plugin.
    """
    return load_plugin('causal_inference.preprocessing', plugin_name)

def load_inference_plugin(plugin_name: str):
    """
    Loads an inference plugin by name.

    Parameters:
    ----------
    plugin_name : str
        The name of the inference plugin to load.

    Returns:
    -------
    tuple:
        - The inference plugin class.
        - A list of required parameter names for the plugin.
    """
    return load_plugin('causal_inference.inference', plugin_name)

def load_transformation_plugin(plugin_name: str):
    """
    Loads a transformation plugin by name.

    Parameters:
    ----------
    plugin_name : str
        The name of the transformation plugin to load.

    Returns:
    -------
    tuple:
        - The transformation plugin class.
        - A list of required parameter names for the plugin.
    """
    return load_plugin('causal_inference.transformation', plugin_name)

def get_plugin_params(plugin_group: str, plugin_name: str):
    """
    Retrieves the parameters of a specified plugin.

    Parameters:
    ----------
    plugin_group : str
        The entry point group under which the plugin is registered.
    plugin_name : str
        The name of the plugin whose parameters are to be retrieved.

    Returns:
    -------
    dict:
        A dictionary of parameter names and their default values.

    Raises:
    ------
    ImportError:
        If the plugin cannot be found or loaded.
    """
    logger.debug(f"Getting parameters for plugin '{plugin_name}' from group '{plugin_group}'.")

    try:
        # Retrieve all entry points for the specified group
        entry_points = metadata.entry_points()
        if hasattr(entry_points, 'select'):  # For Python >=3.10
            group_entries = entry_points.select(group=plugin_group)
        else:  # For older Python versions
            group_entries = entry_points.get(plugin_group, [])

        logger.debug(f"Found {len(group_entries)} entries in group '{plugin_group}'.")

        # Find the entry point with the specified plugin name
        entry_point = next((ep for ep in group_entries if ep.name == plugin_name), None)
        if entry_point is None:
            logger.error(f"Plugin '{plugin_name}' not found in group '{plugin_group}'.")
            raise ImportError(f"Plugin '{plugin_name}' not found in group '{plugin_group}'.")

        # Load the plugin class
        plugin_class = entry_point.load()
        logger.debug(f"Successfully loaded plugin class '{plugin_class.__name__}' from '{entry_point.module}'.")

        # Retrieve plugin parameters
        plugin_params = getattr(plugin_class, 'plugin_params', {})
        logger.debug(f"Retrieved parameters for plugin '{plugin_name}': {plugin_params}")

        return plugin_params

    except ImportError as ie:
        logger.exception(f"ImportError while retrieving parameters for plugin '{plugin_name}' from group '{plugin_group}': {ie}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error while retrieving parameters for plugin '{plugin_name}' from group '{plugin_group}': {e}")
        raise
