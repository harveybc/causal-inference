# app/data_processor.py

import pandas as pd
from typing import Tuple
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
from app.plugin_loader import load_plugin
from app.logger import get_logger

logger = get_logger(__name__)

def process_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes input data for causal inference and transformation plugins.

    This function prepares data by loading the specified CSV files, applying any
    necessary transformations, and ensuring data consistency, including timestamps.

    Parameters
    ----------
    config : dict
        Configuration parameters.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Input data (features)
        - Target data (outcomes)
    """
    logger.info(f"Loading input data from: {config['x_train_file']}")
    x_train_data = load_csv(config['x_train_file'], headers=config['headers'])
    logger.info(f"Input data loaded with shape: {x_train_data.shape}")

    logger.info(f"Loading target data from: {config['y_train_file']}")
    y_train_data = load_csv(config['y_train_file'], headers=config['headers'])
    logger.info(f"Target data loaded with shape: {y_train_data.shape}")

    # Ensure timestamp column is present
    if 'timestamp' not in x_train_data.columns:
        logger.warning("'timestamp' column missing in input data. Generating sequential timestamps.")
        x_train_data['timestamp'] = pd.date_range(start=config['start_date'], periods=len(x_train_data), freq='H')

    # Ensure data integrity
    if len(x_train_data) != len(y_train_data):
        logger.warning("Input and target data lengths do not match. Trimming to minimum length.")
        min_length = min(len(x_train_data), len(y_train_data))
        x_train_data = x_train_data[:min_length]
        y_train_data = y_train_data[:min_length]

    logger.debug(f"Processed data shapes - Inputs: {x_train_data.shape}, Targets: {y_train_data.shape}")
    return x_train_data, y_train_data


def run_causal_pipeline(config: dict) -> None:
    """
    Executes the causal inference pipeline using specified plugins.

    This function orchestrates:
    - Preprocessing with the preprocessor plugin.
    - Causal inference using the inference plugin.
    - Time-series transformation using the transformation plugin.

    Parameters
    ----------
    config : dict
        Configuration parameters.

    Raises
    ------
    Exception
        If any step in the pipeline fails.
    """
    logger.info("Starting causal inference pipeline.")

    # Step 1: Load and apply the preprocessor plugin
    logger.info("Loading and initializing the preprocessor plugin.")
    preprocessor_class, _ = load_plugin('causal_inference.preprocessors', config['preprocessor_plugin'])
    preprocessor_plugin = preprocessor_class()

    # Step 2: Preprocess the data
    try:
        x_train, y_train = process_data(config, preprocessor_plugin)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

    # Step 3: Load and apply the inference plugin
    logger.info("Loading and initializing the inference plugin.")
    inference_class, _ = load_plugin('causal_inference.inferencers', config['inference_plugin'])
    inference_plugin = inference_class()

    logger.info("Applying causal inference methods.")
    try:
        inference_plugin.set_params(**config)
        causal_effects = inference_plugin.estimate_effects(x_train, y_train)
        logger.info(f"Causal effects estimated successfully. First few effects:\n{causal_effects.head()}")
    except Exception as e:
        logger.error(f"Error during causal inference: {e}")
        raise

    # Step 4: Load and apply the transformation plugin
    logger.info("Loading and initializing the transformation plugin.")
    transformation_class, _ = load_plugin('causal_inference.transformers', config['transformation_plugin'])
    transformation_plugin = transformation_class()

    logger.info("Transforming causal effects into time-series data.")
    try:
        transformation_plugin.set_params(**config)
        transformed_data = transformation_plugin.transform(causal_effects)
        logger.info(f"Transformation completed. Transformed data shape: {transformed_data.shape}")
    except Exception as e:
        logger.error(f"Error during transformation: {e}")
        raise

    # Save transformed data
    if config.get('output_file'):
        try:
            write_csv(config['output_file'], transformed_data, headers=config['headers'])
            logger.info(f"Transformed data saved to: {config['output_file']}")
        except Exception as e:
            logger.error(f"Error saving transformed data: {e}")
            raise

    logger.info("Causal inference pipeline completed successfully.")


def evaluate_causal_model(config: dict) -> None:
    """
    Evaluates a causal model using provided data and metrics.

    This function computes causal metrics and validates the model's performance.

    Parameters
    ----------
    config : dict
        Configuration parameters.

    Raises
    ------
    Exception
        If any step in evaluation fails.
    """
    logger.info("Evaluating causal model.")

    # Step 1: Load and apply the preprocessor plugin
    logger.info("Loading and initializing the preprocessor plugin.")
    preprocessor_class, _ = load_plugin('causal_inference.preprocessors', config['preprocessor_plugin'])
    preprocessor_plugin = preprocessor_class()

    # Step 2: Preprocess the data
    try:
        x_train, y_train = process_data(config, preprocessor_plugin)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

    # Step 3: Load and apply the inference plugin
    logger.info("Loading and initializing the inference plugin.")
    inference_class, _ = load_plugin('causal_inference.inferencers', config['inference_plugin'])
    inference_plugin = inference_class()

    try:
        inference_plugin.set_params(**config)
        metrics = causal_validation_metrics(inference_plugin, x_train, y_train)
        logger.info("Causal model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")
    except Exception as e:
        logger.error(f"Error during causal model evaluation: {e}")
        raise
