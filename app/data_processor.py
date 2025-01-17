# app/data_processor.py

import pandas as pd
from typing import Tuple
from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
from app.plugin_loader import load_plugin
from app.logger import get_logger

logger = get_logger(__name__)

def process_data(config: dict, preprocessor_plugin) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes input data for causal inference, including preprocessing.

    This function prepares data by loading the specified CSV files, applying
    preprocessing via the preprocessor plugin, and ensuring data consistency.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    preprocessor_plugin : object
        The loaded preprocessor plugin.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Preprocessed input data (features)
        - Target data (outcomes)

    Raises
    ------
    ValueError
        If data shapes do not match or required data is missing.
    """
    logger.info(f"Loading raw input data from: {config['x_train_file']}")
    raw_data = load_csv(config['x_train_file'], headers=config['headers'])
    logger.info(f"Raw input data loaded with shape: {raw_data.shape}")

    logger.info(f"Preprocessing input data using plugin: {config['preprocessor_plugin']}")
    preprocessor_plugin.set_params(**config)
    preprocessed_data = preprocessor_plugin.process_and_transform(raw_data)
    logger.info(f"Preprocessed input data shape: {preprocessed_data.shape}")

    logger.info(f"Loading target data from: {config['y_train_file']}")
    y_train_data = load_csv(config['y_train_file'], headers=config['headers'])
    logger.info(f"Target data loaded with shape: {y_train_data.shape}")

    # Ensure data integrity
    if len(preprocessed_data) != len(y_train_data):
        logger.warning("Input and target data lengths do not match. Trimming to minimum length.")
        min_length = min(len(preprocessed_data), len(y_train_data))
        preprocessed_data = preprocessed_data[:min_length]
        y_train_data = y_train_data[:min_length]

    logger.debug(f"Processed data shapes - Inputs: {preprocessed_data.shape}, Targets: {y_train_data.shape}")
    return preprocessed_data, y_train_data


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
