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
    """
    logger.info(f"Loading input data from: {config['input_file']}")
    input_data = load_csv(config['input_file'], headers=config.get('headers', True))
    logger.info(f"Input data loaded with shape: {input_data.shape}")

    # Ensure timestamp column
    if 'timestamp' not in input_data.columns:
        logger.warning("Missing 'timestamp' column in input data. Generating timestamps.")
        input_data['timestamp'] = pd.date_range(
            start=config.get('start_date', '2020-01-01'),
            periods=len(input_data),
            freq='H'
        )

    # Target column handling
    target_column = config.get("target_column", "target")
    if target_column not in input_data.columns:
        logger.error(f"Target column '{target_column}' missing from input data.")
        raise ValueError(f"Target column '{target_column}' not found.")

    x_train = input_data.drop(columns=[target_column, 'timestamp'], errors='ignore')
    y_train = input_data[[target_column]]

    logger.debug(f"Processed data shapes - Inputs: {x_train.shape}, Targets: {y_train.shape}")
    return x_train, y_train


def run_causal_pipeline(config: dict) -> None:
    """
    Executes the causal inference pipeline using specified plugins.
    """
    logger.info("Starting causal inference pipeline.")

    try:
        # Load and initialize preprocessing plugin
        preprocessor_class, _ = load_plugin(
            'causal_inference.preprocessing', config['preprocessing_plugin']
        )
        preprocessor_plugin = preprocessor_class()

        # Preprocess the data
        x_train, y_train = process_data(config)
        preprocessed_data = preprocessor_plugin.process(x_train, y_train)

        # Load and initialize inference plugin
        inference_class, _ = load_plugin(
            'causal_inference.inference', config['inference_plugin']
        )
        inference_plugin = inference_class()

        inference_plugin.set_params(**config)
        causal_effects = inference_plugin.estimate_effects(
            preprocessed_data['x'], preprocessed_data['y']
        )

        # Load and initialize transformation plugin
        transformation_class, _ = load_plugin(
            'causal_inference.transformation', config['transformation_plugin']
        )
        transformation_plugin = transformation_class()

        transformed_data = transformation_plugin.transform(causal_effects)
        write_csv(config['output_file'], transformed_data, headers=config.get('headers', True))
        logger.info(f"Transformed data saved to: {config['output_file']}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise



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
    preprocessor_class, _ = load_plugin('causal_inference.preprocessing', config['preprocessor_plugin'])
    preprocessor_plugin = preprocessor_class()

    # Step 2: Preprocess the data
    try:
        x_train, y_train = process_data(config)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

    # Step 3: Load and apply the inference plugin
    logger.info("Loading and initializing the inference plugin.")
    inference_class, _ = load_plugin('causal_inference.inference', config['inference_plugin'])
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


def causal_validation_metrics(inference_plugin, x_train, y_train):
    """
    Computes validation metrics for the causal model.

    Parameters
    ----------
    inference_plugin : object
        The inference plugin used to estimate causal effects.
    x_train : pd.DataFrame
        Training input data.
    y_train : pd.DataFrame
        Training target data.

    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    logger.info("Validating causal model...")

    # Example metric: R-squared
    try:
        predictions = inference_plugin.predict(x_train)
        residuals = y_train.squeeze() - predictions.squeeze()
        r_squared = 1 - (residuals.var() / y_train.squeeze().var())

        # Example additional metrics
        mse = (residuals**2).mean()
        mae = residuals.abs().mean()

        metrics = {
            "R-squared": r_squared,
            "MSE": mse,
            "MAE": mae,
        }

        logger.debug(f"Validation metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error during validation metrics calculation: {e}")
        raise
