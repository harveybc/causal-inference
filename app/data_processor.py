# app/data_processor.py

import pandas as pd
import numpy as np
import os
import time
import json
import pickle
from typing import Tuple, Optional

from app.data_handler import load_csv, write_csv
from app.config_handler import save_debug_info, remote_log
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.utils import (
    calculate_effect_sizes,
    calculate_causal_metrics,
    causal_validation_metrics
)
from app.logger import get_logger

logger = get_logger(__name__)

def process_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes input data for causal inference.

    This function prepares data by loading the specified CSV files, applying any
    necessary transformations, and ensuring data consistency.

    Parameters
    ----------
    config : dict
        Configuration parameters.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Input data (features)
        - Target data (outcomes)

    Raises
    ------
    ValueError
        If data shapes do not match or required data is missing.
    """
    logger.info(f"Loading input data from: {config['x_train_file']}")
    x_train_data = load_csv(config['x_train_file'], headers=config['headers'])
    logger.info(f"Input data loaded with shape: {x_train_data.shape}")

    logger.info(f"Loading target data from: {config['y_train_file']}")
    y_train_data = load_csv(config['y_train_file'], headers=config['headers'])
    logger.info(f"Target data loaded with shape: {y_train_data.shape}")

    # Ensure data integrity
    if len(x_train_data) != len(y_train_data):
        logger.warning("Input and target data lengths do not match. Trimming to minimum length.")
        min_length = min(len(x_train_data), len(y_train_data))
        x_train_data = x_train_data[:min_length]
        y_train_data = y_train_data[:min_length]

    logger.debug(f"Processed data shapes - Inputs: {x_train_data.shape}, Targets: {y_train_data.shape}")
    return x_train_data, y_train_data


def run_causal_pipeline(config: dict, inference_plugin, transformation_plugin) -> None:
    """
    Executes the causal inference pipeline based on the provided configuration.

    This function:
    - Processes input and target data.
    - Applies causal inference methods using the specified plugins.
    - Saves the transformed series or logs debug information.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    inference_plugin : object
        The loaded causal inference plugin.
    transformation_plugin : object
        The loaded transformation plugin.

    Raises
    ------
    Exception
        If any step in the pipeline fails.
    """
    logger.info("Processing data for causal inference.")
    try:
        x_train, y_train = process_data(config)
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        raise

    logger.info("Applying causal inference methods.")
    try:
        inference_plugin.set_params(**config)
        causal_effects = inference_plugin.estimate_effects(x_train, y_train)
        logger.info(f"Causal effects estimated successfully. First few effects:\n{causal_effects.head()}")
    except Exception as e:
        logger.error(f"Error during causal inference: {e}")
        raise

    logger.info("Applying transformation to produce time series outputs.")
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
            output_path = config['output_file']
            write_csv(output_path, transformed_data, headers=config['headers'])
            logger.info(f"Transformed data saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving transformed data to {output_path}: {e}")
            raise

    logger.info("Causal inference pipeline completed successfully.")


def evaluate_causal_model(config: dict, inference_plugin) -> None:
    """
    Evaluates a causal model using provided data and metrics.

    This function computes causal metrics and validates the model's performance.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    inference_plugin : object
        The loaded inference plugin.

    Raises
    ------
    Exception
        If any step in evaluation fails.
    """
    logger.info("Evaluating causal model.")
    try:
        x_train, y_train = process_data(config)
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        raise

    try:
        metrics = causal_validation_metrics(inference_plugin, x_train, y_train)
        logger.info("Causal model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")
    except Exception as e:
        logger.error(f"Error during causal model evaluation: {e}")
        raise
