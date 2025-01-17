import pandas as pd
from typing import Tuple
from app.data_handler import load_csv, write_csv
from app.plugin_loader import load_plugin
from app.logger import get_logger

logger = get_logger(__name__)

def process_data(config: dict, preprocessor_plugin) -> pd.DataFrame:
    """
    Processes input data using the preprocessor plugin.

    This function loads the specified dataset, assigns headers if missing,
    and applies the preprocessor plugin to filter and prepare the data for causal inference.

    Parameters
    ----------
    config : dict
        Configuration parameters.
    preprocessor_plugin : object
        The preprocessor plugin instance.

    Returns
    -------
    pd.DataFrame
        Preprocessed data.
    """
    logger.info(f"Loading input data from: {config['input_file']}")
    input_data = load_csv(config['input_file'], headers=False)
    logger.info(f"Input data loaded with shape: {input_data.shape}")

    # Assign headers based on dataset structure
    logger.info("Assigning headers to input data.")
    input_data.columns = [
        'event_date',         # Event date
        'event_time',         # Event time (New York time)
        'country',            # Country of the event
        'volatility_degree',  # Degree of volatility
        'event_description',  # Description of the event
        'evaluation',         # Evaluation (better/worse/same as forecast)
        'data_format',        # Data format (%, K, M, T)
        'actual_data',        # Actual event data
        'forecast_data',      # Forecasted event data
        'previous_data'       # Previous data with possible comments
    ]

    # Strip whitespace from string columns
    for col in ['country', 'volatility_degree', 'event_description', 'evaluation', 'data_format']:
        input_data[col] = input_data[col].str.strip()

    # Ensure numeric columns are properly parsed
    numeric_cols = ['actual_data', 'forecast_data', 'previous_data']
    for col in numeric_cols:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

    logger.debug(f"Processed dataset after assigning headers and cleaning: {input_data.head()}")

    # Preprocess the data
    logger.info("Applying preprocessor plugin to filter and prepare the data.")
    preprocessed_data = preprocessor_plugin.process(input_data)
    logger.debug(f"Preprocessed data shape: {preprocessed_data.shape}")
    return preprocessed_data




def align_with_hourly_dataset(hourly_dataset: str, time_series: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns the generated time series with the hourly dataset.

    Parameters
    ----------
    hourly_dataset : str
        Path to the hourly dataset CSV file.
    time_series : pd.DataFrame
        DataFrame containing the generated time series.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with the same datetime index as the hourly dataset.
    """
    logger.info(f"Loading hourly dataset from: {hourly_dataset}")
    hourly_data = load_csv(hourly_dataset, headers=True)
    logger.info(f"Hourly dataset loaded with shape: {hourly_data.shape}")

    # Ensure datetime alignment
    hourly_data['timestamp'] = pd.to_datetime(hourly_data['datetime'], errors='coerce')
    time_series['timestamp'] = pd.to_datetime(time_series['timestamp'], errors='coerce')

    aligned_data = pd.merge(hourly_data[['timestamp']], time_series, on='timestamp', how='inner')
    logger.info(f"Aligned time series with shape: {aligned_data.shape}")
    return aligned_data




def run_causal_pipeline(config: dict) -> None:
    """
    Executes the causal inference pipeline using specified plugins.

    The pipeline integrates the preprocessor, inference, and transformation plugins
    to produce a time series of trend and volatility influences aligned with the hourly dataset.
    """
    logger.info("Starting causal inference pipeline.")

    try:
        # Load and initialize preprocessing plugin
        preprocessor_class, _ = load_plugin(
            'causal_inference.preprocessing', config['preprocessing_plugin']
        )
        preprocessor_plugin = preprocessor_class()

        # Preprocess the data
        preprocessed_data = process_data(config, preprocessor_plugin)

        # Load and initialize inference plugin
        inference_class, _ = load_plugin(
            'causal_inference.inference', config['inference_plugin']
        )
        inference_plugin = inference_class()
        inference_plugin.set_params(**config)

        # Estimate causal effects
        logger.info("Estimating causal effects using the inference plugin.")
        causal_effects = inference_plugin.estimate_effects(preprocessed_data)
        logger.debug(f"Causal effects estimated with shape: {causal_effects.shape}")

        # Load and initialize transformation plugin
        transformation_class, _ = load_plugin(
            'causal_inference.transformation', config['transformation_plugin']
        )
        transformation_plugin = transformation_class()
        transformation_plugin.set_params(**config)

        # Transform causal effects into time series
        logger.info("Transforming causal effects into time series using the transformation plugin.")
        transformed_data = transformation_plugin.transform(causal_effects)
        logger.debug(f"Transformed data shape: {transformed_data.shape}")

        # Align with hourly dataset
        aligned_data = align_with_hourly_dataset(config['hourly_dataset'], transformed_data)

        # Save the output
        write_csv(config['output_file'], aligned_data, headers=True)
        logger.info(f"Aligned time series saved to: {config['output_file']}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise




def evaluate_causal_model(config: dict) -> None:
    """
    Evaluates a causal model using provided data and metrics.

    This function validates the causal model by computing relevant metrics.
    """
    logger.info("Evaluating causal model.")

    try:
        # Load and initialize preprocessing plugin
        preprocessor_class, _ = load_plugin(
            'causal_inference.preprocessing', config['preprocessing_plugin']
        )
        preprocessor_plugin = preprocessor_class()

        # Preprocess the data
        preprocessed_data = process_data(config, preprocessor_plugin)

        # Load and initialize inference plugin
        inference_class, _ = load_plugin(
            'causal_inference.inference', config['inference_plugin']
        )
        inference_plugin = inference_class()
        inference_plugin.set_params(**config)

        # Validate the causal model
        logger.info("Validating causal model using validation metrics.")
        metrics = causal_validation_metrics(inference_plugin, preprocessed_data)
        logger.info("Causal model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")

    except Exception as e:
        logger.error(f"Error during causal model evaluation: {e}")
        raise


def causal_validation_metrics(inference_plugin, preprocessed_data: pd.DataFrame) -> dict:
    """
    Computes validation metrics for the causal model.

    Parameters
    ----------
    inference_plugin : object
        The inference plugin used to estimate causal effects.
    preprocessed_data : pd.DataFrame
        The preprocessed input data.

    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    logger.info("Validating causal model...")
    try:
        predictions = inference_plugin.predict(preprocessed_data)
        actuals = preprocessed_data[['Trend', 'Volatility']]  # Example targets
        residuals = actuals - predictions

        # Compute metrics
        r_squared = 1 - (residuals.var() / actuals.var()).mean()
        mse = (residuals**2).mean().mean()
        mae = residuals.abs().mean().mean()

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
