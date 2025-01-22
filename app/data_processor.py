# data_processor.py

import pandas as pd
import numpy as np
from typing import Tuple
from app.data_handler import load_csv, write_csv
from app.plugin_loader import load_plugin
from app.logger import get_logger

logger = get_logger(__name__)

def process_data(config: dict, preprocessor_plugin) -> pd.DataFrame:
    """
    Processes input data using the preprocessor plugin.

    This function loads the specified dataset, dynamically assigns headers based on the dataset,
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
        Preprocessed data with treatment features.
    """
    logger.info(f"Loading input data from: {config['input_file']}")
    input_data = load_csv(config['input_file'], headers=False)
    logger.info(f"Input data loaded with shape: {input_data.shape}")

    # Dynamically assign headers based on the dataset's number of columns
    logger.info("Assigning headers to input data.")
    expected_headers = [
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
    if input_data.shape[1] != len(expected_headers):
        logger.warning(f"Dataset column count mismatch: Expected {len(expected_headers)}, Found {input_data.shape[1]}.")
        expected_headers = expected_headers[:input_data.shape[1]]

    input_data.columns = expected_headers
    logger.debug(f"Assigned headers: {input_data.columns.tolist()}")

    # Strip whitespace only from string columns
    string_columns = ['country', 'volatility_degree', 'event_description', 'evaluation', 'data_format']
    for col in string_columns:
        if col in input_data.columns:
            input_data[col] = input_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Ensure numeric columns are properly parsed
    numeric_cols = ['actual_data', 'forecast_data', 'previous_data']
    for col in numeric_cols:
        if col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

    logger.debug(f"Processed dataset after assigning headers and cleaning: {input_data.head()}")

    # Preprocess the data
    logger.info("Applying preprocessor plugin to filter and prepare the data.")
    preprocessed_data = preprocessor_plugin.preprocess(input_data)
    logger.debug(f"Preprocessed data shape: {preprocessed_data.shape}")

    # Calcular la señal de entrenamiento Q
    logger.info("Calculating training signal Q (Trend and Volatility).")
    preprocessed_data = calculate_training_signal(config['hourly_dataset'], preprocessed_data)

    # Implementar la ventana deslizante para generar tratamientos
    logger.info("Applying sliding window to generate treatments.")
    preprocessed_data = apply_sliding_window(preprocessed_data, window_size=config.get('window_size', 128))

    logger.debug(f"Data after Q calculation and sliding window: {preprocessed_data.head()}")
    return preprocessed_data

def calculate_training_signal(hourly_dataset: str, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the training signal Q (Trend and Volatility) from the hourly dataset.

    Parameters
    ----------
    hourly_dataset : str
        Path to the hourly dataset CSV file.
    preprocessed_data : pd.DataFrame
        Preprocessed data with filtered events.

    Returns
    -------
    pd.DataFrame
        Preprocessed data with 'Trend' and 'Volatility' columns added.
    """
    logger.info(f"Loading hourly dataset from: {hourly_dataset}")
    hourly_data = load_csv(hourly_dataset, headers=True)
    logger.info(f"Hourly dataset loaded with shape: {hourly_data.shape}")

    # Ensure 'timestamp' is in datetime format
    hourly_data['timestamp'] = pd.to_datetime(hourly_data['datetime'], errors='coerce')
    preprocessed_data['timestamp'] = pd.to_datetime(preprocessed_data['event_date'] + ' ' + preprocessed_data['event_time'], errors='coerce')

    # Merge events with hourly market data
    logger.info("Merging event data with hourly market dataset.")
    merged_data = pd.merge_asof(preprocessed_data.sort_values('timestamp'),
                                hourly_data.sort_values('timestamp'),
                                on='timestamp',
                                direction='backward')

    # Sort by timestamp
    merged_data = merged_data.sort_values('timestamp').reset_index(drop=True)

    # Calculate Q: Trend and Volatility
    logger.info("Calculating Trend and Volatility for each tick.")

    # Calculate Trend: Change in close from current tick to 6 ticks ahead
    merged_data['Trend'] = merged_data['close'].shift(-6) - merged_data['close']

    # Calculate Volatility: Standard deviation of the next 6 closes
    merged_data['Volatility'] = merged_data['close'].rolling(window=6, min_periods=6).std().shift(-6)

    # Drop rows where Q cannot be calculated (last 6 ticks)
    merged_data = merged_data.dropna(subset=['Trend', 'Volatility'])

    logger.debug(f"Data after calculating Q: {merged_data[['timestamp', 'Trend', 'Volatility']].head()}")

    return merged_data

def apply_sliding_window(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Applies a sliding window to generate treatments from events within the window.

    Parameters
    ----------
    data : pd.DataFrame
        Data with events and training signals Q.
    window_size : int
        Size of the sliding window in ticks.

    Returns
    -------
    pd.DataFrame
        Data with generated treatment features.
    """
    logger.info(f"Applying sliding window of size {window_size} ticks.")
    data = data.sort_values('timestamp').reset_index(drop=True)

    # Initialize treatment columns
    data['sorpresa'] = data['actual_data'] - data['forecast_data']

    # Initialize lists to store aggregated treatments
    sorpresa_ponderada = []
    distancia_promedio = []

    # Precompute exponential decay factors
    decay_alpha = 0.1  # You can parameterize this if needed

    for i in range(len(data)):
        current_time = data.loc[i, 'timestamp']
        window_start_time = current_time - pd.Timedelta(hours=window_size)

        # Filter events within the sliding window
        window_events = data[(data['timestamp'] > window_start_time) & (data['timestamp'] <= current_time)]

        if not window_events.empty:
            # Calculate distance in ticks (assuming each tick is one hour)
            window_events = window_events.copy()
            window_events['distancia'] = (current_time - window_events['timestamp']).dt.total_seconds() / 3600  # Distance in ticks

            # Apply exponential decay to sorpresas
            window_events['decay'] = np.exp(-decay_alpha * window_events['distancia'])

            # Calculate weighted sorpresa
            weighted_sorpresa = (window_events['sorpresa'] * window_events['decay']).sum()

            # Calculate average distancia
            avg_distancia = window_events['distancia'].mean()

            sorpresa_ponderada.append(weighted_sorpresa)
            distancia_promedio.append(avg_distancia)
        else:
            # No events in the window
            sorpresa_ponderada.append(0)
            distancia_promedio.append(0)

    # Add the aggregated treatment features to the DataFrame
    data['sorpresa_ponderada'] = sorpresa_ponderada
    data['distancia_promedio'] = distancia_promedio

    logger.debug(f"Aggregated treatment features added: {data[['sorpresa_ponderada', 'distancia_promedio']].head()}")

    return data
