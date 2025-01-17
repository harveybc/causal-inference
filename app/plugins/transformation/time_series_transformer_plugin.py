import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any
from app.logger import get_logger

logger = get_logger(__name__)

class Plugin:
    """
    A transformation plugin to generate hourly time series of trend and volatility
    based on trained inference models and events in a sliding window.
    """

    plugin_params = {
        'window_size': 128,  # Number of hours in the sliding window
        'model_dir': 'trained_models/',  # Directory containing inference models
        'output_file': 'output_timeseries.csv',  # File to save the output time series
        'decay_factor': 0.5,  # Weight decay factor for events based on time
    }

    plugin_debug_vars = ['window_size', 'model_dir', 'decay_factor', 'output_file']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.models = {}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def load_models(self):
        """
        Load all trained models from the specified directory.
        """
        model_dir = self.params['model_dir']
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

        logger.info(f"Loading models from {model_dir}")
        for file_name in os.listdir(model_dir):
            if file_name.endswith('.pkl'):
                model_path = os.path.join(model_dir, file_name)
                with open(model_path, 'rb') as f:
                    self.models[file_name] = pickle.load(f)
        logger.info(f"Loaded {len(self.models)} models from {model_dir}")

    def transform(self, causal_effects: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms causal effects into a time series of trend and volatility.

        Parameters
        ----------
        causal_effects : pd.DataFrame
            DataFrame containing causal effects with columns for timestamp,
            treatment effects (trend and volatility), and event-specific data.

        Returns
        -------
        pd.DataFrame
            Time series DataFrame with trend and volatility values for each hourly tick.
        """
        logger.info("Starting transformation of causal effects into time series.")

        # Parameters
        window_size = self.params['window_size']
        decay_factor = self.params['decay_factor']

        # Ensure timestamps are sorted
        causal_effects = causal_effects.sort_values(by='timestamp')

        # Generate hourly time series index
        start_time = causal_effects['timestamp'].min()
        end_time = causal_effects['timestamp'].max()
        hourly_index = pd.date_range(start=start_time, end=end_time, freq='H')

        # Initialize the output DataFrame
        time_series = pd.DataFrame(index=hourly_index, columns=['trend', 'volatility'])
        time_series = time_series.fillna(0)

        # Process each tick
        for current_time in hourly_index:
            # Identify events within the sliding window
            window_start = current_time - pd.Timedelta(hours=window_size)
            window_events = causal_effects[(causal_effects['timestamp'] > window_start) & 
                                           (causal_effects['timestamp'] <= current_time)]

            # Calculate trend and volatility influence
            trend_score = self._calculate_influence(window_events, 'trend_effect', decay_factor, current_time)
            volatility_score = self._calculate_influence(window_events, 'volatility_effect', decay_factor, current_time)

            time_series.loc[current_time, 'trend'] = trend_score
            time_series.loc[current_time, 'volatility'] = volatility_score

        # Reset index and save to file if required
        time_series.reset_index(inplace=True)
        time_series.rename(columns={'index': 'timestamp'}, inplace=True)

        output_file = self.params.get('output_file')
        if output_file:
            time_series.to_csv(output_file, index=False)
            logger.info(f"Transformed time series saved to: {output_file}")

        logger.info("Time series transformation completed successfully.")
        return time_series

    def _calculate_influence(self, events: pd.DataFrame, effect_column: str, decay_factor: float, current_time: pd.Timestamp) -> float:
        """
        Calculate the aggregate influence for a specified effect column (e.g., trend or volatility).

        Parameters
        ----------
        events : pd.DataFrame
            DataFrame of events within the current time window.
        effect_column : str
            Column representing the effect (e.g., 'trend_effect' or 'volatility_effect').
        decay_factor : float
            Factor controlling the exponential decay of influence over time.
        current_time : pd.Timestamp
            The current timestamp for which influence is being calculated.

        Returns
        -------
        float
            The aggregated influence score.
        """
        total_influence = 0.0
        for _, event in events.iterrows():
            time_diff = (current_time - event['timestamp']).total_seconds() / 3600.0  # Time difference in hours
            weight = np.exp(-decay_factor * time_diff)  # Exponential decay
            influence = weight * event[effect_column]
            total_influence += influence
        return total_influence
