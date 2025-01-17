import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from app.logger import get_logger

logger = get_logger(__name__)

class Plugin:
    """
    A transformation plugin to generate hourly time series of trend and volatility
    based on trained inference models and events in a time window.
    """

    plugin_params = {
        'window_size': 128,  # Number of ticks in the time window
        'trend_model_dir': 'models/trend',
        'volatility_model_dir': 'models/volatility',
        'output_file': 'output_timeseries.csv',
    }

    plugin_debug_vars = ['window_size', 'trend_model_dir', 'volatility_model_dir']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.trend_models = {}
        self.volatility_models = {}
        
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def load_models(self):
        """Load trend and volatility models from the specified directories."""
        trend_dir = self.params['trend_model_dir']
        volatility_dir = self.params['volatility_model_dir']

        logger.info(f"Loading trend models from {trend_dir}")
        self.trend_models = self._load_models_from_directory(trend_dir)
        logger.info(f"Loaded {len(self.trend_models)} trend models.")

        logger.info(f"Loading volatility models from {volatility_dir}")
        self.volatility_models = self._load_models_from_directory(volatility_dir)
        logger.info(f"Loaded {len(self.volatility_models)} volatility models.")

    def _load_models_from_directory(self, directory: str) -> Dict[str, Any]:
        models = {}
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist.")
            return models

        for file_name in os.listdir(directory):
            if file_name.endswith('.pkl'):
                model_path = os.path.join(directory, file_name)
                with open(model_path, 'rb') as f:
                    models[file_name] = pickle.load(f)
        return models

    def transform(self, causal_effects: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Converts causal effects into an hourly time series of trend and volatility influence.

        Parameters
        ----------
        causal_effects : pd.DataFrame
            DataFrame containing causal effects estimated by the inference plugin.
        kwargs : dict
            Additional parameters like window size and distance decay factor.

        Returns
        -------
        pd.DataFrame
            Hourly time series DataFrame with trend and volatility influences.
        """
        logger.info("Starting transformation of causal effects into time series.")

        # Parameters
        window_size = self.params.get('window_size', 128)
        decay_factor = self.params.get('decay_factor', 0.5)

        # Ensure timestamps are sorted
        causal_effects = causal_effects.sort_values(by='timestamp')

        # Generate hourly time series index
        start_time = causal_effects['timestamp'].min()
        end_time = causal_effects['timestamp'].max()
        hourly_index = pd.date_range(start=start_time, end=end_time, freq='H')

        # Initialize output DataFrame
        time_series = pd.DataFrame(index=hourly_index, columns=['trend', 'volatility'])
        time_series = time_series.fillna(0)

        # Process each tick
        for tick_time in hourly_index:
            # Select events within the window
            window_start = tick_time - pd.Timedelta(hours=window_size)
            window_events = causal_effects[(causal_effects['timestamp'] > window_start) & (causal_effects['timestamp'] <= tick_time)]

            # Calculate influence using distance decay
            for _, event in window_events.iterrows():
                time_diff = (tick_time - event['timestamp']).total_seconds() / 3600  # Time difference in hours
                weight = np.exp(-decay_factor * time_diff)

                # Update trend and volatility influences
                time_series.loc[tick_time, 'trend'] += weight * event['trend_effect']
                time_series.loc[tick_time, 'volatility'] += weight * event['volatility_effect']

        # Reset index and export as CSV if required
        time_series.reset_index(inplace=True)
        time_series.rename(columns={'index': 'timestamp'}, inplace=True)

        if 'output_file' in kwargs:
            output_file = kwargs['output_file']
            time_series.to_csv(output_file, index=False)
            logger.info(f"Transformed time series saved to: {output_file}")

        logger.info("Time series transformation completed successfully.")
        return time_series


    def _calculate_score(self, events: pd.DataFrame, models: Dict[str, Any], target: str) -> float:
        """
        Calculate the aggregate score for a given target (trend/volatility).

        Parameters
        ----------
        events : pd.DataFrame
            Events within the current time window.
        models : Dict[str, Any]
            Loaded models for the specified target.
        target : str
            Target for which to calculate the score ('trend' or 'volatility').

        Returns
        -------
        float
            Calculated score.
        """
        total_score = 0.0

        for _, event in events.iterrows():
            for model_name, model in models.items():
                try:
                    feature_vector = event[model['features']].values.reshape(1, -1)
                    prediction = model['model'].predict(feature_vector)[0]
                    weight = self._calculate_weight(event['timestamp'], model_name)
                    total_score += prediction * weight
                except Exception as e:
                    logger.warning(f"Error in model {model_name} for target {target}: {e}")

        return total_score

    def _calculate_weight(self, event_time: pd.Timestamp, model_name: str) -> float:
        """
        Calculate the weight of an event based on its distance from the current tick.

        Parameters
        ----------
        event_time : pd.Timestamp
            Timestamp of the event.
        model_name : str
            Name of the model being applied.

        Returns
        -------
        float
            Weight for the event.
        """
        current_time = pd.Timestamp.now()
        time_diff = (current_time - event_time).total_seconds() / 3600.0  # Difference in hours

        # Example weight function: exponential decay
        weight = np.exp(-time_diff / self.params['window_size'])

        return weight
