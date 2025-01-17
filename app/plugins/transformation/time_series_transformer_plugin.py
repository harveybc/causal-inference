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

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Transform events into an hourly time series of trend and volatility.

        Parameters
        ----------
        events : pd.DataFrame
            DataFrame containing the event data with timestamps.

        Returns
        -------
        pd.DataFrame
            Hourly time series of trend and volatility.
        """
        logger.info("Transforming events into time series.")

        if not self.trend_models or not self.volatility_models:
            self.load_models()

        window_size = self.params['window_size']
        time_index = pd.date_range(
            start=events['timestamp'].min(),
            end=events['timestamp'].max(),
            freq='H'
        )
        
        results = []

        for current_time in time_index:
            # Select events within the time window
            window_start = current_time - pd.Timedelta(hours=window_size)
            window_events = events[(events['timestamp'] > window_start) & (events['timestamp'] <= current_time)]

            if window_events.empty:
                results.append({'timestamp': current_time, 'trend': 0.0, 'volatility': 0.0})
                continue

            trend_score = self._calculate_score(window_events, self.trend_models, 'trend')
            volatility_score = self._calculate_score(window_events, self.volatility_models, 'volatility')

            results.append({
                'timestamp': current_time,
                'trend': trend_score,
                'volatility': volatility_score
            })

        time_series = pd.DataFrame(results)

        # Save the resulting time series
        output_file = self.params['output_file']
        time_series.to_csv(output_file, index=False)
        logger.info(f"Time series saved to {output_file}")

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
