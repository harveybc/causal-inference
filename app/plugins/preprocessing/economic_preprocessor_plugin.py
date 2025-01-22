# economic_preprocessor_plugin.py

import pandas as pd
import itertools
import os
from app.logger import get_logger

logger = get_logger(__name__)

class Plugin:
    """
    Economic Preprocessor Plugin

    This plugin processes economic data by filtering based on specified
    countries and impact levels, and prepares the dataset for causal inference.
    """

    plugin_params = {
        'columns_to_combine': ['country', 'event_description'],  # Adjusted to correct column names
        'target_columns': ['Trend', 'Volatility'],  # Columns representing outcomes
        'heterogeneous_columns': ['forecast_data', 'actual_data'],  # Event-specific data
        'filtered_output_path': 'processed_data/',  # Where to save filtered datasets
        'key_countries': ['USA', 'Germany', 'France', 'UK', 'Italy'],  # Key countries
        'impact_levels': ['high', 'medium'],  # Impact levels to filter
    }

    plugin_debug_vars = ['columns_to_combine', 'target_columns', 'heterogeneous_columns', 'key_countries', 'impact_levels']

    def __init__(self):
        self.params = self.plugin_params.copy()

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        """
        Returns debugging information for the plugin's configuration.
        """
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        """
        Adds plugin-specific debugging information to a provided debug_info dictionary.
        """
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the economic data by filtering based on key countries and impact levels.

        Parameters
        ----------
        data : pd.DataFrame
            The input dataset containing economic event data.

        Returns
        -------
        pd.DataFrame
            Filtered and processed DataFrame ready for causal inference.
        """
        logger.info("Starting economic data preprocessing.")

        # Filter by key countries
        logger.info(f"Filtering data for key countries: {self.params['key_countries']}")
        data_filtered = data[data['country'].isin(self.params['key_countries'])]

        # Filter by impact levels
        logger.info(f"Filtering data for impact levels: {self.params['impact_levels']}")
        data_filtered = data_filtered[data_filtered['volatility_degree'].isin(self.params['impact_levels'])]

        logger.debug(f"Data after filtering by country and impact: {data_filtered.shape}")

        # Assign combined columns for grouping (if needed)
        combined_columns = self.params['columns_to_combine']
        logger.info(f"Combining columns for grouping: {combined_columns}")

        # If specific combinations are needed, adjust here
        # Currently, we're returning the filtered DataFrame as a single group
        # since sliding window is handled in data_processor.py

        # Ensure output directory exists
        os.makedirs(self.params['filtered_output_path'], exist_ok=True)

        # Save the filtered dataset
        output_file = os.path.join(self.params['filtered_output_path'], "filtered_events.csv")
        data_filtered.to_csv(output_file, index=False)
        logger.info(f"Filtered events saved to {output_file}")

        return data_filtered
