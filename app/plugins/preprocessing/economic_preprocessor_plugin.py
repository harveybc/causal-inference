import pandas as pd
import itertools
import os

class Plugin:
    """
    Economic Preprocessor Plugin

    This plugin processes economic data by filtering and splitting the dataset
    based on specified combinations of columns. It prepares subsets of data
    with relevant features for further analysis in the inference and transformation plugins.
    """

    plugin_params = {
        'columns_to_combine': ['Country', 'Event Type'],
        'target_columns': ['Trend', 'Volatility'],  # Columns representing outcomes
        'heterogeneous_columns': ['Forecast', 'Actual'],  # Event-specific data
        'filtered_output_path': 'processed_data/',  # Where to save filtered datasets
    }

    plugin_debug_vars = ['columns_to_combine', 'target_columns', 'heterogeneous_columns']

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

    def preprocess(self, data: pd.DataFrame):
        """
        Preprocess the economic data by splitting it into subsets based on specified combinations of columns.

        Parameters
        ----------
        data : pd.DataFrame
            The input dataset containing economic event data.

        Returns
        -------
        dict
            A dictionary containing filtered datasets for each combination of columns.
        """
        # Validate required columns
        required_columns = set(self.params['columns_to_combine'] + self.params['target_columns'] + self.params['heterogeneous_columns'])
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in the dataset: {missing_columns}")

        # Ensure output directory exists
        os.makedirs(self.params['filtered_output_path'], exist_ok=True)

        # Generate combinations of specified columns
        column_combinations = list(itertools.combinations(self.params['columns_to_combine'], len(self.params['columns_to_combine'])))
        processed_data = {}

        for combination in column_combinations:
            combination_name = '_'.join(combination)
            filtered_data = data.groupby(list(combination), group_keys=True).apply(lambda x: x.reset_index(drop=True))

            # Save the filtered dataset
            output_file = os.path.join(self.params['filtered_output_path'], f"{combination_name}_filtered.csv")
            filtered_data.to_csv(output_file, index=False)
            processed_data[combination_name] = filtered_data

            print(f"Processed and saved dataset for combination: {combination_name} to {output_file}")

        return processed_data
