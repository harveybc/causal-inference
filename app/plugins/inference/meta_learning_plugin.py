import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import pickle


class Plugin:
    """
    Economic Inference Plugin

    This plugin trains models for combinations of specified columns to estimate
    the impact of treatments on target outcomes such as Trend and Volatility.
    """

    plugin_params = {
        'max_combination_size': 2,  # Maximum number of columns to combine
        'columns_to_combine': ['Country', 'Event Type'],  # Columns defining treatments
        'target_columns': ['Trend', 'Volatility'],  # Outcome columns
        'heterogeneous_columns': ['Forecast', 'Actual'],  # Features for treatment heterogeneity
        'model_output_path': 'trained_models/',  # Directory to save trained models
    }

    plugin_debug_vars = ['max_combination_size', 'columns_to_combine', 'target_columns', 'heterogeneous_columns']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.models = {}

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

    def train_models(self, data_dict: dict):
        """
        Train models for each combination of columns to predict target outcomes.

        Parameters
        ----------
        data_dict : dict
            A dictionary where keys are combination names (e.g., 'Country_Event Type')
            and values are DataFrames filtered by the preprocessing plugin.

        Raises
        ------
        ValueError
            If required columns are missing from any filtered dataset.
        """
        # Ensure output directory exists
        os.makedirs(self.params['model_output_path'], exist_ok=True)

        for combination_name, data in data_dict.items():
            print(f"Processing dataset for combination: {combination_name}")

            # Validate required columns
            required_columns = set(self.params['target_columns'] + self.params['heterogeneous_columns'])
            missing_columns = required_columns - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in the dataset for {combination_name}: {missing_columns}")

            self.models[combination_name] = {}

            for target in self.params['target_columns']:
                # Prepare training data
                X = data[self.params['heterogeneous_columns']]
                y = data[target]

                # Train linear regression model
                model = LinearRegression()
                model.fit(X, y)

                # Calculate training error
                predictions = model.predict(X)
                mse = mean_squared_error(y, predictions)

                # Store the model and training error
                self.models[combination_name][target] = {
                    'model': model,
                    'features': self.params['heterogeneous_columns'],
                    'mse': mse,
                }

                print(f"Trained model for {combination_name} -> {target} with MSE: {mse:.4f}")

    def save_models(self):
        """
        Save all trained models to the specified output directory.

        Raises
        ------
        Exception
            If saving models fails.
        """
        try:
            for combination, targets in self.models.items():
                for target, model_info in targets.items():
                    model_file = os.path.join(
                        self.params['model_output_path'], f"{combination}_to_{target}.pkl"
                    )
                    with open(model_file, 'wb') as f:
                        pickle.dump(model_info, f)
                    print(f"Model saved: {model_file}")
        except Exception as e:
            raise Exception(f"Error saving models: {e}")

    def load_models(self, model_paths: list):
        """
        Load pre-trained models from specified file paths.

        Parameters
        ----------
        model_paths : list
            List of file paths for the models to load.

        Raises
        ------
        Exception
            If loading models fails.
        """
        try:
            for model_path in model_paths:
                with open(model_path, 'rb') as f:
                    model_info = pickle.load(f)

                # Extract combination and target from file name
                combination, target = model_path.split('/')[-1].replace('.pkl', '').rsplit('_to_', 1)
                if combination not in self.models:
                    self.models[combination] = {}
                self.models[combination][target] = model_info

                print(f"Model loaded: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading models: {e}")

    def get_models(self):
        """
        Return the trained models for use in downstream tasks.

        Returns
        -------
        dict
            Dictionary of trained models.
        """
        return self.models
