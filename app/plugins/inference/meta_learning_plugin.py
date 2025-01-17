import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import itertools
import pickle

class Plugin:
    """
    Meta-learning plugin for training models to estimate expected outcomes.

    This plugin prepares and trains models for combinations of specified columns
    and stores the trained models for later use in transformation plugins.
    """

    plugin_params = {
        'max_combination_size': 3,  # Maximum number of columns to combine
        'columns_to_combine': ['Country', 'Event Type', 'Degree of Volatility'],
        'target_columns': ['Trend', 'Volatility'],  # Columns to predict
        'model_output_path': 'trained_models/',  # Path to save models
    }

    plugin_debug_vars = ['max_combination_size', 'columns_to_combine', 'target_columns']

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

    def train_models(self, data: pd.DataFrame):
        """
        Train models for combinations of specified columns to predict target outcomes.

        Parameters
        ----------
        data : pd.DataFrame
            The input dataset containing the features and targets.

        Raises
        ------
        ValueError
            If required columns are missing from the dataset.
        """
        # Validate dataset columns
        required_columns = set(self.params['columns_to_combine'] + self.params['target_columns'])
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in the dataset: {missing_columns}")

        # Prepare combinations of columns
        max_comb_size = self.params['max_combination_size']
        column_combinations = []
        for r in range(1, max_comb_size + 1):
            column_combinations.extend(itertools.combinations(self.params['columns_to_combine'], r))

        # Train a model for each combination of columns
        for combination in column_combinations:
            combination_name = '_'.join(combination)
            self.models[combination_name] = {}
            
            for target in self.params['target_columns']:
                model_key = f"{combination_name}_to_{target}"

                # Prepare the training data
                X = pd.get_dummies(data[list(combination)], drop_first=True)
                y = data[target]

                # Train linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Store the model and training error
                predictions = model.predict(X)
                mse = mean_squared_error(y, predictions)
                self.models[combination_name][target] = {
                    'model': model,
                    'features': X.columns.tolist(),
                    'mse': mse,
                }
                
                print(f"Trained model: {model_key} with MSE: {mse:.4f}")

    def save_models(self):
        """
        Save the trained models to the specified output path.

        Raises
        ------
        Exception
            If saving models fails.
        """
        output_path = self.params['model_output_path']
        try:
            for combination, targets in self.models.items():
                for target, model_info in targets.items():
                    model_file = f"{output_path}{combination}_to_{target}.pkl"
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
                combination, target = model_path.split('/')[-1].replace('.pkl', '').rsplit('_to_', 1)
                if combination not in self.models:
                    self.models[combination] = {}
                self.models[combination][target] = model_info
                print(f"Model loaded: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading models: {e}")
