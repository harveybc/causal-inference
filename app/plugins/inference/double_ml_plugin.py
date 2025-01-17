import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from typing import Dict
from app.logger import get_logger

logger = get_logger(__name__)

class Plugin:
    """
    A plugin for Double Machine Learning (DML) using EconML.
    """

    plugin_params = {
        'model_y': GradientBoostingRegressor(),
        'model_t': GradientBoostingClassifier(),
        'featurizer': None,
        'cv': 5,
        'random_state': 42
    }

    plugin_debug_vars = ['cv', 'random_state']

    def __init__(self):
        self.params = self.plugin_params.copy()
        self.model = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params[var] for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        plugin_debug_info = self.get_debug_info()
        debug_info.update(plugin_debug_info)

    def estimate_effects(self, x_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """
        Estimate causal effects using Double Machine Learning.

        Parameters
        ----------
        x_train : pd.DataFrame
            The feature set.
        y_train : pd.Series
            The target variable.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing estimated effects.
        """
        if 'treatment' not in x_train.columns:
            raise ValueError("The input data must include a 'treatment' column.")

        treatment = x_train['treatment']
        features = x_train.drop(columns=['treatment'])

        logger.info("Initializing LinearDML model.")
        self.model = LinearDML(
            model_y=self.params['model_y'],
            model_t=self.params['model_t'],
            featurizer=self.params['featurizer'],
            discrete_treatment=True,
            cv=self.params['cv'],
            random_state=self.params['random_state']
        )

        logger.info("Fitting LinearDML model.")
        self.model.fit(y_train, treatment, X=features)

        logger.info("Estimating treatment effects.")
        treatment_effects = self.model.effect(X=features)

        logger.info("Creating DataFrame of treatment effects.")
        effects_df = pd.DataFrame({
            'treatment_effect': treatment_effects,
            'confidence_interval_low': self.model.effect_interval(X=features)[0],
            'confidence_interval_high': self.model.effect_interval(X=features)[1]
        })

        return effects_df

    def save_model(self, file_path: str):
        """
        Save the trained model to a file.

        Parameters
        ----------
        file_path : str
            Path to save the model.
        """
        import joblib
        if self.model is None:
            raise ValueError("No model is available to save. Please train the model first.")
        joblib.dump(self.model, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str):
        """
        Load a trained model from a file.

        Parameters
        ----------
        file_path : str
            Path to the model file.
        """
        import joblib
        self.model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")

    def transform(self, causal_effects: pd.DataFrame) -> pd.DataFrame:
        """
        Transform causal effects into time series format.

        Parameters
        ----------
        causal_effects : pd.DataFrame
            DataFrame of causal effects.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame in time series format.
        """
        logger.info("Transforming causal effects to time series.")
        causal_effects['time'] = pd.date_range(start='2020-01-01', periods=len(causal_effects), freq='H')
        return causal_effects.set_index('time')
