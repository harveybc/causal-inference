# double_ml_plugin.py

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor
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
        'model_t': GradientBoostingRegressor(),  # Changed to Regressor for continuous treatments
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

    def estimate_effects(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate causal effects using Double Machine Learning.

        Parameters
        ----------
        x_train : pd.DataFrame
            The feature set.
        y_train : pd.DataFrame
            The target variables (Trend and Volatility).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing estimated effects for each target.
        """
        required_treatments = ['sorpresa', 'distancia']
        for treat in required_treatments:
            if treat not in x_train.columns:
                raise ValueError(f"The input data must include a '{treat}' column.")

        treatments = x_train[required_treatments]
        features = x_train.drop(columns=required_treatments)

        logger.info("Initializing LinearDML model.")
        self.model = LinearDML(
            model_y=self.params['model_y'],
            model_t=self.params['model_t'],
            featurizer=self.params['featurizer'],
            discrete_treatment=False,  # Treatments are continuous
            cv=self.params['cv'],
            random_state=self.params['random_state']
        )

        logger.info("Fitting LinearDML model.")
        self.model.fit(y_train, treatments, X=features)

        logger.info("Estimating treatment effects.")
        treatment_effects = self.model.effect(X=features)

        logger.info("Creating DataFrame of treatment effects.")
        effects_df = pd.DataFrame(treatment_effects, columns=y_train.columns)
        lower, upper = self.model.effect_interval(X=features)
        effects_df['confidence_interval_low'] = lower
        effects_df['confidence_interval_high'] = upper

        return effects_df

    # The remaining methods (save_model, load_model, transform) remain unchanged
