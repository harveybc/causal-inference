"""A bounded, constant-effect binary-treatment ATE adapter for EconML."""

from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
import math
import warnings

ASSUMPTIONS = (
    "sufficient_adjustment", "pre_treatment_adjustment", "consistency",
    "no_interference", "positivity", "iid_sampling", "constant_effect",
    "nuisance_models_correct",
)
MAX_ROWS = 10_000
MAX_ADJUSTMENTS = 20
SEED = 1729


def _response(status, reason=None, payload=None):
    return {
        "schema_version": "causal-inference.provider.v1",
        "family": "causal_inference",
        "output_kind": "causal_effect",
        "status": status,
        "reason": reason,
        "payload": payload,
    }


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class CausalInferenceProvider:
    """One in-memory analysis per instance; not a concurrent job scheduler.

    load(config) never fits. fit(data) explicitly estimates and caches one result.
    infer() returns that result by value, never fits or accepts a new population.
    """

    def __init__(self):
        self._config = None
        self._phase = "NEW"
        self._result = None
        self._analysis_id = None

    @staticmethod
    def capabilities():
        return {
            "provider": "causal_inference",
            "schema_version": "causal-inference.provider.v1",
            "family": "causal_inference",
            "output_kinds": ["causal_effect"],
            "operations": ["load", "fit", "infer"],
            "infer_requires_fit": True,
            "estimands": ["ATE"],
            "contrast": {"control": 0, "treated": 1},
            "input_kinds": ["dataframe", "records"],
            "required_config": ["estimand", "treatment", "outcome", "adjustments", "unit", "assumptions"],
            "required_assumptions": list(ASSUMPTIONS),
            "uncertainty_methods": ["econml_statsmodels_HC1_normal"],
            "limits": {"min_rows": 100, "max_rows": MAX_ROWS,
                       "max_adjustments": MAX_ADJUSTMENTS, "min_per_arm": 20,
                       "numeric_threads": 1},
            "persistent_state": False,
            "text_identification": False,
        }

    @property
    def state(self):
        return deepcopy({"phase": self._phase, "config": self._config,
                         "analysis_id": self._analysis_id})

    def _refuse(self, status, reason):
        self._phase = "REFUSED"
        self._analysis_id = None
        self._result = _response(status, reason)
        return deepcopy(self._result)

    def load(self, config):
        """Load a complete explicit config, invalidating any previous analysis."""
        self._config = self._result = self._analysis_id = None
        self._phase = "NEW"
        if not isinstance(config, dict):
            return self._refuse("INVALID_INPUT", "Config must be a mapping, not text.")
        allowed = {"estimand", "treatment", "outcome", "adjustments", "unit", "assumptions", "alpha"}
        if set(config) - allowed:
            return self._refuse("INVALID_INPUT", "Unknown config fields; no implicit interpretation is supported.")
        if not allowed.difference({"alpha", "assumptions"}).issubset(config):
            return self._refuse("INVALID_INPUT", "Explicit estimand, roles, adjustment list and outcome unit are required.")
        if config["estimand"] != "ATE":
            return self._refuse("UNSUPPORTED_TASK", "Only constant-effect binary-treatment ATE (1 versus 0) is supported.")
        names = [config["treatment"], config["outcome"], config["unit"]]
        if any(not isinstance(n, str) or not n.strip() or len(n) > 128 for n in names):
            return self._refuse("INVALID_INPUT", "Roles and unit must be nonempty strings of at most 128 characters.")
        adjustment = config["adjustments"]
        if not isinstance(adjustment, list) or any(
            not isinstance(n, str) or not n.strip() or len(n) > 128 for n in adjustment
        ):
            return self._refuse("INVALID_INPUT", "Adjustments must be an explicit list of column names; [] is allowed.")
        if len(adjustment) > MAX_ADJUSTMENTS:
            return self._refuse("RESOURCE_EXCEEDED", "At most 20 adjustments are supported.")
        roles = [config["treatment"], config["outcome"], *adjustment]
        if len(set(roles)) != len(roles):
            return self._refuse("INVALID_INPUT", "Treatment, outcome and adjustments must be distinct.")
        assumptions = config.get("assumptions")
        if not isinstance(assumptions, dict) or any(assumptions.get(a) is not True for a in ASSUMPTIONS):
            return self._refuse("NOT_IDENTIFIED", "All identifying and model assumptions must be explicitly declared true.")
        if set(assumptions) != set(ASSUMPTIONS):
            return self._refuse("INVALID_INPUT", "Unknown assumptions cannot be interpreted by this provider.")
        alpha = config.get("alpha", 0.05)
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < 1:
            return self._refuse("INVALID_INPUT", "alpha must be a finite number strictly between 0 and 1.")
        self._config = deepcopy(config)
        self._config["alpha"] = float(alpha)
        self._phase = "CONFIGURED"
        return _response("OK", "Configured only; explicit fit(data) is required.")

    def fit(self, data):
        """Run one explicitly requested IID causal study on numeric records."""
        self._result = self._analysis_id = None
        if self._config is None:
            return self._refuse("MODEL_NOT_FITTED", "Load a valid identifying config before fit.")
        self._phase = "CONFIGURED"
        import numpy as np
        import pandas as pd
        from pandas.api.types import is_complex_dtype, is_numeric_dtype

        if not isinstance(data, (pd.DataFrame, list)):
            return self._refuse("INVALID_INPUT", "Expected a DataFrame or list of records, never a prompt or path.")
        if len(data) > MAX_ROWS:
            return self._refuse("RESOURCE_EXCEEDED", "At most 10000 rows are supported; no automatic subsampling.")
        if len(data) < 100:
            return self._refuse("INVALID_INPUT", "At least 100 rows are required.")
        if isinstance(data, list):
            if any(not isinstance(row, dict) for row in data):
                return self._refuse("INVALID_INPUT", "Every record must be a mapping.")
            data = pd.DataFrame.from_records(data)
        cfg = self._config
        columns = [cfg["treatment"], cfg["outcome"], *cfg["adjustments"]]
        if not data.columns.is_unique or not set(columns).issubset(data.columns):
            return self._refuse("INVALID_INPUT", "Selected columns must exist and column labels must be unique.")
        selected = data.loc[:, columns]
        if any(not is_numeric_dtype(dtype) or is_complex_dtype(dtype) for dtype in selected.dtypes):
            return self._refuse("INVALID_INPUT", "Selected columns must be real numeric values; no implicit coercion.")
        values = selected.to_numpy(dtype=float, na_value=np.nan, copy=True)
        if not np.isfinite(values).all():
            return self._refuse("INVALID_INPUT", "Missing or nonfinite data are not accepted; no rows were dropped.")
        t, y = values[:, 0], values[:, 1]
        if not np.isin(t, [0, 1]).all():
            return self._refuse("UNSUPPORTED_TASK", "Treatment must already be coded as binary 0/1.")
        n_treated = int(t.sum())
        if min(n_treated, len(t) - n_treated) < 20:
            return self._refuse("NOT_IDENTIFIED", "Both treatment arms need at least 20 observations.")
        w = values[:, 2:]

        from econml.dml import LinearDML
        from econml.inference import StatsModelsInference
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits

        try:
            with threadpool_limits(limits=1), warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                if w.shape[1]:
                    scaled = StandardScaler().fit_transform(w)
                    design = np.column_stack([np.ones(len(t)), scaled])
                    if np.linalg.matrix_rank(design) != design.shape[1]:
                        return self._refuse("NOT_IDENTIFIED", "Adjustment design is rank deficient.")
                else:
                    # With no adjustments, both nuisances are intercept-only.
                    w = np.ones((len(t), 1))
                model_t = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=1000))
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
                propensity = cross_val_predict(model_t, w, t, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
                pmin, pmax = float(propensity.min()), float(propensity.max())
                if not np.isfinite(propensity).all() or pmin < 0.05 or pmax > 0.95:
                    return self._refuse("NOT_IDENTIFIED", "Cross-fitted propensity screen failed [0.05, 0.95]; no trimming or clipping applied.")
                model = LinearDML(
                    model_y=make_pipeline(StandardScaler(), LinearRegression()),
                    model_t=model_t, discrete_treatment=True, cv=cv, random_state=SEED,
                )
                model.fit(y, t, W=w, inference=StatsModelsInference(cov_type="HC1"))
                estimate = float(np.asarray(model.ate()).item())
                low, high = model.ate_interval(alpha=cfg["alpha"])
                low, high = float(np.asarray(low).item()), float(np.asarray(high).item())
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, ConvergenceWarning):
            return self._refuse("NOT_IDENTIFIED", "Numerical fit or nuisance convergence failed; no estimate released.")
        if not all(math.isfinite(v) for v in (estimate, low, high)) or not low < estimate < high:
            return self._refuse("NOT_IDENTIFIED", "The engine did not return a finite, nondegenerate uncertainty interval.")

        versions = {name: version(name) for name in (
            "causal-inference-m5phet", "econml", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels"
        )}
        data_hash = sha256(json.dumps(columns).encode() + values.astype("<f8").tobytes()).hexdigest()
        config_hash = _digest(cfg)
        self._analysis_id = _digest({"data": data_hash, "config": config_hash, "versions": versions})
        payload = {
            "estimand": "ATE", "estimate": estimate, "interval": [low, high],
            "unit": cfg["unit"], "assumptions": sorted(cfg["assumptions"]),
            "diagnostics": {
                "identification": "conditional_on_user_assumptions",
                "engine": "econml.dml.LinearDML",
                "uncertainty_method": "econml_statsmodels_HC1_normal",
                "confidence_level": 1 - cfg["alpha"],
                "contrast": {"control": 0, "treated": 1},
                "n_rows": len(t), "n_treated": n_treated, "n_control": len(t) - n_treated,
                "propensity_min": pmin, "propensity_max": pmax,
                "overlap_screen": "passed_not_proven", "cv_folds": 3, "seed": SEED,
                "data_sha256": data_hash, "config_sha256": config_hash,
                "analysis_id": self._analysis_id, "versions": versions,
            },
        }
        self._phase = "FITTED"
        self._result = _response("OK", payload=payload)
        return deepcopy(self._result)

    def infer(self):
        """Return the fitted study's result; no fitting, new data or side effects."""
        if self._result is None:
            return _response("MODEL_NOT_FITTED", "Explicit fit(data) is required before infer().")
        return deepcopy(self._result)
