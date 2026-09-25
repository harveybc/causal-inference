"""A bounded binary-treatment adapter for EconML: a constant average effect, or an effect that varies with one
explicitly declared binary modifier.

Two estimands, and the difference between them is a declaration, not a guess. `ATE` is the constant-effect study: one
number for the whole population, and the caller declares `constant_effect`. `CATE` is fitted with an explicit effect
modifier and carries, alongside that average, the effect within each level of the modifier -- computed here, while
fitting, so that reporting it later reads an artifact instead of touching data again. A study that declares a modifier
may not also declare the effect constant; it declares instead that the effect is linear in the modifiers it named, which
for a single binary modifier is saturated and so restricts nothing.
"""

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
#: A heterogeneous study cannot claim the effect is constant: that is the very claim it sets aside. It claims instead
#: that the effect is linear in the declared modifiers, which a single binary modifier satisfies exactly.
MODIFIER_ASSUMPTIONS = tuple(a for a in ASSUMPTIONS if a != "constant_effect") + ("effect_linear_in_modifiers",)
ESTIMANDS = ("ATE", "CATE")
MODIFIER_LEVELS = (0, 1)
MAX_MODIFIERS = 1
MIN_PER_CELL = 20
MAX_ROWS = 10_000
MAX_ADJUSTMENTS = 20
SEED = 1729


def assumptions_for(estimand):
    """The assumption set a study of this estimand must declare, all of it and nothing else."""
    return MODIFIER_ASSUMPTIONS if estimand == "CATE" else ASSUMPTIONS


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
            "estimands": list(ESTIMANDS),
            "contrast": {"control": 0, "treated": 1},
            "input_kinds": ["dataframe", "records"],
            "required_config": ["estimand", "treatment", "outcome", "adjustments", "unit", "assumptions"],
            "required_assumptions": list(ASSUMPTIONS),
            "required_assumptions_cate": list(MODIFIER_ASSUMPTIONS),
            "effect_modifier_levels": list(MODIFIER_LEVELS),
            "uncertainty_methods": ["econml_statsmodels_HC1_normal"],
            "limits": {"min_rows": 100, "max_rows": MAX_ROWS,
                       "max_adjustments": MAX_ADJUSTMENTS, "min_per_arm": 20,
                       "max_effect_modifiers": MAX_MODIFIERS, "min_per_subgroup_arm": MIN_PER_CELL,
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
        allowed = {"estimand", "treatment", "outcome", "adjustments", "unit", "assumptions", "alpha",
                   "effect_modifiers"}
        if set(config) - allowed:
            return self._refuse("INVALID_INPUT", "Unknown config fields; no implicit interpretation is supported.")
        if not allowed.difference({"alpha", "assumptions", "effect_modifiers"}).issubset(config):
            return self._refuse("INVALID_INPUT", "Explicit estimand, roles, adjustment list and outcome unit are required.")
        if config["estimand"] not in ESTIMANDS:
            return self._refuse("UNSUPPORTED_TASK", "Only binary-treatment (1 versus 0) constant-effect ATE and CATE "
                                                    "over declared binary effect modifiers are supported.")
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
        modifiers = config.get("effect_modifiers", [])
        if not isinstance(modifiers, list) or any(
            not isinstance(n, str) or not n.strip() or len(n) > 128 for n in modifiers
        ):
            return self._refuse("INVALID_INPUT", "Effect modifiers must be an explicit list of column names.")
        if config["estimand"] == "CATE" and not modifiers:
            return self._refuse("INVALID_INPUT", "A CATE study must declare the effect modifier it was fitted with; "
                                                 "a conditional effect over nothing is not a question.")
        if config["estimand"] == "ATE" and modifiers:
            return self._refuse("INVALID_INPUT", "An ATE study carries no effect modifier; declare estimand CATE to "
                                                 "fit an effect that varies with a modifier.")
        if len(modifiers) > MAX_MODIFIERS:
            return self._refuse("UNSUPPORTED_TASK", "Exactly one binary effect modifier is supported; a subgroup over "
                                                    "several modifiers would have to name the level of each.")
        roles = [config["treatment"], config["outcome"], *adjustment, *modifiers]
        if len(set(roles)) != len(roles):
            return self._refuse("INVALID_INPUT", "Treatment, outcome, adjustments and effect modifiers must be distinct.")
        required = assumptions_for(config["estimand"])
        assumptions = config.get("assumptions")
        if not isinstance(assumptions, dict) or any(assumptions.get(a) is not True for a in required):
            return self._refuse("NOT_IDENTIFIED", "All identifying and model assumptions must be explicitly declared true.")
        if set(assumptions) != set(required):
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
        modifiers = cfg.get("effect_modifiers") or []
        columns = [cfg["treatment"], cfg["outcome"], *cfg["adjustments"], *modifiers]
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
        w = values[:, 2:2 + len(cfg["adjustments"])]
        x = values[:, 2 + len(cfg["adjustments"]):]
        if x.size and not np.isin(x, MODIFIER_LEVELS).all():
            return self._refuse("UNSUPPORTED_TASK", "Effect modifiers must already be coded as binary 0/1; a subgroup "
                                                    "is read at a declared level, never at a cut this provider chose.")
        for index, name in enumerate(modifiers):
            for level in MODIFIER_LEVELS:
                inside = x[:, index] == level
                treated_inside = int(t[inside].sum())
                if min(treated_inside, int(inside.sum()) - treated_inside) < MIN_PER_CELL:
                    return self._refuse("NOT_IDENTIFIED", f"Subgroup {name} == {level} needs at least "
                                                          f"{MIN_PER_CELL} treated and {MIN_PER_CELL} control "
                                                          f"observations; no subgroup is pooled to reach it.")

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
                covariates = np.column_stack([w, x]) if x.size else w
                if covariates.shape[1]:
                    scaled = StandardScaler().fit_transform(covariates)
                    design = np.column_stack([np.ones(len(t)), scaled])
                    if np.linalg.matrix_rank(design) != design.shape[1]:
                        return self._refuse("NOT_IDENTIFIED", "Adjustment design is rank deficient.")
                if not w.shape[1]:
                    # With no adjustments, the control nuisance is intercept-only.
                    w = np.ones((len(t), 1))
                # the modifiers are part of the nuisance design too, exactly as the estimator uses them
                nuisance = np.column_stack([w, x]) if x.size else w
                model_t = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=1000))
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
                propensity = cross_val_predict(model_t, nuisance, t, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
                pmin, pmax = float(propensity.min()), float(propensity.max())
                if not np.isfinite(propensity).all() or pmin < 0.05 or pmax > 0.95:
                    return self._refuse("NOT_IDENTIFIED", "Cross-fitted propensity screen failed [0.05, 0.95]; no trimming or clipping applied.")
                model = LinearDML(
                    model_y=make_pipeline(StandardScaler(), LinearRegression()),
                    model_t=model_t, discrete_treatment=True, cv=cv, random_state=SEED,
                )
                model.fit(y, t, X=(x if x.size else None), W=w, inference=StatsModelsInference(cov_type="HC1"))
                # the average effect of a heterogeneous study is the mean of its conditional effects over the very rows
                # it was fitted on, which is why the fitted modifier matrix is passed back in here
                population = {"X": x} if x.size else {}
                estimate = float(np.asarray(model.ate(**population)).item())
                low, high = model.ate_interval(alpha=cfg["alpha"], **population)
                low, high = float(np.asarray(low).item()), float(np.asarray(high).item())
                # every subgroup effect the study will ever report is computed HERE, while the data are in hand, so
                # that reporting one later is a read of this artifact and never a second look at a population
                conditional = []
                for index, name in enumerate(modifiers):
                    for level in MODIFIER_LEVELS:
                        point = np.zeros((1, x.shape[1]))
                        point[0, index] = level
                        inside = x[:, index] == level
                        treated_inside = int(t[inside].sum())
                        cell_low, cell_high = model.effect_interval(point, T0=0, T1=1, alpha=cfg["alpha"])
                        conditional.append({
                            "modifier": name, "level": level, "subgroup": f"{name} == {level}",
                            "estimate": float(np.asarray(model.effect(point, T0=0, T1=1)).item()),
                            "interval": [float(np.asarray(cell_low).item()), float(np.asarray(cell_high).item())],
                            "n_subgroup": int(inside.sum()), "n_treated": treated_inside,
                            "n_control": int(inside.sum()) - treated_inside,
                        })
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, ConvergenceWarning):
            return self._refuse("NOT_IDENTIFIED", "Numerical fit or nuisance convergence failed; no estimate released.")
        if not all(math.isfinite(v) for v in (estimate, low, high)) or not low < estimate < high:
            return self._refuse("NOT_IDENTIFIED", "The engine did not return a finite, nondegenerate uncertainty interval.")
        for cell in conditional:
            bounds = (cell["estimate"], *cell["interval"])
            if not all(math.isfinite(v) for v in bounds) or not cell["interval"][0] < cell["estimate"] < cell["interval"][1]:
                return self._refuse("NOT_IDENTIFIED", f"The engine did not return a finite, nondegenerate interval for "
                                                      f"subgroup {cell['subgroup']}; no subgroup effect is released "
                                                      f"without one.")

        versions = {name: version(name) for name in (
            "causal-inference-m5phet", "econml", "numpy", "pandas", "scikit-learn", "scipy", "statsmodels"
        )}
        data_hash = sha256(json.dumps(columns).encode() + values.astype("<f8").tobytes()).hexdigest()
        config_hash = _digest(cfg)
        self._analysis_id = _digest({"data": data_hash, "config": config_hash, "versions": versions})
        payload = {
            "estimand": cfg["estimand"], "estimate": estimate, "interval": [low, high],
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
        if modifiers:
            # the average stays where it always was; the subgroup effects are additional, named and countable
            payload["effect_modifiers"] = list(modifiers)
            payload["conditional_effects"] = conditional
            payload["diagnostics"]["effect_modifiers"] = list(modifiers)
            payload["diagnostics"]["final_model"] = "linear_in_declared_effect_modifiers"
        self._phase = "FITTED"
        self._result = _response("OK", payload=payload)
        return deepcopy(self._result)

    def infer(self):
        """Return the fitted study's result; no fitting, new data or side effects."""
        if self._result is None:
            return _response("MODEL_NOT_FITTED", "Explicit fit(data) is required before infer().")
        return deepcopy(self._result)
