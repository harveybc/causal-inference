"""The configuration space a causal study is chosen FROM, declared once and offered as choice options.

WP20 of the M5PHET work plan asks for the thing a decision chooses among, not for the decision. So this module holds
the DECLARED alternatives -- estimators, nuisance models, column roles, identifying assumptions, confidence levels --
each as a `[key, label]` pair, so a `choice` question can be built from any dimension verbatim and the answer is a key
this repository already understands. Nothing here chooses, asks, fits or scores.

Two rules keep the declaration honest.

**An option that cannot be built is not offered.** `study_space()` probes every candidate by importing it, so the space
this function returns in the fit interpreter contains exactly the EconML and scikit-learn classes that interpreter can
actually construct; a class that disappears from the environment disappears from the space by name. The probe is the
only place this module touches a fit dependency.

**Serving never imports a fit dependency.** The workbench serves the causal provider from an interpreter that has no
EconML at all, and a catalog call there must not import one (`test_serving_does_not_import_fit_dependencies`). The
catalog therefore shows `study_space(probe=declared)` -- the same space with every declared candidate present and no
import attempted -- and says so in the space itself: availability is decided in the fit interpreter, where an
unavailable estimator is refused by name before anything is fitted.

Every parameter point below is small and fixed on purpose: it is a declared starting point a study can be reproduced
from, not a tuned configuration, and this module never searches for a better one.
"""

from copy import deepcopy
import importlib

SPACE_SCHEMA = "m5phet.causal_study_space.v1"

#: The estimand a study is fitted for. `ATE` is one number for the whole population; `CATE` is an effect that varies
#: with a declared modifier. The engine's own vocabulary, repeated here so an option can name it.
ATE, CATE = "ATE", "CATE"

#: The seed every declared estimator and every randomised nuisance model is built with, so two fits of one spec agree.
SEED = 1729

#: Bootstrap resamples for the estimators whose only interval in this engine is a bootstrap one. Declared, not tuned.
BOOTSTRAP_SAMPLES = 40

#: The candidate estimators, in the order WP20 names them. Each row: key, module, class, label, the estimands this
#: adapter can serve WITH AN INTERVAL, the inference it asks EconML for, and the name the study's diagnostics will
#: carry for that inference. An estimator whose interval this adapter has not verified is not listed: the engine never
#: releases an estimate without an interval, so "supported" here means "supported with its uncertainty".
_ESTIMATORS = (
    {"key": "LinearDML", "module": "econml.dml", "class": "LinearDML",
     "label": "Double machine learning with a linear final model and HC1 robust intervals",
     "estimands": (ATE, CATE), "inference": "statsmodels_HC1",
     "uncertainty_method": "econml_statsmodels_HC1_normal"},
    {"key": "CausalForestDML", "module": "econml.dml", "class": "CausalForestDML",
     "label": "Causal forest final model with bootstrap-of-little-bags intervals",
     # a causal forest is fitted ON the modifiers: EconML refuses `X=None`, so this estimator serves a conditional
     # study and has nothing to say about a study that declares no modifier at all
     "estimands": (CATE,), "inference": "blb",
     "uncertainty_method": "econml_blb_normal"},
    {"key": "DRLearner", "module": "econml.dr", "class": "DRLearner",
     "label": "Doubly robust learner (propensity and outcome regression) with bootstrap intervals",
     "estimands": (ATE, CATE), "inference": "bootstrap",
     "uncertainty_method": "econml_bootstrap_pivot"},
    {"key": "SparseLinearDML", "module": "econml.dml", "class": "SparseLinearDML",
     "label": "Double machine learning with a debiased-lasso final model and its normal intervals",
     "estimands": (ATE, CATE), "inference": "debiasedlasso",
     "uncertainty_method": "econml_debiasedlasso_normal"},
    {"key": "DML", "module": "econml.dml", "class": "DML",
     "label": "Double machine learning with an explicit linear final model and bootstrap intervals",
     "estimands": (ATE, CATE), "inference": "bootstrap",
     "uncertainty_method": "econml_bootstrap_pivot"},
)

#: The candidate nuisance models. A nuisance model is fitted twice in every study -- once for the outcome (a
#: regression) and once for the treatment (a classification, because the treatment is binary) -- so each key declares
#: BOTH forms, and a key is offered only when both of its classes import. `standardize` says whether the declared
#: parameter point is meant on standardised inputs; it is true exactly for the penalised linear models, whose penalty
#: is not scale free.
_NUISANCE_MODELS = (
    {"key": "lasso", "label": "L1-penalised linear models (Lasso / L1 logistic regression)", "standardize": True,
     "regressor": {"module": "sklearn.linear_model", "class": "Lasso",
                   "parameters": {"alpha": 0.01, "max_iter": 10000}},
     "classifier": {"module": "sklearn.linear_model", "class": "LogisticRegression",
                    "parameters": {"penalty": "l1", "solver": "liblinear", "C": 1.0, "max_iter": 1000}}},
    {"key": "ridge", "label": "L2-penalised linear models (Ridge / L2 logistic regression)", "standardize": True,
     "regressor": {"module": "sklearn.linear_model", "class": "Ridge", "parameters": {"alpha": 1.0}},
     "classifier": {"module": "sklearn.linear_model", "class": "LogisticRegression",
                    "parameters": {"penalty": "l2", "C": 1.0, "max_iter": 1000}}},
    {"key": "gradient_boosting", "label": "Gradient-boosted regression trees", "standardize": False,
     "regressor": {"module": "sklearn.ensemble", "class": "GradientBoostingRegressor",
                   "parameters": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": SEED}},
     "classifier": {"module": "sklearn.ensemble", "class": "GradientBoostingClassifier",
                    "parameters": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": SEED}}},
    {"key": "random_forest", "label": "Random forest, leaves held large enough to keep the fit stable",
     "standardize": False,
     "regressor": {"module": "sklearn.ensemble", "class": "RandomForestRegressor",
                   "parameters": {"n_estimators": 200, "min_samples_leaf": 20, "random_state": SEED, "n_jobs": 1}},
     "classifier": {"module": "sklearn.ensemble", "class": "RandomForestClassifier",
                    "parameters": {"n_estimators": 200, "min_samples_leaf": 20, "random_state": SEED, "n_jobs": 1}}},
)

#: The role every column of the dataset is given. Exactly the five WP20 declares, and the whole vocabulary of a role
#: choice: a column is the treatment, the outcome, something adjusted for, something the effect varies with, or left out.
_ROLES = (
    ("treatment", "The intervention whose effect is asked about (binary, coded 0/1)"),
    ("outcome", "The variable the effect is measured on"),
    ("confounder", "Adjusted for: it moves both the treatment and the outcome"),
    ("modifier", "The effect varies with it; the study reports an effect per level (binary, coded 0/1)"),
    ("exclude", "Not used by the study at all"),
)

#: The identifying and model assumptions, in the engine's own words -- the keys a retained study already carries in its
#: config and its manifest -- with a one-line reading of each. Nothing is added to the list here and nothing renamed:
#: a study declares these and no others, and an assumption outside this list is refused by name.
_ASSUMPTIONS = (
    ("sufficient_adjustment", "The declared confounders close every back-door path between treatment and outcome"),
    ("pre_treatment_adjustment", "Every adjusted variable is fixed before the treatment; none is a consequence of it"),
    ("consistency", "The observed outcome is the outcome under the treatment actually received"),
    ("no_interference", "One unit's treatment does not change another unit's outcome"),
    ("positivity", "Both treatment arms occur with positive probability at every adjusted value"),
    ("iid_sampling", "The rows are an independent sample from the population the effect is about"),
    ("constant_effect", "The effect is the same for every unit (an average-effect study only)"),
    ("effect_linear_in_modifiers", "The effect is linear in the declared modifiers (a conditional-effect study only)"),
    ("nuisance_models_correct", "The declared nuisance models are adequate for the outcome and the treatment"),
)

#: The confidence levels a study may be fitted at, as option keys (strings, because a choice option's key is a string)
#: and as the numbers the fit uses.
_CONFIDENCE_LEVELS = (
    ("0.95", 0.95, "95% interval (alpha 0.05)"),
    ("0.90", 0.90, "90% interval (alpha 0.10)"),
)


def _import_class(module, name):
    """Import one declared class. The single place a fit dependency is imported, so a test can take one away."""
    return getattr(importlib.import_module(module), name)


def importable(module, name):
    """Whether this interpreter can actually build a declared class: the probe that imports it.

    This is the only function in this module that imports a fit dependency, and it is never reached from the serving
    path -- the catalog probes with `declared` instead."""
    try:
        _import_class(module, name)
    except Exception:
        return False
    return True


def declared(module, name):
    """The import-free probe: every declared candidate is present, and nothing is imported to find out.

    Used where importing is forbidden or pointless -- the inference interpreter, which holds no EconML and fits
    nothing. The space it yields is the DECLARATION; whether an option can be built is settled in the fit interpreter,
    which refuses an unavailable estimator by name before it fits."""
    return True


def _options(entries):
    return [[entry["key"], entry["label"]] for entry in entries]


def study_space(*, probe=importable):
    """The declared configuration space, with every dimension's options as `[key, label]` pairs.

    `probe(module, class_name) -> bool` decides which candidates are offered. The default imports them, so in the fit
    interpreter the space contains exactly the classes that interpreter can construct."""
    estimators = [entry for entry in _ESTIMATORS if probe(entry["module"], entry["class"])]
    models = [entry for entry in _NUISANCE_MODELS
              if probe(entry["regressor"]["module"], entry["regressor"]["class"])
              and probe(entry["classifier"]["module"], entry["classifier"]["class"])]
    return {
        "schema": SPACE_SCHEMA,
        "availability": ("Options are offered when the interpreter that will FIT can build them; the serving "
                         "interpreter shows the declaration and fits nothing."),
        "probe": "imported" if probe is importable else "declared",
        "estimators": {
            "options": _options(estimators),
            "detail": {entry["key"]: {"class": f"{entry['module']}.{entry['class']}",
                                      "estimands": list(entry["estimands"]),
                                      "inference": entry["inference"],
                                      "uncertainty_method": entry["uncertainty_method"]}
                       for entry in estimators},
        },
        "nuisance_models": {
            "options": _options(models),
            "detail": {entry["key"]: {"standardize": entry["standardize"],
                                      "regressor": deepcopy(entry["regressor"]),
                                      "classifier": deepcopy(entry["classifier"])}
                       for entry in models},
            "used_for": [["model_y", "The outcome regression"], ["model_t", "The treatment classification"]],
        },
        "roles": {"options": [[key, label] for key, label in _ROLES]},
        "identification_assumptions": {
            "options": [[key, label] for key, label in _ASSUMPTIONS],
            # which of them a study of each estimand must declare -- all of them and nothing else
            "required_for": {ATE: list(assumptions_for(ATE)), CATE: list(assumptions_for(CATE))},
        },
        "confidence_levels": {
            "options": [[key, label] for key, _, label in _CONFIDENCE_LEVELS],
            "values": {key: value for key, value, _ in _CONFIDENCE_LEVELS},
        },
    }


def assumptions_for(estimand):
    """The assumption set a study of this estimand declares, read from the engine so the two cannot drift apart."""
    from .provider import assumptions_for as engine_assumptions
    return tuple(engine_assumptions(estimand))


def estimator_detail(key, space=None):
    """One declared estimator's detail, or None when the space does not offer it."""
    space = space if space is not None else study_space()
    return (space["estimators"]["detail"] or {}).get(key)


def confidence_level_value(key, space=None):
    """The number a confidence-level option key stands for, or None. A number already in the declared set passes."""
    space = space if space is not None else study_space()
    values = space["confidence_levels"]["values"]
    if isinstance(key, str):
        return values.get(key)
    if isinstance(key, (int, float)) and not isinstance(key, bool):
        return next((value for value in values.values() if float(key) == value), None)
    return None


# --- building what was chosen (the fit interpreter only) ---------------------------------------------------------


def build_nuisance(key, role):
    """The scikit-learn estimator a declared nuisance key stands for, in the role it is asked for.

    `role` is `model_y` (the outcome regression) or `model_t` (the treatment classification). The declared parameter
    point is passed verbatim; the penalised linear models are wrapped in the standardiser they are declared on."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    entry = next((candidate for candidate in _NUISANCE_MODELS if candidate["key"] == key), None)
    if entry is None:
        raise ValueError(f"{key!r} is not a declared nuisance model; the declared ones are "
                         f"{[candidate['key'] for candidate in _NUISANCE_MODELS]}.")
    if role not in ("model_y", "model_t"):
        raise ValueError("A nuisance model is built for `model_y` or `model_t`.")
    declaration = entry["regressor"] if role == "model_y" else entry["classifier"]
    model = _import_class(declaration["module"], declaration["class"])(**declaration["parameters"])
    return make_pipeline(StandardScaler(), model) if entry["standardize"] else model


def build_estimator(key, *, model_y, model_t, cv, random_state=SEED):
    """The EconML estimator a declared key stands for, together with the inference it is fitted with.

    Returns `(estimator, inference, detail)`. The inference is chosen per estimator and declared in the space, because
    it is what decides whether the study can carry an interval -- and this engine releases no estimate without one."""
    entry = next((candidate for candidate in _ESTIMATORS if candidate["key"] == key), None)
    if entry is None:
        raise ValueError(f"{key!r} is not a declared estimator; the declared ones are "
                         f"{[candidate['key'] for candidate in _ESTIMATORS]}.")
    factory = _import_class(entry["module"], entry["class"])
    if entry["inference"] == "statsmodels_HC1":
        from econml.inference import StatsModelsInference
        inference = StatsModelsInference(cov_type="HC1")
    elif entry["inference"] == "bootstrap":
        from econml.inference import BootstrapInference
        inference = BootstrapInference(n_bootstrap_samples=BOOTSTRAP_SAMPLES, n_jobs=1, bootstrap_type="pivot")
    else:
        inference = entry["inference"]
    if entry["key"] == "DRLearner":
        estimator = factory(model_propensity=model_t, model_regression=model_y, cv=cv, random_state=random_state)
    elif entry["key"] == "DML":
        from sklearn.linear_model import LinearRegression
        estimator = factory(model_y=model_y, model_t=model_t, model_final=LinearRegression(fit_intercept=False),
                            discrete_treatment=True, cv=cv, random_state=random_state)
    else:
        estimator = factory(model_y=model_y, model_t=model_t, discrete_treatment=True, cv=cv,
                            random_state=random_state)
    return estimator, inference, deepcopy(entry)
