"""A treatment that is not an arm: the continuous kind WP22's calendar surprise needs, declared and fitted.

The engine was built around a 0/1 intervention, and everything about that shape is load bearing -- the propensity
screen, the classifier nuisance, the two arms counted inside every subgroup. A standardized calendar surprise is none
of those things: it has no arms, no propensity, and its effect is read per one unit of it. So the kind is DECLARED,
per estimator, and everything that differs between the two kinds differs by name.

Three things are checked here, in this order of importance:

1. **Nothing that was already retained changes.** The identifying config of a binary study must still be exactly the
   object it was before this dimension existed, because that object's digest IS the task identity every retained
   study answers under. A test for that is the first test in this file for a reason.
2. **An option that is not declared is refused, not attempted.** A doubly robust learner is built on a propensity; a
   continuous treatment has none, and EconML's own DRLearner requires a discrete treatment. So the space declares
   DRLearner for the binary kind only and the spec refuses it by name rather than letting EconML raise.
3. **On a world where the effect was planted, the fit recovers it.** Not a claim about markets: a recovery test says
   the adapter is estimating the quantity it says it is, which is exactly what the calendar study cannot prove about
   itself.
"""

import json

import numpy as np
import pandas as pd
import pytest

from causal_inference_provider import study_space as space
from causal_inference_provider import study_spec as spec
from causal_inference_provider.provider import CausalInferenceProvider

SEED = 20260925
ROWS = 2000


def continuous_world(*, heterogeneous, rows=ROWS, determined=False):
    """A confounded continuous treatment with a planted effect: 2 everywhere, or 1 and 3 by the modifier's level."""
    rng = np.random.default_rng(SEED)
    modifier = rng.integers(0, 2, size=rows).astype(float)
    confounder = rng.normal(size=rows)
    noise = np.zeros(rows) if determined else rng.normal(size=rows)
    treatment = 0.5 * confounder + noise
    effect = (1.0 + 2.0 * modifier) if heterogeneous else np.full(rows, 2.0)
    outcome = effect * treatment + 1.5 * confounder + rng.normal(scale=0.5, size=rows)
    return pd.DataFrame({"confounder": confounder, "modifier": modifier,
                         "surprise": treatment, "response": outcome})


def config_for(*, heterogeneous, estimator, kind="continuous"):
    config = {
        "estimand": "CATE" if heterogeneous else "ATE",
        "treatment": "surprise", "outcome": "response", "adjustments": ["confounder"],
        "unit": "response units per unit of surprise", "alpha": 0.05,
        "estimator": estimator, "treatment_kind": kind,
        "nuisance": {"model_y": "gradient_boosting", "model_t": "gradient_boosting"},
        "assumptions": {name: True for name in (
            space.assumptions_for("CATE" if heterogeneous else "ATE"))},
    }
    if heterogeneous:
        config["effect_modifiers"] = ["modifier"]
    return config


# ---------------------------------------------------------------- 1. nothing that was already retained changes

def test_a_binary_study_keeps_the_identifying_config_it_had_before_the_kind_existed():
    """The config's digest is the task identity of every retained study; the default kind must stay ABSENT from it."""
    document = {
        "schema": spec.SPEC_SCHEMA,
        "dataset": {"id": "synthetic-constant", "columns": ["baseline", "treatment", "outcome"]},
        "roles": {"baseline": "confounder", "treatment": "treatment", "outcome": "outcome"},
        "estimator": "LinearDML", "nuisance": {"model_y": "lasso", "model_t": "lasso"},
        "confidence_level": "0.95", "identification": list(space.assumptions_for("ATE")),
        "decisions": [], "provenance": "DEVELOPMENT",
    }
    config = spec.config_from_spec(document)
    assert "treatment_kind" not in config
    provider = CausalInferenceProvider()
    assert provider.load(config)["status"] == "OK"
    assert "treatment_kind" not in provider.state["config"]


def test_declaring_the_binary_kind_explicitly_is_the_same_study_by_name_and_says_so():
    document = {
        "schema": spec.SPEC_SCHEMA,
        "dataset": {"id": "synthetic-constant", "columns": ["baseline", "treatment", "outcome"]},
        "roles": {"baseline": "confounder", "treatment": "treatment", "outcome": "outcome"},
        "estimator": "LinearDML", "nuisance": {"model_y": "lasso", "model_t": "lasso"},
        "confidence_level": "0.95", "identification": list(space.assumptions_for("ATE")),
        "decisions": [], "provenance": "DEVELOPMENT", "treatment_kind": "binary",
    }
    assert spec.validate_spec(document)["treatment_kind"] == "binary"
    assert "treatment_kind" not in spec.config_from_spec(document)


# ------------------------------------------------------------------- 2. an undeclared option is refused by name

def test_the_space_declares_which_estimators_serve_a_continuous_treatment():
    declared = space.study_space(probe=space.declared)
    kinds = declared["treatment_kinds"]
    assert [key for key, _ in kinds["options"]] == ["binary", "continuous"]
    assert kinds["default"] == "binary"
    assert "DRLearner" in kinds["served_by"]["binary"]
    assert "DRLearner" not in kinds["served_by"]["continuous"]
    assert {"LinearDML", "CausalForestDML"} <= set(kinds["served_by"]["continuous"])
    for detail in declared["estimators"]["detail"].values():
        assert set(detail["treatments"]) <= {"binary", "continuous"} and detail["treatments"]


@pytest.mark.parametrize("kind", ["ordinal", "", None, 3])
def test_a_kind_the_space_does_not_declare_is_refused_by_name(kind):
    document = {
        "schema": spec.SPEC_SCHEMA,
        "dataset": {"path": "/nowhere.csv", "columns": ["baseline", "treatment", "outcome"]},
        "roles": {"baseline": "confounder", "treatment": "treatment", "outcome": "outcome"},
        "estimator": "LinearDML", "nuisance": {"model_y": "lasso", "model_t": "lasso"},
        "confidence_level": "0.95", "identification": list(space.assumptions_for("ATE")),
        "decisions": [], "provenance": "UNDECLARED", "treatment_kind": kind,
    }
    with pytest.raises(spec.SpecRefused) as refused:
        spec.validate_spec(document)
    assert refused.value.refusal == spec.UNKNOWN_TREATMENT_KIND


def test_a_doubly_robust_learner_is_refused_for_a_continuous_treatment_rather_than_attempted():
    document = {
        "schema": spec.SPEC_SCHEMA,
        "dataset": {"path": "/nowhere.csv", "columns": ["baseline", "treatment", "outcome"]},
        "roles": {"baseline": "confounder", "treatment": "treatment", "outcome": "outcome"},
        "estimator": "DRLearner", "nuisance": {"model_y": "lasso", "model_t": "lasso"},
        "confidence_level": "0.95", "identification": list(space.assumptions_for("ATE")),
        "decisions": [], "provenance": "UNDECLARED", "treatment_kind": "continuous",
    }
    with pytest.raises(spec.SpecRefused) as refused:
        spec.validate_spec(document)
    assert refused.value.refusal == spec.ESTIMATOR_HAS_NO_TREATMENT_KIND
    assert "DRLearner" in refused.value.why and "continuous" in refused.value.why


def test_the_provider_refuses_an_estimator_not_declared_for_the_kind_before_it_reads_a_row():
    provider = CausalInferenceProvider()
    result = provider.load(config_for(heterogeneous=False, estimator="DRLearner"))
    assert result["status"] == "UNSUPPORTED_TASK" and "DRLearner" in result["reason"]


def test_the_treatment_model_of_a_continuous_study_is_a_regression_not_a_classification():
    regression = space.build_nuisance("gradient_boosting", "model_t", treatment_kind="continuous")
    classification = space.build_nuisance("gradient_boosting", "model_t", treatment_kind="binary")
    assert type(regression).__name__ == "GradientBoostingRegressor"
    assert type(classification).__name__ == "GradientBoostingClassifier"


# --------------------------------------------------------------------------- 3. the planted effect is recovered

@pytest.mark.parametrize("estimator", ["LinearDML"])
def test_the_average_effect_per_unit_of_a_continuous_treatment_is_recovered_within_its_interval(estimator):
    provider = CausalInferenceProvider()
    assert provider.load(config_for(heterogeneous=False, estimator=estimator))["status"] == "OK"
    result = provider.fit(continuous_world(heterogeneous=False))
    assert result["status"] == "OK", result["reason"]
    payload = result["payload"]
    low, high = payload["interval"]
    assert low < 2.0 < high, f"the planted effect 2 is outside [{low}, {high}]"
    diagnostics = payload["diagnostics"]
    assert diagnostics["treatment_kind"] == "continuous"
    assert diagnostics["n_treated"] is None and diagnostics["n_control"] is None
    assert "propensity_min" not in diagnostics and "overlap_screen" not in diagnostics
    assert diagnostics["residual_screen"] == "passed_not_proven"
    assert 0.0 < diagnostics["treatment_residual_fraction"] <= 1.0 + 1e-9
    assert "one unit more" in diagnostics["contrast_reading"]
    assert "no arms to overlap" in diagnostics["positivity_reading"]


def test_the_linear_final_model_covers_the_planted_effect_at_both_levels_of_the_modifier():
    """LinearDML's final model is linear in the modifier, which is exactly how this world was generated, so its
    interval is the one that must cover the planted number."""
    provider = CausalInferenceProvider()
    assert provider.load(config_for(heterogeneous=True, estimator="LinearDML"))["status"] == "OK"
    result = provider.fit(continuous_world(heterogeneous=True))
    assert result["status"] == "OK", result["reason"]
    cells = {entry["subgroup"]: entry for entry in result["payload"]["conditional_effects"]}
    assert set(cells) == {"modifier == 0", "modifier == 1"}
    for subgroup, planted in (("modifier == 0", 1.0), ("modifier == 1", 3.0)):
        low, high = cells[subgroup]["interval"]
        assert low < planted < high, f"{subgroup}: the planted effect {planted} is outside [{low}, {high}]"
        assert cells[subgroup]["n_treated"] is None and cells[subgroup]["n_control"] is None
        assert cells[subgroup]["treatment_std"] > 0


def test_the_causal_forest_recovers_the_planted_effect_to_a_declared_tolerance():
    """Measured, not assumed: at 2000 rows the forest's point estimate for `modifier == 1` came out 2.9662 with a
    bootstrap-of-little-bags interval of [2.9378, 2.9950] -- close to the planted 3, and NOT covering it. The forest's
    final model is not the linear one this world was generated from and its BLB interval is not a coverage guarantee
    for a point estimate at this size, so the check here is a declared distance and the interval is only required to
    be finite and ordered. A test that quietly widened the interval instead would be claiming coverage nobody saw."""
    tolerance = 0.1
    provider = CausalInferenceProvider()
    assert provider.load(config_for(heterogeneous=True, estimator="CausalForestDML"))["status"] == "OK"
    result = provider.fit(continuous_world(heterogeneous=True))
    assert result["status"] == "OK", result["reason"]
    cells = {entry["subgroup"]: entry for entry in result["payload"]["conditional_effects"]}
    for subgroup, planted in (("modifier == 0", 1.0), ("modifier == 1", 3.0)):
        low, high = cells[subgroup]["interval"]
        assert abs(cells[subgroup]["estimate"] - planted) < tolerance, f"{subgroup}: {cells[subgroup]['estimate']}"
        assert low < cells[subgroup]["estimate"] < high
        assert cells[subgroup]["n_treated"] is None and cells[subgroup]["treatment_std"] > 0


def test_a_constant_continuous_treatment_is_refused_because_it_carries_no_contrast():
    provider = CausalInferenceProvider()
    assert provider.load(config_for(heterogeneous=False, estimator="LinearDML"))["status"] == "OK"
    data = continuous_world(heterogeneous=False)
    data["surprise"] = 0.7
    result = provider.fit(data)
    assert result["status"] == "NOT_IDENTIFIED" and "no contrast" in result["reason"]


def test_a_treatment_the_adjustment_set_already_determines_fails_the_residual_screen_by_name():
    """The continuous counterpart of the overlap screen: no variation left is no effect to identify."""
    provider = CausalInferenceProvider()
    assert provider.load(config_for(heterogeneous=False, estimator="LinearDML"))["status"] == "OK"
    result = provider.fit(continuous_world(heterogeneous=False, determined=True))
    assert result["status"] == "NOT_IDENTIFIED"
    assert "treatment-residual screen failed" in result["reason"]
    assert str(__import__("causal_inference_provider.provider", fromlist=["x"]).MIN_TREATMENT_RESIDUAL_FRACTION) \
        in result["reason"]


def test_a_continuous_study_answers_ate_and_cate_from_the_artifact_it_retained(tmp_path):
    """The whole path: a spec on a CSV, fitted explicitly, then asked -- with no second look at the population."""
    from causal_inference_provider import questions
    from causal_inference_provider.chat import save_study

    data = continuous_world(heterogeneous=True)
    csv_path = tmp_path / "events.csv"
    data.to_csv(csv_path, index=False)
    document = {
        "schema": spec.SPEC_SCHEMA,
        "dataset": {"path": str(csv_path), "columns": list(data.columns)},
        "roles": {"confounder": "confounder", "modifier": "modifier", "surprise": "treatment",
                  "response": "outcome"},
        "estimator": "LinearDML", "nuisance": {"model_y": "gradient_boosting", "model_t": "gradient_boosting"},
        "confidence_level": "0.95", "identification": list(space.assumptions_for("CATE")),
        "decisions": [], "provenance": "DEVELOPMENT", "treatment_kind": "continuous",
        "study_id": "continuous-recovery-test",
    }
    validated = spec.validate_spec(document, space.study_space())
    config = spec.config_from_spec(validated, space.study_space())
    assert config["treatment_kind"] == "continuous"
    provider = CausalInferenceProvider()
    assert provider.load(config)["status"] == "OK"
    assert provider.fit(data)["status"] == "OK"
    state_ref = save_study(provider, tmp_path / "state", development=True,
                           study_id=spec.study_identifier(validated), spec=validated)

    from causal_inference_provider.chat import M5PHETCausalProvider
    served = M5PHETCausalProvider(directory=tmp_path / "state")
    answers = questions.answer_questions(
        served, {"state_ref": state_ref},
        {"average": {"type": "ate"}, "high": {"type": "cate", "subgroup": "modifier == 1"}}, None, None)
    assert answers["average"]["status"] == "OK"
    assert answers["high"]["status"] == "OK" and answers["high"]["subgroup"] == "modifier == 1"
    assert answers["high"]["execution_authorized"] is False
    assert json.loads(json.dumps(answers))            # every answer is JSON, with no NaN anywhere in it


# ------------------------------------------------- what the data could not deliver, carried onto every answer

def test_a_declared_caveat_reaches_every_answer_verbatim(tmp_path):
    """WP22's studies rest on a clock nobody observed. No assumption name says that, so the spec carries the sentence
    itself and every answer repeats it -- a caveat a reader has to go and look up is a caveat that gets dropped."""
    from causal_inference_provider import questions
    from causal_inference_provider.chat import M5PHETCausalProvider, save_study

    caveat = ("ASSUMED_SCHEDULED_PUBLICATION_LOCALIZED: nobody observed when anything was published. NOT_IDENTIFIED.")
    data = continuous_world(heterogeneous=True)
    csv_path = tmp_path / "events.csv"
    data.to_csv(csv_path, index=False)
    document = {
        "schema": spec.SPEC_SCHEMA,
        "dataset": {"path": str(csv_path), "columns": list(data.columns)},
        "roles": {"confounder": "confounder", "modifier": "modifier", "surprise": "treatment",
                  "response": "outcome"},
        "estimator": "LinearDML", "nuisance": {"model_y": "lasso", "model_t": "lasso"},
        "confidence_level": "0.95", "identification": list(space.assumptions_for("CATE")),
        "decisions": [], "provenance": "DEVELOPMENT", "treatment_kind": "continuous",
        "study_id": "caveat-carrier", "identification_caveat": caveat,
    }
    validated = spec.validate_spec(document, space.study_space())
    provider = CausalInferenceProvider()
    assert provider.load(spec.config_from_spec(validated, space.study_space()))["status"] == "OK"
    assert provider.fit(data)["status"] == "OK"
    state_ref = save_study(provider, tmp_path / "state", development=True,
                           study_id="caveat-carrier", spec=validated)
    served = M5PHETCausalProvider(directory=tmp_path / "state")
    answers = questions.answer_questions(
        served, {"state_ref": state_ref},
        {"a": {"type": "ate"}, "c": {"type": "cate", "subgroup": "modifier == 1"}}, None, None)
    for name in ("a", "c"):
        assert answers[name]["status"] == "OK"
        assert answers[name]["identification_caveat"] == caveat


@pytest.mark.parametrize("caveat", ["", "   ", 7, "x" * 4097])
def test_a_caveat_that_is_not_a_sentence_is_refused(caveat):
    document = {
        "schema": spec.SPEC_SCHEMA,
        "dataset": {"id": "synthetic-constant", "columns": ["baseline", "treatment", "outcome"]},
        "roles": {"baseline": "confounder", "treatment": "treatment", "outcome": "outcome"},
        "estimator": "LinearDML", "nuisance": {"model_y": "lasso", "model_t": "lasso"},
        "confidence_level": "0.95", "identification": list(space.assumptions_for("ATE")),
        "decisions": [], "provenance": "DEVELOPMENT", "identification_caveat": caveat,
    }
    with pytest.raises(spec.SpecRefused) as refused:
        spec.validate_spec(document)
    assert refused.value.refusal == spec.MALFORMED_SPEC
