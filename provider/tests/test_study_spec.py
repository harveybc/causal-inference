"""What a study spec must say, and what it is refused for -- each refusal by its own name.

A spec is the document a chooser produces and the fit consumes, so the only interesting question about it is which
specs never become a study. Every test below asserts the NAME of the refusal, not merely that something was refused:
a caller that has to read prose to find out what went wrong cannot repair the spec, and neither can a chooser.

The valid spec these tests start from is built from the space's OWN option keys, so a spec assembled by picking one
option per dimension -- which is exactly what a `choice` question yields -- is a spec that validates.
"""

from copy import deepcopy

import pytest

from causal_inference_provider import study_spec as spec_module
from causal_inference_provider.study_space import declared, study_space
from causal_inference_provider.study_spec import (SPEC_SCHEMA, SpecRefused, config_from_spec, spec_digest,
                                                 study_identifier, validate_spec)

SPACE = study_space(probe=declared)


def option_keys(dimension):
    return [key for key, _ in SPACE[dimension]["options"]]


def valid_spec(**changes):
    """A spec assembled from the space's own keys: one estimator, one nuisance model per role, one confidence level,
    the roles the space declares and the assumptions a CATE study is required to state."""
    spec = {
        "schema": SPEC_SCHEMA,
        "dataset": {"id": "synthetic-modifier", "columns": ["baseline", "confounder", "treatment", "outcome"]},
        "roles": {"treatment": "treatment", "outcome": "outcome",
                  "confounder": "confounder", "baseline": "modifier"},
        "estimator": "LinearDML",
        "nuisance": {"model_y": "lasso", "model_t": "lasso"},
        "confidence_level": "0.95",
        "identification": list(SPACE["identification_assumptions"]["required_for"]["CATE"]),
        "decisions": [],
        "provenance": "DEVELOPMENT",
    }
    spec.update(changes)
    return spec


def refusal_of(spec):
    with pytest.raises(SpecRefused) as raised:
        validate_spec(spec, SPACE)
    return raised.value.refusal


def test_a_spec_built_from_the_space_s_own_keys_validates():
    spec = valid_spec()
    assert spec["estimator"] in option_keys("estimators")
    assert set(spec["nuisance"].values()) <= set(option_keys("nuisance_models"))
    assert set(spec["roles"].values()) <= set(option_keys("roles"))
    assert set(spec["identification"]) <= set(option_keys("identification_assumptions"))
    assert spec["confidence_level"] in option_keys("confidence_levels")
    validated = validate_spec(spec, SPACE)
    assert validated["confidence_level"] == 0.95
    assert spec_digest(validated) != spec_digest(valid_spec(estimator="DML"))


def test_every_pairing_of_the_space_s_own_options_validates():
    """One option per dimension, in every combination that the roles admit: the space offers nothing that the spec
    schema then rejects."""
    for estimator in option_keys("estimators"):
        estimands = SPACE["estimators"]["detail"][estimator]["estimands"]
        for model_y in option_keys("nuisance_models"):
            for level in option_keys("confidence_levels"):
                spec = valid_spec(estimator=estimator, confidence_level=level,
                                  nuisance={"model_y": model_y, "model_t": "lasso"})
                if "CATE" in estimands:
                    assert validate_spec(spec, SPACE)["estimator"] == estimator
                else:
                    assert refusal_of(spec) == spec_module.ESTIMATOR_HAS_NO_CATE


def test_the_config_the_fit_receives_follows_from_the_roles():
    config = config_from_spec(valid_spec(), SPACE)
    assert config["estimand"] == "CATE"
    assert config["treatment"] == "treatment" and config["outcome"] == "outcome"
    assert config["adjustments"] == ["confounder"] and config["effect_modifiers"] == ["baseline"]
    assert config["estimator"] == "LinearDML"
    assert config["nuisance"] == {"model_y": "lasso", "model_t": "lasso"}
    assert config["alpha"] == pytest.approx(0.05)
    assert config["assumptions"] == {name: True for name in SPACE["identification_assumptions"]["required_for"]["CATE"]}
    # no modifier: the same spec asks for one average effect, and the estimand follows the roles, not a separate field
    average = config_from_spec(valid_spec(
        dataset={"id": "synthetic-constant", "columns": ["baseline", "treatment", "outcome"]},
        roles={"treatment": "treatment", "outcome": "outcome", "baseline": "confounder"},
        identification=list(SPACE["identification_assumptions"]["required_for"]["ATE"])), SPACE)
    assert average["estimand"] == "ATE" and "effect_modifiers" not in average


@pytest.mark.parametrize("change, refusal", [
    ({"estimator": "MyOwnEstimator"}, spec_module.UNKNOWN_ESTIMATOR),
    ({"estimator": None}, spec_module.UNKNOWN_ESTIMATOR),
    ({"nuisance": {"model_y": "neural_net", "model_t": "lasso"}}, spec_module.UNKNOWN_NUISANCE_MODEL),
    ({"nuisance": {"model_y": "lasso", "model_t": "deep_forest"}}, spec_module.UNKNOWN_NUISANCE_MODEL),
    ({"nuisance": {"model_y": "lasso"}}, spec_module.MALFORMED_SPEC),
    ({"roles": {"treatment": "instrument", "outcome": "outcome", "confounder": "confounder",
                "baseline": "modifier"}}, spec_module.UNKNOWN_ROLE),
    ({"roles": {"outcome": "outcome", "confounder": "confounder", "baseline": "modifier",
                "treatment": "exclude"}}, spec_module.NO_TREATMENT),
    ({"roles": {"treatment": "treatment", "outcome": "treatment", "confounder": "confounder",
                "baseline": "modifier"}}, spec_module.TWO_TREATMENTS),
    ({"roles": {"treatment": "treatment", "outcome": "exclude", "confounder": "confounder",
                "baseline": "modifier"}}, spec_module.NO_OUTCOME),
    ({"roles": {"treatment": "treatment", "outcome": "outcome", "confounder": "outcome",
                "baseline": "modifier"}}, spec_module.TWO_OUTCOMES),
    ({"roles": {"treatment": "treatment", "outcome": "outcome", "confounder": "modifier",
                "baseline": "modifier"}}, spec_module.TWO_MODIFIERS),
    ({"roles": {"treatment": "treatment", "outcome": "outcome", "baseline": "modifier"}},
     spec_module.UNASSIGNED_COLUMN),
    ({"roles": {"treatment": "treatment", "outcome": "outcome", "confounder": "confounder",
                "baseline": "modifier", "rainfall": "confounder"}}, spec_module.UNKNOWN_COLUMN),
    ({"identification": []}, spec_module.NOT_IDENTIFIED),
    ({"identification": ["sufficient_adjustment", "telepathy"]}, spec_module.UNKNOWN_ASSUMPTION),
    ({"identification": ["sufficient_adjustment"]}, spec_module.INCOMPLETE_IDENTIFICATION),
    ({"confidence_level": 0.5}, spec_module.UNKNOWN_CONFIDENCE_LEVEL),
    ({"confidence_level": "0.99"}, spec_module.UNKNOWN_CONFIDENCE_LEVEL),
    ({"decisions": ["not-a-digest"]}, spec_module.MALFORMED_DECISION_DIGEST),
    ({"decisions": "abc"}, spec_module.MALFORMED_SPEC),
    ({"provenance": "PRODUCTION"}, spec_module.UNKNOWN_PROVENANCE),
    ({"provenance": "UNDECLARED"}, spec_module.UNKNOWN_PROVENANCE),
    ({"schema": "m5phet.causal_study_spec.v2"}, spec_module.MALFORMED_SPEC),
    ({"dataset": {"columns": ["treatment", "outcome"]}}, spec_module.UNKNOWN_DATASET),
    ({"dataset": {"id": "eurusd-2019", "columns": ["baseline", "confounder", "treatment", "outcome"]}},
     spec_module.UNKNOWN_DATASET),
    ({"study_id": "NOT A STUDY ID"}, spec_module.MALFORMED_SPEC),
])
def test_refused_by_name(change, refusal):
    assert refusal_of(valid_spec(**change)) == refusal


def test_a_constant_effect_study_may_not_state_the_assumption_it_sets_aside():
    """A CATE study cannot claim the effect is constant; an ATE study cannot claim linearity in modifiers it has none
    of. Both are assumptions the space declares, and both are refused where they do not belong."""
    assert refusal_of(valid_spec(identification=list(SPACE["identification_assumptions"]["required_for"]["ATE"]))) \
        == spec_module.INAPPLICABLE_ASSUMPTION
    average = valid_spec(dataset={"id": "synthetic-constant", "columns": ["baseline", "treatment", "outcome"]},
                         roles={"treatment": "treatment", "outcome": "outcome", "baseline": "confounder"})
    assert refusal_of(average) == spec_module.INAPPLICABLE_ASSUMPTION


def test_a_modifier_with_an_estimator_that_has_no_cate_is_refused_by_name():
    """No estimator this repository offers today is average-effect-only, so the refusal is shown against a space that
    declares one -- which is what `validate_spec(spec, space)` takes a space for."""
    narrowed = deepcopy(SPACE)
    narrowed["estimators"]["detail"]["LinearDML"]["estimands"] = ["ATE"]
    with pytest.raises(SpecRefused) as raised:
        validate_spec(valid_spec(), narrowed)
    assert raised.value.refusal == spec_module.ESTIMATOR_HAS_NO_CATE
    assert "baseline" in raised.value.why


def test_an_estimator_that_cannot_serve_an_average_effect_is_refused_by_name():
    average = valid_spec(estimator="CausalForestDML",
                         dataset={"id": "synthetic-constant", "columns": ["baseline", "treatment", "outcome"]},
                         roles={"treatment": "treatment", "outcome": "outcome", "baseline": "confounder"},
                         identification=list(SPACE["identification_assumptions"]["required_for"]["ATE"]))
    assert refusal_of(average) == spec_module.ESTIMATOR_HAS_NO_ATE


def test_a_synthetic_dataset_cannot_be_declared_anything_but_development():
    assert refusal_of(valid_spec(provenance="UNDECLARED")) == spec_module.UNKNOWN_PROVENANCE
    # a CSV path may be either, because this repository does not know where the file came from
    on_disk = valid_spec(provenance="UNDECLARED",
                         dataset={"path": "/tmp/whatever.csv",
                                  "columns": ["baseline", "confounder", "treatment", "outcome"]})
    assert validate_spec(on_disk, SPACE)["provenance"] == "UNDECLARED"


def test_a_spec_names_its_own_study_or_is_named_by_its_content():
    assert study_identifier(valid_spec(study_id="wp20-demo-v1")) == "wp20-demo-v1"
    derived = study_identifier(valid_spec())
    assert derived.startswith("study-") and derived[len("study-"):] == spec_digest(valid_spec())[:12]


def test_decision_digests_travel_with_the_spec():
    digest = "a" * 64
    spec = validate_spec(valid_spec(decisions=[digest]), SPACE)
    assert spec["decisions"] == [digest]
    assert refusal_of(valid_spec(decisions=[digest, digest])) == spec_module.MALFORMED_SPEC


def test_the_space_the_catalog_shows_imports_no_fit_dependency(tmp_path):
    """The workbench serves this provider from an interpreter that holds no EconML, and a catalog call there must not
    try to import one. The space it shows is the declaration; availability is settled where the fitting happens."""
    import subprocess
    import sys

    script = '''
import builtins, json, sys
real_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split(".")[0] in {"numpy", "pandas", "scipy", "sklearn", "econml", "statsmodels"}:
        raise AssertionError("serving imported fit dependency: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guard
from causal_inference_provider.chat import M5PHETCausalProvider
space = M5PHETCausalProvider(sys.argv[1]).capabilities()["study_space"]
assert space["probe"] == "declared", space["probe"]
assert [key for key, _ in space["estimators"]["options"]], space
assert [key for key, _ in space["roles"]["options"]] == ["treatment", "outcome", "confounder", "modifier", "exclude"]
json.dumps(space, allow_nan=False)
'''
    empty = tmp_path / "studies"
    empty.mkdir()
    finished = subprocess.run([sys.executable, "-c", script, str(empty)], capture_output=True, text=True, timeout=60)
    assert finished.returncode == 0, finished.stderr
