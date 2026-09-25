"""The configuration space, checked against what this interpreter can actually build.

The point of these tests is that the space is not a wish list. An option is offered only when the class behind it
imports here, and an estimator is offered only when it really produces an effect AND an interval on data -- because the
engine releases no estimate without one. A class that leaves the environment leaves the space by name.
"""

import pytest

from causal_inference_provider import study_space as space_module
from causal_inference_provider.study_space import SPACE_SCHEMA, declared, importable, study_space

pytest.importorskip("econml")
pytest.importorskip("sklearn")

CANDIDATE_ESTIMATORS = {"LinearDML", "CausalForestDML", "DRLearner", "SparseLinearDML", "DML"}
CANDIDATE_NUISANCE = {"lasso", "ridge", "gradient_boosting", "random_forest"}


@pytest.fixture(scope="module")
def space():
    return study_space()


def keys(dimension):
    return [key for key, _ in dimension["options"]]


def test_every_option_is_a_key_and_a_one_line_label(space):
    assert space["schema"] == SPACE_SCHEMA
    for name in ("estimators", "nuisance_models", "roles", "identification_assumptions", "confidence_levels"):
        options = space[name]["options"]
        assert options, name
        for option in options:
            assert isinstance(option, list) and len(option) == 2, (name, option)
            key, label = option
            assert isinstance(key, str) and key.strip()
            assert isinstance(label, str) and label.strip() and "\n" not in label
        assert len(set(keys(space[name]))) == len(options), name


def test_the_space_holds_only_classes_this_interpreter_can_import(space):
    """Every offered class imports, and nothing outside the declared candidates is offered."""
    assert set(keys(space["estimators"])) <= CANDIDATE_ESTIMATORS
    assert set(keys(space["nuisance_models"])) <= CANDIDATE_NUISANCE
    for key in keys(space["estimators"]):
        module, _, name = space["estimators"]["detail"][key]["class"].rpartition(".")
        assert importable(module, name), key
    for key in keys(space["nuisance_models"]):
        detail = space["nuisance_models"]["detail"][key]
        for form in ("regressor", "classifier"):
            assert importable(detail[form]["module"], detail[form]["class"]), (key, form)


def test_this_interpreter_builds_every_declared_candidate(space):
    """In the fit venv nothing is missing, so the probed space and the declaration agree option for option."""
    assert set(keys(space["estimators"])) == CANDIDATE_ESTIMATORS
    assert set(keys(space["nuisance_models"])) == CANDIDATE_NUISANCE
    assert space["estimators"]["options"] == study_space(probe=declared)["estimators"]["options"]
    assert space["nuisance_models"]["options"] == study_space(probe=declared)["nuisance_models"]["options"]


def test_a_class_that_cannot_be_imported_disappears_by_name(monkeypatch, space):
    """Take one class out of the environment and the option it stood for is gone -- by name, not silently renamed."""
    real = space_module._import_class

    def missing(module, name):
        if name in ("CausalForestDML", "Ridge"):
            raise ImportError(f"no {name} here")
        return real(module, name)

    monkeypatch.setattr(space_module, "_import_class", missing)
    reduced = study_space()
    assert "CausalForestDML" not in keys(reduced["estimators"])
    assert "CausalForestDML" not in reduced["estimators"]["detail"]
    assert "ridge" not in keys(reduced["nuisance_models"])
    assert "ridge" not in reduced["nuisance_models"]["detail"]
    # and nothing else moved: the options that remain are the ones that still import
    assert set(keys(reduced["estimators"])) == set(keys(space["estimators"])) - {"CausalForestDML"}
    assert set(keys(reduced["nuisance_models"])) == set(keys(space["nuisance_models"])) - {"ridge"}
    # the declaration is unaffected: it is what the serving interpreter shows, and it imports nothing
    assert "CausalForestDML" in keys(study_space(probe=declared)["estimators"])


def test_serving_probe_imports_nothing(monkeypatch):
    """`declared` is the probe the catalog uses; it must not reach a fit dependency."""
    def forbidden(module, name):
        raise AssertionError("the declared space imported " + module)

    monkeypatch.setattr(space_module, "_import_class", forbidden)
    shown = study_space(probe=declared)
    assert set(keys(shown["estimators"])) == CANDIDATE_ESTIMATORS
    assert shown["probe"] == "declared"


def test_roles_are_the_five_a_column_can_take(space):
    assert keys(space["roles"]) == ["treatment", "outcome", "confounder", "modifier", "exclude"]


def test_assumptions_are_the_engine_s_own_words(space):
    from causal_inference_provider.provider import ASSUMPTIONS, MODIFIER_ASSUMPTIONS, assumptions_for

    assert set(keys(space["identification_assumptions"])) == set(ASSUMPTIONS) | set(MODIFIER_ASSUMPTIONS)
    for estimand in ("ATE", "CATE"):
        assert space["identification_assumptions"]["required_for"][estimand] == list(assumptions_for(estimand))


def test_confidence_levels_are_the_two_declared(space):
    assert sorted(space["confidence_levels"]["values"].values()) == [0.9, 0.95]
    assert set(keys(space["confidence_levels"])) == set(space["confidence_levels"]["values"])


def test_every_estimand_an_estimator_declares_is_one_the_engine_knows(space):
    for key in keys(space["estimators"]):
        estimands = space["estimators"]["detail"][key]["estimands"]
        assert estimands and set(estimands) <= {"ATE", "CATE"}, key


@pytest.mark.parametrize("estimator", sorted(CANDIDATE_ESTIMATORS))
def test_every_offered_estimator_really_returns_an_effect_with_an_interval(estimator, space):
    """An option that cannot produce an interval is an option that would always be refused. None is offered."""
    from causal_inference_provider.example import modifier_example_config, modifier_example_data
    from causal_inference_provider.provider import CausalInferenceProvider

    if estimator not in keys(space["estimators"]):
        pytest.skip(f"{estimator} does not import in this interpreter")
    config = modifier_example_config() | {"estimator": estimator,
                                          "nuisance": {"model_y": "lasso", "model_t": "lasso"}}
    provider = CausalInferenceProvider()
    assert provider.load(config)["status"] == "OK"
    result = provider.fit(modifier_example_data())
    assert result["status"] == "OK", result["reason"]
    payload = result["payload"]
    low, high = payload["interval"]
    assert low < payload["estimate"] < high
    assert payload["diagnostics"]["engine"] == space["estimators"]["detail"][estimator]["class"]
    assert payload["diagnostics"]["uncertainty_method"] == \
        space["estimators"]["detail"][estimator]["uncertainty_method"]
    carried = {entry["subgroup"]: entry for entry in payload["conditional_effects"]}
    assert sorted(carried) == ["baseline == 0", "baseline == 1"]
    for entry in carried.values():
        assert entry["interval"][0] < entry["estimate"] < entry["interval"][1]


def test_an_estimator_that_cannot_serve_an_estimand_says_so(space):
    """CausalForestDML is fitted ON the modifiers; EconML refuses X=None, so it declares CATE and not ATE."""
    assert space["estimators"]["detail"]["CausalForestDML"]["estimands"] == ["CATE"]
    from causal_inference_provider.example import example_config
    from causal_inference_provider.provider import CausalInferenceProvider

    refused = CausalInferenceProvider().load(example_config() | {"estimator": "CausalForestDML"})
    assert refused["status"] == "UNSUPPORTED_TASK"
    assert "CausalForestDML" in refused["reason"]


def test_an_undeclared_estimator_or_nuisance_model_is_refused_by_name():
    from causal_inference_provider.example import example_config
    from causal_inference_provider.provider import CausalInferenceProvider

    provider = CausalInferenceProvider()
    assert provider.load(example_config() | {"estimator": "MyOwnEstimator"})["status"] == "UNSUPPORTED_TASK"
    refused = provider.load(example_config() | {"nuisance": {"model_y": "neural_net", "model_t": "lasso"}})
    assert refused["status"] == "UNSUPPORTED_TASK" and "neural_net" in refused["reason"]
    assert provider.load(example_config() | {"nuisance": {"model_y": "lasso"}})["status"] == "INVALID_INPUT"


def test_a_study_that_declares_nothing_is_fitted_exactly_as_before():
    """The two studies retained before the space existed declare no estimator and no nuisance pair; their config, and
    so their identity, must not acquire one."""
    from causal_inference_provider.example import example_config, example_data
    from causal_inference_provider.provider import CausalInferenceProvider

    provider = CausalInferenceProvider()
    assert provider.load(example_config())["status"] == "OK"
    assert "estimator" not in provider.state["config"] and "nuisance" not in provider.state["config"]
    payload = provider.fit(example_data())["payload"]
    assert payload["diagnostics"]["engine"] == "econml.dml.LinearDML"
    assert payload["diagnostics"]["uncertainty_method"] == "econml_statsmodels_HC1_normal"
    assert payload["estimate"] == pytest.approx(2.0, abs=0.2)
