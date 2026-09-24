"""Real numerical alpha tests; all data here are synthetic and IID."""

from copy import deepcopy
import json
import time

import numpy as np
import pandas as pd
import pytest

from causal_inference_provider import CausalInferenceProvider


@pytest.fixture
def config():
    return {
        "estimand": "ATE",
        "treatment": "treatment",
        "outcome": "outcome",
        "adjustments": ["baseline"],
        "unit": "synthetic outcome units",
        "assumptions": {
            "sufficient_adjustment": True,
            "pre_treatment_adjustment": True,
            "consistency": True,
            "no_interference": True,
            "positivity": True,
            "iid_sampling": True,
            "constant_effect": True,
            "nuisance_models_correct": True,
        },
    }


@pytest.fixture
def data():
    rng = np.random.default_rng(17)
    z = rng.binomial(1, 0.5, 2400)
    t = rng.binomial(1, 0.2 + 0.6 * z)
    y = 2.0 * t + 4.0 * z + rng.normal(size=len(t))
    return pd.DataFrame({"baseline": z, "treatment": t, "outcome": y})


def fitted(config, data):
    provider = CausalInferenceProvider()
    assert provider.load(config)["status"] == "OK"
    assert provider.fit(data)["status"] == "OK"
    return provider


def test_core_lifecycle(config, data, monkeypatch):
    provider = CausalInferenceProvider()
    caps = provider.capabilities()
    assert caps["family"] == "causal_inference"
    assert caps["output_kinds"] == ["causal_effect"]
    assert set(caps["operations"]) == {"load", "fit", "infer"}
    assert caps["infer_requires_fit"] is True
    assert provider.state["phase"] == "NEW"
    assert provider.infer()["status"] == "MODEL_NOT_FITTED"
    assert provider.fit(data)["status"] == "MODEL_NOT_FITTED"
    assert provider.load(config)["status"] == "OK"
    assert provider.state["phase"] == "CONFIGURED"
    assert provider.fit(data)["status"] == "OK"
    state = provider.state
    assert state["phase"] == "FITTED"
    from econml.dml import LinearDML

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("infer must never fit")

    monkeypatch.setattr(LinearDML, "fit", forbidden_fit)
    first = provider.infer()
    assert first == provider.infer()
    assert state == provider.state
    first["payload"]["estimate"] = -999
    assert provider.infer()["payload"]["estimate"] != -999
    json.dumps(provider.infer(), allow_nan=False)


def test_known_effect_and_actual_uncertainty(config, data):
    start = time.monotonic()
    result = fitted(config, data).infer()
    assert time.monotonic() - start < 30
    assert result["status"] == "OK"
    p = result["payload"]
    naive = data.groupby("treatment")["outcome"].mean().diff().iloc[-1]
    assert abs(naive - 2.0) > 1.5
    assert p["estimate"] == pytest.approx(2.0, abs=0.2)
    assert p["interval"][0] < 2.0 < p["interval"][1]
    assert 0 < p["interval"][1] - p["interval"][0] < 1.0
    assert p["estimand"] == "ATE"
    assert p["unit"] == config["unit"]
    assert p["assumptions"] == sorted(config["assumptions"])
    assert p["diagnostics"]["identification"] == "conditional_on_user_assumptions"
    assert p["diagnostics"]["n_rows"] == len(data)
    assert p["diagnostics"]["engine"] == "econml.dml.LinearDML"


def test_matches_real_library(config, data):
    from econml.dml import LinearDML
    from econml.inference import StatsModelsInference
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    direct = LinearDML(
        model_y=make_pipeline(StandardScaler(), LinearRegression()),
        model_t=make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=1000)),
        discrete_treatment=True,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=1729),
        random_state=1729,
    )
    direct.fit(data.outcome.to_numpy(), data.treatment.to_numpy(),
               W=data[["baseline"]].to_numpy(),
               inference=StatsModelsInference(cov_type="HC1"))
    p = fitted(config, data).infer()["payload"]
    assert p["estimate"] == pytest.approx(float(direct.ate()), abs=1e-10)
    lo, hi = direct.ate_interval(alpha=0.05)
    assert p["interval"] == pytest.approx([float(lo), float(hi)], abs=1e-10)


@pytest.mark.parametrize("effect", [0.0, -1.5])
def test_negative_control_and_sign(config, data, effect):
    data["outcome"] += (effect - 2.0) * data.treatment
    p = fitted(config, data).infer()["payload"]
    assert p["estimate"] == pytest.approx(effect, abs=0.2)
    assert p["interval"][0] < effect < p["interval"][1]


@pytest.mark.parametrize("assumption", [
    "sufficient_adjustment", "pre_treatment_adjustment", "consistency",
    "no_interference", "positivity", "iid_sampling", "constant_effect",
    "nuisance_models_correct",
])
@pytest.mark.parametrize("declaration", [False, "true", 1, None])
def test_no_implicit_assumptions(config, assumption, declaration):
    config["assumptions"][assumption] = declaration
    p = CausalInferenceProvider()
    result = p.load(config)
    assert result["status"] == "NOT_IDENTIFIED"
    assert result["payload"] is None
    assert p.infer()["status"] == "NOT_IDENTIFIED"


def test_missing_assumptions(config):
    del config["assumptions"]
    assert CausalInferenceProvider().load(config)["status"] == "NOT_IDENTIFIED"
    config["assumptions"] = {}
    assert CausalInferenceProvider().load(config)["status"] == "NOT_IDENTIFIED"


@pytest.mark.parametrize("change,status", [
    ({"estimand": "ATT"}, "UNSUPPORTED_TASK"),
    ({"adjustments": ["outcome"]}, "INVALID_INPUT"),
    ({"adjustments": ["treatment"]}, "INVALID_INPUT"),
    ({"adjustments": ["baseline", "baseline"]}, "INVALID_INPUT"),
    ({"outcome": "treatment"}, "INVALID_INPUT"),
    ({"adjustments": "baseline"}, "INVALID_INPUT"),
    ({"unit": ""}, "INVALID_INPUT"),
    ({"alpha": 1}, "INVALID_INPUT"),
    ({"alpha": float("nan")}, "INVALID_INPUT"),
    ({"prompt": "pick controls for me"}, "INVALID_INPUT"),
])
def test_config_validation(config, change, status):
    config.update(change)
    result = CausalInferenceProvider().load(config)
    assert result["status"] == status
    assert result["payload"] is None


@pytest.mark.parametrize("bad_input", ["infer cause from this text", [], [{"treatment": 1}], None])
def test_bad_data(config, bad_input):
    p = CausalInferenceProvider()
    p.load(config)
    result = p.fit(bad_input)
    assert result["status"] == "INVALID_INPUT"
    assert result["payload"] is None


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "2.5"])
def test_no_coercion_or_silent_drop(config, data, bad_value):
    data["outcome"] = data.outcome.astype(object)
    data.loc[0, "outcome"] = bad_value
    p = CausalInferenceProvider()
    p.load(config)
    assert p.fit(data)["status"] == "INVALID_INPUT"


def test_unsupported_treatment(config, data):
    data["treatment"] *= 0.5
    p = CausalInferenceProvider()
    p.load(config)
    assert p.fit(data)["status"] == "UNSUPPORTED_TASK"


@pytest.mark.parametrize("mode", ["single_arm", "no_overlap", "rare_arm"])
def test_overlap_refusal(config, data, mode):
    if mode == "single_arm":
        data["treatment"] = 1
    elif mode == "no_overlap":
        data["treatment"] = data.baseline
    else:
        data["treatment"] = 0
        data.loc[:4, "treatment"] = 1
    p = CausalInferenceProvider()
    p.load(config)
    assert p.fit(data)["status"] == "NOT_IDENTIFIED"
    assert p.infer()["payload"] is None


def test_resource_limit(config, data):
    p = CausalInferenceProvider()
    p.load(config)
    assert p.fit(pd.concat([data] * 5))["status"] == "RESOURCE_EXCEEDED"


def test_reload_failed_fit_and_defensive_state(config, data):
    original = data.copy(deep=True)
    p = fitted(config, data)
    before = p.infer()
    config["unit"] = "changed"
    state = p.state
    state["config"]["unit"] = "also changed"
    assert p.infer() == before
    pd.testing.assert_frame_equal(data, original)
    assert p.fit("bad")["status"] == "INVALID_INPUT"
    assert p.state["phase"] == "REFUSED"
    assert p.infer()["payload"] is None
    assert p.fit(data)["status"] == "OK"
    p.load(config)
    assert p.infer()["status"] == "MODEL_NOT_FITTED"
    assert p.load({})["status"] != "OK"
    assert p.infer()["payload"] is None


def test_records_replay_units_and_interval_level(config, data):
    a = fitted(config, data).infer()
    assert a == fitted(config, data.to_dict("records")).infer()
    wider = deepcopy(config)
    wider["alpha"] = 0.01
    b = fitted(wider, data).infer()["payload"]
    assert b["interval"][0] < a["payload"]["interval"][0]
    assert b["interval"][1] > a["payload"]["interval"][1]
    scaled = data.copy()
    scaled["outcome"] *= 10
    c = fitted(config, scaled).infer()["payload"]
    assert c["estimate"] == pytest.approx(10 * a["payload"]["estimate"])
    assert c["interval"] == pytest.approx(np.array(a["payload"]["interval"]) * 10)


def test_randomized_no_adjustment(config):
    config["adjustments"] = []
    rng = np.random.default_rng(27)
    t = rng.binomial(1, 0.5, 2400)
    data = pd.DataFrame({"treatment": t, "outcome": 2 * t + rng.normal(size=len(t))})
    assert fitted(config, data).infer()["payload"]["estimate"] == pytest.approx(2, abs=0.2)
