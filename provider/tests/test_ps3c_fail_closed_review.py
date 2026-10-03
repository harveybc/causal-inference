"""Synthetic regression probes for the PS3-C methodological audit.

These tests use planted rows and patched nuisance predictions only. They never load campaign
artifacts, fit real-market data, or invoke a GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference_provider import ps3c
from causal_inference_provider import ps3c_batch
from causal_inference_provider import ps3c_review


DAG = {"nodes": ["W", "A", "Y"], "edges": [["W", "A"], ["W", "Y"], ["A", "Y"]]}


def _binary_fixture(n=240):
    rng = np.random.default_rng(19)
    w = rng.normal(size=n)
    a = (rng.random(n) < 0.5).astype(float)
    y = a + 0.1 * w + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"episode_id": [f"e{i}" for i in range(n)],
                         "decision_time": pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC"),
                         "W": w, "A": a, "Y": y, "Ypre": rng.normal(size=n)})


def _identified_call(frame, **kwargs):
    return ps3c.rung2_effect(
        frame, treatment="A", outcome="Y", adjustment=["W"], contrast=(1.0, 0.0), dag=DAG,
        treatment_kind="BINARY", node_columns={"W": ["W"]}, placebo_outcomes=[], n_boot=5,
        assumptions={name: True for name in ps3c.REQUIRED_ASSUMPTIONS},
        assumption_evidence={name: "planted synthetic SCM" for name in ps3c.REQUIRED_ASSUMPTIONS},
        **kwargs,
    )


def test_balance_above_declared_smd_bound_cannot_be_identified(monkeypatch):
    monkeypatch.setattr(ps3c.st, "crossfit_predict", lambda x, y, logistic_model=False: np.full(len(y), 0.5))
    monkeypatch.setattr(ps3c.st, "smd", lambda *args, **kwargs: 0.10001)
    out = _identified_call(_binary_fixture())
    assert out["state"] == ps3c.NOT_IDENTIFIED
    assert out["estimate"] is None
    assert "IMBALANCE" in out["reasons"]


def test_caller_cannot_relax_smd_bound_above_spec(monkeypatch):
    monkeypatch.setattr(ps3c.st, "crossfit_predict", lambda x, y, logistic_model=False: np.full(len(y), 0.5))
    monkeypatch.setattr(ps3c.st, "smd", lambda *args, **kwargs: 0.2)
    out = _identified_call(_binary_fixture(), support={"balance_bound": 1.0})
    assert out["state"] == ps3c.NOT_IDENTIFIED and out["estimate"] is None
    assert "BALANCE_BOUND_EXCEEDS_SPEC" in out["reasons"]


def test_trimming_option_is_rejected_even_when_all_rows_are_in_overlap(monkeypatch):
    monkeypatch.setattr(ps3c.st, "crossfit_predict", lambda x, y, logistic_model=False: np.full(len(y), 0.5))
    out = _identified_call(_binary_fixture(), support={"restrict_to_overlap": True})
    assert out["state"] == ps3c.NOT_IDENTIFIED and out["estimate"] is None
    assert "TRIMMING_FORBIDDEN" in out["reasons"]


def test_caller_cannot_widen_propensity_overlap_bounds(monkeypatch):
    monkeypatch.setattr(ps3c.st, "crossfit_predict", lambda x, y, logistic_model=False: np.full(len(y), 0.5))
    out = _identified_call(_binary_fixture(), support={"propensity_bounds": (0.01, 0.99)})
    assert out["state"] == ps3c.NOT_IDENTIFIED and out["estimate"] is None
    assert "PROPENSITY_BOUNDS_EXCEED_SPEC" in out["reasons"]


def test_propensity_outside_bounds_fails_without_trimming(monkeypatch):
    frame = _binary_fixture()

    def propensities(x, y, logistic_model=False):
        values = np.full(len(y), 0.5)
        values[0] = 0.01
        return values

    monkeypatch.setattr(ps3c.st, "crossfit_predict", propensities)
    out = _identified_call(frame)
    assert out["state"] == ps3c.NOT_IDENTIFIED
    assert out["estimate"] is None
    assert "OVERLAP_SCREEN_FAILED" in out["reasons"]
    assert out["support"]["n_population"] == len(frame)
    assert "population_restricted_to_overlap_dropped" not in out["sensitivity"]


def test_missing_assumption_evidence_is_named_and_never_defaults_true():
    frame = _binary_fixture()
    out = ps3c.rung2_effect(frame, treatment="A", outcome="Y", adjustment=["W"], contrast=(1.0, 0.0),
                            dag=DAG, treatment_kind="BINARY", node_columns={"W": ["W"]},
                            assumptions=None, placebo_outcomes=[], n_boot=5)
    assert out["state"] == ps3c.NOT_IDENTIFIED
    assert out["estimate"] is None
    assert set(out["assumptions_unverified"]) == set(ps3c.REQUIRED_ASSUMPTIONS)
    assert all(f"ASSUMPTION_NOT_EVIDENCED_{name}" in out["reasons"] for name in ps3c.REQUIRED_ASSUMPTIONS)


def test_confirmation_requires_same_population_and_estimand():
    linear = {"population_n": 120, "population_sha256": "a" * 64, "estimand_id": "ATE:A:Y:1:0"}
    same = dict(linear)
    different_population = {**linear, "population_n": 119, "population_sha256": "b" * 64}
    different_estimand = {**linear, "estimand_id": "ATT:A:Y:1:0"}
    assert ps3c_review.confirmation_identity(linear, same)["compatible"] is True
    assert ps3c_review.confirmation_identity(linear, different_population)["compatible"] is False
    assert ps3c_review.confirmation_identity(linear, different_estimand)["compatible"] is False


def test_nonlinear_propensity_failure_preserves_population(monkeypatch):
    frame = _binary_fixture()

    def predictions(model_fn, x, y, k=5, classify=False, rows=None):
        values = np.full(len(y), 0.5)
        values[0] = 0.01
        return values

    monkeypatch.setattr(ps3c_review, "_crossfit", predictions)
    out = ps3c_review.nonlinear_aipw(frame, "Y", ["W"], n_boot=5)
    assert out["state"] == "OVERLAP_SCREEN_FAILED"
    assert out["population_n"] == len(frame)
    assert out["dropped_outside_overlap"] == 0


def test_nonlinear_confirmation_cannot_widen_overlap_bounds():
    out = ps3c_review.nonlinear_aipw(_binary_fixture(), "Y", ["W"], bounds=(0.01, 0.99))
    assert out["state"] == "OVERLAP_SCREEN_FAILED"
    assert out["reason"] == "PROPENSITY_BOUNDS_EXCEED_SPEC"


def test_multiplicity_scope_is_explicitly_batch_local():
    meta = ps3c_review.multiplicity_metadata(batch_id="batch_001", family_size=4, q=0.05)
    assert meta["scope"] == "BATCH_LOCAL"
    assert meta["global_campaign_adjustment"] == "NOT_APPLIED"


def test_batch_runner_does_not_assert_assumptions_for_every_cell():
    assert not hasattr(ps3c_batch, "ASSUMPTIONS")


def test_legacy_estimated_row_fails_closed_under_current_identification_gate():
    legacy = {"rung2": "ESTIMATED", "r2_support": "SUPPORTED", "r2_balance_max_smd": 1.2182,
              "r2_treatment_kind": "BINARY", "r2_propensity_range": "[0.041, 0.999]",
              "r2_assumptions_declared": "{}", "r2_assumptions_unverified": "[]",
              "r2_assumptions_evidence": "{}",
              "r2_population_n": None, "r2_population_sha256": None, "r2_estimand_id": None}
    gate = ps3c_review.summary_identification_gate(legacy)
    assert gate["eligible"] is False
    assert {"BALANCE_UNVERIFIED_OR_OVER_BOUND", "OVERLAP_SCREEN_FAILED",
            "ASSUMPTIONS_UNVERIFIED", "POPULATION_OR_ESTIMAND_IDENTITY_MISSING"} <= set(gate["reasons"])


def test_nonlinear_confirmation_identity_matches_linear_complete_case_population(monkeypatch):
    frame = _binary_fixture()

    def predictions(model_fn, x, y, k=5, classify=False, rows=None):
        return np.full(len(y), 0.5)

    monkeypatch.setattr(ps3c_review, "_crossfit", predictions)
    nl = ps3c_review.nonlinear_aipw(frame, "Y", ["W"], n_boot=5)
    linear = ps3c.estimand_population_identity(frame.episode_id.astype(str), treatment="A", outcome="Y",
                                               contrast=(1.0, 0.0), treatment_kind="BINARY")
    assert ps3c_review.confirmation_identity(linear, nl)["compatible"] is True
