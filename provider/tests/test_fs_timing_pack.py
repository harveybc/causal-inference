"""EURUSD_TIMING pack: validation, HAC lag screen, PCMCI record, matched-control contrast (TP01-TP06)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference_provider import feature_selection_worker as W
from causal_inference_provider import fs_timing_pack as T


def _bars(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    r = rng.normal(0, 1e-3, n)
    close = 1.1 * np.exp(np.cumsum(r))
    high = close * (1 + np.abs(rng.normal(0, 3e-4, n)))
    low = close * (1 - np.abs(rng.normal(0, 3e-4, n)))
    return idx, close, high, low, rng


def test_tp01_pack_is_registered_beside_the_existing_packs_unchanged():
    """Append-only registration: the pinned EURUSD/ETH entries, cell counts, denominators and feature count are
    byte-identical to the pre-extension values (owner constraint: nothing that exists for prediction changes)."""
    assert T.PACK_ID in W.TARGET_PACKS
    assert W.TARGET_CELL_COUNTS[T.PACK_ID] == len(T.timing_target_definitions(2))
    assert W.TARGET_PACKS - {T.PACK_ID} == {"EURUSD", "ETH"}
    assert W.TARGET_CELL_COUNTS["EURUSD"] == 14 and W.TARGET_CELL_COUNTS["ETH"] == 6
    assert W.EURUSD_FEATURE_COUNT == 366 and len(W.EURUSD_TARGET_DENOMINATOR) == 14


def test_tp02_definitions_load_as_target_definitions():
    defs = [W.TargetDefinition.from_dict(d) for d in T.timing_target_definitions(2)]
    assert {d.head for d in defs} == {"classification", "regression"}
    assert all(d.family == T.FAMILY and d.horizon_hours == 2 for d in defs)


def test_tp03_validation_refuses_multiclass_labels_and_wrong_family():
    targets = pd.DataFrame({"timing_valley_k2": [0.0, 1.0, 2.0], "timing_expert_leg_pnl_pips": [1.0, -2.0, np.nan]})
    pack = {"id": T.PACK_ID, "definitions": [T.timing_target_definitions(2)[0]]}
    with pytest.raises(T.TimingPackError, match="indicator"):
        T.validate_timing_pack(pack, targets)
    bad = {"id": T.PACK_ID, "definitions": [dict(T.timing_target_definitions(2)[3], family="Y_s")]}
    with pytest.raises(T.TimingPackError, match="family"):
        T.validate_timing_pack(bad, targets)
    with pytest.raises(T.TimingPackError, match="target_pack.id"):
        T.validate_timing_pack({"id": "EURUSD", "definitions": []}, targets)


def test_tp04_hac_lag_screen_finds_a_lagged_parent_and_controls_the_null():
    idx, close, high, low, rng = _bars()
    n = len(idx)
    cause = rng.normal(size=n)
    y = (cause.copy()[np.r_[0, np.arange(n - 1)]] * 0.8 + rng.normal(size=n) > 1.0).astype(float)   # parent at lag 1
    X = pd.DataFrame({"cause": cause, **{f"noise{i}": rng.normal(size=n) for i in range(10)},
                      "px.rv_24": np.abs(rng.normal(size=n))})
    rep = T.hac_lag_screen(X, y, k=2, history=["px.rv_24"], q=0.1)
    assert "cause" in rep["parents_at_lags_1_to_k"]
    assert len([f for f in rep["parents_at_lags_1_to_k"] if f.startswith("noise")]) <= 2


def test_tp05_pcmci_records_not_run_without_tigramite():
    X = pd.DataFrame({"a": np.arange(300.0), "px.rv_24": np.ones(300)})
    rec = T.pcmci_parents(X, np.zeros(300), k=2, history=["px.rv_24"], target_name="timing_valley_k2")
    assert rec["state"] in ("NOT_RUN", "RUN")
    if rec["state"] == "NOT_RUN":
        assert "tigramite" in rec["reason"]


def test_tp06_matched_control_contrast_recovers_a_planted_excess_and_is_null_on_noise():
    idx, close, high, low, rng = _bars(6000, seed=3)
    n = len(idx)
    bars = T.bars_table(idx, close, high, low)
    feat = rng.normal(size=n)
    base = rng.random(n) < 0.2
    thr = np.quantile(feat, 0.8)
    above = feat > thr
    cross = above & ~np.r_[False, above[:-1]]
    planted = base.copy()
    planted[cross] = rng.random(cross.sum()) < 0.6
    rec = T.matched_control_contrast(bars, feat, planted.astype(float), k=2, seed=1)
    assert rec["state"] == "ESTIMATED" and rec["excess_pivot_rate"] > 0.2 and rec["significant"]
    null = T.matched_control_contrast(bars, rng.normal(size=n), base.astype(float), k=2, seed=1)
    assert null["state"] == "ESTIMATED" and abs(null["excess_pivot_rate"]) < 0.08
