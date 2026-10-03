"""End-to-end PS3-C batch runner on a synthetic lane-A batch (synthetic bytes only, CPU)."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from causal_inference_provider import ps3c, ps3c_batch


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def make_lane_a_batch(root, n_hours=24 * 900, seed=5):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2016-01-01T01:00:00Z", periods=n_hours, freq="h")
    t = t[t.dayofweek < 5]
    n = len(t)
    r = rng.normal(scale=1e-3, size=n)
    lc = np.cumsum(r)
    X = pd.DataFrame({"t_decision_utc": t, "row_id": np.arange(n)})
    s = pd.Series(lc)
    for k in (1, 6, 24, 120):
        X[f"px.logret_{k}h"] = s - s.shift(k)
    rr = pd.Series(r)
    X["px.ewma_vol_24"] = np.sqrt((rr ** 2).ewm(halflife=24).mean())
    X["px.ewma_vol_168"] = np.sqrt((rr ** 2).ewm(halflife=168).mean())
    h = t.hour
    X["cal.hour_sin"], X["cal.hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    X["cal.dow_sin"], X["cal.dow_cos"] = np.sin(2 * np.pi * t.dayofweek / 7), np.cos(2 * np.pi * t.dayofweek / 7)
    X["ta.osc"] = pd.Series(rng.normal(size=n)).rolling(12, min_periods=1).mean()  # autocorrelated, no effect
    Y = pd.DataFrame({"t_decision_utc": t, "row_id": np.arange(n)})
    for hh in (1, 2, 3, 4, 5, 6, 24, 48, 72, 96, 120, 144):
        fam = "Y_s" if hh <= 6 else "Y_l"
        Y[f"{fam}_{hh}h"] = s.shift(-hh) - s
    Y["Y_b_s6"] = np.sign(Y["Y_s_6h"]).fillna(0)
    Y["Y_b_l144"] = np.sign(Y["Y_l_144h"]).fillna(0)
    os.makedirs(root, exist_ok=True)
    X.to_parquet(os.path.join(root, "features_train.parquet"), index=False)
    Y.to_parquet(os.path.join(root, "targets_train.parquet"), index=False)
    years = sorted(set(t.year))[1:]
    folds = []
    for y in years:
        tr = np.where(t < pd.Timestamp(f"{y}-01-01", tz="UTC") - pd.Timedelta(hours=150))[0]
        va = np.where(t.year == y)[0]
        folds.append({"name": f"inner_{y}", "train_rows": [int(tr.min()), int(tr.max()) + 1],
                      "val_rows": [int(va.min()), int(va.max()) + 1]})
    json.dump({"folds": folds}, open(os.path.join(root, "folds.json"), "w"))
    json.dump({"schema": "eurusd_business_contract.v1", "asset": "EURUSD", "contract_sha256": "a" * 64,
               "periods": {"train": [str(t[0]), str(t[-1] + pd.Timedelta(hours=1))]}},
              open(os.path.join(root, "contract.json"), "w"))
    feats = [{"feature_id": c, "family": c.split(".")[0], "source": "synthetic", "role": "feature",
              "admissibility": "ADMISSIBLE"} for c in X.columns if "." in c]
    json.dump({"features": feats}, open(os.path.join(root, "admissible_features.json"), "w"))
    arts = {n_: _sha(os.path.join(root, n_)) for n_ in os.listdir(root)}
    json.dump({"artifacts_sha256": arts, "inputs_sha256": {"eurusd_5m.parquet": "b" * 64}, "code_commit": "synthetic"},
              open(os.path.join(root, "digests.json"), "w"))
    open(os.path.join(root, "READY"), "w").write(json.dumps({"digests_sha256": _sha(os.path.join(root, "digests.json"))}))
    return root


def test_batch_runner_end_to_end_on_synthetic_lane_a(tmp_path):
    lane = make_lane_a_batch(str(tmp_path / "laneA"))
    out = str(tmp_path / "out")
    ps3c_batch.main(["--lane-a-batch", lane, "--inputs", str(tmp_path), "--lane-a-code", str(tmp_path / "none"),
                     "--out", out, "--batch", "batch_test", "--revision", "0123456",
                     "--only-features", "ta.osc,px.logret_1h,cal.hour_sin", "--permutations", "50"])
    rep = json.load(open(os.path.join(out, "batch_report.json")))
    assert os.path.exists(os.path.join(out, "READY"))
    assert rep["denominators"]["cells"] == 3 * len(ps3c_batch.TARGETS)
    assert rep["schema_validation"]["with_errors"] == 0
    assert "error" in rep["events"]  # no lane-A code: the event study is recorded, not faked
    summ = pd.read_csv(os.path.join(out, "summary.csv"))
    cal = summ[summ.subject == "cal.hour_sin"]
    assert (cal.rung2 == "NOT_APPLICABLE").all() and (cal.rung3 == "NOT_APPLICABLE").all()
    states = {"ESTIMATED", "NOT_IDENTIFIED", "NOT_APPLICABLE", "FAILED", "PENDING"}
    assert set(summ.rung1) | set(summ.rung2) | set(summ.rung3) <= states
    assert (summ[summ.subject != "cal.hour_sin"].rung1 == "ESTIMATED").all()
    osc = rep["crossing"]["ta.osc"]
    assert osc["treated"] > 20 and "auc_crossfit" in osc["upstream_mechanism"]
    rec = json.load(open(os.path.join(out, "records", os.listdir(os.path.join(out, "records"))[0])))
    card = rec["candidate_card"]
    for field in ("question", "A", "Y", "H", "rung2", "rung3"):
        assert field in card
    for field in ("DAG", "adjustment_set", "support_overlap", "estimator", "diagnostics"):
        assert card["rung2"][field] is not None
    for field in ("SCM", "abduction", "historically_supported_alternative", "propagation",
                  "factual_reconstruction", "placebos", "sensitivity"):
        assert card["rung3"][field] is not None  # filled, or an explicit {"abstention": reason}
    for p in os.listdir(os.path.join(out, "dossiers")):
        doc = json.load(open(os.path.join(out, "dossiers", p)))
        assert ps3c.validate_dossier(doc) == []
        if doc["rung2"]["state"] != ps3c.IDENTIFIED:
            assert doc["rung2"].get("estimate") is None
    with pytest.raises(SystemExit, match="already READY"):
        ps3c_batch.main(["--lane-a-batch", lane, "--inputs", str(tmp_path), "--lane-a-code", "x", "--out", out,
                         "--batch", "b", "--revision", "0123456"])


def test_batch_runner_refuses_tampered_lane_a_bytes(tmp_path):
    lane = make_lane_a_batch(str(tmp_path / "laneA"), n_hours=24 * 300)
    t = pd.read_parquet(os.path.join(lane, "targets_train.parquet"))
    t.loc[5, "Y_s_1h"] = 9.0
    t.to_parquet(os.path.join(lane, "targets_train.parquet"), index=False)
    with pytest.raises(SystemExit, match="digest mismatch"):
        ps3c_batch.main(["--lane-a-batch", lane, "--inputs", str(tmp_path), "--lane-a-code", "x",
                         "--out", str(tmp_path / "o"), "--batch", "b", "--revision", "0123456"])


def test_crossing_episode_membership_never_depends_on_the_future(tmp_path):
    """With the TRAIN threshold fixed, perturbing the feature after row k leaves every earlier episode unchanged."""
    lane = make_lane_a_batch(str(tmp_path / "laneA"), n_hours=24 * 400)
    X, Y, *_ = ps3c_batch.load_batch(lane)
    x = X["ta.osc"].to_numpy()
    thr, band = np.nanquantile(x, 0.8), np.nanquantile(x, 0.6)
    k = len(X) // 2
    a, _ = ps3c_batch.crossing_episodes(X, Y, "ta.osc", threshold=thr, band=band)
    X2 = X.copy()
    X2.loc[X2.index[k:], "ta.osc"] = x[k:] * -3.0 + 1.0
    b, _ = ps3c_batch.crossing_episodes(X2, Y, "ta.osc", threshold=thr, band=band)
    cut = X["t_decision_utc"].iloc[k - 1]
    cols = ["decision_time", "A", "W_x_prev"]
    ea = a[a.decision_time <= cut][cols].reset_index(drop=True)
    eb = b[b.decision_time <= cut][cols].reset_index(drop=True)
    assert len(ea) > 50
    pd.testing.assert_frame_equal(ea, eb)


def test_controls_are_not_selected_on_a_future_crossing():
    t = pd.date_range("2020-01-06T01:00:00Z", periods=400, freq="h")
    x = np.full(400, 0.65)
    x[::50] = 0.0
    x[200] = 1.0  # one crossing at row 200
    X = pd.DataFrame({"t_decision_utc": t, "row_id": np.arange(400), "f": x})
    Y = pd.DataFrame({"t_decision_utc": t, "row_id": np.arange(400),
                      **{name: np.zeros(400) for name, *_ in ps3c_batch.TARGETS}})
    ep, info = ps3c_batch.crossing_episodes(X, Y, "f", threshold=0.8, band=0.5, stride_h=1)
    assert ep is not None, info
    ctrl_times = ep.loc[ep.A == 0, "decision_time"]
    cross_t = t[200]
    # rows in the 24h BEFORE the crossing remain eligible controls; rows within 24h AFTER it are excluded
    assert ((ctrl_times < cross_t) & (ctrl_times >= cross_t - pd.Timedelta(hours=24))).any()
    assert not ((ctrl_times > cross_t) & (ctrl_times < cross_t + pd.Timedelta(hours=24))).any()
