"""FS-CAUSAL selector on planted synthetic worlds where the truth is known (closure order 2026-10-05 section 5).

* a true cause ends SUPPORTED (rung 1 after family BH; rung 2 identified + nonlinear confirmation; rung 3 counterfactual);
* a confounded pair with no overlap stays NOT_IDENTIFIED (FS10, FS12);
* a planted anti-effect (association sign opposite to the identified effect) ends CONTRADICTED; a precise null too;
* FS11: the counterfactual keeps the episode's own noise, propagates descendants, refuses the live future;
* refusals: future timestamp, missing/assumed availability clock, empty strata;
* the three assumption-evidence references name tests that exist here and pass.

Synthetic, seeded, numpy/pandas only, CPU, hundreds to a few thousand rows; numbers are arbitrary and never a market.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from causal_inference_provider import fs_causal as fc
from causal_inference_provider import ps3c
from causal_inference_provider import ps3c_batch as B
from causal_inference_provider import ps3c_review as R
from causal_inference_provider import ps3c_stats as st

SEED = 1729


# ------------------------------------------------------------------------------------------ worlds


def hourly_world(n=6000, effect=0.0, phi=0.9, seed=SEED, n_null=0):
    """Decision grid with history H, a candidate x (AR(1)) whose q80 crossing moves the next-hour return by ``effect``."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    r = rng.normal(scale=1.0, size=n)
    thr = np.quantile(x, 0.8)
    cross = np.r_[False, (x[:-1] < thr) & (x[1:] >= thr)]
    r[1:] += effect * cross[:-1]  # the return of the bar after the crossing
    H = np.column_stack([rng.normal(size=n) for _ in range(3)])
    t = pd.date_range("2016-01-04", periods=n, freq="h", tz="UTC")
    X = pd.DataFrame({"t_decision_utc": t, "row_id": np.arange(n), "x": x, "h1": H[:, 0], "h2": H[:, 1], "h3": H[:, 2]})
    for j in range(n_null):
        z = np.zeros(n)
        for i in range(1, n):
            z[i] = phi * z[i - 1] + rng.normal()
        X[f"null{j}"] = z
    s = pd.Series(np.cumsum(r))
    Y = pd.DataFrame({"t_decision_utc": t, "row_id": np.arange(n)})
    for name, fam, head, h in fc.TARGETS:
        Y[name] = (s.shift(-h) - s).to_numpy() if fam != "Y_b" else np.sign((s.shift(-h) - s).fillna(0)).to_numpy()
    return X, Y, thr


def folds_of(n, k=4):
    edges = np.linspace(0, n, k + 2).astype(int)
    return [(np.arange(0, max(edges[j] - 150, 50)), np.arange(edges[j], edges[j + 1])) for j in range(1, k + 1)]


def binary_episodes(n=1500, effect=1.0, slope=0.6, confound=1.5, seed=7, deterministic=False):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    t = (w > 0).astype(float) if deterministic else (rng.random(n) < 1 / (1 + np.exp(-slope * w))).astype(float)
    y = effect * t + confound * w + rng.normal(scale=0.5, size=n)
    ep = pd.DataFrame({"episode_id": [f"e{i}" for i in range(n)], "A": t, "W_w": w, "Y": y,
                       "Ypre_fixed": 0.4 * w + rng.normal(scale=0.5, size=n),
                       "M_first_hour": 0.3 * t + 0.2 * w + rng.normal(scale=0.3, size=n),
                       "decision_time": pd.date_range("2018-01-01", periods=n, freq="30h", tz="UTC")})
    return ep


def run_r2(ep, target="Y", clock="OBSERVED", disjoint=True):
    info = {"windows_disjoint": disjoint}
    r2 = fc.rung2_cell(ep, fid="x", target=target, horizon_h=24, clock=clock, info=info)
    rec = fc.rung2_summary(r2, y_sd_train=float(np.std(ep[target])))
    nl = None
    if r2["state"] == ps3c.IDENTIFIED:
        w_cols = [c for c in ep.columns if c.startswith("W_")]
        nl = R.nonlinear_aipw(ep, target, w_cols)
        nl["confirmation_identity"] = R.confirmation_identity(
            {k: r2["population"][k] for k in ("population_n", "population_sha256", "estimand_id")}, nl)
    return r2, rec, nl


# ------------------------------------------------------------------------------------- HAC test calibration


def test_hac_partial_test_is_calibrated_on_overlapping_outcomes_where_naive_ols_is_not():
    rng = np.random.default_rng(3)
    h, n, reps = 6, 1500, 60
    naive_rej, hac_rej = 0, 0
    for _ in range(reps):
        e = rng.normal(size=n + h)
        y = np.array([e[i:i + h].sum() for i in range(n)])  # h-step overlapping outcome: MA(h-1) errors
        a = rng.normal(size=n)
        hz = rng.normal(size=(n, 2))
        _, t_hac, p_hac, _ = fc.hac_partial_test(a, y, hz, bandwidth=h)
        f = st.ols(st.add_const(np.column_stack([hz, a])), y)
        p_naive = st.normal_two_sided_p(f["beta"][-1] / f["se"][-1])
        naive_rej += p_naive <= 0.05
        hac_rej += p_hac <= 0.05
    # independent A: HC1 is actually fine here (A is iid); the HAC must not over-reject either
    assert hac_rej / reps <= 0.15
    # with an autocorrelated A the naive SE under-states and over-rejects; HAC corrects it
    naive_rej, hac_rej = 0, 0
    for _ in range(reps):
        e = rng.normal(size=n + h)
        y = np.array([e[i:i + h].sum() for i in range(n)])
        a = np.zeros(n)
        for i in range(1, n):
            a[i] = 0.95 * a[i - 1] + rng.normal()
        hz = np.zeros((n, 0))
        _, _, p_hac, _ = fc.hac_partial_test(a, y, hz, bandwidth=24)
        f = st.ols(st.add_const(a[:, None]), y)
        naive_rej += st.normal_two_sided_p(f["beta"][-1] / f["se"][-1]) <= 0.05
        hac_rej += p_hac <= 0.05
    assert naive_rej > hac_rej and hac_rej / reps <= 0.2


# ------------------------------------------------------------------------------------------ rung 1


def test_true_cause_is_supported_at_rung1_after_family_bh_and_nulls_are_not():
    X, Y, _ = hourly_world(effect=1.2, n_null=12)
    folds = folds_of(len(X))
    hist = ["h1", "h2", "h3"]
    H = X[hist].to_numpy(float)
    cells = {}
    for fid in ["x"] + [c for c in X if c.startswith("null")]:
        cells[fid] = fc.rung1_cell(X[fid].to_numpy(float), Y["Y_s_1h"].to_numpy(float), H, horizon_h=1, folds=folds,
                                   hist_names=hist, permutations=60)
    qs = st.bh_q([cells[f]["p"] for f in cells])
    for (f, c), q in zip(cells.items(), qs):
        fc.assign_rung1_state(c, q)
    assert cells["x"]["state"] == fc.SUPPORTED and cells["x"]["sign"] == 1
    assert cells["x"]["ci_test"].startswith("HAC(") and cells["x"]["max_lag_rows"] == fc.MAX_LAG
    assert cells["x"]["minimal_separating_set"]["state"] == "NO_SEPARATING_SET_FOUND"
    nulls = [c["state"] for f, c in cells.items() if f != "x"]
    assert nulls.count(fc.NOT_IDENTIFIED) >= len(nulls) - 1
    assert all(c["abstention_reason"] for f, c in cells.items() if c["state"] == fc.NOT_IDENTIFIED)


def test_rung1_states_vocabulary_and_abstention_names():
    X, Y, _ = hourly_world(n=400)
    c = fc.rung1_cell(X["x"].to_numpy(float), Y["Y_s_1h"].to_numpy(float), X[["h1"]].to_numpy(float), horizon_h=1,
                      folds=folds_of(400), hist_names=["h1"], permutations=20, min_rows=1000)
    fc.assign_rung1_state(c, None)
    assert c["state"] == fc.NOT_IDENTIFIED and c["abstention_reason"] == "TOO_FEW_ROWS"
    assert set(fc.STATES) == {"SUPPORTED", "CONTRADICTED", "NOT_IDENTIFIED"}


def test_sypi_two_conditions_find_the_planted_lag_and_reject_a_null():
    rng = np.random.default_rng(11)
    n = 5000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.normal()
    y = rng.normal(size=n)
    y[2:] += 0.8 * x[:-2]  # x_{t-2} -> y_t
    S = np.column_stack([rng.normal(size=n), fc._shift_rows(y, 1)])
    d = fc.sypi_conditions(x, y, S, horizon_h=1)
    assert d["state"] == "RUN" and d["w"] == 2 and d["condition1_p"] < 1e-6 and d["condition2_independent"]
    fc.assign_sypi_state(d, d["condition1_p"])
    assert d["verdict"].startswith("SYPI_CANDIDATE_CAUSE")
    z = rng.normal(size=n)
    dn = fc.sypi_conditions(z, y, S, horizon_h=1)
    fc.assign_sypi_state(dn, 0.5)
    assert dn["verdict"] == "SYPI_CONDITION1_FAILED"


# ------------------------------------------------------------------------------------------ rung 2


def test_true_cause_identified_supported_and_rung3_counterfactual_supported():
    ep = binary_episodes(effect=1.0)
    r2, rec, nl = run_r2(ep)
    assert r2["state"] == ps3c.IDENTIFIED, r2["reasons"]
    assert rec["assumptions_unverified"] == [] and all(isinstance(v, str) and v for v in rec["assumptions_evidence"].values())
    assert rec["assumption_strength"]["CAUSAL_SUFFICIENCY_OF_DECLARED_DAG"].startswith("DECLARED_WITH_SENSITIVITY_ONLY")
    fc.assign_rung2_state(rec, q=rec["p_linear"], nonlinear=nl, rung1_sign=1)
    assert rec["state"] == fc.SUPPORTED and abs(rec["estimate"]["value"] - 1.0) < 0.25
    assert rec["support"]["state"] == "SUPPORTED" and rec["placebo"]["state"] == "PASSED"
    # naive filtering of means is NOT the effect (FS10)
    naive = ep.Y[ep.A == 1].mean() - ep.Y[ep.A == 0].mean()
    assert abs(naive - 1.0) > 0.3
    w_cols = [c for c in ep.columns if c.startswith("W_")]
    r3, n3 = fc.rung3_cell(ep, target="Y", r2_state=r2["state"], w_cols=w_cols)
    rec3 = fc.rung3_summary(r3, n3)
    fc.assign_rung3_state(rec3, fc.SUPPORTED, 1)
    assert rec3["raw_state"] == ps3c.CF_STATE and rec3["state"] == fc.SUPPORTED, rec3
    assert rec3["sensitivity"]["reconstruction_max_abs_error"] < 1e-8  # FS11: abduction keeps the episode's own noise
    assert "never an observation" in rec3["never_observed"]


def test_confounded_pair_without_overlap_stays_not_identified_fs10_fs12():
    ep = binary_episodes(deterministic=True)
    r2, rec, nl = run_r2(ep)
    assert r2["state"] == ps3c.NOT_IDENTIFIED and r2["estimate"] is None
    fc.assign_rung2_state(rec, q=None, nonlinear=None, rung1_sign=1)
    assert rec["state"] == fc.NOT_IDENTIFIED and rec["estimate"] is None
    assert "OVERLAP_SCREEN_FAILED" in rec["abstention_reason"] or "NO_COMMON_SUPPORT" in rec["abstention_reason"]
    rec3 = fc.rung3_summary({"state": ps3c.NOT_IDENTIFIED, "label": "NONE", "reasons": ["RUNG2_NOT_IDENTIFIED"]}, 0)
    fc.assign_rung3_state(rec3, rec["state"], None)
    assert rec3["state"] == fc.NOT_IDENTIFIED and rec3["abstention_reason"] == "RUNG2_NOT_IDENTIFIED"


def test_planted_anti_effect_is_contradicted_when_identified_sign_opposes_the_association():
    ep = binary_episodes(n=2500, effect=-1.0, slope=0.6, confound=3.0)
    # the association ignoring W is positive (confounding), the identified effect is negative
    assert np.corrcoef(ep.A, ep.Y)[0, 1] > 0
    r2, rec, nl = run_r2(ep)
    assert r2["state"] == ps3c.IDENTIFIED, r2["reasons"]
    fc.assign_rung2_state(rec, q=rec["p_linear"], nonlinear=nl, rung1_sign=+1)
    assert rec["state"] == fc.CONTRADICTED and rec["contradiction_kind"].startswith("IDENTIFIED_EFFECT_OPPOSITE")
    assert rec["estimate"]["value"] < 0
    # the same evidence with a consistent association is SUPPORTED, not contradicted
    rec_b = fc.rung2_summary(r2, y_sd_train=float(np.std(ep.Y)))
    fc.assign_rung2_state(rec_b, q=rec_b["p_linear"], nonlinear=nl, rung1_sign=-1)
    assert rec_b["state"] == fc.SUPPORTED


def test_precise_null_is_contradicted_only_inside_the_declared_equivalence_margin():
    ep = binary_episodes(n=6000, effect=0.0, slope=0.5, confound=0.3)
    r2, rec, nl = run_r2(ep)
    assert r2["state"] == ps3c.IDENTIFIED, r2["reasons"]
    fc.assign_rung2_state(rec, q=0.9, nonlinear=nl, rung1_sign=None)
    lo, hi = rec["estimate"]["interval"]
    m = rec["equivalence_margin"]
    if -m <= lo and hi <= m and nl["interval"][0] >= -m and nl["interval"][1] <= m:
        assert rec["state"] == fc.CONTRADICTED and rec["contradiction_kind"].startswith("PRECISE_NULL")
    else:
        assert rec["state"] == fc.NOT_IDENTIFIED


def test_unevidenced_assumption_or_assumed_clock_keeps_rung2_not_identified():
    ep = binary_episodes(effect=1.0)
    r2, rec, _ = run_r2(ep, disjoint=False)
    assert r2["state"] == ps3c.NOT_IDENTIFIED
    assert "ASSUMPTION_NOT_EVIDENCED_NO_INTERFERENCE_BETWEEN_EPISODES" in r2["reasons"]
    r2c, _, _ = run_r2(ep, clock="ASSUMED")
    assert r2c["state"] == ps3c.NOT_IDENTIFIED and "ASSUMED_PUBLICATION_CLOCK" in r2c["reasons"]


def test_empty_stratum_is_an_abstention_not_a_number():
    ep = binary_episodes(effect=1.0)
    ep["A"] = 0.0
    r2, rec, _ = run_r2(ep)
    assert r2["state"] == ps3c.NOT_IDENTIFIED and "EMPTY_TREATMENT_STRATUM" in r2["reasons"]
    fc.assign_rung2_state(rec, None, None, None)
    assert rec["state"] == fc.NOT_IDENTIFIED and rec["estimate"] is None


# ------------------------------------------------------------------------------ episode construction contract


def test_episode_outcome_windows_are_disjoint():
    X, Y, _ = hourly_world(n=8000)
    for h in (24, 72, 144):
        ep, info = fc.crossing_episodes_h(X, Y, "x", h)
        assert ep is not None and info["windows_disjoint"] is True and info["min_gap_h"] == h
        t = np.sort(pd.DatetimeIndex(ep["decision_time"]).asi8)
        assert np.all(np.diff(t) >= h * 3600 * 10**9)
    assert fc.episode_windows_disjoint(np.array([0, 10 * 3600 * 10**9]), 24) is False


def test_temporal_order_enforced():
    X, Y, thr = hourly_world(n=4000)
    ep, info = fc.crossing_episodes_h(X, Y, "x", 24)
    x = X["x"].to_numpy()
    rows = X.index[X["t_decision_utc"].isin(ep["decision_time"])].to_numpy()
    assert np.allclose(ep["W_x_prev"].to_numpy(), x[rows - 1])  # W from the row BEFORE the decision
    assert np.allclose(ep["Y_s_1h"].to_numpy(), Y["Y_s_1h"].to_numpy()[rows])  # Y realised after the decision row
    assert set(c[2:] for c in ep.columns if c.startswith("W_") and c[2:] in X) <= set(X.columns)


def test_treatment_is_deterministic_function_of_observed_path():
    X, Y, thr = hourly_world(n=4000)
    ep, info = fc.crossing_episodes_h(X, Y, "x", 24)
    x = X["x"].to_numpy()
    rows = X.index[X["t_decision_utc"].isin(ep["decision_time"])].to_numpy()
    treated = ep["A"].to_numpy() == 1
    assert np.all((x[rows[treated] - 1] < info["threshold"]) & (x[rows[treated]] >= info["threshold"]))
    assert np.all(x[rows[treated] - 1] >= info["band"])  # common support by construction: same pre-row band as controls
    assert np.all((x[rows[~treated]] < info["threshold"]) & (x[rows[~treated] - 1] >= info["band"]))
    ep2, _ = fc.crossing_episodes_h(X, Y, "x", 24)
    assert ep2["A"].tolist() == ep["A"].tolist()  # deterministic, reproducible
    assert info["threshold"] == pytest.approx(thr)


def test_future_timestamp_is_refused_by_the_lane_a_loader(tmp_path):
    from test_ps3c_batch import make_lane_a_batch

    root = make_lane_a_batch(str(tmp_path / "laneA"))
    c = json.load(open(os.path.join(root, "contract.json")))
    X = pd.read_parquet(os.path.join(root, "features_train.parquet"))
    c["periods"]["train"][1] = str(pd.Timestamp(X["t_decision_utc"].iloc[-1]) - pd.Timedelta(hours=1))
    json.dump(c, open(os.path.join(root, "contract.json"), "w"))
    with pytest.raises(SystemExit, match="REFUSED: a decision row at or after TRAIN end"):
        B.load_batch(root)


def test_known_in_advance_calendar_feature_has_no_episode_set_and_constant_feature_abstains():
    X, Y, _ = hourly_world(n=3000)
    X["cal.hour_sin"] = np.sin(2 * np.pi * X["t_decision_utc"].dt.hour / 24)
    X["flat"] = 1.0
    ep, info = fc.crossing_episodes_h(X, Y, "flat", 24)
    assert ep is None and info["reason"] == "TOO_FEW_FINITE_OR_CONSTANT"


def test_feature_weight_against_only_counts_robust_contradicted():
    cells = [{"rung1": {"state": fc.NOT_IDENTIFIED}, "rung2": {"state": fc.CONTRADICTED, "robust": False}, "rung3": {"state": fc.NOT_IDENTIFIED}},
             {"rung1": {"state": fc.SUPPORTED}, "rung2": {"state": fc.NOT_IDENTIFIED}, "rung3": {"state": fc.NOT_IDENTIFIED}}]
    agg = fc.feature_weight_against(cells)
    assert agg["weighs_against"] is False and agg["supported_cells"] == 1 and agg["not_identified_never_eliminates"]
    cells[0]["rung2"]["robust"] = True
    assert fc.feature_weight_against(cells)["weighs_against"] is True


def test_rung2_propensity_is_scale_invariant_small_unit_covariates():
    """A confounder measured in units of 1e-3 (a log return) must be adjusted as well as the same confounder in units of 1."""
    ep = binary_episodes(effect=1.0, seed=7)
    tiny = ep.copy()
    tiny["W_w"] = tiny["W_w"] * 1e-3
    r2a, _, _ = run_r2(ep)
    r2b, _, _ = run_r2(tiny)
    assert r2a["state"] == ps3c.IDENTIFIED and r2b["state"] == ps3c.IDENTIFIED, (r2a["reasons"], r2b["reasons"])
    assert r2a["estimate"]["value"] == pytest.approx(r2b["estimate"]["value"], abs=1e-5)
    assert r2a["support"]["balance_max_smd"] == pytest.approx(r2b["support"]["balance_max_smd"], abs=1e-5)
