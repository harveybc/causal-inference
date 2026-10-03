"""PS3-C three-rung ladder on synthetic SCM worlds where the truth is planted and known.

Covers FS10 (filtering is not do(); adjustment + support), FS11 (abduction keeps the episode's
own perturbations, descendants propagate, no live future), FS12 (confounding / missing support
keeps NOT_IDENTIFIED, which is never a rejection), plus the episode contract: future timestamps,
missing availability, empty strata, history strictly before t, TRAIN-only thresholds/scales.

Synthetic, seeded, numpy/pandas only, CPU. Numbers are arbitrary and never compared to a market.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from causal_inference_provider import ps3c
from causal_inference_provider import ps3c_episodes as epi
from causal_inference_provider import ps3c_graph as graph

SEED = 1729
DAG = {"nodes": ["W", "A", "M", "Y"], "edges": [["W", "A"], ["W", "Y"], ["A", "M"], ["M", "Y"], ["A", "Y"], ["W", "M"]]}
NODE_COLUMNS = {"W": ["W1", "W2"]}


def continuous_world(n=800, seed=SEED):
    """A = 0.6 W1 - 0.3 W2 + e; M = 0.5 A + 0.2 W1 + U_M; Y = 0.3 A + 0.8 W1 + 0.5 M + U_Y. Total effect 0.55."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = 0.6 * w1 - 0.3 * w2 + rng.normal(scale=0.7, size=n)
    um, uy = rng.normal(scale=0.3, size=n), rng.normal(scale=0.3, size=n)
    m = 0.5 * a + 0.2 * w1 + um
    y = 0.3 * a + 0.8 * w1 + 0.5 * m + uy
    ypre = 0.5 * w1 + rng.normal(scale=0.5, size=n)  # fixed before t: depends on W, never on A
    t = pd.date_range("2018-01-01", periods=n, freq="7h", tz="UTC")
    df = pd.DataFrame({"episode_id": [f"e{i}" for i in range(n)], "decision_time": t, "W1": w1, "W2": w2,
                       "A": a, "M": m, "Y": y, "Ypre": ypre, "U_M": um, "U_Y": uy})
    return df


def binary_world(n=1200, effect=1.0, seed=SEED, deterministic=False):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    t = (w > 0).astype(float) if deterministic else (rng.random(n) < 1 / (1 + np.exp(-0.8 * w))).astype(float)
    y = effect * t + 1.5 * w + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"W": w, "A": t, "Y": y, "Ypre": 0.4 * w + rng.normal(scale=0.5, size=n),
                         "decision_time": pd.date_range("2018-01-01", periods=n, freq="5h", tz="UTC")})


BDAG = {"nodes": ["W", "A", "Y"], "edges": [["W", "A"], ["W", "Y"], ["A", "Y"]]}


# ----------------------------------------------------------------------------------------------- graph


def test_backdoor_rejects_mediator_and_accepts_confounder():
    ok, reasons, excluded = graph.backdoor_check(DAG, "A", "Y", ["W"])
    assert ok and reasons == [] and excluded == ["M"]
    ok, reasons, _ = graph.backdoor_check(DAG, "A", "Y", ["M"])
    assert not ok and "BACKDOOR_NOT_SATISFIED" in reasons and "ADJUSTMENT_CONTAINS_DESCENDANT" in reasons
    ok, reasons, _ = graph.backdoor_check(DAG, "A", "Y", [])
    assert not ok and reasons == ["BACKDOOR_NOT_SATISFIED"]


# ----------------------------------------------------------------------------------------------- rung 1


def test_rung1_signal_vs_null_and_temporal_placebo():
    df = continuous_world()
    sig = ps3c.rung1_association(df, treatment="A", outcome="Y", history=["W1", "W2"], placebo_outcomes=["Ypre"],
                                 time_key="decision_time")
    assert sig["state"] == "ASSOCIATION_REPORTED"
    gain = [e for e in sig["evidence"] if e["measure"] == "oof_relative_mse_gain_of_A_over_H"][0]
    assert gain["value"] > 0.2 and gain["signed_direction_stable"] is True
    pc = [e for e in sig["evidence"] if e["measure"] == "partial_corr_given_H"][0]
    assert pc["p"] <= 2 / 201
    assert sig["diagnostics"]["placebo_state"] == "PASSED"
    rng = np.random.default_rng(7)
    null = df.assign(A=rng.normal(size=len(df)))
    nb = ps3c.rung1_association(null, treatment="A", outcome="Y", history=["W1", "W2"], time_key="decision_time")
    ngain = [e for e in nb["evidence"] if e["measure"] == "oof_relative_mse_gain_of_A_over_H"][0]
    assert ngain["value"] < 0.01
    assert sig["multiplicity"]["p_floor"] == pytest.approx(1 / 201)


def test_rung1_too_few_events_and_zero_variance_are_states_not_zero():
    df = continuous_world(n=20)
    assert ps3c.rung1_association(df, treatment="A", outcome="Y")["state"] == "TOO_FEW_EVENTS"
    df = continuous_world().assign(A=1.0)
    assert ps3c.rung1_association(df, treatment="A", outcome="Y")["state"] == "ZERO_VARIANCE"


# ----------------------------------------------------------------------------------------------- rung 2


def test_rung2_continuous_recovers_total_effect_only_with_valid_adjustment():
    df = continuous_world()
    naive = np.polyfit(df["A"], df["Y"], 1)[0]
    assert abs(naive - 0.55) > 0.2  # the planted world is confounded
    out = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=["W"], contrast=(1.0, 0.0), dag=DAG,
                            node_columns=NODE_COLUMNS, placebo_outcomes=["Ypre"], time_key="decision_time")
    assert out["state"] == ps3c.IDENTIFIED, out["reasons"]
    assert abs(out["estimate"]["value"] - 0.55) < 0.06
    lo, hi = out["estimate"]["interval"]
    assert lo < 0.55 < hi
    assert out["excluded_from_adjustment"] == ["M"]
    assert out["placebo"]["state"] == "PASSED"
    assert out["sensitivity"]["robustness_value_q1"] > 0
    bad = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=["M"], contrast=(1.0, 0.0), dag=DAG)
    assert bad["state"] == ps3c.NOT_IDENTIFIED and bad["estimate"] is None
    assert "BACKDOOR_NOT_SATISFIED" in bad["reasons"]


def test_rung2_binary_aipw_recovers_ate_where_filtering_fails():
    df = binary_world()
    filtered = df.loc[df.A == 1, "Y"].mean() - df.loc[df.A == 0, "Y"].mean()
    assert abs(filtered - 1.0) > 0.5
    out = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=["W"], contrast=(1, 0), dag=BDAG,
                            placebo_outcomes=["Ypre"], modifiers=["W"], time_key="decision_time")
    assert out["state"] == ps3c.IDENTIFIED, out["reasons"]
    assert abs(out["estimate"]["value"] - 1.0) < 0.15
    assert abs(out["sensitivity"]["att_aipw"] - 1.0) < 0.2
    assert out["support"]["propensity_range"][0] > 0
    assert len(out["estimate"]["heterogeneity"]) == 3


def test_rung2_empty_stratum_and_overlap_failure_keep_not_identified():
    df = binary_world().assign(A=0.0)
    out = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=["W"], contrast=(1, 0), dag=BDAG,
                            treatment_kind="BINARY")
    assert out["state"] == ps3c.NOT_IDENTIFIED and "EMPTY_TREATMENT_STRATUM" in out["reasons"]
    assert out["estimate"] is None
    det = binary_world(deterministic=True)
    out = ps3c.rung2_effect(det, treatment="A", outcome="Y", adjustment=["W"], contrast=(1, 0), dag=BDAG)
    assert out["state"] == ps3c.NOT_IDENTIFIED
    assert "OVERLAP_SCREEN_FAILED" in out["reasons"] or "NO_COMMON_SUPPORT" in out["reasons"]
    assert out["rung1"]["state"] == "ASSOCIATION_REPORTED"  # rung-1 evidence survives the refusal


def test_rung2_anticipation_leak_fails_placebo():
    df = continuous_world()
    df["Ypre"] = 0.8 * df["A"] + np.random.default_rng(3).normal(scale=0.3, size=len(df))
    out = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=["W"], contrast=(1.0, 0.0), dag=DAG,
                            node_columns=NODE_COLUMNS, placebo_outcomes=["Ypre"], time_key="decision_time")
    assert out["state"] == ps3c.NOT_IDENTIFIED and "PLACEBO_FAILED" in out["reasons"]
    assert out["estimate"] is None


def test_rung2_model_based_expectation_and_assumed_clock_block_identification():
    df = continuous_world()
    out = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=["W"], contrast=(1.0, 0.0), dag=DAG,
                            node_columns=NODE_COLUMNS, context={"expectation_kind": "MODEL_BASED_EXPECTATION",
                                                                "publication_clock": "ASSUMED_SCHEDULED_PUBLICATION"})
    assert out["state"] == ps3c.NOT_IDENTIFIED
    assert {"EXPECTATION_IS_MODEL_BASED", "ASSUMED_PUBLICATION_CLOCK"} <= set(out["reasons"])
    assert out["estimate"] is None


def test_rung2_undeclared_assumption_keeps_not_identified():
    df = continuous_world()
    out = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=["W"], contrast=(1.0, 0.0), dag=DAG,
                            node_columns=NODE_COLUMNS, assumptions={"CONSISTENCY": True})
    assert out["state"] == ps3c.NOT_IDENTIFIED and "ASSUMPTIONS_NOT_DECLARED_TRUE" in out["reasons"]


# ----------------------------------------------------------------------------------------------- rung 3


def test_rung3_population_counterfactual_matches_planted_truth():
    df = continuous_world()
    r2 = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=["W"], contrast=(1.0, 0.0), dag=DAG,
                           node_columns=NODE_COLUMNS, placebo_outcomes=["Ypre"], time_key="decision_time")
    block, rows = ps3c.rung3_population(df, treatment="A", outcome="Y", adjustment_cols=["W1", "W2"], a0=0.0,
                                        rung2_state=r2["state"], mediators=["M"], placebo_outcome="Ypre",
                                        time_key="decision_time")
    assert block["state"] == ps3c.CF_STATE, block["reasons"]
    assert block["label"] == ps3c.CF_LABEL
    est = pd.DataFrame(rows).set_index("episode_id")
    truth = df.set_index("episode_id")
    # planted same-episode counterfactual: A -> 0, M recomputed with its own U_M, Y with its own U_Y
    m_cf = 0.2 * truth["W1"] + truth["U_M"]
    y_cf = 0.8 * truth["W1"] + 0.5 * m_cf + truth["U_Y"]
    err = (est["y_counterfactual"] - y_cf.loc[est.index]).abs()
    assert err.mean() < 0.05
    assert np.corrcoef(est["delta"], 0.55 * truth.loc[est.index, "A"])[0, 1] > 0.99
    assert block["sensitivity"]["reconstruction_max_abs_error"] < 1e-9
    assert block["sensitivity"]["analog_state"] == "CONSISTENT"
    # model_based (no U) and same-episode counterfactual are different objects
    assert (est["y_counterfactual"] - est["model_based"] - est["u_y"]).abs().max() < 1e-9


def test_rung3_refuses_without_rung2_and_outside_support():
    df = continuous_world()
    block, _ = ps3c.rung3_population(df, treatment="A", outcome="Y", adjustment_cols=["W1", "W2"], a0=0.0,
                                     rung2_state=ps3c.NOT_IDENTIFIED, mediators=["M"], placebo_outcome="Ypre")
    assert block["state"] == ps3c.NOT_IDENTIFIED and "RUNG2_NOT_IDENTIFIED" in block["reasons"]
    block, rows = ps3c.rung3_population(df, treatment="A", outcome="Y", adjustment_cols=["W1", "W2"], a0=50.0,
                                        rung2_state=ps3c.IDENTIFIED, mediators=["M"])
    assert block["state"] == ps3c.NOT_IDENTIFIED and "NO_COMMON_SUPPORT" in block["reasons"] and rows == []


def test_counterfactual_support_and_operational_contract():
    scm = ps3c.AdditiveSCM(order=["W", "A", "Y"], mechanisms={"Y": lambda A, W: 2 * A + W},
                           residual_sd={"Y": 0.5}, support={"A": (-1.0, 1.0)})
    with pytest.raises(ps3c.CounterfactualRefusal, match="NO_COMMON_SUPPORT"):
        ps3c.counterfactual_same_episode(scm, {"W": 0.0, "A": 0.5, "Y": 1.2}, intervention={"A": 3.0})
    with pytest.raises(ps3c.CounterfactualRefusal, match="ABDUCTION_NEEDS_OBSERVED_OUTCOME"):
        ps3c.counterfactual_same_episode(scm, {"W": 0.0, "A": 0.5}, intervention={"A": 0.0})
    live = ps3c.counterfactual_same_episode(scm, {"W": 0.1, "A": 0.5}, intervention={"A": 0.0}, mode="OPERATIONAL")
    assert live["abduction"] is None and live["prediction"]["Y"]["mean"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        ps3c.AdditiveSCM(order=["Y", "A"], mechanisms={"Y": lambda A: A})  # parent after child


# ----------------------------------------------------------------------------------------------- dossier


def _manifest(state="CONTRACTED"):
    if state == "CONTRACTED":
        app = {"state": "CONTRACTED", "appearance_id": "app_" + "a" * 24,
               "dataset_id": "financial_data.census_appearance.app_" + "a" * 24, "entity": "SYNTHETIC",
               "resource_sha256": "b" * 64, "contract_id": "synthetic-fixture", "contract_sha256": "c" * 64,
               "train_rows": [0, 800], "frequency": "1h", "period": ["2018-01-01T00:00:00Z", "2018-08-01T00:00:00Z"],
               "contracts_document_sha256": "d" * 64}
    else:
        app = {"state": "NOT_EXECUTABLE_NO_CONTRACTED_PRICE", "entity": "SYNTHETIC", "reason": "fixture"}
    return {"sources": [{"role": "event_rows", "resource": "synthetic", "sha256": "e" * 64, "rows": 800}],
            "asset_appearance": app, "publication_clock": "OBSERVED_PUBLICATION_CLOCK",
            "consensus_clock": "OBSERVED", "expectation_kind": "PUBLISHED_CONSENSUS", "n_episodes": 800,
            "exclusions": {}, "train_folds": ["chrono_expanding_5"]}


def _dossier(r1, r2, r3, state="CONTRACTED"):
    return ps3c.dossier(
        dossier_id="synthetic.ps3c.test", producer_revision="0123456",
        subject={"kind": "EPISODE_POPULATION", "population": "synthetic", "event_type": "SYN | release",
                 "asset": "SYNTHETIC", "head": "short", "target": "Y_s", "horizon_minutes": 60},
        data_manifest=_manifest(state),
        treatment={"name": "A", "definition": "planted", "kind": "CONTINUOUS", "standardization": "none",
                   "dose_support": {"min": -3.0, "max": 3.0, "n": 800}},
        rung1=r1, rung2=r2, rung3=r3,
        emission={"operational_use": "RETROSPECTIVE_ONLY", "emittable_from": {"rung3": "2018-08-01T00:00:00Z"}},
        limitations=["synthetic fixture"])


def test_dossier_validates_and_never_turns_not_identified_into_rejection():
    df = continuous_world()
    r2 = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=["W"], contrast=(1.0, 0.0), dag=DAG,
                           node_columns=NODE_COLUMNS, placebo_outcomes=["Ypre"], time_key="decision_time")
    r3, _ = ps3c.rung3_population(df, treatment="A", outcome="Y", adjustment_cols=["W1", "W2"], a0=0.0,
                                  rung2_state=r2["state"], mediators=["M"], placebo_outcome="Ypre")
    doc = _dossier(r2["rung1"], r2, r3)
    assert ps3c.validate_dossier(doc) == []
    assert doc["selection"]["causal_evidence_level"] == "COUNTERFACTUAL_SENSITIVITY" and doc["selection"]["cf_eligible"]
    ni = ps3c.rung2_effect(df, treatment="A", outcome="Y", adjustment=None, contrast=(1.0, 0.0), dag=DAG)
    doc2 = _dossier(ni["rung1"], ni, None)
    assert ps3c.validate_dossier(doc2) == []
    assert doc2["selection"]["causal_evidence_level"] == "ASSOCIATION"
    assert doc2["selection"]["reason_code"] == "NOT_IDENTIFIED_IS_NOT_REJECTION"
    assert "rejected" not in str(doc2).lower()
    doc3 = _dossier(r2["rung1"], r2, r3, state="NOT_EXECUTABLE")
    assert ps3c.validate_dossier(doc3) == []
    assert doc3["rung1"]["state"] == "NOT_EVALUATED" and doc3["selection"]["causal_evidence_level"] == "NONE"


# ----------------------------------------------------------------------------------------------- episodes


def _bars(n_hours=24 * 400, seed=SEED, start="2019-01-01"):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    lc = np.cumsum(rng.normal(scale=1e-3, size=n_hours)) + np.log(1.1)
    c = np.exp(lc)
    return pd.DataFrame({"timestamp": t, "close": c, "high": c * (1 + 5e-4), "low": c * (1 - 5e-4)})


def _events(bars, n=150, seed=SEED):
    rng = np.random.default_rng(seed)
    times = bars["timestamp"].iloc[200:-400].sample(n, random_state=seed).sort_values() + pd.Timedelta(minutes=30)
    cons = rng.normal(size=n)
    act = cons + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"event_type": np.where(np.arange(n) % 2 == 0, "USD | cpi", "EUR | pmi"),
                         "currency": np.where(np.arange(n) % 2 == 0, "USD", "EUR"),
                         "published_at": times.values, "actual": act, "consensus": cons, "previous": cons})


def test_episodes_ignore_everything_after_train_end():
    bars = _bars()
    ev = _events(bars)
    train_end = bars["timestamp"].iloc[-2000]
    a = epi.build_event_episodes(ev, bars, train_end=train_end)
    bars2 = bars.copy()
    late = bars2["timestamp"] > train_end
    bars2.loc[late, ["close", "high", "low"]] *= 3.0
    ev2 = ev.copy()
    ev2.loc[pd.to_datetime(ev2["published_at"], utc=True) > train_end, "actual"] += 100.0
    b = epi.build_event_episodes(ev2, bars2, train_end=train_end)
    pd.testing.assert_frame_equal(a.episodes, b.episodes)
    assert a.fitted["scales"] == b.fitted["scales"]
    assert a.exclusions["BEYOND_TRAIN_END_NOT_READ"] > 0
    assert any(k.startswith("OUTCOME_WINDOW_CROSSES_TRAIN_END") for k in a.exclusions)
    assert (pd.to_datetime(a.episodes["decision_time"]) <= train_end).all()


def test_history_is_strictly_before_t_and_outcomes_after():
    bars = _bars()
    ev = _events(bars)
    a = epi.build_event_episodes(ev, bars, train_end=bars["timestamp"].iloc[-1])
    e0 = a.episodes.iloc[10]
    t = e0["decision_time"]
    bars2 = bars.copy()
    after = bars2["timestamp"] + pd.Timedelta(hours=1) > t  # bars closing after t
    bars2.loc[after, ["close", "high", "low"]] *= 1.05
    b = epi.build_event_episodes(ev, bars2, train_end=bars["timestamp"].iloc[-1])
    e1 = b.episodes.set_index("episode_id").loc[e0["episode_id"]]
    wcols = [c for c in a.episodes.columns if c.startswith("W_") or c.startswith("Ypre_")]
    for c in wcols:
        if c == "W_regime_code":
            continue  # regime cut points are fitted on the TRAIN population, which changed
        assert (np.isnan(e0[c]) and np.isnan(e1[c])) or e0[c] == pytest.approx(e1[c], abs=1e-12), c
    assert e0["entry_time"] >= t


def test_missing_availability_and_inconsistent_clocks_are_excluded_by_name():
    bars = _bars()
    ev = _events(bars)
    ev.loc[3, "published_at"] = pd.NaT
    ev["received_at"] = pd.to_datetime(ev["published_at"], utc=True)
    ev.loc[5, "received_at"] = pd.to_datetime(ev.loc[5, "published_at"], utc=True) - pd.Timedelta(hours=1)
    out = epi.build_event_episodes(ev, bars, train_end=bars["timestamp"].iloc[-1])
    assert out.exclusions["MISSING_AVAILABILITY"] == 1
    assert out.exclusions["RECEIVED_BEFORE_PUBLISHED"] == 1
    with pytest.raises(epi.EpisodeError, match="MISSING_AVAILABILITY"):
        epi.build_event_episodes(ev, bars, train_end=bars["timestamp"].iloc[-1], strict=True)


def test_surprise_zero_means_as_expected_and_model_based_is_labelled():
    bars = _bars()
    ev = _events(bars)
    ev.loc[0, "actual"] = ev.loc[0, "consensus"]
    ev.loc[1:20, "consensus"] = np.nan
    out = epi.build_event_episodes(ev, bars, train_end=bars["timestamp"].iloc[-1], expectation="consensus_or_previous")
    first = out.episodes.sort_values("decision_time").iloc[0]
    assert first["A_surprise"] == 0.0 and first["expectation_kind"] == "PUBLISHED_CONSENSUS"
    assert (out.episodes["expectation_kind"] == "MODEL_BASED_EXPECTATION").sum() > 0
    strict = epi.build_event_episodes(ev, bars, train_end=bars["timestamp"].iloc[-1], expectation="consensus")
    assert strict.exclusions["NO_EXPECTATION"] == 20


def test_crossing_threshold_is_train_only_and_empty_strata_stay_not_identified():
    bars = _bars()
    train_end = bars["timestamp"].iloc[-2000]
    a = epi.build_crossing_episodes(bars, train_end=train_end, feature_values=epi.realized_vol_series(120),
                                    feature_name="rv120")
    bars2 = bars.copy()
    bars2.loc[bars2["timestamp"] > train_end, "close"] *= np.linspace(1, 2, int((bars2["timestamp"] > train_end).sum()))
    b = epi.build_crossing_episodes(bars2, train_end=train_end, feature_values=epi.realized_vol_series(120),
                                    feature_name="rv120")
    assert a.fitted["threshold"] == b.fitted["threshold"]
    pd.testing.assert_frame_equal(a.episodes, b.episodes)
    assert a.manifest["treated"] > 0 and a.manifest["controls"] > 0
    only_ctrl = a.episodes[a.episodes["A_crossing"] == 0]
    out = ps3c.rung2_effect(only_ctrl, treatment="A_crossing", outcome="Y_s_6h", adjustment=["W"],
                            node_columns={"W": ["W_rv_24h", "W_ret_24h"]}, contrast=(1, 0), treatment_kind="BINARY",
                            dag={"nodes": ["W", "A_crossing", "Y_s_6h"],
                                 "edges": [["W", "A_crossing"], ["W", "Y_s_6h"], ["A_crossing", "Y_s_6h"]]})
    assert out["state"] == ps3c.NOT_IDENTIFIED and "EMPTY_TREATMENT_STRATUM" in out["reasons"]
