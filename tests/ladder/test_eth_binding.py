"""Synthetic tests for the ETH 4h point-in-time binding of the causal ladder."""
import json

import numpy as np
import pandas as pd
import pytest

from causal_ladder.eth_binding import (BindingError, CALENDAR_STRATUM, PROTECTED_TEST_START_ROW,
                                       bind_clocks, load_train_rows, run_ladder)

MANIFEST = {"date_column": "DATE_TIME", "feature_columns": ["f1", "f2", "flag"], "timeframe": "4h",
            "splits": {"train_end": "2020-01-31T23:59:59"}}


def synth(n=300, start="2020-01-01 00:00:00"):
    r = np.random.default_rng(0)
    t = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.01, n)))
    return pd.DataFrame({"DATE_TIME": t, "CLOSE": close, "f1": r.normal(size=n),
                         "f2": r.normal(size=n), "flag": r.integers(0, 2, n)})


def test_decision_time_is_bar_close_and_outcome_never_crosses_train_end():
    d = synth()
    train = d[d.DATE_TIME <= pd.Timestamp(MANIFEST["splits"]["train_end"], tz="UTC")]
    b = bind_clocks(train.reset_index(drop=True), MANIFEST, ["f1"], horizon=3)
    assert (b.decision_time - b.bar_open == pd.Timedelta("4h")).all()
    assert (b.available_at <= b.decision_time).all()
    end = pd.Timestamp(MANIFEST["splits"]["train_end"], tz="UTC")
    assert (b.outcome_realized_at <= end).all()
    # last bar open 20:00 closes at 00:00 next day, beyond 23:59:59: dropped.
    assert b.decision_time.max() <= end
    assert b.attrs["clock_audit"]["dropped_outcome_after_train_end"] >= 2


def test_outcome_is_forward_log_return_by_time_not_row():
    d = synth(50).drop(index=10).reset_index(drop=True)  # a missing bar
    b = bind_clocks(d, {**MANIFEST, "splits": {"train_end": "2030-01-01"}}, ["f1"], horizon=1)
    assert pd.Timestamp("2020-01-02 12:00", tz="UTC") not in set(b.bar_open)  # bar before gap has no t+1
    row = b.iloc[0]
    c = d.set_index("DATE_TIME").CLOSE
    assert row.outcome == pytest.approx(np.log(c[row.bar_open + pd.Timedelta("4h")] / c[row.bar_open]))
    assert b.attrs["clock_audit"]["bar_gaps_over_4h"] == 1


def test_negative_lag_and_missing_feature_rejected():
    d = synth()
    with pytest.raises(BindingError):
        bind_clocks(d, MANIFEST, ["f1"], 1, {"f1": -1})
    with pytest.raises(BindingError):
        bind_clocks(d, MANIFEST, ["nope"], 1)


def test_read_guard_never_reaches_protected_rows(tmp_path):
    p = tmp_path / "x.csv"
    synth().assign(DATE_TIME=lambda f: f.DATE_TIME.dt.tz_localize(None)).to_csv(p, index=False)
    with pytest.raises(BindingError):
        load_train_rows(str(p), MANIFEST, max_rows=PROTECTED_TEST_START_ROW + 1)
    t = load_train_rows(str(p), MANIFEST, max_rows=100)
    assert len(t) == 100


def test_ladder_sections_are_separate_and_never_causal():
    d = synth(600)
    b = bind_clocks(d, {**MANIFEST, "splits": {"train_end": "2030-01-01"}}, ["f1", "f2", "flag"], 1)
    rep = run_ladder(b, ["f1", "f2", "flag"], "2030-01-01", binary_features=["flag"])
    assert rep["causal_claim"] == "not_asserted"
    assert {"association", "observed_interventions", "counterfactual_support"} <= set(rep)
    assert rep["observed_interventions"]["interventions_counted"] == 0
    assert all(not s["counted_as_intervention"] for s in rep["observed_interventions"]["strata"])
    names = [s["stratum"] for s in rep["counterfactual_support"]]
    assert names == [CALENDAR_STRATUM, "flag"]
    assert all(a["label"] == "association_only" for a in rep["association"])
    assert "effect" not in json.dumps(rep["association"])
