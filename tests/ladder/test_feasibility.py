import numpy as np
import pandas as pd
import pytest

from causal_ladder import FeasibilityError, feasibility_report

END = "2020-12-31"


def make(n=200, seed=0):
    r = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"decision_time": t, "available_at": t - pd.Timedelta("1min"),
                         "treatment": r.integers(0, 2, n), "outcome": r.normal(size=n),
                         "x": r.normal(size=n)})


def test_report_never_asserts_causality():
    rep = feasibility_report(make(), END, ["x"], [0, 1], min_stratum=10)
    assert rep["causal_claim"] == "not_asserted"
    assert rep["stage1_association_candidates"]["label"] == "association_only"
    assert rep["verdict"] == "support_present_assumptions_unverified"


def test_rejects_future_timestamps():
    d = make()
    d.loc[0, "decision_time"] = pd.Timestamp("2021-06-01", tz="UTC")
    with pytest.raises(FeasibilityError, match="future"):
        feasibility_report(d, END, ["x"], [0, 1])


def test_rejects_availability_after_decision():
    d = make()
    d.loc[3, "available_at"] = d.loc[3, "decision_time"] + pd.Timedelta("1h")
    with pytest.raises(FeasibilityError, match="point-in-time"):
        feasibility_report(d, END, ["x"], [0, 1])


def test_rejects_missing_availability():
    d = make()
    d.loc[5, "available_at"] = pd.NaT
    with pytest.raises(FeasibilityError, match="availability"):
        feasibility_report(d, END, ["x"], [0, 1])
    with pytest.raises(FeasibilityError, match="missing columns"):
        feasibility_report(make().drop(columns="available_at"), END, ["x"], [0, 1])


def test_rejects_empty_stratum():
    with pytest.raises(FeasibilityError, match="empty intervention strata"):
        feasibility_report(make(), END, ["x"], [0, 1, 2])
    d = make()
    d["treatment"] = 0
    with pytest.raises(FeasibilityError, match="empty"):
        feasibility_report(d, END, ["x"], [0, 1])


def test_thin_support_reported_not_hidden():
    d = make()
    d["treatment"] = 0
    d.loc[:2, "treatment"] = 1
    rep = feasibility_report(d, END, ["x"], [0, 1], min_stratum=10)
    assert rep["verdict"] == "insufficient_support"
    assert rep["stage2_intervention_strata"]["thin_strata"] == ["1"]
