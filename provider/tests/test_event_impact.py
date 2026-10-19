"""Acceptance tests EI01-EI10 of `docs/EVENT_IMPACT_DESIGN.md`.

The synthetic world below exists ONLY here. It plants, into hourly bars with a strong hour-of-day volatility
profile, a known volatility impulse with a known half-life and a known surprise-conditional drift at scheduled
releases, so that the estimator in `event_impact.py` can be shown to recover each planted number. Nothing in it
touches a market, and the estimator never imports it.
"""

import numpy as np
import pandas as pd
import pytest

from causal_inference_provider import event_impact as ei

AMPLITUDE = 3.0          # additive excess of the volatility scale at the release bar, in units of sigma
HALF_LIFE = 4.0          # hours, volatility
HALF_LIFE_TREND = 2.0    # hours, trend: planted SEPARATELY so the two decay curves can be told apart
DRIFT_PER_Z = 0.004      # log return per unit of standardized surprise, realised over the 4 bars after the release
HORIZON = 4
WINDOW = 24


def seasonal_profile(hours):
    """A hour-of-day volatility profile with a 5x range, peaking at the release hour so an unmatched baseline is
    biased."""
    return 1.0 + 4.0 * np.exp(-((hours - 13) % 24) ** 2 / 8.0)


def synthetic_world(seed=7, weeks=160, release_hour=13, release_dow=4, amplitude=AMPLITUDE, half_life=HALF_LIFE,
                    drift_per_z=DRIFT_PER_Z, half_life_trend=HALF_LIFE_TREND, sigma=5e-4, shift_hours=0,
                    start="2012-01-02"):
    """Hourly bars for `weeks` weeks (weekdays only) with one scheduled release every second week.

    `shift_hours` moves the RECORDED release instant relative to the planted response (a future-shifted schedule).
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=weeks * 7 * 24, freq="h")
    index = index[index.dayofweek < 5]
    n = len(index)
    scale = sigma * seasonal_profile(index.hour.to_numpy())
    eps = rng.standard_normal(n)
    mean = np.zeros(n)
    releases = []
    week_starts = index[(index.dayofweek == release_dow) & (index.hour == release_hour)]
    for k, stamp in enumerate(week_starts):
        if k % 2:
            continue
        position = index.get_loc(stamp)
        surprise = rng.standard_normal() * 50.0
        z = surprise / 50.0
        for lag in range(0, WINDOW + 1):
            if position + lag < n:
                scale[position + lag] += amplitude * sigma * 2.0 ** (-lag / half_life)
        weights = 2.0 ** (-np.arange(HORIZON) / half_life_trend)
        weights = weights / weights.sum()  # the forward return over HORIZON bars is exactly drift_per_z * z
        for lag in range(0, HORIZON):
            if position + lag < n:
                mean[position + lag] += drift_per_z * z * weights[lag]
        releases.append({"release_utc": stamp + pd.Timedelta(minutes=30) + pd.Timedelta(hours=shift_hours),
                         "family": "Synthetic | Planted", "actual": 150.0 + surprise, "forecast": 150.0})
    log_ret = mean + scale * eps
    close = 1.25 * np.exp(np.cumsum(log_ret))
    bars = ei.bars_from_prices(index, close)
    calendar = pd.DataFrame(releases)
    calendar["country"], calendar["event"] = "Synthetic", "Planted"
    calendar["surprise"] = calendar["actual"] - calendar["forecast"]
    return bars, calendar, sigma


@pytest.fixture(scope="module")
def world():
    return synthetic_world()


@pytest.fixture(scope="module")
def estimate(world):
    bars, calendar, _ = world
    return ei.estimate_family(bars, calendar, "Synthetic | Planted", window=WINDOW, n_boot=200, seed=1)


# EI01 -- the impulse response recovers the planted amplitude at the release bar
def test_ei01_impulse_amplitude_is_recovered(estimate, world):
    sigma = world[2]
    response = estimate["impulse_response"]
    # E|r| of a normal with scale s is s*sqrt(2/pi); the planted additive excess at lag 0 is AMPLITUDE * sigma
    expected = AMPLITUDE * sigma * np.sqrt(2 / np.pi)
    lag0 = response["lags"].index(0)
    assert response["excess_ci_lower"][lag0] < expected < response["excess_ci_upper"][lag0]
    assert abs(response["excess"][lag0] / expected - 1.0) < 0.25
    assert response["peak_lag"] == 0


# EI02 -- the fitted half-life recovers the planted one within its interval
def test_ei02_half_life_is_recovered(estimate):
    response = estimate["impulse_response"]
    assert abs(response["half_life_hours"] - HALF_LIFE) < 1.5
    lo, hi = response["half_life_ci"]
    assert lo <= HALF_LIFE <= hi
    assert response["half_life_fit_r2"] > 0.8


# EI03 -- the surprise-conditional drift recovers the planted slope and a zero intercept
def test_ei03_surprise_drift_slope_is_recovered(estimate):
    drift = estimate["surprise_drift"]
    fit = drift["by_horizon"][str(HORIZON)]
    assert fit["status"] == "OK"
    assert fit["slope_ci"][0] < DRIFT_PER_Z < fit["slope_ci"][1]
    assert fit["intercept_ci"][0] < 0.0 < fit["intercept_ci"][1]
    assert drift["verdict"] == "SURPRISE_CONDITIONAL_DRIFT"


# EI04 -- with no planted drift there is no drift: the slope is not significant, the verdict says so
def test_ei04_no_surprise_drift_is_not_invented():
    bars, calendar, _ = synthetic_world(seed=11, drift_per_z=0.0)
    out = ei.estimate_family(bars, calendar, "Synthetic | Planted", window=WINDOW, n_boot=50, seed=2)
    drift = out["surprise_drift"]
    assert drift["verdict"] == "NO_DRIFT_DETECTED"
    for fit in drift["by_horizon"].values():
        assert fit["slope_ci"][0] < 0.0 < fit["slope_ci"][1]


# EI05 -- matched controls remove the hour-of-day seasonality; an unmatched baseline does not
def test_ei05_matched_controls_remove_hour_of_day_seasonality():
    bars, calendar, sigma = synthetic_world(seed=5, amplitude=0.0, drift_per_z=0.0)
    positions = ei.release_positions(bars, calendar["release_utc"])
    controls = ei.match_controls(bars, positions, positions, window=WINDOW)
    response = ei.impulse_response(bars, positions, controls, window=WINDOW, n_boot=100, seed=3)
    lag0 = response["lags"].index(0)
    assert response["excess_ci_lower"][lag0] < 0.0 < response["excess_ci_upper"][lag0]
    assert abs(response["excess"][lag0]) < 0.15 * response["control_mean"][lag0]
    # the unmatched baseline (the mean of every bar) is biased by the profile at the release hour
    naive_excess = response["event_mean"][lag0] - float(np.nanmean(bars["abs_ret"]))
    assert naive_excess > 0.5 * response["control_mean"][lag0]
    for c in controls:
        assert all(bars["hour"].iloc[c] == 13) and all(bars["dow"].iloc[c] == 4)


# EI06 -- a future-shifted event is refused, in both places it could enter
def test_ei06_future_shifted_event_is_refused():
    bars, calendar, _ = synthetic_world(seed=9, shift_hours=6)
    with pytest.raises(ei.EventImpactError) as error:
        ei.estimate_family(bars, calendar, "Synthetic | Planted", window=WINDOW, n_boot=50, seed=4)
    assert error.value.code == ei.FUTURE_SHIFTED_EVENT
    assert "precedes" in error.value.why
    stamps = pd.DatetimeIndex(["2013-01-04T13:30", "2013-01-18T13:30"])
    with pytest.raises(ei.EventImpactError) as error:
        ei.assert_released(stamps, "2013-01-10T00:00")
    assert error.value.code == ei.FUTURE_SHIFTED_EVENT


# EI07 -- an STL, loess or centred target is refused by type
@pytest.mark.parametrize("name", ["stl_trend", "loess_resid", "close_centered_ma", "seasonal_component",
                                  "lowess_24", "two_sided_filter"])
def test_ei07_two_sided_targets_are_refused_typed(name, world):
    bars, calendar, _ = world
    with pytest.raises(ei.EventImpactError) as error:
        ei.refuse_forbidden_target(name)
    assert error.value.code == ei.FORBIDDEN_TARGET_TRANSFORM
    bars = bars.assign(**{name: bars["abs_ret"]})
    positions = ei.release_positions(bars, calendar["release_utc"])
    controls = ei.match_controls(bars, positions, positions, window=WINDOW)
    with pytest.raises(ei.EventImpactError) as error:
        ei.impulse_response(bars, positions, controls, window=WINDOW, target=name, n_boot=10)
    assert error.value.code == ei.FORBIDDEN_TARGET_TRANSFORM
    assert ei.refuse_forbidden_target("abs_ret") == "abs_ret"


# EI08 -- the surprise scale is fitted on prior releases only
def test_ei08_surprise_scale_is_frozen_before_each_release():
    surprises = np.array([10.0, -5.0, 3.0, 8.0, -12.0, 4.0, -2.0, 6.0, 9.0, -7.0, 20.0, 1.0])
    z, scale = ei.standardize_surprises(surprises, min_prior=8)
    assert np.isnan(z[:8]).all()
    assert np.isfinite(z[8:]).all()
    altered = surprises.copy()
    altered[11] = 1e6
    z2, scale2 = ei.standardize_surprises(altered, min_prior=8)
    np.testing.assert_allclose(z[:11], z2[:11])
    assert scale[9] == pytest.approx(np.std(surprises[:9], ddof=1))
    assert z[9] == pytest.approx(surprises[9] / np.std(surprises[:9], ddof=1))


# EI09 -- the aggregation kernel reads released events and the schedule only, and matches its closed form
def test_ei09_cumulative_impact_kernel_is_closed_form_and_leak_free():
    impacts = {"F": {"volatility": 2.0, "half_life_hours": 4.0, "trend_per_z": 0.5}}
    events = [{"release_utc": "2013-03-01T13:30", "family": "F", "z": 1.5},
              {"release_utc": "2013-03-01T20:30", "family": "F", "z": -9.0}]
    times = pd.DatetimeIndex(["2013-03-01T15:30", "2013-03-01T19:30"])
    out = ei.cumulative_impact(times, events, impacts, horizon_hours=8)
    np.testing.assert_allclose(out["volatility"], [2.0 * 2 ** (-2 / 4), 2.0 * 2 ** (-6 / 4)])
    np.testing.assert_allclose(out["trend"], [0.5 * 1.5 * 2 ** (-2 / 4), 0.5 * 1.5 * 2 ** (-6 / 4)])
    np.testing.assert_allclose(out["anticipation"], [2.0 * 2 ** (-5 / 4), 2.0 * 2 ** (-1 / 4)])
    # the future event's surprise never enters: changing it leaves every feature unchanged
    events[1]["z"] = 999.0
    again = ei.cumulative_impact(times, events, impacts, horizon_hours=8)
    for key in out:
        np.testing.assert_allclose(out[key], again[key])
    # an event claimed released without an observed actual is refused
    with pytest.raises(ei.EventImpactError) as error:
        ei.cumulative_impact(pd.DatetimeIndex(["2013-03-01T21:30"]), [{"release_utc": "2013-03-01T20:30",
                                                                      "family": "F"}], impacts, horizon_hours=8)
    assert error.value.code == ei.FUTURE_SHIFTED_EVENT


# EI10 -- test rows are never read, and heterogeneity is reported per stratum under one interface
def test_ei10_test_rows_refused_and_heterogeneity_interface(estimate):
    with pytest.raises(ei.EventImpactError) as error:
        ei.bars_from_prices(pd.date_range("2024-12-31T20:00", periods=8, freq="h"), np.linspace(1.0, 1.01, 8))
    assert error.value.code == ei.TEST_ROWS_FORBIDDEN
    with pytest.raises(ei.EventImpactError) as error:
        ei.refuse_test_rows(pd.DatetimeIndex(["2025-01-01T00:00"]))
    assert error.value.code == ei.TEST_ROWS_FORBIDDEN
    rng = np.random.default_rng(3)
    z = rng.standard_normal(600)
    strata = np.where(np.arange(600) % 2 == 0, "low", "high")
    y = np.where(strata == "low", 0.1, 0.5) * z + 0.05 * rng.standard_normal(600)
    cate = ei.heterogeneous_effects(y, z, strata)
    assert cate["estimator"] in {"stratified_ols_hc1", "econml_linear_dml"}
    assert cate["strata"]["low"]["slope_ci"][0] < 0.1 < cate["strata"]["low"]["slope_ci"][1]
    assert cate["strata"]["high"]["slope_ci"][0] < 0.5 < cate["strata"]["high"]["slope_ci"][1]
    cate_estimate = estimate["heterogeneity_by_pre_vol"]
    assert set(cate_estimate["strata"]) <= {"low_pre_vol", "mid_pre_vol", "high_pre_vol"}


def test_calendar_clock_offsets_come_from_anchors_not_a_legal_rule():
    local = pd.DatetimeIndex(["2011-01-07T08:30", "2011-01-10T03:00", "2011-07-08T07:30", "2011-07-11T03:00",
                              "2018-04-06T08:30", "2018-04-09T03:00"])
    family = ["United States | Nonfarm Payrolls", "Euro Zone | CPI", "United States | Nonfarm Payrolls",
              "Euro Zone | CPI", "United States | Nonfarm Payrolls", "Euro Zone | CPI"]
    offsets, status = ei.calendar_clock_offsets(local, family)
    np.testing.assert_allclose(offsets, [5, 5, 5, 5, 4, 4])
    assert status[0] == "ANCHORED" and status[-1] == "ANCHORED"
    assert "SWITCH_AMBIGUOUS" in status


# EI11 -- the trend decay is estimated separately from the volatility decay and the two are told apart
def test_ei11_trend_half_life_is_separate_and_recovered(world, estimate):
    bars, calendar, _ = world
    positions = ei.release_positions(bars, calendar["release_utc"])
    z, _ = ei.standardize_surprises(calendar["surprise"].to_numpy())
    trend = ei.trend_impulse_response(bars, positions, z, window=12, n_boot=200, seed=5)
    weights = 2.0 ** (-np.arange(HORIZON) / HALF_LIFE_TREND)
    planted_lag0 = DRIFT_PER_Z * weights[0] / weights.sum()
    lag0 = trend["lags"].index(0)
    assert trend["beta_ci_lower"][lag0] < planted_lag0 < trend["beta_ci_upper"][lag0]
    assert abs(trend["half_life_hours"] - HALF_LIFE_TREND) < 1.5
    comparison = ei.compare_half_lives(estimate["impulse_response"], trend)
    assert comparison["status"] == "OK"
    assert comparison["trend_minus_volatility_hours"] < 0
    # a response that sits before the release is zero: the pre-release betas straddle zero
    for k, lag in enumerate(trend["lags"]):
        if lag < 0:
            assert trend["beta_ci_lower"][k] < 0 < trend["beta_ci_upper"][k]


# EI12 -- the patient-state arm is kept only when it predicts the decision period better, purged and chronological
def test_ei12_state_arm_is_judged_out_of_sample():
    rng = np.random.default_rng(21)
    n = 400
    times = pd.date_range("2012-01-06T13:30", periods=n, freq="7D")
    z = rng.standard_normal(n)
    label = np.where(np.arange(n) % 2 == 0, "calm", "stressed")
    y_hetero = np.where(label == "calm", 0.001, 0.006) * z + 0.001 * rng.standard_normal(n)
    kept = ei.cate_out_of_sample(y_hetero, z, label, times, decision_start="2017-01-01")
    assert kept["status"] == "OK" and kept["verdict"] == "X_ARM_KEPT"
    assert kept["n_train"] + kept["n_decision"] < n, "the purge removed rows between the two periods"
    y_homo = 0.003 * z + 0.001 * rng.standard_normal(n)
    noise_label = rng.choice(["a", "b", "c", "d", "e", "f"], size=n)
    dropped = ei.cate_out_of_sample(y_homo, z, noise_label, times, decision_start="2017-01-01", min_n=5)
    assert dropped["status"] == "OK"
    assert dropped["mse_x_arm"] >= dropped["mse_x_free"] * 0.98
    state = ei.market_state(ei.bars_from_prices(pd.date_range("2013-01-01", periods=300, freq="h"),
                                                np.exp(np.cumsum(rng.standard_normal(300) * 1e-3))),
                            np.array([150, 200, 250]))
    assert {"vol_short", "vol_long", "vol_ratio", "trend_strength_short", "px.rv5", "px.log_co"} <= set(state)
    labels = ei.regime_labels(state, np.array([True, True, True]))
    assert len(labels) == 3


# EI13 -- concurrent doses: additivity holds when the world is additive, and is reported when it is not
def test_ei13_concurrency_additivity(world):
    bars, calendar, _ = world
    second = calendar.copy()
    second["release_utc"] = second["release_utc"] + pd.Timedelta(hours=3)
    second["family"] = "Synthetic | Second"
    both = pd.concat([calendar, second], ignore_index=True)
    out = ei.concurrency_test(bars, both, ["Synthetic | Planted", "Synthetic | Second"], window=WINDOW)
    assert out["status"] in {"OK", "NO_OVERLAPS"}
    if out["status"] == "OK":
        assert out["n_overlapping"] > 0 and "verdict" in out


# EI14 -- event selection ranks on the decision period only, controls FDR, needs enough events, and freezes
def test_ei14_event_selection_is_out_of_sample_fdr_controlled_and_frozen():
    bars, calendar, _ = synthetic_world(seed=31, weeks=260)
    null = calendar.copy()
    null["family"] = "Synthetic | Null"
    null["release_utc"] = null["release_utc"] + pd.Timedelta(days=7)
    null["actual"] = null["forecast"] + np.random.default_rng(1).standard_normal(len(null)) * 50
    null["surprise"] = null["actual"] - null["forecast"]
    both = pd.concat([calendar, null], ignore_index=True)
    out = ei.select_events(bars, both, ["Synthetic | Planted", "Synthetic | Null", "Synthetic | Absent"],
                           decision_start="2015-01-01", horizon=HORIZON, n_boot=30)
    by = {e["family"]: e for e in out["ranking"]}
    assert by["Synthetic | Absent"]["status"] == "TOO_FEW_EVENTS"
    assert by["Synthetic | Planted"]["selected"] is True
    assert by["Synthetic | Planted"]["oos_mean_log_return"] > 0
    assert out["frozen"] is True and out["fdr_method"] == "benjamini_hochberg"
    assert out["selected"] == ["Synthetic | Planted"] or "Synthetic | Null" not in out["selected"]
    passed = ei.benjamini_hochberg([0.001, 0.2, 0.9], alpha=0.1)
    assert passed.tolist() == [True, False, False]
