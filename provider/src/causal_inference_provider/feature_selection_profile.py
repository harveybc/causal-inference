"""Frequency-aware PS1 profiling for one TRAIN time series.

The metric families originate in lane-A's established PS1 profiler. This port
adds calendar-aware gaps, outlier severity, PACF and entropy while refusing to
hide irregular sampling through silent gap compression or zero filling.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal, stats


METRICS_VERSION = "generic_ps1_metrics.v1"
METRICS = (
    "missingness", "timestamp_gaps", "constant", "scale_tails", "outliers",
    "volatility", "acf", "pacf", "trend", "adf", "kpss", "seasonality",
    "spectrum", "information_entropy", "cost",
)
LAG_DURATIONS = ("1h", "6h", "24h", "120h", "168h")
MAX_STAT_N = 20_000

try:
    from statsmodels.tsa.stattools import adfuller, kpss, pacf as sm_pacf
    _STATSMODELS = True
except Exception:  # pragma: no cover - dependency state is reported in cells
    _STATSMODELS = False


def _json_safe(value: Any) -> Any:
    """Convert profiler output to strict JSON without inventing replacements."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _cell(feature_id: str, metric: str, state: str, value: Any = None, reason: str = "") -> dict[str, Any]:
    return {
        "feature_id": feature_id,
        "metric": metric,
        "state": state,
        "value": _json_safe(value),
        "reason": reason,
        "metrics_version": METRICS_VERSION,
    }


def _frequency_delta(frequency: str) -> pd.Timedelta:
    try:
        delta = pd.Timedelta(frequency)
    except ValueError as trouble:
        raise ValueError(f"frequency must be a fixed pandas duration, got {frequency!r}") from trouble
    if delta <= pd.Timedelta(0):
        raise ValueError("frequency must be positive")
    return delta


def _calendar_grid(start: pd.Timestamp, end: pd.Timestamp, frequency: str, calendar: str) -> pd.DatetimeIndex:
    grid = pd.date_range(start, end, freq=frequency)
    rule = calendar.strip().lower()
    if rule in {"24x7", "continuous", "crypto"}:
        return grid
    if rule in {"24x5", "weekday_24h", "fx"}:
        return grid[grid.dayofweek < 5]
    if rule == "observed_only":
        return pd.DatetimeIndex([])
    raise ValueError(f"unsupported calendar rule: {calendar}")


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_start = best_end = current_start = 0
    for index, value in enumerate(mask):
        if value:
            if index == 0 or not mask[index - 1]:
                current_start = index
            if index + 1 - current_start > best_end - best_start:
                best_start, best_end = current_start, index + 1
    return best_start, best_end


def _duration_lags(frequency: str) -> dict[str, dict[str, Any]]:
    step = _frequency_delta(frequency)
    result = {}
    for duration in LAG_DURATIONS:
        ratio = pd.Timedelta(duration) / step
        rows = int(round(ratio))
        if ratio < 1 or not math.isclose(ratio, rows, rel_tol=0.0, abs_tol=1e-12):
            result[duration] = {"state": "NOT_APPLICABLE", "rows": None,
                                "reason": "DURATION_NOT_REPRESENTABLE_ON_GRID"}
        else:
            result[duration] = {"state": "READY", "rows": rows}
    return result


def _histogram_entropy(values: np.ndarray) -> tuple[float | None, int]:
    if len(values) < 2:
        return None, 0
    bins = max(2, min(128, int(np.sqrt(len(values)))))
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(bins), bins


def profile_series(feature_id: str, frame: pd.DataFrame, timestamp_column: str, feature_column: str,
                   frequency: str, calendar: str, build_cost_s: float | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    step = _frequency_delta(frequency)
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    numeric = pd.to_numeric(frame[feature_column], errors="coerce")
    valid_timestamp = timestamps.notna()
    ordered = pd.DataFrame({"timestamp": timestamps[valid_timestamp], "value": numeric[valid_timestamp]})
    was_monotonic = bool(ordered.timestamp.is_monotonic_increasing)
    duplicate_count = int(ordered.timestamp.duplicated(keep=False).sum())
    ordered = ordered.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if len(ordered) and calendar.strip().lower() != "observed_only":
        grid = _calendar_grid(ordered.timestamp.iloc[0], ordered.timestamp.iloc[-1], frequency, calendar)
        series = ordered.set_index("timestamp").value.reindex(grid)
        missing_grid = grid.difference(pd.DatetimeIndex(ordered.timestamp))
    else:
        grid = pd.DatetimeIndex(ordered.timestamp)
        series = ordered.set_index("timestamp").value
        missing_grid = pd.DatetimeIndex([])
    values = series.to_numpy(float)
    finite = np.isfinite(values)
    finite_values = values[finite]
    run_start, run_end = _longest_true_run(finite)
    contiguous = values[run_start:run_end]
    cells: list[dict[str, Any]] = []

    cells.append(_cell(feature_id, "missingness", "MEASURED", {
        "source_rows": int(len(frame)),
        "invalid_timestamp_count": int((~valid_timestamp).sum()),
        "grid_rows": int(len(grid)),
        "finite_count": int(finite.sum()),
        "missing_or_nonfinite_count": int((~finite).sum()),
        "missing_fraction": float((~finite).mean()) if len(finite) else None,
    }))
    gap_state = "NOT_APPLICABLE" if calendar.strip().lower() == "observed_only" else "MEASURED"
    cells.append(_cell(feature_id, "timestamp_gaps", gap_state, {
        "calendar_rule": calendar,
        "frequency": frequency,
        "frequency_seconds": float(step.total_seconds()),
        "expected_timestamp_count": int(len(grid)),
        "missing_timestamp_count": int(len(missing_grid)),
        "duplicate_timestamp_rows": duplicate_count,
        "monotonic_in_source_order": was_monotonic,
        "longest_contiguous_finite_rows": int(len(contiguous)),
        "excluded_from_contiguous_analysis": int(len(values) - len(contiguous)),
    }, "CALENDAR_HAS_NO_EXPECTED_GRID" if gap_state == "NOT_APPLICABLE" else ""))

    if not len(finite_values):
        for metric in METRICS[2:-1]:
            cells.append(_cell(feature_id, metric, "NOT_APPLICABLE", reason="NO_FINITE_TRAIN_VALUES"))
        cells.append(_cell(feature_id, "cost", "MEASURED", {
            "build_s": build_cost_s,
            "profile_s": float(time.perf_counter() - started),
            "bytes_float64": int(len(values) * 8),
        }))
        return {"schema": "feature_selection_ps1_profile.v1", "feature_id": feature_id,
                "frequency": frequency, "calendar_rule": calendar, "cells": cells}

    unique = np.unique(finite_values)
    is_constant = len(unique) == 1
    top_share = float(pd.Series(finite_values).value_counts(normalize=True).iloc[0])
    cells.append(_cell(feature_id, "constant", "MEASURED", {
        "n_unique": int(len(unique)), "is_constant": is_constant, "top_value_share": top_share,
    }))

    def measured(metric: str, function) -> None:
        try:
            cells.append(_cell(feature_id, metric, "MEASURED", function()))
        except Exception as trouble:  # a failed metric is never replaced by zero
            cells.append(_cell(feature_id, metric, "FAILED", reason=f"{type(trouble).__name__}: {trouble}"[:300]))

    if is_constant:
        for metric in METRICS[3:-1]:
            cells.append(_cell(feature_id, metric, "NOT_APPLICABLE", reason="CONSTANT_IN_TRAIN"))
    else:
        quantile = np.percentile(finite_values, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
        median, iqr = float(quantile[4]), float(quantile[5] - quantile[3])
        mad = float(np.median(np.abs(finite_values - median)))

        measured("scale_tails", lambda: {
            "mean": float(np.mean(finite_values)), "std": float(np.std(finite_values)),
            "min": float(np.min(finite_values)), "max": float(np.max(finite_values)),
            "p001": float(quantile[0]), "p01": float(quantile[1]), "p05": float(quantile[2]),
            "p25": float(quantile[3]), "median": median, "p75": float(quantile[5]),
            "p95": float(quantile[6]), "p99": float(quantile[7]), "p999": float(quantile[8]),
            "mad": mad, "iqr": iqr, "skew": float(stats.skew(finite_values)),
            "excess_kurtosis": float(stats.kurtosis(finite_values)),
            "tail_ratio_p99_mad": float((quantile[7] - median) / mad) if mad else None,
            "share_beyond_5mad": float(np.mean(np.abs(finite_values - median) > 5 * mad)) if mad else None,
        })

        def outliers() -> dict[str, Any]:
            robust_scale = 1.4826 * mad
            robust_z = np.abs(finite_values - median) / robust_scale if robust_scale else np.zeros(len(finite_values))
            lower, upper = quantile[3] - 1.5 * iqr, quantile[5] + 1.5 * iqr
            iqr_distance = np.maximum(lower - finite_values, finite_values - upper)
            return {
                "iqr_1_5_count": int(np.sum((finite_values < lower) | (finite_values > upper))),
                "iqr_max_severity": float(max(0.0, np.max(iqr_distance) / iqr)) if iqr else None,
                "robust_z_3_5_count": int(np.sum(robust_z > 3.5)),
                "robust_z_max": float(np.max(robust_z)) if robust_scale else None,
            }
        measured("outliers", outliers)

        def volatility() -> dict[str, Any]:
            differences = np.diff(contiguous)
            duration = pd.Timedelta("168h") / step
            window = int(round(duration)) if duration >= 2 and math.isclose(duration, round(duration)) else None
            rolling = pd.Series(contiguous).rolling(window, min_periods=max(2, window // 2)).std() if window else pd.Series(dtype=float)
            rolling = rolling.dropna()
            return {
                "std_first_difference_contiguous": float(np.std(differences)) if len(differences) else None,
                "rolling_168h_rows": window,
                "rolling_168h_std_median": float(rolling.median()) if len(rolling) else None,
                "volatility_of_volatility_cv": float(rolling.std() / rolling.mean()) if len(rolling) and rolling.mean() > 0 else None,
                "diagnostics": {"gaps_compressed": False, "contiguous_rows": int(len(contiguous))},
            }
        measured("volatility", volatility)

        lag_specs = _duration_lags(frequency)

        def autocorrelation() -> dict[str, Any]:
            output = {}
            contiguous_series = pd.Series(contiguous)
            for duration, spec in lag_specs.items():
                if spec["state"] != "READY":
                    output[duration] = spec
                elif len(contiguous) <= spec["rows"] + 10:
                    output[duration] = {**spec, "state": "NOT_APPLICABLE", "value": None,
                                        "reason": "TOO_FEW_CONTIGUOUS_ROWS"}
                else:
                    output[duration] = {**spec, "state": "MEASURED",
                                        "value": float(contiguous_series.autocorr(spec["rows"]))}
            return {"lags": output, "lag_unit": "declared duration converted to rows at the declared frequency",
                    "diagnostics": {"gaps_compressed": False, "contiguous_rows": int(len(contiguous))}}
        measured("acf", autocorrelation)

        if not _STATSMODELS:
            cells.append(_cell(feature_id, "pacf", "PENDING", reason="DEPENDENCY_MISSING:statsmodels"))
        else:
            def partial_autocorrelation() -> dict[str, Any]:
                ready = {duration: spec["rows"] for duration, spec in lag_specs.items() if spec["state"] == "READY"}
                measurable = {duration: lag for duration, lag in ready.items() if len(contiguous) > 2 * lag + 1}
                if not measurable:
                    return {"lags": {duration: {**spec, "state": "NOT_APPLICABLE",
                                                "reason": "TOO_FEW_CONTIGUOUS_ROWS"}
                                      for duration, spec in lag_specs.items()},
                            "diagnostics": {"gaps_compressed": False, "contiguous_rows": int(len(contiguous))}}
                values_pacf = sm_pacf(contiguous, nlags=max(measurable.values()), method="ywm")
                output = {}
                for duration, spec in lag_specs.items():
                    lag = spec.get("rows")
                    if duration in measurable:
                        output[duration] = {**spec, "state": "MEASURED", "value": float(values_pacf[lag])}
                    else:
                        output[duration] = {**spec, "state": "NOT_APPLICABLE", "value": None,
                                            "reason": spec.get("reason", "TOO_FEW_CONTIGUOUS_ROWS")}
                return {"lags": output, "method": "Yule-Walker without adjustment",
                        "diagnostics": {"gaps_compressed": False, "contiguous_rows": int(len(contiguous))}}
            measured("pacf", partial_autocorrelation)

        def trend() -> dict[str, Any]:
            finite_positions = np.flatnonzero(finite)
            elapsed_years = (series.index[finite_positions] - series.index[finite_positions][0]).total_seconds() / (365.25 * 86400)
            slope, _, correlation, _, _ = stats.linregress(elapsed_years, finite_values)
            sample_index = np.linspace(0, len(finite_values) - 1, min(len(finite_values), 3000)).astype(int)
            tau, tau_p = stats.kendalltau(elapsed_years[sample_index], finite_values[sample_index])
            return {"ols_slope_per_year": float(slope),
                    "slope_in_std_per_year": float(slope / np.std(finite_values)),
                    "ols_r": float(correlation), "kendall_tau_3000": float(tau),
                    "kendall_p_3000": float(tau_p),
                    "diagnostics": {"uses_actual_elapsed_time": True, "gaps_compressed": False}}
        measured("trend", trend)

        stationary_sample = contiguous[-MAX_STAT_N:]
        stationary_diagnostics = {"sample": "last contiguous finite TRAIN segment",
                                  "sample_rows": int(len(stationary_sample)), "gaps_compressed": False,
                                  "excluded_noncontiguous_rows": int(len(values) - len(contiguous))}
        if not _STATSMODELS:
            cells.append(_cell(feature_id, "adf", "PENDING", reason="DEPENDENCY_MISSING:statsmodels"))
            cells.append(_cell(feature_id, "kpss", "PENDING", reason="DEPENDENCY_MISSING:statsmodels"))
        elif len(unique) < 3 or len(stationary_sample) < 30:
            reason = "FEWER_THAN_3_DISTINCT_VALUES" if len(unique) < 3 else "TOO_FEW_CONTIGUOUS_ROWS"
            cells.append(_cell(feature_id, "adf", "NOT_APPLICABLE", reason=reason))
            cells.append(_cell(feature_id, "kpss", "NOT_APPLICABLE", reason=reason))
        else:
            max_lag = min(24, max(1, len(stationary_sample) // 10))
            measured("adf", lambda: _adf(stationary_sample, max_lag, stationary_diagnostics))
            measured("kpss", lambda: _kpss(stationary_sample, stationary_diagnostics))

        def seasonality() -> dict[str, Any]:
            observed = series.dropna()
            total_variance = float(observed.var())
            output = {}
            for name, key in (("hour_of_day", observed.index.hour),
                              ("day_of_week", observed.index.dayofweek),
                              ("month", observed.index.month)):
                group_mean = observed.groupby(key).transform("mean")
                output[f"eta2_{name}"] = float(group_mean.var() / total_variance) if total_variance > 0 else None
            output["diagnostics"] = {"timestamps": "event timestamps as stored", "gaps_compressed": False}
            return output
        measured("seasonality", seasonality)

        spectrum_value: dict[str, Any] = {}
        spectrum_diagnostics = {"zero_filled": False, "gaps_compressed": False,
                                "contiguous_rows": int(len(contiguous)),
                                "excluded_rows": int(len(values) - len(contiguous))}

        def spectrum() -> dict[str, Any]:
            frequencies, power = signal.welch(contiguous - np.mean(contiguous), fs=1.0 / step.total_seconds(),
                                               nperseg=min(2048, len(contiguous)))
            frequencies, power = frequencies[1:], power[1:]
            if not len(power) or float(np.sum(power)) <= 0:
                raise ValueError("NO_POSITIVE_SPECTRAL_POWER")
            normalized = power / np.sum(power)
            top = np.argsort(power)[::-1][:3]
            entropy = -float(np.sum(normalized * np.log(normalized + 1e-300))) / math.log(len(normalized))
            spectrum_value.update({"top_period_seconds": [float(1.0 / frequencies[index]) for index in top],
                                   "top_power_share": [float(normalized[index]) for index in top],
                                   "spectral_entropy_normalized": entropy,
                                   "method": "Welch on longest contiguous finite segment",
                                   "diagnostics": spectrum_diagnostics})
            return spectrum_value
        if len(contiguous) < 16:
            cells.append(_cell(feature_id, "spectrum", "NOT_APPLICABLE",
                               {"diagnostics": spectrum_diagnostics}, "TOO_FEW_CONTIGUOUS_ROWS"))
        else:
            measured("spectrum", spectrum)

        def information_entropy() -> dict[str, Any]:
            histogram_entropy, bins = _histogram_entropy(finite_values)
            return {"histogram_entropy_normalized": histogram_entropy, "histogram_bins": bins,
                    "spectral_entropy_normalized": spectrum_value.get("spectral_entropy_normalized"),
                    "diagnostics": {"histogram_uses_finite_values": True,
                                    "spectrum_uses_contiguous_values": True,
                                    "gaps_compressed": False, "zero_filled": False}}
        measured("information_entropy", information_entropy)

    cells.append(_cell(feature_id, "cost", "MEASURED", {
        "build_s": build_cost_s, "profile_s": float(time.perf_counter() - started),
        "bytes_float64": int(len(values) * 8),
    }))
    return {"schema": "feature_selection_ps1_profile.v1", "feature_id": feature_id,
            "frequency": frequency, "calendar_rule": calendar, "cells": cells}


def _adf(sample: np.ndarray, max_lag: int, diagnostics: dict[str, Any]) -> dict[str, Any]:
    result = adfuller(sample, maxlag=max_lag, autolag=None, regression="c")
    return {"stat": float(result[0]), "pvalue": float(result[1]), "lags": int(result[2]),
            "nobs": int(result[3]), "critical_5pct": float(result[4]["5%"]),
            "null_hypothesis": "unit root", "regression": "constant", "diagnostics": diagnostics}


def _kpss(sample: np.ndarray, diagnostics: dict[str, Any]) -> dict[str, Any]:
    import warnings
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = kpss(sample, regression="c", nlags="auto")
    return {"stat": float(result[0]), "pvalue": float(result[1]), "lags": int(result[2]),
            "null_hypothesis": "level stationary",
            "pvalue_truncated_to_table": any("p-value" in str(item.message) for item in captured),
            "diagnostics": diagnostics}
