"""Causal impact of scheduled macro releases on EUR/USD hourly volatility and trend, from data only.

This module is the estimator the design `docs/EVENT_IMPACT_DESIGN.md` specifies. It is deliberately separate from
`event_study.py`, which registers and answers from a projections document that feature-eng fitted; this one FITS,
offline, from a calendar and an hourly bar table, and writes plain dictionaries that a tool serialises. It climbs
Pearl's ladder with observed data only:

* association -- releases co-occur with elevated absolute returns (`impulse_response` raw path `event_mean`);
* intervention -- the release contrasted with MATCHED CONTROL windows: same hour of day, same day of week, no release
  of any covered family inside the window, nearest in pre-release realized volatility (`match_controls`). The
  contrast is the excess path, and the half-life is fitted on it;
* counterfactual -- only through matched observed cases. No synthetic generator exists in this module. The one in the
  tests plants a known impulse to prove the estimator recovers it, and never informs an estimate.

Trend is estimated only CONDITIONAL on the surprise (actual minus consensus, standardized on prior releases of the
same family), because the announcement literature (Andersen, Bollerslev, Diebold and Vega 2003) finds the signed
response is a function of the surprise and the unconditional drift is indistinguishable from zero.

Leakage rules this module enforces by refusal, each with a typed code on `EventImpactError`:

* `FORBIDDEN_TARGET_TRANSFORM` -- any target or input named like an STL, loess/lowess, centred or two-sided
  smoother. Those filters read bars AFTER t to describe t; a model fitted on them learns the future.
* `FUTURE_SHIFTED_EVENT` -- an event whose release instant lies after the `as_of` instant it is used at, or a family
  whose matched excess is already significant BEFORE the release (the response precedes the cause: the clock or the
  schedule is shifted).
* `TEST_ROWS_FORBIDDEN` -- any row at or after 2025-01-01, which no estimator in this programme may read.
* `CONSENSUS_NOT_FROZEN` -- a standardization scale is fitted only on releases strictly before the one it scales.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re

import numpy as np
import pandas as pd

SCHEMA = "causal-inference.event_impact.v1"

#: rows at or after this instant are never read by anything in this programme
TEST_BOUNDARY = pd.Timestamp("2025-01-01T00:00:00")

#: the calendar fixture's clock is NOT a legal time zone. Read from the data: a fixed UTC-5 clock without daylight
#: saving for 2011-01 to 2018-01 (Nonfarm Payrolls, published 08:30 US Eastern, appears at 8:30 in winter and 7:30
#: in summer) and an exporter pseudo-DST afterwards (UTC-4 over roughly February/March to September/November of
#: 2018-2021, switching on dates no legal rule produces). So the offset is a PER-PERIOD TABLE built from anchor
#: releases whose true publication instant is known (the US 08:30 Eastern families below); every other row takes
#: the offset of the anchor nearest in time, and a row between two anchors that disagree is marked
#: SWITCH_AMBIGUOUS. This constant is only the fallback for rows with no anchor at all.
CALENDAR_UTC_OFFSET_HOURS = 5

#: US families published at 08:30 US Eastern (Bureau of Labor Statistics / Census / BEA release times)
ANCHOR_FAMILIES_0830_ET = (
    "United States | Nonfarm Payrolls", "United States | CPI", "United States | Core CPI",
    "United States | Retail Sales", "United States | Core Retail Sales", "United States | PPI",
    "United States | GDP", "United States | Initial Jobless Claims", "United States | Building Permits",
    "United States | Unemployment Rate", "United States | Core Durable Goods Orders",
)

#: column names of the headerless calendar fixture, in file order
CALENDAR_COLUMNS = ("date", "time", "country", "impact", "event", "actual_text", "unit", "actual", "forecast",
                    "previous")

FORBIDDEN_TARGET_PATTERN = re.compile(
    r"(stl|loess|lowess|centred|centered|two[_-]?sided|seasonal|trend[_-]?comp|resid|smooth|savgol|hp[_-]?filter)",
    re.IGNORECASE)

FORBIDDEN_TARGET_TRANSFORM = "FORBIDDEN_TARGET_TRANSFORM"
FUTURE_SHIFTED_EVENT = "FUTURE_SHIFTED_EVENT"
TEST_ROWS_FORBIDDEN = "TEST_ROWS_FORBIDDEN"
CONSENSUS_NOT_FROZEN = "CONSENSUS_NOT_FROZEN"
TOO_FEW_EVENTS = "TOO_FEW_EVENTS"
MALFORMED_INPUT = "MALFORMED_INPUT"

#: the fewest prior releases a standardization scale may be fitted on
MIN_PRIOR_RELEASES = 8

#: the fewest events an impulse response is fitted on
MIN_EVENTS = 20


class EventImpactError(ValueError):
    """A typed refusal. `code` is one of the module constants; `why` says what was seen."""

    def __init__(self, code, why):
        super().__init__(f"{code}: {why}")
        self.code = code
        self.why = why


# ---------------------------------------------------------------------------------------------------- leakage guards

def refuse_forbidden_target(name):
    """Refuse a target or input column whose name says it was produced by a two-sided filter."""
    if not isinstance(name, str) or not name:
        raise EventImpactError(MALFORMED_INPUT, "a target name must be a non-empty string")
    match = FORBIDDEN_TARGET_PATTERN.search(name)
    if match:
        raise EventImpactError(
            FORBIDDEN_TARGET_TRANSFORM,
            f"target {name!r} names a two-sided or decomposition filter ({match.group(0)!r}); such a series at bar t "
            f"reads bars after t and is forbidden as an input or a target. STL is allowed only as a retrospective "
            f"diagnostic on closed data, never as a fitted quantity.")
    return name


def refuse_test_rows(timestamps, what="rows"):
    """Refuse any timestamp at or after the test boundary."""
    stamps = pd.DatetimeIndex(timestamps)
    if len(stamps) and stamps.max() >= TEST_BOUNDARY:
        raise EventImpactError(TEST_ROWS_FORBIDDEN,
                               f"{what} reach {stamps.max()} but nothing at or after {TEST_BOUNDARY.date()} may be read")


def assert_released(release_times, as_of):
    """Refuse an event whose release lies after the instant its actual value is used at."""
    stamps = pd.DatetimeIndex(release_times)
    as_of = pd.Timestamp(as_of)
    late = stamps[stamps > as_of]
    if len(late):
        raise EventImpactError(FUTURE_SHIFTED_EVENT,
                               f"{len(late)} event(s) are released after as_of={as_of} (first: {late.min()}); their "
                               f"actual value is not available at as_of")


# ------------------------------------------------------------------------------------------------------------ loading

def _to_float(series):
    cleaned = series.astype(str).str.strip().str.replace(",", "", regex=False)
    cleaned = cleaned.str.extract(r"^(-?\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(cleaned, errors="coerce")


def load_calendar(path, *, utc_offset_hours=CALENDAR_UTC_OFFSET_HOURS, end_exclusive=None):
    """Read the headerless calendar fixture; return one row per release with `release_utc` and numeric surprise.

    `end_exclusive` bounds what is read; it may not pass the test boundary.
    """
    end = TEST_BOUNDARY if end_exclusive is None else pd.Timestamp(end_exclusive)
    if end > TEST_BOUNDARY:
        raise EventImpactError(TEST_ROWS_FORBIDDEN, f"end_exclusive={end} passes the test boundary {TEST_BOUNDARY}")
    raw = pd.read_csv(path, header=None, names=list(CALENDAR_COLUMNS), dtype=str, keep_default_na=False)
    for column in CALENDAR_COLUMNS:
        raw[column] = raw[column].astype(str).str.strip()
    local = pd.to_datetime(raw["date"] + " " + raw["time"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    frame = pd.DataFrame({
        "release_local": local,
        "country": raw["country"],
        "impact": raw["impact"],
        "event": raw["event"],
        "unit": raw["unit"],
        "actual": _to_float(raw["actual"]),
        "forecast": _to_float(raw["forecast"]),
        "previous": _to_float(raw["previous"]),
    })
    frame = frame[frame["release_local"].notna()]
    frame = frame[frame["release_local"] < end].copy()
    frame["family"] = frame["country"] + " | " + frame["event"]
    frame["surprise"] = frame["actual"] - frame["forecast"]
    frame = frame.sort_values("release_local").reset_index(drop=True)
    offsets, status = calendar_clock_offsets(frame["release_local"], frame["family"], fallback=utc_offset_hours)
    frame["utc_offset_hours"] = offsets
    frame["clock_status"] = status
    frame["release_utc"] = frame["release_local"] + pd.to_timedelta(offsets, unit="h")
    return frame.sort_values("release_utc").reset_index(drop=True)


def us_eastern_utc_offset(dates):
    """UTC minus US Eastern, in hours, for the TRUE publication instant: 4 under daylight saving (second Sunday of
    March to first Sunday of November, the rule in force since 2007), 5 otherwise. This is applied to the known
    publication clock of the anchors, never to the fixture's own clock."""
    dates = pd.DatetimeIndex(dates)
    out = np.full(len(dates), 5, dtype=int)
    for k, day in enumerate(dates):
        march = pd.Timestamp(year=day.year, month=3, day=1)
        start = march + pd.Timedelta(days=(6 - march.dayofweek) % 7 + 7)
        november = pd.Timestamp(year=day.year, month=11, day=1)
        stop = november + pd.Timedelta(days=(6 - november.dayofweek) % 7)
        if start <= day.normalize() < stop:
            out[k] = 4
    return out


def calendar_clock_offsets(release_local, family, *, fallback=CALENDAR_UTC_OFFSET_HOURS,
                           anchor_families=ANCHOR_FAMILIES_0830_ET):
    """Per-row UTC offset of the fixture clock, from anchor releases of known publication instant.

    An anchor at local time L on day D was really published at 08:30 Eastern = (08:30 + eastern_offset(D)) UTC, so
    the fixture clock's offset on that day is (08:30 UTC-of-publication) - L. Every row takes the offset of the
    nearest anchor; a row whose previous and next anchors disagree is SWITCH_AMBIGUOUS. Returns (offsets, status).
    """
    local = pd.DatetimeIndex(release_local)
    family = np.asarray(family)
    is_anchor = np.isin(family, list(anchor_families))
    n = len(local)
    offsets = np.full(n, fallback, dtype=float)
    status = np.full(n, "NO_ANCHOR", dtype=object)
    if not is_anchor.any():
        return offsets, status
    anchor_times = local[is_anchor]
    eastern = us_eastern_utc_offset(anchor_times)
    publication_utc_minutes = 8 * 60 + 30 + eastern * 60
    local_minutes = anchor_times.hour * 60 + anchor_times.minute
    anchor_offsets = (publication_utc_minutes - local_minutes) / 60.0
    anchor_offsets = np.where(np.abs(anchor_offsets - 5.0) <= 2.0, anchor_offsets, np.nan)
    good = np.isfinite(anchor_offsets)
    anchor_times, anchor_offsets = anchor_times[good], anchor_offsets[good]
    if len(anchor_times) == 0:
        return offsets, status
    position = anchor_times.searchsorted(local, side="right")
    previous = np.clip(position - 1, 0, len(anchor_times) - 1)
    following = np.clip(position, 0, len(anchor_times) - 1)
    prev_off = anchor_offsets[previous]
    next_off = anchor_offsets[following]
    agree = prev_off == next_off
    offsets[:] = np.where(agree, prev_off, prev_off)
    status[:] = np.where(agree, "ANCHORED", "SWITCH_AMBIGUOUS")
    return offsets, status


def load_hourly(path, *, end_exclusive=None):
    """Read the hourly OHLC fixture (mixed D/M/Y stamps, UTC clock) into the bar table the estimators consume."""
    end = TEST_BOUNDARY if end_exclusive is None else pd.Timestamp(end_exclusive)
    if end > TEST_BOUNDARY:
        raise EventImpactError(TEST_ROWS_FORBIDDEN, f"end_exclusive={end} passes the test boundary {TEST_BOUNDARY}")
    raw = pd.read_csv(path)
    stamps = pd.to_datetime(raw["datetime"], dayfirst=True)
    raw = raw.assign(datetime=stamps).sort_values("datetime")
    raw = raw[raw["datetime"] < end]
    return bars_from_prices(raw["datetime"].to_numpy(), raw["close"].to_numpy(dtype=float),
                            raw["high"].to_numpy(dtype=float), raw["low"].to_numpy(dtype=float))


def bars_from_prices(timestamps, close, high=None, low=None):
    """The bar table: UTC index, close, log return, absolute log return (realized-vol unit), log range."""
    index = pd.DatetimeIndex(timestamps)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise EventImpactError(MALFORMED_INPUT, "bar timestamps must be strictly increasing")
    refuse_test_rows(index, "bars")
    close = np.asarray(close, dtype=float)
    log_ret = np.diff(np.log(close), prepend=np.nan)
    bars = pd.DataFrame({"close": close, "log_ret": log_ret, "abs_ret": np.abs(log_ret)}, index=index)
    if high is not None and low is not None:
        bars["log_range"] = np.log(np.asarray(high, dtype=float) / np.asarray(low, dtype=float))
    bars["hour"] = index.hour
    bars["dow"] = index.dayofweek
    return bars


def release_positions(bars, release_times):
    """The positional index of the bar that CONTAINS each release (the last bar opening at or before it).

    An event with no bar inside 2 hours after its release (weekend, holiday, gap) gets -1.
    """
    stamps = pd.DatetimeIndex(release_times)
    positions = bars.index.searchsorted(stamps, side="right") - 1
    out = np.full(len(stamps), -1, dtype=int)
    for k, (position, stamp) in enumerate(zip(positions, stamps)):
        if position < 0:
            continue
        bar_open = bars.index[position]
        if stamp - bar_open <= pd.Timedelta(hours=2):
            out[k] = position
    return out


# ------------------------------------------------------------------------------------------------------ surprises

def standardize_surprises(surprises, *, min_prior=MIN_PRIOR_RELEASES):
    """z_i = surprise_i / scale_i with scale_i the standard deviation of surprises of releases STRICTLY before i.

    The scale of a release never sees that release or any later one (CONSENSUS_NOT_FROZEN otherwise). The first
    `min_prior` releases get NaN: their scale could not be fitted on enough prior data.
    """
    s = pd.Series(np.asarray(surprises, dtype=float))
    prior_std = s.expanding(min_periods=min_prior).std(ddof=1).shift(1)
    prior_std = prior_std.where(prior_std > 0)
    z = s / prior_std
    return z.to_numpy(), prior_std.to_numpy()


# ------------------------------------------------------------------------------------------------------- matching

def pre_release_vol(bars, positions, pre_window):
    abs_ret = bars["abs_ret"].to_numpy()
    cumulative = np.concatenate([[0.0], np.nancumsum(abs_ret)])
    positions = np.asarray(positions)
    lo = np.clip(positions - pre_window, 0, None)
    return cumulative[positions] - cumulative[lo]


def blocked_mask(n_bars, exclusion_positions, window, pre_window=0):
    """True at every bar c whose response window [c, c + window] contains an excluded release, or that lies inside
    the response window of one: [p - window, p + window] for any excluded release position p. `pre_window` is
    accepted for interface stability and not used: a release inside the matching covariate's window only perturbs
    the covariate, and blocking it too leaves no control bar on a dense calendar."""
    mask = np.zeros(n_bars, dtype=bool)
    for p in np.asarray(exclusion_positions, dtype=int):
        if p < 0:
            continue
        mask[max(0, p - window): min(n_bars, p + window + 1)] = True
    return mask


def match_controls(bars, event_positions, exclusion_positions, *, window, pre_window=24, n_controls=5,
                   max_distance_days=365):
    """For each event bar, the `n_controls` bars of the same hour and weekday, free of any covered release, within
    `max_distance_days`, nearest in log pre-release realized volatility. Returns a list of int arrays."""
    n = len(bars)
    event_positions = np.asarray(event_positions, dtype=int)
    blocked = blocked_mask(n, exclusion_positions, window, pre_window)
    hour = bars["hour"].to_numpy()
    dow = bars["dow"].to_numpy()
    usable = ~blocked
    usable[: pre_window] = False
    usable[n - window:] = False
    all_pre = pre_release_vol(bars, np.arange(n), pre_window)
    log_pre = np.log(np.where(all_pre > 0, all_pre, np.nan))
    seconds = bars.index.asi8 / 1e9
    max_distance = max_distance_days * 86400.0
    controls = []
    for p in event_positions:
        candidates = np.flatnonzero(usable & (hour == hour[p]) & (dow == dow[p])
                                    & (np.abs(seconds - seconds[p]) <= max_distance) & np.isfinite(log_pre))
        candidates = candidates[candidates != p]
        if len(candidates) == 0:
            controls.append(np.array([], dtype=int))
            continue
        distance = np.abs(log_pre[candidates] - log_pre[p])
        order = np.argsort(distance, kind="stable")[: n_controls]
        controls.append(candidates[order])
    return controls


# ------------------------------------------------------------------------------------------- impulse response

def _paths(values, positions, lags):
    out = np.full((len(positions), len(lags)), np.nan)
    n = len(values)
    for row, p in enumerate(positions):
        idx = p + lags
        ok = (idx >= 0) & (idx < n)
        out[row, ok] = values[idx[ok]]
    return out


#: the half-life grid: below `HALF_LIFE_FLOOR` hours the excess is gone inside the release bar itself (hourly bars
#: cannot resolve it), above the window length it is not identified by the window; both ends are reported CENSORED
HALF_LIFE_FLOOR = 0.25


def fit_half_life(lags, excess, *, max_half_life=None):
    """Fit excess_k = A * 2^(-k / H) on k >= 0 by least squares over a grid of H, A closed-form per H.

    Returns (A, H, r2); H is NaN when no positive decaying fit exists (the excess never rises above zero). H at the
    grid floor means the excess dissipates within the release bar; at the cap it is not resolved by the window.
    """
    lags = np.asarray(lags, dtype=float)
    excess = np.asarray(excess, dtype=float)
    keep = (lags >= 0) & np.isfinite(excess)
    k, y = lags[keep], excess[keep]
    if len(k) < 3 or np.nanmax(y) <= 0:
        return float("nan"), float("nan"), float("nan")
    cap = max_half_life if max_half_life is not None else max(2.0, float(k.max()))
    grid = np.exp(np.linspace(np.log(HALF_LIFE_FLOOR), np.log(cap), 400))
    best = (np.inf, np.nan, np.nan)
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    for half in grid:
        basis = np.exp2(-k / half)
        amplitude = float(np.dot(basis, y) / np.dot(basis, basis))
        if amplitude <= 0:
            continue
        sse = float(np.sum((y - amplitude * basis) ** 2))
        if sse < best[0]:
            best = (sse, amplitude, half)
    if not np.isfinite(best[2]):
        return float("nan"), float("nan"), float("nan")
    return best[1], best[2], 1.0 - best[0] / sst


def half_life_censoring(half_life, window):
    if not np.isfinite(half_life):
        return "UNDEFINED"
    if half_life <= HALF_LIFE_FLOOR * 1.01:
        return "LOW_WITHIN_RELEASE_BAR"
    if half_life >= max(2.0, float(window)) * 0.99:
        return "HIGH_NOT_RESOLVED_BY_WINDOW"
    return None


def impulse_response(bars, event_positions, controls, *, window, pre_lags=6, n_boot=400, seed=0,
                     target="abs_ret", leak_ratio=0.5):
    """Excess absolute return per lag after the release, against matched controls, with bootstrap intervals and the
    fitted half-life. Lags run from -pre_lags (placebo, before the release) to +window.

    Refuses `FUTURE_SHIFTED_EVENT` when the pre-release placebo excess is significantly positive AND at least
    `leak_ratio` of the post-release peak: a response that precedes its cause is a shifted clock, not an effect.
    """
    refuse_forbidden_target(target)
    values = bars[target].to_numpy(dtype=float)
    positions = np.asarray(event_positions, dtype=int)
    keep = np.array([len(c) > 0 for c in controls], dtype=bool)
    positions = positions[keep]
    controls = [c for c, k in zip(controls, keep) if k]
    if len(positions) < MIN_EVENTS:
        raise EventImpactError(TOO_FEW_EVENTS, f"{len(positions)} matched event(s); {MIN_EVENTS} are the fewest")
    lags = np.arange(-pre_lags, window + 1)
    event_paths = _paths(values, positions, lags)
    control_paths = np.stack([np.nanmean(_paths(values, c, lags), axis=0) for c in controls])
    diff = event_paths - control_paths

    rng = np.random.default_rng(seed)
    n = len(positions)
    excess = np.nanmean(diff, axis=0)
    event_mean = np.nanmean(event_paths, axis=0)
    control_mean = np.nanmean(control_paths, axis=0)
    amplitude, half_life, r2 = fit_half_life(lags, excess)
    post = lags >= 0
    boot_excess = np.empty((n_boot, len(lags)))
    boot_half = np.empty(n_boot)
    boot_cum = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, n, n)
        path = np.nanmean(diff[draw], axis=0)
        boot_excess[b] = path
        boot_half[b] = fit_half_life(lags, path)[1]
        boot_cum[b] = np.nansum(path[post])
    lo, hi = np.nanpercentile(boot_excess, [2.5, 97.5], axis=0)
    pre = lags < 0
    placebo = float(np.nanmean(excess[pre])) if pre.any() else 0.0
    placebo_boot = np.nanmean(boot_excess[:, pre], axis=1) if pre.any() else np.zeros(n_boot)
    placebo_lo, placebo_hi = (np.nanpercentile(placebo_boot, [2.5, 97.5]) if pre.any() else (0.0, 0.0))
    peak = float(np.nanmax(excess[post]))
    if pre.any() and placebo_lo > 0 and peak > 0 and placebo >= leak_ratio * peak:
        raise EventImpactError(
            FUTURE_SHIFTED_EVENT,
            f"matched excess is already {placebo:.3e} over the {pre_lags} bars BEFORE the release (95% interval "
            f"[{placebo_lo:.3e}, {placebo_hi:.3e}]) against a post-release peak of {peak:.3e}: the response precedes "
            f"the recorded release instant, so the schedule or the clock is shifted")
    finite_half = boot_half[np.isfinite(boot_half)]
    return {
        "schema": SCHEMA, "target": target, "n_events": int(n), "window": int(window), "pre_lags": int(pre_lags),
        "n_controls_per_event": float(np.mean([len(c) for c in controls])),
        "lags": lags.tolist(),
        "event_mean": event_mean.tolist(), "control_mean": control_mean.tolist(),
        "excess": excess.tolist(), "excess_ci_lower": lo.tolist(), "excess_ci_upper": hi.tolist(),
        "ratio": (event_mean / control_mean).tolist(),
        "peak_excess": peak, "peak_lag": int(lags[post][int(np.nanargmax(excess[post]))]),
        "cumulative_excess": float(np.nansum(excess[post])),
        "cumulative_excess_ci": [float(np.nanpercentile(boot_cum, 2.5)), float(np.nanpercentile(boot_cum, 97.5))],
        "half_life_hours": float(half_life), "amplitude": float(amplitude), "half_life_fit_r2": float(r2),
        "half_life_censored": half_life_censoring(half_life, window),
        "half_life_ci": ([float(np.percentile(finite_half, 2.5)), float(np.percentile(finite_half, 97.5))]
                         if len(finite_half) >= 20 else [float("nan"), float("nan")]),
        "half_life_boot_defined_fraction": float(len(finite_half) / n_boot),
        "placebo_pre_release": {"mean_excess": placebo, "ci": [float(placebo_lo), float(placebo_hi)]},
        "identification": "MATCHED_CONTROLS_SAME_HOUR_SAME_WEEKDAY_NEAREST_PRE_VOL",
        "_boot_half": boot_half,
    }


def strip_private(obj):
    """Drop every key starting with `_` (bootstrap draws kept for comparisons) and make the rest JSON-ready."""
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [strip_private(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return strip_private(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# --------------------------------------------------------------------------------------------- surprise and drift

def forward_returns(bars, positions, horizon):
    """log(close[p + horizon] / close[p - 1]): from the last pre-release close to the close `horizon` bars after the
    release bar. Nothing before p - 1 and nothing after p + horizon enters."""
    close = bars["close"].to_numpy(dtype=float)
    positions = np.asarray(positions, dtype=int)
    out = np.full(len(positions), np.nan)
    ok = (positions - 1 >= 0) & (positions + horizon < len(close))
    out[ok] = np.log(close[positions[ok] + horizon] / close[positions[ok] - 1])
    return out


def ols_hc1(y, x):
    """y = a + b x by OLS with HC1 (heteroskedasticity-robust) standard errors and normal p-values."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(y) & np.isfinite(x)
    y, x = y[keep], x[keep]
    n = len(y)
    if n < 5 or np.nanstd(x) == 0:
        return {"n": int(n), "status": "TOO_FEW_ROWS"}
    design = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    meat = design.T @ (design * (resid ** 2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv * (n / (n - 2))
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    p = 2.0 * (1.0 - _normal_cdf(np.abs(tstat)))
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {
        "n": int(n), "status": "OK",
        "intercept": float(beta[0]), "intercept_se": float(se[0]), "intercept_p": float(p[0]),
        "intercept_ci": [float(beta[0] - 1.96 * se[0]), float(beta[0] + 1.96 * se[0])],
        "slope": float(beta[1]), "slope_se": float(se[1]), "slope_p": float(p[1]),
        "slope_ci": [float(beta[1] - 1.96 * se[1]), float(beta[1] + 1.96 * se[1])],
        "r2": float(1.0 - np.sum(resid ** 2) / sst),
    }


def _normal_cdf(value):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(value, dtype=float) / math.sqrt(2.0)))


def surprise_drift(bars, event_positions, z, *, horizons=(1, 2, 4, 8), alpha=0.05):
    """Forward return at each horizon regressed on the standardized surprise. The slope is the surprise-conditional
    trend; the intercept is the unconditional drift the literature says is ~0."""
    positions = np.asarray(event_positions, dtype=int)
    z = np.asarray(z, dtype=float)
    table = {}
    for h in horizons:
        fit = ols_hc1(forward_returns(bars, positions, h), z)
        if fit["status"] == "OK":
            fit["verdict"] = _drift_verdict(fit, alpha)
        table[str(h)] = fit
    fitted = [f for f in table.values() if f["status"] == "OK"]
    # the family verdict reads several horizons at once, so it is Bonferroni-corrected across them
    threshold = alpha / max(len(fitted), 1)
    any_slope = any(f["slope_p"] < threshold for f in fitted)
    any_intercept = any(f["intercept_p"] < threshold for f in fitted)
    return {
        "schema": SCHEMA, "horizons": list(horizons), "alpha": alpha, "family_threshold_bonferroni": threshold,
        "by_horizon": table,
        "verdict": ("SURPRISE_CONDITIONAL_DRIFT" if any_slope and not any_intercept else
                    "UNCONDITIONAL_DRIFT_PRESENT" if any_intercept else
                    "NO_DRIFT_DETECTED" if fitted else "NOT_FITTED"),
        "unit": "log return of EUR/USD from the last pre-release close to the close h bars after the release bar, "
                "per unit of standardized surprise (surprise / standard deviation of the family's prior surprises)",
    }


def _drift_verdict(fit, alpha):
    slope = fit["slope_p"] < alpha
    intercept = fit["intercept_p"] < alpha
    if slope and not intercept:
        return "SURPRISE_CONDITIONAL_DRIFT"
    if intercept:
        return "UNCONDITIONAL_DRIFT_PRESENT"
    return "NO_DRIFT_DETECTED"


# ---------------------------------------------------------------------------------------- heterogeneous effects

#: the fewest rows for which the cross-fitted EconML estimator is preferred over the stratified fallback
ECONML_MIN_ROWS = 60


def heterogeneous_effects(y, z, strata, *, min_n=15, covariates=None, prefer_econml=True):
    """Effect of the surprise on the outcome per stratum (regime, hour, pre-vol tercile).

    With EconML installed and at least `ECONML_MIN_ROWS` rows, a cross-fitted `LinearDML` (5 folds, random-forest
    nuisances, continuous treatment z, the stratum one-hot as effect modifier X and `covariates` as W) gives the
    per-stratum effect and its interval; otherwise a stratified HC1 OLS does. Both return the same mapping and
    `estimator` names which one was used, so a reader can see it per family."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    strata = np.asarray(strata)
    keep = np.isfinite(y) & np.isfinite(z)
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=float)
        keep &= np.isfinite(covariates).all(axis=1)
        covariates = covariates[keep]
    y, z, strata = y[keep], z[keep], strata[keep]
    names = sorted(set(strata.tolist()), key=str)
    result = {"schema": SCHEMA, "strata": {}, "n": int(len(y))}
    engine = _econml_cate(y, z, strata, names, covariates) if (prefer_econml and len(y) >= ECONML_MIN_ROWS) else None
    if engine is not None:
        result.update(engine)
        return result
    result["estimator"] = "stratified_ols_hc1"
    result["estimator_reason"] = ("fewer than %d rows" % ECONML_MIN_ROWS if len(y) < ECONML_MIN_ROWS else
                                  "econml unavailable or failed" if prefer_econml else "econml not requested")
    for name in names:
        mask = strata == name
        if mask.sum() < min_n:
            result["strata"][str(name)] = {"n": int(mask.sum()), "status": "TOO_FEW_ROWS"}
            continue
        result["strata"][str(name)] = ols_hc1(y[mask], z[mask])
    return result


def _econml_cate(y, z, strata, names, covariates):
    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor
        import econml
    except Exception:
        return None
    try:
        x = np.column_stack([(strata == name).astype(float) for name in names])
        # the one-hot is complete, so the intercept column is dropped: effect(stratum k) = const + coef_k
        model = LinearDML(model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=0),
                          model_t=RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=0),
                          discrete_treatment=False, cv=5, random_state=0, fit_cate_intercept=True)
        model.fit(y, z, X=x[:, 1:] if len(names) > 1 else np.zeros((len(y), 1)), W=covariates)
        out = {"estimator": "econml_linear_dml_crossfit", "estimator_version": econml.__version__,
               "nuisance": "RandomForestRegressor(200, min_samples_leaf=10)", "cv_folds": 5, "strata": {}}
        for k, name in enumerate(names):
            point = np.zeros((1, max(len(names) - 1, 1)))
            if k > 0:
                point[0, k - 1] = 1.0
            effect = float(model.effect(point)[0])
            lower, upper = model.effect_interval(point, alpha=0.05)
            out["strata"][str(name)] = {"n": int((strata == name).sum()), "status": "OK", "slope": effect,
                                        "slope_ci": [float(np.ravel(lower)[0]), float(np.ravel(upper)[0])]}
        return out
    except Exception as error:  # a failed fit is reported by the fallback's reason, never silently replaced
        return None


# ------------------------------------------------------------------------------------------- aggregation kernel

def decay(hours_since, half_life_hours):
    hours_since = np.asarray(hours_since, dtype=float)
    return np.where(hours_since >= 0, np.exp2(-hours_since / half_life_hours), 0.0)


def cumulative_impact(times, events, impacts, *, horizon_hours, anticipation_half_life_hours=None):
    """The aggregate causal feature at every t in `times`.

    `events` is an iterable of mappings with `release_utc`, `family` and, for released events, `z` (the standardized
    surprise). `impacts[family]` carries `volatility` (peak excess), `half_life_hours` and `trend_per_z`.

    realized volatility feature at t = sum over events with release <= t of volatility_f * 2^(-(t - release)/H_f)
    realized trend feature at t      = sum over the same events of trend_per_z_f * z * 2^(-(t - release)/H_f)
    anticipation feature at t        = sum over SCHEDULED events with t < release <= t + horizon of
                                       volatility_f * 2^(-(release - t)/H_a); it reads the schedule only, never z.

    An event released after t contributes NOTHING realized; using its z there is a FUTURE_SHIFTED_EVENT refusal,
    enforced by the fact that z is never read for release > t. An event with release <= t and no z is refused
    `FUTURE_SHIFTED_EVENT` too: the caller claims it was released but carries no observed actual.
    """
    times = pd.DatetimeIndex(times)
    seconds = times.asi8 / 3600e9
    vol = np.zeros(len(times))
    trend = np.zeros(len(times))
    anticipation = np.zeros(len(times))
    ant_half = anticipation_half_life_hours if anticipation_half_life_hours else max(horizon_hours / 2.0, 1e-9)
    for event in events:
        family = event["family"]
        if family not in impacts:
            continue
        spec = impacts[family]
        release = pd.Timestamp(event["release_utc"]).value / 3600e9
        since = seconds - release
        released = since >= 0
        kernel = decay(since, spec["half_life_hours"])
        vol += spec["volatility"] * kernel
        if released.any():
            z = event.get("z")
            if z is None or not np.isfinite(float(z)):
                raise EventImpactError(FUTURE_SHIFTED_EVENT,
                                       f"{family} at {event['release_utc']} is at or before some t but carries no "
                                       f"observed surprise; a released event without an actual cannot feed t")
            trend += spec.get("trend_per_z", 0.0) * float(z) * kernel
        ahead = (-since > 0) & (-since <= horizon_hours)
        anticipation[ahead] += spec["volatility"] * np.exp2(-(-since[ahead]) / ant_half)
    return {"volatility": vol, "trend": trend, "anticipation": anticipation}


# ----------------------------------------------------------------------------------------------- clock check

def clock_check(bars, release_times, *, offsets=range(-3, 4)):
    """Where, around the recorded release instant, the absolute return actually spikes: the ratio of the bar at
    each offset to the same bar one week before and after. A best offset other than 0 says the family's recorded
    instant and the bars disagree (CLOCK_MISMATCH); the estimate is then still computed but flagged."""
    abs_ret = bars["abs_ret"]
    out = {}
    for offset in offsets:
        values, controls = [], []
        for stamp in pd.DatetimeIndex(release_times):
            bar = (stamp + pd.Timedelta(hours=offset)).floor("h")
            if bar in abs_ret.index:
                values.append(abs_ret[bar])
                for days in (-7, 7):
                    other = bar + pd.Timedelta(days=days)
                    if other in abs_ret.index:
                        controls.append(abs_ret[other])
        if values and controls:
            out[int(offset)] = float(np.nanmean(values) / np.nanmean(controls))
    if not out:
        return {"status": "NO_BARS"}
    best = max(out, key=out.get)
    return {"status": "OK", "ratio_by_offset_hours": {str(k): v for k, v in out.items()}, "best_offset_hours": best,
            "verdict": "CLOCK_CONSISTENT" if best == 0 else "CLOCK_MISMATCH"}


# --------------------------------------------------------------------------------------------------------- family

def estimate_family(bars, calendar, family, *, window=24, pre_window=24, n_controls=5, horizons=(1, 2, 4, 8),
                    as_of=None, n_boot=400, seed=0, exclusion_families=None, exclusion_calendar=None):
    """The whole estimate for one family: matched impulse response, half-life, surprise-conditional drift, and the
    stratified heterogeneity by pre-release volatility tercile. `as_of` refuses any release after it.

    Control windows must be free of every release in `exclusion_calendar` (default: the calendar given, which is
    meant to be the COVERED high-impact families; passing an entire low-impact calendar blocks every bar)."""
    rows = calendar[calendar["family"] == family].sort_values("release_utc")
    refuse_test_rows(rows["release_utc"], f"{family} releases")
    if as_of is not None:
        assert_released(rows["release_utc"], as_of)
    positions = release_positions(bars, rows["release_utc"])
    exclusion = calendar if exclusion_calendar is None else exclusion_calendar
    if exclusion_families is not None:
        exclusion = exclusion[exclusion["family"].isin(exclusion_families)]
    exclusion_positions = release_positions(bars, exclusion["release_utc"])
    ok = positions >= 0
    rows = rows[ok]
    positions = positions[ok]
    controls = match_controls(bars, positions, exclusion_positions, window=window, pre_window=pre_window,
                              n_controls=n_controls)
    response = impulse_response(bars, positions, controls, window=window, n_boot=n_boot, seed=seed)
    z, scale = standardize_surprises(rows["surprise"].to_numpy())
    drift = surprise_drift(bars, positions, z, horizons=horizons)
    pre_vol = pre_release_vol(bars, positions, pre_window)
    terciles = np.full(len(pre_vol), "mid", dtype=object)
    if len(pre_vol) >= 9:
        lo_q, hi_q = np.nanquantile(pre_vol, [1 / 3, 2 / 3])
        terciles[pre_vol <= lo_q] = "low_pre_vol"
        terciles[pre_vol > hi_q] = "high_pre_vol"
        terciles[(pre_vol > lo_q) & (pre_vol <= hi_q)] = "mid_pre_vol"
    longest = max(horizons)
    state = market_state(bars, positions, short=pre_window)
    cate = heterogeneous_effects(forward_returns(bars, positions, longest), z, terciles,
                                 covariates=state[["vol_long", "trend_strength_short", "trend_sign_short"]].to_numpy())
    return {
        "schema": SCHEMA, "family": family, "n_releases": int(len(rows)), "n_with_bar": int(ok.sum()),
        "clock_check": clock_check(bars, rows["release_utc"]),
        "n_with_surprise": int(np.isfinite(z).sum()),
        "first_release_utc": str(rows["release_utc"].min()), "last_release_utc": str(rows["release_utc"].max()),
        "impulse_response": response, "surprise_drift": drift,
        "heterogeneity_by_pre_vol": {"horizon": longest, **cate},
        "surprise_scale_last": float(scale[-1]) if len(scale) and np.isfinite(scale[-1]) else None,
    }


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# =================================================================================================== patient state

#: the feature set the I6-DMO campaign froze for the RL/price-regression line (predictor, MG2_SELECTED): 19 names
#: under feature manifest sha256 2ff34fa21d5841da1befdd4d898db911128d959a5e460d04cc45b54955416470, recorded in
#: predictor docs/audits/evidence/I6N_STLFSI4_EXCLUSION_20261012/REGENERATION.json and
#: docs/i6n/direct_multi_output_static/best_registry/MG2M_REFERENCE.json. It was selected for PRICE REGRESSION on
#: 2024 development data, not for effect estimation; here it is a DECLARED ARM only.
FS_RL_FEATURE_MANIFEST_SHA256 = "2ff34fa21d5841da1befdd4d898db911128d959a5e460d04cc45b54955416470"
FS_RL_FEATURES = (
    "fred.inflation_expectations.t5yie.level", "fred.rates.dgs3mo.diff_5d", "fred.rates.dgs5.level",
    "fred.rates.dgs7.logret_5d", "fred.rates.dprime.logret_5d", "fx.gbpusd.logret_1h", "px.close_loc", "px.log_co",
    "px.rv5", "rg.atr_ratio", "yh.eem.logret_5d", "yh.iyr.logret_1d", "yh.kc_f.logret_1d", "yh.n225.logret_1d",
    "yh.si_f.logret_1d", "yh.slv.logret_1d", "yh.vnq.logret_1d", "yh.xlc.logret_1d", "yh.xlre.logret_1d",
)
#: the members of that set an OHLC-only fixture can compute causally (backward-looking only)
FS_RL_OHLC_SUBSET = ("px.close_loc", "px.log_co", "px.rv5", "rg.atr_ratio")


def _trailing_sum(values, window):
    cumulative = np.concatenate([[0.0], np.nancumsum(np.nan_to_num(values))])
    idx = np.arange(len(values))
    lo = np.clip(idx - window, 0, None)
    return cumulative[idx] - cumulative[lo]


def market_state(bars, positions, *, short=24, long=120):
    """Backward-looking state at the bar BEFORE each release (nothing at or after the release bar enters):
    volatility at two scales, their ratio, trend strength at two scales, and the OHLC-computable members of the
    frozen RL feature set."""
    positions = np.asarray(positions, dtype=int)
    before = positions  # trailing windows end at position-1 (exclusive of the release bar)
    abs_ret = bars["abs_ret"].to_numpy()
    log_ret = bars["log_ret"].to_numpy()
    vol_s = _trailing_sum(abs_ret, short)[before]
    vol_l = _trailing_sum(abs_ret, long)[before]
    sum_s = _trailing_sum(log_ret, short)[before]
    sum_l = _trailing_sum(log_ret, long)[before]
    state = pd.DataFrame({
        "vol_short": vol_s, "vol_long": vol_l,
        "vol_ratio": np.where(vol_l > 0, vol_s / np.where(vol_l > 0, vol_l, 1.0) * (long / short), np.nan),
        "trend_strength_short": np.where(vol_s > 0, np.abs(sum_s) / np.where(vol_s > 0, vol_s, 1.0), np.nan),
        "trend_strength_long": np.where(vol_l > 0, np.abs(sum_l) / np.where(vol_l > 0, vol_l, 1.0), np.nan),
        "trend_sign_short": np.sign(sum_s),
    })
    prev = np.clip(positions - 1, 0, None)
    state["px.rv5"] = _trailing_sum(log_ret ** 2, 5)[before] ** 0.5
    if "log_range" in bars:
        rng = bars["log_range"].to_numpy()
        atr_s = _trailing_sum(rng, 14)[before] / 14.0
        atr_l = _trailing_sum(rng, 14 * 5)[before] / (14.0 * 5)
        state["rg.atr_ratio"] = np.where(atr_l > 0, atr_s / np.where(atr_l > 0, atr_l, 1.0), np.nan)
    close = bars["close"].to_numpy()
    state["px.log_co"] = log_ret[prev]
    if "log_range" in bars and "high" in bars and "low" in bars:
        high, low = bars["high"].to_numpy(), bars["low"].to_numpy()
        span = high[prev] - low[prev]
        state["px.close_loc"] = np.where(span > 0, (close[prev] - low[prev]) / np.where(span > 0, span, 1.0), 0.5)
    return state


def regime_labels(state, fit_mask, *, columns=("vol_ratio", "trend_strength_short")):
    """An unsupervised regime label from backward-looking state: tercile cells of the chosen columns, with the
    tercile cuts fitted on `fit_mask` rows only (the training period) and applied everywhere."""
    label = np.full(len(state), "", dtype=object)
    for column in columns:
        values = state[column].to_numpy(dtype=float)
        fit_values = values[fit_mask & np.isfinite(values)]
        lo, hi = np.quantile(fit_values, [1 / 3, 2 / 3]) if len(fit_values) >= 9 else (-np.inf, np.inf)
        bucket = np.where(values <= lo, "L", np.where(values > hi, "H", "M"))
        bucket = np.where(np.isfinite(values), bucket, "?")
        label = np.array([f"{a}{column[:3]}{b}" if a else f"{column[:3]}{b}" for a, b in zip(label, bucket)],
                         dtype=object)
    return label


# ============================================================================================ trend decay curve

def trend_impulse_response(bars, positions, z, *, window, pre_lags=6, n_boot=400, seed=0):
    """The signed drift per unit of standardized surprise at every lag: beta_tau from log_ret[p + tau] on z.
    The decay curve is fitted on beta_tau * sign(beta_0) so a negative response (a USD-positive surprise lowers
    EUR/USD) decays like a positive one. Its half-life is SEPARATE from the volatility one."""
    log_ret = bars["log_ret"].to_numpy(dtype=float)
    positions = np.asarray(positions, dtype=int)
    z = np.asarray(z, dtype=float)
    keep = np.isfinite(z)
    positions, z = positions[keep], z[keep]
    if len(z) < MIN_EVENTS:
        raise EventImpactError(TOO_FEW_EVENTS, f"{len(z)} event(s) with a surprise; {MIN_EVENTS} are the fewest")
    lags = np.arange(-pre_lags, window + 1)
    paths = _paths(log_ret, positions, lags)
    rng = np.random.default_rng(seed)

    def slopes(rows):
        zz = z[rows]
        zc = zz - zz.mean()
        denominator = float(np.dot(zc, zc))
        out = np.full(len(lags), np.nan)
        for col in range(len(lags)):
            y = paths[rows, col]
            ok = np.isfinite(y)
            if ok.sum() > 5:
                out[col] = float(np.dot(zc[ok], y[ok] - y[ok].mean()) / max(np.dot(zc[ok], zc[ok]), 1e-300))
        return out

    all_rows = np.arange(len(z))
    beta = slopes(all_rows)
    lag0 = int(np.flatnonzero(lags == 0)[0])
    sign = float(np.sign(beta[lag0])) or 1.0
    amplitude, half_life, r2 = fit_half_life(lags, beta * sign)
    boot = np.empty((n_boot, len(lags)))
    boot_half = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, len(z), len(z))
        path = slopes(draw)
        boot[b] = path
        boot_half[b] = fit_half_life(lags, path * sign)[1]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    finite = boot_half[np.isfinite(boot_half)]
    return {
        "schema": SCHEMA, "target": "log_ret_per_z", "n_events": int(len(z)), "lags": lags.tolist(),
        "beta": beta.tolist(), "beta_ci_lower": lo.tolist(), "beta_ci_upper": hi.tolist(),
        "sign_at_release": sign, "amplitude_per_z": float(amplitude) * sign,
        "half_life_hours": float(half_life), "half_life_fit_r2": float(r2),
        "half_life_censored": half_life_censoring(half_life, window),
        "half_life_ci": ([float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
                         if len(finite) >= 20 else [float("nan"), float("nan")]),
        "half_life_boot_defined_fraction": float(len(finite) / n_boot),
        "_boot_half": boot_half,
    }


def compare_half_lives(volatility, trend):
    """Whether the two decay curves differ: the bootstrap distribution of (trend - volatility) half-life."""
    a = np.asarray(trend.get("_boot_half", []), dtype=float)
    b = np.asarray(volatility.get("_boot_half", []), dtype=float)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    diff = diff[np.isfinite(diff)]
    if len(diff) < 20:
        return {"status": "NOT_COMPARABLE", "why": "fewer than 20 bootstrap draws define both half-lives"}
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"status": "OK", "trend_minus_volatility_hours": float(trend["half_life_hours"] - volatility["half_life_hours"]),
            "ci": [float(lo), float(hi)], "verdict": "DIFFER" if (lo > 0 or hi < 0) else "NOT_DISTINGUISHABLE",
            "n_draws": int(len(diff))}


# ====================================================================================== CATE out of sample

def cate_out_of_sample(y, z, state_label, times, *, decision_start, purge_hours=24 * 7, min_n=15):
    """The X arm is kept only if a per-state slope fitted on the training period predicts the decision period's
    outcomes better than the X-free slope. Chronological, purged: rows inside `purge_hours` before the decision
    start are used by neither side."""
    times = pd.DatetimeIndex(times)
    start = pd.Timestamp(decision_start)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    label = np.asarray(state_label)
    ok = np.isfinite(y) & np.isfinite(z)
    train = ok & (times < start - pd.Timedelta(hours=purge_hours))
    decide = ok & (times >= start)
    if train.sum() < min_n or decide.sum() < min_n:
        return {"status": "TOO_FEW_ROWS", "n_train": int(train.sum()), "n_decision": int(decide.sum())}
    pooled = ols_hc1(y[train], z[train])
    pred_free = pooled["intercept"] + pooled["slope"] * z[decide]
    pred_x = pred_free.copy()
    per_state = {}
    for name in sorted(set(label[train].tolist())):
        rows = train & (label == name)
        if rows.sum() < min_n:
            continue
        fit = ols_hc1(y[rows], z[rows])
        if fit["status"] != "OK":
            continue
        per_state[name] = {"n_train": int(rows.sum()), "slope": fit["slope"], "slope_ci": fit["slope_ci"]}
        target = decide & (label == name)
        pred_x[target[decide]] = fit["intercept"] + fit["slope"] * z[target]
    truth = y[decide]
    mse_free = float(np.mean((truth - pred_free) ** 2))
    mse_x = float(np.mean((truth - pred_x) ** 2))
    mse_zero = float(np.mean(truth ** 2))
    return {"status": "OK", "n_train": int(train.sum()), "n_decision": int(decide.sum()), "purge_hours": purge_hours,
            "decision_start": str(start), "mse_zero": mse_zero, "mse_x_free": mse_free, "mse_x_arm": mse_x,
            "per_state_train": per_state,
            "verdict": "X_ARM_KEPT" if mse_x < mse_free else "X_ARM_DROPPED_ADDS_NOISE",
            "x_free_beats_zero": mse_free < mse_zero}


# ================================================================================ concurrent events: additivity

def concurrency_test(bars, calendar, families, *, window=24, min_n=15):
    """Do two overlapping doses add? Realized volatility over the window after every covered release is regressed
    on one indicator per family present in that window plus an indicator of two-or-more; the latter's interval
    says whether superposition holds."""
    cal = calendar[calendar["family"].isin(families)].copy()
    positions = release_positions(bars, cal["release_utc"])
    cal = cal.assign(position=positions)
    cal = cal[cal["position"] >= 0]
    abs_ret = bars["abs_ret"].to_numpy()
    n = len(bars)
    rows = []
    starts = sorted(set(cal["position"].tolist()))
    for p in starts:
        if p + window >= n:
            continue
        inside = cal[(cal["position"] >= p) & (cal["position"] <= p + window)]
        present = sorted(set(inside["family"].tolist()))
        rows.append((float(np.nansum(abs_ret[p: p + window + 1])), present))
    if len(rows) < min_n:
        return {"status": "TOO_FEW_ROWS", "n": len(rows)}
    fams = list(families)
    design = np.array([[1.0] + [1.0 if f in present else 0.0 for f in fams] + [1.0 if len(present) >= 2 else 0.0]
                       for _, present in rows])
    y = np.array([v for v, _ in rows])
    keep_cols = design.std(axis=0) > 0
    keep_cols[0] = True
    design = design[:, keep_cols]
    names = (["const"] + fams + ["two_or_more"])
    names = [nm for nm, k in zip(names, keep_cols) if k]
    if "two_or_more" not in names:
        return {"status": "NO_OVERLAPS", "n": len(rows)}
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    meat = design.T @ (design * (resid ** 2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv * (len(y) / max(len(y) - design.shape[1], 1))
    se = np.sqrt(np.diag(cov))
    k = names.index("two_or_more")
    lo, hi = beta[k] - 1.96 * se[k], beta[k] + 1.96 * se[k]
    return {"status": "OK", "n": len(rows), "n_overlapping": int(design[:, k].sum()),
            "interaction": float(beta[k]), "interaction_ci": [float(lo), float(hi)],
            "verdict": "ADDITIVE_HOLDS" if lo <= 0 <= hi else "INTERACTIONS_PRESENT",
            "coefficients": {nm: float(b) for nm, b in zip(names, beta)}}


# ========================================================================================== event selection

def benjamini_hochberg(p_values, alpha=0.1):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    passed = np.zeros(n, dtype=bool)
    below = np.flatnonzero(ranked <= alpha)
    if len(below):
        passed[order[: below.max() + 1]] = True
    return passed


def select_events(bars, calendar, families, *, decision_start, purge_hours=24 * 7, horizon=4, window=24,
                  min_events=20, alpha=0.1, n_boot=200, seed=0, exclusion_calendar=None):
    """Rank families by OUT-OF-SAMPLE tradable effect per unit risk on the decision period, with the direction of
    the trade fitted on the training period only; control false discoveries with Benjamini-Hochberg; freeze.

    tradable effect of a family = mean over decision-period releases of sign(slope_train) * sign(z) * forward
    return over `horizon`, divided by its standard deviation (a per-release Sharpe-like ratio), with its t-statistic
    and p-value; volatility impulse = decision-period matched excess at the release bar over the control mean.
    """
    start = pd.Timestamp(decision_start)
    table = []
    exclusion = calendar if exclusion_calendar is None else exclusion_calendar
    all_positions = release_positions(bars, exclusion["release_utc"])
    for family in families:
        rows = calendar[calendar["family"] == family].sort_values("release_utc")
        positions = release_positions(bars, rows["release_utc"])
        ok = positions >= 0
        rows, positions = rows[ok], positions[ok]
        z, _ = standardize_surprises(rows["surprise"].to_numpy())
        y = forward_returns(bars, positions, horizon)
        times = pd.DatetimeIndex(rows["release_utc"])
        train = (times < start - pd.Timedelta(hours=purge_hours)) & np.isfinite(z) & np.isfinite(y)
        decide = (times >= start) & np.isfinite(z) & np.isfinite(y)
        entry = {"family": family, "n_train": int(train.sum()), "n_decision": int(decide.sum())}
        if train.sum() < min_events or decide.sum() < min_events:
            entry["status"] = "TOO_FEW_EVENTS"
            table.append(entry)
            continue
        fit = ols_hc1(y[train], z[train])
        direction = float(np.sign(fit["slope"])) or 1.0
        pnl = direction * np.sign(z[decide]) * y[decide]
        mean, std = float(pnl.mean()), float(pnl.std(ddof=1))
        t = mean / (std / np.sqrt(len(pnl))) if std > 0 else 0.0
        p = float(2.0 * (1.0 - _normal_cdf(abs(t))))
        controls = match_controls(bars, positions[decide], all_positions, window=window)
        try:
            response = impulse_response(bars, positions[decide], controls, window=window, n_boot=n_boot, seed=seed)
            lag0 = response["lags"].index(0)
            vol_impulse = {"excess_over_control": response["ratio"][lag0] - 1.0,
                           "ci_excludes_zero": response["excess_ci_lower"][lag0] > 0,
                           "half_life_hours": response["half_life_hours"]}
        except EventImpactError as error:
            vol_impulse = {"status": error.code}
        entry.update({"status": "OK", "train_slope": fit["slope"], "train_slope_p": fit["slope_p"],
                      "trade_direction": direction, "oos_mean_log_return": mean, "oos_std": std,
                      "oos_effect_per_unit_risk": mean / std if std > 0 else 0.0, "oos_t": float(t), "oos_p": p,
                      "volatility_impulse_oos": vol_impulse})
        table.append(entry)
    fitted = [e for e in table if e["status"] == "OK"]
    if fitted:
        passed = benjamini_hochberg([e["oos_p"] for e in fitted], alpha)
        for e, ok in zip(fitted, passed):
            e["bh_pass"] = bool(ok)
            e["selected"] = bool(ok and e["oos_mean_log_return"] > 0)
    table.sort(key=lambda e: -e.get("oos_effect_per_unit_risk", -np.inf))
    return {"schema": SCHEMA + ".selection", "decision_start": str(start), "purge_hours": purge_hours,
            "horizon": horizon, "min_events": min_events, "fdr_alpha": alpha, "fdr_method": "benjamini_hochberg",
            "ranking": table, "selected": [e["family"] for e in table if e.get("selected")],
            "frozen": True, "rule": "direction from the training period; score, p and FDR on the decision period "
                                    "only; selected = BH pass with positive out-of-sample mean; frozen here, "
                                    "before any confirmation period is read"}
