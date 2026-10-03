"""Bind the train-only causal-ladder feasibility components to ETHUSDT 4h data.

Point-in-time clocks (decisions recorded in C2_LADDER_REPORT):
- ``DATE_TIME`` is the Binance kline OPEN time (export metadata: the last row is
  ``2025-12-31 20:00:00`` for data ending 2025-12-31, so labels are bar opens).
- ``decision_time`` = bar close = ``DATE_TIME + 4h``.
- every manifest feature is a trailing technical/statistical transform of bars
  that closed at or before ``decision_time``; its ``available_at`` lag is 0
  (available at the decision bar close).  Per-feature lags may be overridden.
- the outcome is the forward log return ``log(CLOSE[t+h] / CLOSE[t])``; it is
  realized at ``decision_time + h * 4h`` and a row is kept only when that
  realization time is <= ``train_end`` (no validation/test value enters).
- crypto clock: 24/7 UTC, no session close, no weekend closure.

Read guard: the CSV is read with ``nrows`` bounded by the protected-test start
row, so test rows are never parsed.

Observed interventions: the dataset holds no event, news, policy or calendar
columns, so NO column is an observed intervention.  The only exogenous strata
available are clock-derived (UTC weekend vs weekday), which are observed
calendar conditions, not interventions anyone performed.  Binary feature
columns (EMA crosses, volatility regimes) are endogenous price states and are
reported as endogenous strata, never as interventions.

Nothing here estimates or asserts a causal effect.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .feasibility import FeasibilityError, feasibility_report

BAR = pd.Timedelta("4h")
PROTECTED_TEST_START_ROW = 15895  # ETH test rows [15895, 18085) are never read.
CALENDAR_STRATUM = "calendar_weekend_utc"


class BindingError(ValueError):
    """Input violates the ETH point-in-time binding contract."""


def load_manifest(path: str) -> dict:
    with open(path) as handle:
        manifest = json.load(handle)
    for key in ("date_column", "feature_columns", "splits", "timeframe"):
        if key not in manifest:
            raise BindingError(f"manifest lacks {key}")
    if manifest["timeframe"] != "4h":
        raise BindingError("binding is defined for 4h bars only")
    return manifest


def load_train_rows(csv_path: str, manifest: Mapping[str, Any],
                    max_rows: int = PROTECTED_TEST_START_ROW) -> pd.DataFrame:
    """Read at most ``max_rows`` data rows and keep bars whose open <= train_end."""
    if max_rows > PROTECTED_TEST_START_ROW:
        raise BindingError("read bound would reach the protected test rows")
    frame = pd.read_csv(csv_path, nrows=max_rows)
    date_col = manifest["date_column"]
    frame[date_col] = pd.to_datetime(frame[date_col], utc=True)
    train_end = pd.Timestamp(manifest["splits"]["train_end"], tz="UTC")
    train = frame[frame[date_col] <= train_end].reset_index(drop=True)
    if train.empty:
        raise BindingError("no train rows")
    return train


def bind_clocks(train: pd.DataFrame, manifest: Mapping[str, Any], features: Sequence[str],
                horizon: int, feature_lag_bars: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Return decision_time/available_at/outcome/calendar columns, PIT-filtered."""
    if horizon < 1:
        raise BindingError("horizon must be >= 1 bar")
    date_col = manifest["date_column"]
    missing = [f for f in features if f not in train.columns]
    if missing:
        raise BindingError(f"features absent from data: {missing}")
    lags = {f: int((feature_lag_bars or {}).get(f, 0)) for f in features}
    if any(v < 0 for v in lags.values()):
        raise BindingError("negative availability lag would read the future")
    train_end = pd.Timestamp(manifest["splits"]["train_end"], tz="UTC")
    opens = train[date_col]
    gaps = opens.diff().iloc[1:]
    out = pd.DataFrame({"bar_open": opens})
    out["decision_time"] = opens + BAR
    # A row's availability is the latest availability of any feature it uses.
    out["available_at"] = out["decision_time"] - BAR * min(lags.values()) if lags else out["decision_time"]
    close = train["CLOSE"].astype(float)
    # Forward close by TIME, not by row, so a data gap never shortens the horizon.
    target_open = opens + BAR * horizon
    fwd = pd.Series(close.to_numpy(), index=opens).reindex(target_open.to_numpy())
    out["outcome"] = np.log(fwd.to_numpy() / close.to_numpy())
    out["outcome_realized_at"] = out["decision_time"] + BAR * horizon
    out[CALENDAR_STRATUM] = (opens.dt.dayofweek >= 5).astype(int)
    for f in features:
        out[f] = pd.to_numeric(train[f], errors="coerce").shift(lags[f]) if lags[f] else pd.to_numeric(train[f], errors="coerce")
    keep = (out["decision_time"] <= train_end) & (out["outcome_realized_at"] <= train_end)
    keep &= out["outcome"].notna() & out[list(features)].notna().all(axis=1)
    bound = out[keep].reset_index(drop=True)
    bound.attrs["clock_audit"] = {
        "rows_read_train": int(len(train)),
        "rows_bound": int(len(bound)),
        "dropped_decision_after_train_end": int((out["decision_time"] > train_end).sum()),
        "dropped_outcome_after_train_end": int(((out["decision_time"] <= train_end) & (out["outcome_realized_at"] > train_end)).sum()),
        "dropped_missing_outcome_or_feature": int(((out["outcome_realized_at"] <= train_end) & ~(out["outcome"].notna() & out[list(features)].notna().all(axis=1))).sum()),
        "bar_gaps_over_4h": int((gaps > BAR).sum()),
        "max_gap": str(gaps.max()) if len(gaps) else None,
        "first_decision_time": str(bound["decision_time"].min()) if len(bound) else None,
        "last_decision_time": str(bound["decision_time"].max()) if len(bound) else None,
        "last_outcome_realized_at": str(bound["outcome_realized_at"].max()) if len(bound) else None,
        "train_end": str(train_end),
    }
    return bound


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    if x.nunique() < 2 or y.nunique() < 2:
        return None
    return float(np.corrcoef(x.rank(), y.rank())[0, 1])


def association_section(bound: pd.DataFrame, features: Sequence[str]) -> list[dict]:
    rows = []
    for f in features:
        x, y = bound[f], bound["outcome"]
        pr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
        rows.append({"feature": f, "n": int(len(x)), "pearson_r": pr,
                     "spearman_rho": _spearman(x, y), "label": "association_only"})
    return rows


def run_ladder(bound: pd.DataFrame, features: Sequence[str], train_end: str,
               binary_features: Iterable[str] = (), min_stratum: int = 30,
               n_bins: int = 4, min_cell: int = 5) -> dict:
    """Three separate stages; no causal claim is produced."""
    features = list(features)
    binary = [f for f in binary_features if f in features]
    continuous = [f for f in features if f not in binary]
    frame = bound
    report: dict[str, Any] = {"causal_claim": "not_asserted",
                              "association": association_section(bound, features)}
    # Stage 2: observed historical strata. Interventions counted: none exist in the data.
    strata = []
    for name, kind in [(CALENDAR_STRATUM, "exogenous_clock_stratum_not_intervention")] + \
            [(b, "endogenous_state_stratum_not_intervention") for b in binary]:
        counts = bound[name].value_counts().sort_index()
        strata.append({"stratum": name, "kind": kind, "counts": {str(k): int(v) for k, v in counts.items()},
                       "thin": [str(k) for k, v in counts.items() if v < min_stratum],
                       "counted_as_intervention": False})
    report["observed_interventions"] = {
        "interventions_counted": 0,
        "statement": "No observed intervention column exists in the ETH 4h data (no event, news, "
                     "policy or calendar columns). Strata below are observed conditions only.",
        "strata": strata,
    }
    # Stage 3: counterfactual-support overlap for each stratum and each continuous survivor.
    support = []
    for s in strata:
        name = s["stratum"]
        covs = [c for c in continuous if c != name]
        df = frame.assign(treatment=frame[name])
        per = []
        for c in covs:
            try:
                rep = feasibility_report(df, train_end, [c], None, min_stratum, n_bins, min_cell, top_k=1)
                per.append({"feature": c, **rep["stage3_overlap"], "verdict": rep["verdict"]})
            except FeasibilityError as e:
                per.append({"feature": c, "verdict": "rejected", "reason": str(e)})
        joint = None
        if covs:
            try:
                rep = feasibility_report(df, train_end, covs, None, min_stratum, n_bins, min_cell, top_k=1)
                joint = {**rep["stage3_overlap"], "verdict": rep["verdict"], "n_covariates": len(covs)}
            except FeasibilityError as e:
                joint = {"verdict": "rejected", "reason": str(e)}
        support.append({"stratum": name, "univariate": per, "joint": joint})
    report["counterfactual_support"] = support
    report["notes"] = ["Association, observed strata and overlap are reported separately; none of them "
                       "identifies an effect. Exchangeability, consistency and positivity are not checked.",
                       "Rows are serially dependent 4h bars; n is not an effective sample size and no "
                       "p-values are reported."]
    return report
