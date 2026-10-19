"""EURUSD_TIMING target pack: the causal adapter for the ZigZag-expert timing targets.

The feature-selection worker validated only the EURUSD (14 cells) and ETH (6 cells) target packs.  This
module declares the timing pack and the three causal screens the timing selection runs on it, reusing the
existing estimators (``fs_causal.hac_partial_test``, ``fs_causal_discovery.pcmci_feature``,
``event_impact.match_controls``) rather than new ones.

Targets (predictor ``tools/timing_pivot_targets.py``; family ``timing_pivot``; labels only, hindsight):

  timing_valley_k{k}, timing_peak_k{k}   0/1: a valley / peak of the final ZigZag lies in (t, t+k]
  timing_action_confirmed               -1/0/+1 expert action at t from the CONFIRMED timing
  timing_expert_leg_pnl_pips            signed pips from close(t) to the next pivot level (regression)

Heads: a ``classification`` head is screened as a LINEAR-PROBABILITY model of its 0/1 indicator (HAC OLS,
the rung-1 estimator); multi-class labels must be supplied as indicators.  This is a screening comparator
under declared assumptions, never identification; NOT_IDENTIFIED stays neutral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import event_impact as ei
from . import fs_causal as fc
from . import fs_causal_discovery as fd

PACK_ID = "EURUSD_TIMING"
FAMILY = "timing_pivot"
ALLOWED_HEADS = ("classification", "regression")
TEST_BOUNDARY = ei.TEST_BOUNDARY


class TimingPackError(ValueError):
    pass


def timing_target_definitions(k_bars: int) -> list[dict]:
    k = int(k_bars)
    return [
        {"name": f"timing_valley_k{k}", "column": f"timing_valley_k{k}", "family": FAMILY, "head": "classification", "horizon_hours": k},
        {"name": f"timing_peak_k{k}", "column": f"timing_peak_k{k}", "family": FAMILY, "head": "classification", "horizon_hours": k},
        {"name": "timing_action_confirmed", "column": "timing_action_confirmed", "family": FAMILY, "head": "classification", "horizon_hours": k},
        {"name": "timing_expert_leg_pnl_pips", "column": "timing_expert_leg_pnl_pips", "family": FAMILY, "head": "regression", "horizon_hours": k},
    ]


def validate_timing_pack(pack: dict, targets: pd.DataFrame) -> list[dict]:
    """Refuse a pack whose id, heads or label values are not the timing contract."""
    if str(pack.get("id", "")).upper() != PACK_ID:
        raise TimingPackError(f"target_pack.id must be {PACK_ID}")
    defs = pack.get("definitions") or []
    if not defs:
        raise TimingPackError("timing pack needs definitions")
    for d in defs:
        if d.get("family") != FAMILY or d.get("head") not in ALLOWED_HEADS:
            raise TimingPackError(f"{d.get('name')}: family must be {FAMILY} and head one of {ALLOWED_HEADS}")
        if d["column"] not in targets.columns:
            raise TimingPackError(f"{d['name']}: column {d['column']} absent")
        v = targets[d["column"]].to_numpy(float)
        v = v[np.isfinite(v)]
        if d["head"] == "classification" and not np.all(np.isin(v, (-1.0, 0.0, 1.0))):
            raise TimingPackError(f"{d['name']}: a classification head must be a -1/0/1 indicator (multi-class labels go as indicators)")
    return defs


def _bh(p, q):
    p = np.asarray(p, float)
    out = np.zeros(len(p), bool)
    idx = np.flatnonzero(np.isfinite(p))
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(idx) / (np.arange(len(idx)) + 1)
    below = np.flatnonzero(ranked <= q)
    if len(below):
        out[order[: below.max() + 1]] = True
    return out


def hac_lag_screen(X: pd.DataFrame, y, *, k: int, history: list[str], q: float = 0.1, min_rows: int = 200) -> dict:
    """For every candidate a and lag 0..k: HAC partial t-test of a_{t-lag} in y ~ 1 + H_t (+ a_t when lag > 0).
    BH at ``q`` over all (feature, lag) tests; ``parents_at_lags_1_to_k`` lists features significant at a lag >= 1."""
    y = np.asarray(y, float)
    hist = [h for h in history if h in X.columns]
    H = X[hist].to_numpy(float) if hist else np.zeros((len(X), 0))
    bw = max(int(k), 24)
    rows = []
    for f in X.columns:
        if f in hist:
            continue
        a = X[f].to_numpy(float)
        for lag in range(0, int(k) + 1):
            ak = np.r_[np.full(lag, np.nan), a[:-lag]] if lag else a
            m = np.isfinite(ak) & np.isfinite(y) & np.isfinite(a) & np.all(np.isfinite(H), axis=1)
            if m.sum() < min_rows or np.std(ak[m]) == 0 or np.std(y[m]) == 0:
                rows.append({"feature": f, "lag": lag, "state": "TOO_FEW_ROWS_OR_ZERO_VARIANCE"})
                continue
            Hk = np.column_stack([H[m], a[m]]) if lag else H[m]
            coef, t, p, n = fc.hac_partial_test(ak[m], y[m], Hk, bw)
            rows.append({"feature": f, "lag": lag, "coef": coef, "t": t, "p": p, "n": n, "state": "TESTED"})
    tested = [r for r in rows if r["state"] == "TESTED"]
    for r, ok in zip(tested, _bh([r["p"] for r in tested], q)):
        r["pass_fdr"] = bool(ok)
    parents = sorted({r["feature"] for r in tested if r.get("pass_fdr") and r["lag"] >= 1})
    return {"estimand": "conditional dependence of the future timing label on the lagged candidate given H_t and a_t "
                        "(association screen; not identification)", "ci_test": f"HAC(Bartlett, bandwidth={bw}) partial t-test",
            "model": "linear-probability regression of the 0/1 indicator", "history": hist, "lags": list(range(0, int(k) + 1)),
            "q": q, "rows": rows, "parents_at_lags_1_to_k": parents}


def pcmci_parents(X: pd.DataFrame, y, *, k: int, history: list[str], target_name: str, pc_alpha=0.05, alpha=0.01,
                  max_features: int | None = None) -> dict:
    """PCMCI+ per candidate through the existing comparator stage; NOT_RUN (recorded) without tigramite."""
    pin = fd.tigramite_pin()
    if "error" in pin:
        return {"state": "NOT_RUN", "reason": pin["error"], "pin": pin}
    hist = [h for h in history if h in X.columns]
    H = X[hist].to_numpy(float) if hist else np.zeros((len(X), 0))
    recs = {}
    for f in [c for c in X.columns if c not in hist][:max_features]:
        recs[f] = fd.pcmci_feature(X[f].to_numpy(float), np.asarray(y, float), H, tau_max=int(k), pc_alpha=pc_alpha, alpha=alpha,
                                   var_names=[f, target_name, *hist])
    return {"state": "RUN", "pin": pin, "records": recs}


def matched_control_contrast(bars: pd.DataFrame, feature, pivot_indicator, *, k: int, event_quantile: float = 0.8,
                             n_controls: int = 5, min_events: int = 20, n_boot: int = 400, seed: int = 0) -> dict:
    """Upward crossings of the feature's q-quantile as events; controls = same hour and weekday, free of another
    crossing within k bars, nearest in 24-bar pre-event realized vol (``event_impact.match_controls``); outcome =
    the pivot-within-k indicator.  Excess pivot rate with a bootstrap 95% interval over events."""
    a = np.asarray(feature, float)
    yv = np.asarray(pivot_indicator, float)
    fin = np.isfinite(a)
    if fin.sum() < 500 or np.nanstd(a) == 0:
        return {"state": "TOO_FEW_ROWS_OR_ZERO_VARIANCE"}
    thr = float(np.nanquantile(a, event_quantile))
    above = fin & (a > thr)
    events = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    events = events[(events >= 24) & (events < len(a) - int(k))]
    if len(events) < min_events:
        return {"state": "TOO_FEW_EVENTS", "events": int(len(events))}
    controls = ei.match_controls(bars, events, events, window=int(k), pre_window=24, n_controls=n_controls)
    ev_y = yv[events]
    ct_y = np.array([np.nanmean(yv[c]) if len(c) else np.nan for c in controls])
    ok = np.isfinite(ev_y) & np.isfinite(ct_y)
    if ok.sum() < min_events:
        return {"state": "TOO_FEW_MATCHED_EVENTS", "events": int(ok.sum())}
    diff = ev_y[ok] - ct_y[ok]
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"state": "ESTIMATED", "events": int(ok.sum()), "threshold": thr, "event_pivot_rate": float(ev_y[ok].mean()),
            "control_pivot_rate": float(ct_y[ok].mean()), "excess_pivot_rate": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0),
            "design": "association under matching on hour, weekday and pre-event realized vol; crossings of other features are "
                      "not excluded; no placebo; screening only"}


def bars_table(timestamps, close, high, low) -> pd.DataFrame:
    """The event_impact bar table from lake arrays (refuses TEST rows)."""
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return ei.bars_from_prices(idx, close, high, low)
