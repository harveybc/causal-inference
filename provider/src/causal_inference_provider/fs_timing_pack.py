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


# ------------------------------------------------------------------------- mixed-data CI stage (plan audit F07)
EVIDENCE_LABEL = "CONDITIONAL_PREDICTIVE_ASSOCIATION"
LAG0_MAPPING = ("lag 0 = the candidate's value on the COMPLETED bar t (available at bar end) against the label of the "
                "window (t, t+k]; the label's support starts at bar t+1, so a lag-0 link is time-ordered x -> y")


def _is_discrete(v) -> bool:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return bool(v.size and np.unique(v).size <= 3 and np.all(v == np.round(v)))


def pcmci_mixed_record(x, y, H, *, var_names, tau_max, pc_alpha=0.05, alpha=0.01, missing_flag=999.0) -> dict:
    """PCMCI+ with tigramite RegressionCI on the system [x, y, H...]; the type mask marks every discrete column
    (y = the 3-class T1 label as ONE categorical variable, never two dummies; 0/1 candidates discrete).
    Returns the x -> y links at every lag and a verdict, labelled CONDITIONAL_PREDICTIVE_ASSOCIATION."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.regressionCI import RegressionCI
    from tigramite.pcmci import PCMCI

    data = np.column_stack([np.asarray(x, float), np.asarray(y, float), np.asarray(H, float).reshape(len(x), -1)])
    dtype = np.zeros(data.shape, dtype=int)
    for j in range(data.shape[1]):
        if _is_discrete(data[:, j]):
            dtype[:, j] = 1
    bad = ~np.all(np.isfinite(data), axis=1)
    for j in range(data.shape[1]):          # affine z-score of continuous columns: the regression CI test is
        if dtype[0, j] == 0:                 # invariant to it, the multinomial solver converges better
            v = data[~bad, j]
            sd = v.std()
            data[:, j] = (data[:, j] - v.mean()) / (sd if sd > 0 else 1.0)
    data[bad] = missing_flag
    df = pp.DataFrame(data, var_names=list(var_names), data_type=dtype, missing_flag=missing_flag)
    res = PCMCI(dataframe=df, cond_ind_test=RegressionCI(significance="analytic"), verbosity=0).run_pcmciplus(
        tau_min=0, tau_max=int(tau_max), pc_alpha=pc_alpha)
    g, pm, vm = res["graph"], res["p_matrix"], res["val_matrix"]
    links = [{"lag": int(tau), "edge": str(g[0, 1, tau]), "p": float(pm[0, 1, tau]), "val": float(vm[0, 1, tau])}
             for tau in range(int(tau_max) + 1) if g[0, 1, tau] != ""]
    directed = [lk for lk in links if lk["edge"] == "-->" or (lk["lag"] == 0 and lk["edge"] in ("-->", "o-o", "x-x"))]
    lagged = [lk for lk in links if lk["lag"] >= 1 and lk["edge"] == "-->"]
    verdict = ("LAGGED_PARENT" if lagged else "LAG0_TIME_ORDERED_LINK" if directed else
               "CONFLICTING_OR_REVERSED" if links else "NO_LINK")
    return {"evidence_label": EVIDENCE_LABEL, "ci_test": "RegressionCI (tigramite; linear / multinomial regressions, chi2 deviance)",
            "assumptions": "linear-in-parameters regressions per variable type; causal sufficiency NOT assumed or claimed; "
                           "stationarity over the sample assumed", "lag0_mapping": LAG0_MAPPING,
            "variables": list(var_names), "data_type": [int(dtype[0, j]) for j in range(data.shape[1])],
            "tau_max": int(tau_max), "pc_alpha": pc_alpha, "alpha_reported": alpha, "n_rows": int(len(data)),
            "n_masked": int(bad.sum()), "links_x_to_y": links, "verdict": verdict,
            "significant_at_alpha": bool(any(lk["p"] <= alpha for lk in links))}


def control_series(y, reference, *, seed: int = 0, shift_bars: int = 720) -> dict:
    """Known controls: POSITIVE = noisy copy of the next row's label (a deliberate label leak at lag 1, must be
    detected); NEGATIVE_NOISE = iid Gaussian; NEGATIVE_SHIFT = the reference candidate circularly shifted by
    ``shift_bars`` (keeps its autocorrelation, breaks the alignment)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    nxt = np.r_[y[1:], np.nan]
    pos = (nxt == 1).astype(float) - (nxt == 2).astype(float) + rng.normal(0, 1.0, len(y))
    pos[~np.isfinite(nxt)] = np.nan
    return {"POSITIVE_CONTROL_LABEL_LEAK_LAG1": pos, "NEGATIVE_CONTROL_NOISE": rng.normal(size=len(y)),
            "NEGATIVE_CONTROL_CIRCULAR_SHIFT": np.roll(np.asarray(reference, float), int(shift_bars))}


def mixed_ci_stage(X: pd.DataFrame, y, *, k: int, history: list[str], target_name: str, reference: str,
                   halves: tuple | None = None, pc_alpha=0.05, alpha=0.01, seed: int = 0, workers: int = 1) -> dict:
    """Every candidate through ``pcmci_mixed_record`` (tau_max = k), the three known controls, and a temporal
    resampling (the same tests on two disjoint contiguous halves) for stability.  Mutually exclusive indicator
    candidates (e.g. sessions) are tested one at a time, never jointly, so exclusivity cannot create a
    deterministic relation inside one system."""
    hist = [h for h in history if h in X.columns]
    H = X[hist].to_numpy(float)
    y = np.asarray(y, float)
    cands = [c for c in X.columns if c not in hist]
    ctrl = control_series(y, X[reference].to_numpy(float), seed=seed)
    jobs = [(c, X[c].to_numpy(float)) for c in cands] + list(ctrl.items())

    def run(xv, rows=slice(None)):
        return pcmci_mixed_record(xv[rows], y[rows], H[rows], var_names=["x", target_name, *hist], tau_max=k,
                                  pc_alpha=pc_alpha, alpha=alpha)

    records = {}
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {name: ex.submit(pcmci_mixed_record, xv, y, H, var_names=["x", target_name, *hist], tau_max=k,
                                    pc_alpha=pc_alpha, alpha=alpha) for name, xv in jobs}
            records = {name: f.result() for name, f in futs.items()}
    else:
        records = {name: run(xv) for name, xv in jobs}
    stability = {}
    if halves:
        a, b = halves
        for name, xv in jobs:
            ra, rb = run(xv, a), run(xv, b)
            stability[name] = {"first_half": ra["verdict"], "second_half": rb["verdict"],
                               "agree": ra["verdict"] == rb["verdict"]}
    # a history member that is itself a candidate is tested against H WITHOUT itself (never silently skipped)
    for h in hist:
        rest = [x for x in hist if x != h]
        rec = pcmci_mixed_record(X[h].to_numpy(float), y, X[rest].to_numpy(float) if rest else np.zeros((len(y), 0)),
                                 var_names=["x", target_name, *rest], tau_max=k, pc_alpha=pc_alpha, alpha=alpha)
        rec["note"] = "history member tested against H without itself"
        records[h] = rec
        cands.append(h)
    controls = {name: records[name]["verdict"] for name in ctrl}
    controls_ok = (records["POSITIVE_CONTROL_LABEL_LEAK_LAG1"]["verdict"] in ("LAGGED_PARENT", "LAG0_TIME_ORDERED_LINK")
                   and not records["NEGATIVE_CONTROL_NOISE"]["significant_at_alpha"])
    return {"evidence_label": EVIDENCE_LABEL, "target": target_name, "history": hist, "tau_max": k,
            "records": {c: records[c] for c in cands}, "controls": {n: records[n] for n in ctrl},
            "control_verdicts": controls, "controls_behave": bool(controls_ok), "temporal_resampling": stability,
            "neutrality": "NO_LINK / NOT_IDENTIFIED never rejects a feature; association is not intervention"}
