"""FS-CAUSAL: auditable causal selector over the lane-A EURUSD candidates (closure order 2026-10-05 section 5).

One cell = (feature, target/horizon). Each cell carries THREE rungs, each ending in exactly one of
``SUPPORTED`` / ``CONTRADICTED`` / ``NOT_IDENTIFIED`` plus the estimand, the declared assumptions with
their evidence references, the conditioning set, the max lag, the CI test, the multiplicity family,
support/overlap/balance, placebos, sensitivity and the abstention reason.

Rung 1 (association, never causal words): lagged conditional dependence of A on Y given the pre-decision
history H with a HAC (Bartlett/Newey-West, bandwidth >= horizon) partial-regression t-test -- an analytic
p that survives Benjamini-Hochberg over a family of hundreds of candidates, unlike the 1/(B+1) floor of a
200-permutation null (subplan 5.2 point 4). The circular-shift permutation p is kept as a calibration
check. Out-of-fold gain on the contract's purged inner folds, signed-direction stability, a lag scan
(max lag declared), a minimal separating set search and a clock-shift robustness check are recorded.

Rung 2 (historical intervention): treated episodes A=1 (first available crossing of the TRAIN q80
threshold) against controls A=0 (stayed below, previous value in [q60, q80)), with episode spacing
>= max(24h, horizon) so outcome windows are DISJOINT (that is what evidences NO_INTERFERENCE), W from
row t-1 (market state, the feature's own level and trend, TRAIN volatility regime, calendar density
used only as a locator), the repaired fail-closed ``ps3c.rung2_effect`` gate (declared DAG, back-door
check, support, overlap without trimming, balance <= 0.1, placebo battery, evidenced assumptions),
AIPW/g-computation/matching, BH over the declared family, and a TRAIN-only cross-fitted nonlinear AIPW
confirmation. CATE by volatility regime only inside support.

Rung 3 (same-episode counterfactual): ``ps3c.rung3_population`` -- linear additive-noise temporal SCM
fitted in TRAIN, abduction of the episode's own noise, action A := 0 inside historical support,
propagation of the first-hour mediator and Y, factual reconstruction, historical analog pairs, placebos
and model sensitivity. The individual counterfactual is never claimed observed.

State semantics (declared before any data was read, tested on planted worlds):
* SUPPORTED      -- the rung's evidence passed every declared gate at the declared FDR.
* CONTRADICTED   -- the rung's evidence robustly refutes the feature's claimed relevance (rung 1: a
                    BH-significant dependence whose sign is unstable and whose OOF gain is negative in
                    every fold; rung 2: an identified effect whose sign is opposite to the stable rung-1
                    association, or a precise null inside the declared equivalence margin, both confirmed
                    by the nonlinear estimator; rung 3: a counterfactual whose sign robustly opposes the
                    identified rung-2 effect). Only ``robust`` CONTRADICTED may weigh against a feature.
* NOT_IDENTIFIED -- everything else, with a named abstention reason. Never a rejection.

Calendar columns are episode locators only (I11 deferred); nothing here feeds a predictor.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import ps3c
from . import ps3c_stats as st

SUPPORTED, CONTRADICTED, NOT_IDENTIFIED = "SUPPORTED", "CONTRADICTED", "NOT_IDENTIFIED"
STATES = (SUPPORTED, CONTRADICTED, NOT_IDENTIFIED)
SEED = 1729
FDR_Q = 0.05
LAGS = (1, 2, 3, 6, 12, 24)
MAX_LAG = max(LAGS)
EQUIVALENCE_MARGIN_SD = 0.1          # smallest effect of interest for a PRECISE_NULL: 0.1 SD of Y in TRAIN
RV_FLOOR = 0.01                      # robustness value below which an identified effect is not called robust
H_BASE = ["px.logret_24h", "px.logret_120h", "px.ewma_vol_24", "px.ewma_vol_168",
          "cal.hour_sin", "cal.hour_cos", "cal.dow_sin", "cal.dow_cos"]
PRE_RETURNS = ["px.logret_1h", "px.logret_6h"]
CALENDAR_LOCATORS = ["ev.USD.high.count_24h", "ev.EUR.high.count_24h", "ev.USD.high.hours_since", "ev.EUR.high.hours_since"]
TARGETS = ([(f"Y_s_{h}h", "Y_s", "short", h) for h in (1, 2, 3, 4, 5, 6)]
           + [(f"Y_l_{h}h", "Y_l", "long", h) for h in (24, 48, 72, 96, 120, 144)]
           + [("Y_b_s6", "Y_b", "barrier", 6), ("Y_b_l144", "Y_b", "barrier", 144)])
HORIZON_OF = {t: h for t, _, _, h in TARGETS}
DAG = {"nodes": ["W", "A", "M", "Y"],
       "edges": [["W", "A"], ["W", "Y"], ["W", "M"], ["A", "M"], ["M", "Y"], ["A", "Y"]]}
ASSUMPTIONS = {name: True for name in ps3c.REQUIRED_ASSUMPTIONS}
ASSUMPTION_STRENGTH = {
    "CONSISTENCY": "EVIDENCED_BY_CONSTRUCTION",
    "NO_INTERFERENCE_BETWEEN_EPISODES": "EVIDENCED_BY_CONSTRUCTION (disjoint outcome windows checked per episode set)",
    "CAUSAL_SUFFICIENCY_OF_DECLARED_DAG": "DECLARED_WITH_SENSITIVITY_ONLY (not verifiable from data; RV and placebos recorded)",
    "TEMPORAL_ORDER_W_BEFORE_A_BEFORE_Y": "EVIDENCED_BY_CONSTRUCTION (observed bar-end clock only)",
}


def assumption_evidence(fid, horizon_h, windows_disjoint, clock):
    """Traceability references for the four required assumptions (a reference is not a proof)."""
    return {
        "CONSISTENCY": (f"A is a deterministic function of the observed path of {fid} at rows t-1,t "
                        "(fs_causal.crossing_episodes_h: cross of the TRAIN q80 threshold from the pre-row band [q60,q80)); one version "
                        "of treatment; test_fs_causal::test_treatment_is_deterministic_function_of_observed_path"),
        "NO_INTERFERENCE_BETWEEN_EPISODES": (f"episode decision rows spaced >= max(24h, {horizon_h}h) so every "
                                             f"outcome window is disjoint; checked: windows_disjoint={windows_disjoint}; "
                                             "test_fs_causal::test_episode_outcome_windows_are_disjoint"),
        "CAUSAL_SUFFICIENCY_OF_DECLARED_DAG": ("DECLARED, NOT VERIFIABLE FROM DATA: W = pre-row market state (H_BASE), "
                                               f"{fid} level and 24h trend at t-1, TRAIN volatility-regime dummies, "
                                               "calendar density locators; back-door check ps3c_graph.backdoor_check on "
                                               "the declared DAG; sensitivity = Cinelli-Hazlett robustness value + placebo "
                                               "battery recorded in this cell"),
        "TEMPORAL_ORDER_W_BEFORE_A_BEFORE_Y": (f"W from row t-1 (bar closed before t), A from rows t-1->t, Y realised after "
                                               f"t (lane-A target at row t); clock={clock}; "
                                               "test_fs_causal::test_temporal_order_enforced"),
    }


# ------------------------------------------------------------------------------------------- HAC statistics


def hac_ols(x, y, bandwidth):
    """OLS with Bartlett-kernel (Newey-West) HAC covariance; ``x`` includes the constant.

    Returns dict(beta, se, t, p) for every coefficient. ``bandwidth`` is the number of lags whose
    autocovariance is kept (>= horizon for overlapping h-step outcomes).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    e = y - x @ beta
    g = x * e[:, None]
    s = g.T @ g
    L = int(min(max(bandwidth, 0), n - 1))
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        c = g[lag:].T @ g[:-lag]
        s += w * (c + c.T)
    cov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    t = np.where(se > 0, beta / np.where(se > 0, se, 1.0), 0.0)
    p = np.array([st.normal_two_sided_p(v) for v in t])
    return {"beta": beta, "se": se, "t": t, "p": p, "resid": e, "n": n, "bandwidth": L}


def hac_partial_test(a, y, h, bandwidth):
    """HAC t-test of A in Y ~ 1 + H + A. Returns (coef, t, p, n)."""
    x = st.add_const(np.column_stack([h, a]) if h.shape[1] else a[:, None])
    f = hac_ols(x, y, bandwidth)
    return float(f["beta"][-1]), float(f["t"][-1]), float(f["p"][-1]), int(f["n"])


def _shift_rows(v, k):
    """Row lag: value k decision rows earlier (NaN where unavailable). Rows are the lane-A decision grid."""
    out = np.full(len(v), np.nan)
    if k == 0:
        return np.array(v, dtype=float)
    if k < len(v):
        out[k:] = v[:-k]
    return out


# ------------------------------------------------------------------------------------------------- rung 1


def rung1_cell(a, y, H, *, horizon_h, folds, hist_names, pool=None, pool_names=(), permutations=200, seed=SEED,
               min_rows=200):
    """Rung-1 evidence for one (feature, target): HAC partial test, lag scan, OOF gain, placebos, MSS, clock shift.

    ``a``, ``y`` 1-D arrays on the time-ordered decision grid; ``H`` (n, k) history matrix; ``pool`` extra
    conditioning columns for the minimal-separating-set search (pre-decision returns). Returns the raw
    evidence block; the state is assigned after BH at the family level (``assign_rung1_state``).
    """
    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(a)
    bw = max(int(horizon_h), 24)
    base_mask = np.isfinite(a) & np.isfinite(y) & np.all(np.isfinite(H), axis=1)
    out = {"estimand": "conditional dependence of Y on A given the pre-decision history H (association, not causal)",
           "ci_test": f"HAC(Bartlett, bandwidth={bw}) partial-regression t-test of A in Y~1+H+A; two-sided normal p",
           "conditioning_set": list(hist_names), "max_lag_rows": MAX_LAG, "lags_tested": list(LAGS),
           "bandwidth": bw, "effective_rows": int(base_mask.sum()), "multiplicity": {"family": "rung1:(target)",
           "correction": "benjamini_hochberg", "q_level": FDR_Q}}
    if base_mask.sum() < min_rows:
        out.update(raw_state="TOO_FEW_ROWS", p=None)
        return out
    if np.std(a[base_mask]) == 0 or np.std(y[base_mask]) == 0:
        out.update(raw_state="ZERO_VARIANCE", p=None)
        return out
    am, ym, Hm = a[base_mask], y[base_mask], H[base_mask]
    coef, t, p, m = hac_partial_test(am, ym, Hm, bw)
    out.update(coef=coef, t_hac=t, p=p, n=m, sign=int(np.sign(coef)))
    # calibration check: circular-shift permutation p and OOF gain through the ps3c rung-1 (explicit folds)
    cols = {"A": a, "Y": y, **{nm: H[:, j] for j, nm in enumerate(hist_names)}}
    r1 = ps3c.rung1_association(cols, treatment="A", outcome="Y", history=list(hist_names), explicit_folds=folds,
                                permutations=permutations, min_events=min_rows, seed=seed)
    ev = {e["measure"]: e for e in r1.get("evidence", [])}
    pc = ev.get("partial_corr_given_H")
    oof = ev.get("oof_relative_mse_gain_of_A_over_H")
    d = r1.get("diagnostics") or {}
    out.update(partial_corr=pc["value"] if pc else None, p_perm_circular_shift=pc["p"] if pc else None,
               oof_gain=oof["value"] if oof else None, oof_positive_folds=d.get("oof_positive_folds"),
               oof_folds=d.get("oof_folds"), oof_signs=d.get("oof_signs"),
               sign_stable=(pc or {}).get("signed_direction_stable"))
    # lag scan: A at k rows earlier, given H and A_t
    lag_scan = []
    for k in LAGS:
        ak = _shift_rows(a, k)
        mk = base_mask & np.isfinite(ak)
        if mk.sum() < min_rows:
            continue
        Hk = np.column_stack([H[mk], a[mk]])
        c_k, t_k, p_k, _ = hac_partial_test(ak[mk], y[mk], Hk, bw)
        lag_scan.append({"lag_rows": k, "coef": c_k, "t": t_k, "p": p_k})
    out["lag_scan_given_H_and_A_t"] = lag_scan
    best = max(lag_scan, key=lambda r: abs(r["t"]), default=None)
    out["best_extra_lag"] = None if best is None else best["lag_rows"]
    out["best_extra_lag_p"] = None if best is None else best["p"]
    # clock-shift robustness: A delayed one more row (24h for daily series = 24 rows) must keep the dependence
    shift = 24
    ak = _shift_rows(a, shift)
    mk = base_mask & np.isfinite(ak)
    if mk.sum() >= min_rows:
        c_s, t_s, p_s, _ = hac_partial_test(ak[mk], y[mk], H[mk], bw)
        out["clock_shift_24rows"] = {"coef": c_s, "t": t_s, "p": p_s, "same_sign": bool(np.sign(c_s) == np.sign(coef))}
    # minimal separating set within H (+ pool): smallest S with p > 0.10 (sizes 0..2)
    cand_cols = list(hist_names) + list(pool_names)
    cand = np.column_stack([H] + ([pool] if pool is not None and len(pool_names) else []))
    mss = None
    tested = 0
    for size in range(0, 3):
        for S in itertools.combinations(range(cand.shape[1]), size):
            mk = base_mask & (np.all(np.isfinite(cand[:, list(S)]), axis=1) if S else True)
            if mk.sum() < min_rows:
                continue
            _, _, p_s, _ = hac_partial_test(a[mk], y[mk], cand[mk][:, list(S)], bw)
            tested += 1
            if p_s > 0.10:
                mss = [cand_cols[j] for j in S]
                break
        if mss is not None:
            break
    out["minimal_separating_set"] = {"set": mss, "pool": cand_cols, "max_size": 2, "alpha_independence": 0.10,
                                     "tests": tested, "state": "SEPARATED" if mss is not None else "NO_SEPARATING_SET_FOUND"}
    out["raw_state"] = "ASSOCIATION_REPORTED"
    return out


def assign_rung1_state(cell, q):
    """Final rung-1 state after the family BH q is known."""
    r = cell
    r["q"] = q
    if r.get("raw_state") != "ASSOCIATION_REPORTED":
        r["state"] = NOT_IDENTIFIED
        r["abstention_reason"] = r["raw_state"]
        return r
    sig = q is not None and q <= FDR_Q
    oof_pos = (r.get("oof_positive_folds") or 0)
    oof_n = (r.get("oof_folds") or 0)
    gain = r.get("oof_gain")
    stable = r.get("sign_stable")
    if sig and gain is not None and gain > 0 and oof_n and oof_pos * 2 > oof_n and stable:
        r["state"] = SUPPORTED
        r["abstention_reason"] = None
        r["robust"] = bool((r.get("clock_shift_24rows") or {}).get("same_sign") and
                           (r.get("p_perm_circular_shift") or 1.0) <= 0.05)
    elif sig and oof_n and oof_pos == 0 and stable is False:
        r["state"] = CONTRADICTED
        r["contradiction_kind"] = "SIGNIFICANT_BUT_SIGN_UNSTABLE_AND_OOF_NEGATIVE_ALL_FOLDS"
        r["abstention_reason"] = None
        r["robust"] = bool(oof_n >= 3)
    else:
        r["state"] = NOT_IDENTIFIED
        why = []
        if not sig:
            why.append(f"NO_CONDITIONAL_DEPENDENCE_AT_FDR_{FDR_Q}")
        if gain is None or gain <= 0 or not (oof_n and oof_pos * 2 > oof_n):
            why.append("NO_OOF_GAIN_MAJORITY")
        if not stable:
            why.append("SIGN_NOT_STABLE_ACROSS_FOLDS")
        r["abstention_reason"] = ";".join(why) or "GATE_NOT_MET"
    return r


# ---------------------------------------------------------------------------------------- discovery comparators


def sypi_conditions(a, y, S, *, horizon_h, max_w=6, alpha_indep=0.05, min_rows=200):
    """Adapted SyPI (Mastakouri, Schoelkopf, Janzing 2021) two-condition screen for one candidate, linear CI tests.

    Scope (declared): the conditioning set S is the pre-decision history H plus the previous realised
    target value, NOT the full set of other candidates at their lags; CI test = HAC partial regression.
    Latent confounders that affect only Y are tolerated by SyPI's theorem; the official code is not
    public, so this is a faithful adaptation of the two conditions, used for screening only.
    Condition 1: X_{t-w} not independent of Y_t given S.  Condition 2: X_{t-w-1} independent of Y_t given S + X_{t-w}.
    """
    bw = max(int(horizon_h), 24)
    base = np.isfinite(y) & np.all(np.isfinite(S), axis=1)
    best = None
    for w in range(0, max_w + 1):
        aw = _shift_rows(a, w)
        mk = base & np.isfinite(aw)
        if mk.sum() < min_rows:
            continue
        c, t, p, _ = hac_partial_test(aw[mk], y[mk], S[mk], bw)
        if best is None or abs(t) > abs(best["t"]):
            best = {"w": w, "coef": c, "t": t, "p": p}
    if best is None:
        return {"state": "NOT_RUN", "reason": "TOO_FEW_ROWS"}
    w = best["w"]
    aw, aw1 = _shift_rows(a, w), _shift_rows(a, w + 1)
    mk = base & np.isfinite(aw) & np.isfinite(aw1)
    c2, t2, p2, _ = hac_partial_test(aw1[mk], y[mk], np.column_stack([S[mk], aw[mk]]), bw)
    return {"state": "RUN", "w": w, "condition1_p": best["p"], "condition1_coef": best["coef"],
            "condition2_p": p2, "condition2_coef": c2, "condition2_independent": bool(p2 > alpha_indep),
            "ci_test": f"HAC(Bartlett, bandwidth={bw}) partial regression", "max_w": max_w,
            "scope": "single-candidate conditioning on H + previous realised target; not the full SyPI set"}


def assign_sypi_state(d, q1):
    if d.get("state") != "RUN":
        d["verdict"] = "NOT_RUN"
        return d
    d["condition1_q"] = q1
    if q1 is not None and q1 <= FDR_Q and d["condition2_independent"]:
        d["verdict"] = "SYPI_CANDIDATE_CAUSE_UNDER_DECLARED_RESTRICTIONS"
    elif q1 is not None and q1 <= FDR_Q:
        d["verdict"] = "SYPI_CONDITION2_FAILED"
    else:
        d["verdict"] = "SYPI_CONDITION1_FAILED"
    return d


# ------------------------------------------------------------------------------------------------- rung 2


def vol_regime_dummies(vol_prev, train_vol):
    """TRAIN tercile regime of ewma_vol_168 at t-1 -> two dummies (mid, high); low is the reference."""
    q1, q2 = np.nanquantile(train_vol, [1 / 3, 2 / 3])
    code = np.where(vol_prev > q2, 2, np.where(vol_prev > q1, 1, 0)).astype(float)
    return code, (code == 1).astype(float), (code == 2).astype(float), [float(q1), float(q2)]


def crossing_episodes_h(X, Y, fid, horizon_h, q=0.8, band_q=0.6, threshold=None, band=None,
                        locators=CALENDAR_LOCATORS, targets=None, history_columns=None,
                        pre_return_columns=None, mediator_target="Y_s_1h",
                        volatility_regime_column="px.ewma_vol_168",
                        placebo_outcome_column="px.logret_24h"):
    """Horizon-aware crossing episodes: decision rows spaced >= max(24, horizon_h) hours so outcome windows are disjoint.

    Common support BY CONSTRUCTION: both arms start from the same pre-row band [band, q80) with
    band = max(q60, q80 - 2*SD_TRAIN(one-step change)), the states from which a one-step crossing is plausible.
    A=1: first available crossing of the TRAIN q80 threshold (row t-1 inside the band, row t at/above).
    A=0: rows that stayed below with the previous value inside the band. W from row t-1.
    A crossing from far below the band has no control counterpart and is not an episode (the repaired gate
    forbids trimming, so support is declared in the design, not recovered by dropping rows afterwards).
    Episodes of both arms are chosen sequentially in time with past-only spacing rules (never on the future path).
    """
    gap_h = max(24, int(horizon_h))
    x = X[fid].to_numpy(float)
    ok = np.isfinite(x)
    if ok.sum() < 200 or np.nanstd(x) == 0:
        return None, {"reason": "TOO_FEW_FINITE_OR_CONSTANT"}
    thr = float(np.nanquantile(x, q)) if threshold is None else float(threshold)
    tn = pd.DatetimeIndex(pd.to_datetime(X["t_decision_utc"], utc=True)).as_unit("ns").asi8
    H_NS = 3600 * 10**9
    # Pre-row band shared by both arms: at least q60, but no further below the threshold than two TRAIN standard
    # deviations of the feature's one-step change -- the states from which a crossing within one step is plausible.
    # For a persistent feature the q60 band is far wider than one step can bridge; its lower part holds controls
    # that no treated episode can resemble (pilot: weighted SMD of the pre-row level ~1.0), so the band narrows.
    step = np.diff(x)
    step = step[np.isfinite(step) & ((tn[1:] - tn[:-1]) <= 72 * H_NS)]
    sd_step = float(np.std(step)) if len(step) else 0.0
    band_q_value = float(np.nanquantile(x, band_q))
    band = max(band_q_value, thr - 2.0 * sd_step) if band is None else float(band)
    if not band < thr:
        return None, {"reason": "THRESHOLD_BAND_DEGENERATE", "threshold": thr, "band": band}
    prev, now = x[:-1], x[1:]
    with np.errstate(invalid="ignore"):
        cross = np.where((prev < thr) & (now >= thr) & (prev >= band))[0] + 1
        ctrl = np.where((prev < thr) & (now < thr) & (prev >= band))[0] + 1
    gap = (tn[1:] - tn[:-1]) / H_NS
    # Sequential, outcome-blind, PAST-ONLY selection over the candidate rows of both arms in time order: a row is
    # accepted when it lies >= gap_h after the last accepted episode (disjoint outcome windows). A repeat of the
    # same arm must wait >= 2*gap_h, which gives the opposite arm a window of exclusivity so the frequent arm does
    # not starve the other at long horizons. Nothing here looks at rows after the candidate (no selection on the
    # future path of the feature or of the price).
    cand = sorted([(int(i), 1) for i in cross if gap[i - 1] <= 72] + [(int(i), 0) for i in ctrl if gap[i - 1] <= 72])
    treated, controls = [], []
    last_t, last_arm = None, None
    for i, arm in cand:
        if last_t is not None:
            wait = (tn[i] - last_t) / H_NS
            if wait < gap_h or (arm == last_arm and wait < 2 * gap_h):
                continue
        (treated if arm == 1 else controls).append(i)
        last_t, last_arm = tn[i], arm
    rows = np.array(sorted(treated + controls), dtype=int)
    if len(rows) == 0:
        return None, {"reason": "NO_EPISODES", "threshold": thr}
    pre = rows - 1
    ep = pd.DataFrame({"episode_id": [f"{fid}|h{horizon_h}|{pd.Timestamp(tn[i], tz='UTC').isoformat()}" for i in rows],
                       "decision_time": pd.to_datetime(X["t_decision_utc"], utc=True).to_numpy()[rows],
                       "A": np.isin(rows, treated).astype(float),
                       "W_x_prev": x[pre],
                       "W_x_gap_to_thr_sq": (thr - x[pre]) ** 2,  # crossing propensity is nonlinear in the distance to the threshold
                       "W_x_trend_24": x[pre] - x[np.clip(pre - 24, 0, None)]})
    targets = TARGETS if targets is None else targets
    history_columns = H_BASE if history_columns is None else history_columns
    pre_return_columns = PRE_RETURNS if pre_return_columns is None else pre_return_columns
    for c in [*history_columns, *pre_return_columns]:
        if c in X and c != fid:
            ep[f"W_{c}"] = X[c].to_numpy(float)[pre]
    if volatility_regime_column and volatility_regime_column in X:
        vol = X[volatility_regime_column].to_numpy(float)
        code, mid, high, cuts = vol_regime_dummies(vol[pre], vol[np.isfinite(vol)])
        ep["W_vol_regime_mid"], ep["W_vol_regime_high"], ep["vol_regime_code"] = mid, high, code
    else:
        cuts = None
    for c in locators:
        if c in X and c != fid:
            ep[f"W_{c}"] = X[c].to_numpy(float)[pre]
    if placebo_outcome_column and placebo_outcome_column in X:
        j = np.searchsorted(tn, tn[pre] - 144 * H_NS, side="right") - 1
        far = X[placebo_outcome_column].to_numpy(float)[np.clip(j, 0, None)]
        ep["Ypre_distant_24h_ending_t_minus_145h"] = np.where(j >= 0, far, np.nan)
    for name, *_ in targets:
        if name in Y:
            ep[name] = Y[name].to_numpy(float)[rows]
    ep["M_first_hour"] = Y[mediator_target].to_numpy(float)[rows] if mediator_target and mediator_target in Y else np.nan
    disjoint = episode_windows_disjoint(tn[rows], horizon_h)
    info = {"threshold_q": q, "threshold": thr, "band_q": band_q, "band": band, "band_q_value": band_q_value,
            "band_rule": "max(q60, thr - 2*SD_TRAIN(one-step change))", "sd_one_step": sd_step, "treated": len(treated),
            "controls": len(controls), "min_gap_h": gap_h, "horizon_h": int(horizon_h), "windows_disjoint": disjoint,
            "vol_regime_cuts_train": cuts}
    return ep, info


def episode_windows_disjoint(times_ns, horizon_h):
    """True when consecutive episode decision instants are >= horizon_h hours apart (outcome windows disjoint)."""
    t = np.sort(np.asarray(times_ns, dtype=np.int64))
    if len(t) < 2:
        return True
    return bool(np.all((t[1:] - t[:-1]) >= int(horizon_h) * 3600 * 10**9))


def rung2_cell(ep, *, fid, target, horizon_h, clock, info, seed=SEED):
    """Repaired fail-closed rung 2 on horizon-aware episodes, with declared + evidenced assumptions."""
    w_cols = [c for c in ep.columns if c.startswith("W_")]
    ctx = {} if clock == "OBSERVED" else {"publication_clock": "ASSUMED_SCHEDULED_PUBLICATION"}
    evidence = assumption_evidence(fid, horizon_h, info.get("windows_disjoint"), clock)
    declared = dict(ASSUMPTIONS)
    if not info.get("windows_disjoint"):
        declared["NO_INTERFERENCE_BETWEEN_EPISODES"] = False  # the gate then abstains by name
    r2 = ps3c.rung2_effect(ep, treatment="A", outcome=target, adjustment=["W"], contrast=(1.0, 0.0), dag=DAG,
                           node_columns={"W": w_cols}, treatment_node="A", outcome_node="Y", context=ctx,
                           assumptions=declared, assumption_evidence=evidence,
                           placebo_outcomes=[c for c in ep.columns if c.startswith("Ypre_")],
                           modifiers=["vol_regime_code"] if "vol_regime_code" in ep else (),
                           time_key="decision_time", treatment_kind="BINARY", seed=seed,
                           support={"min_episodes_per_side": 20})
    r2["assumption_strength"] = ASSUMPTION_STRENGTH
    r2["conditioning_set"] = w_cols
    return r2


def p_from_interval(est, lo, hi):
    se = (hi - lo) / 3.92
    if not se > 0:
        return None
    return float(2 * (1 - NormalDist().cdf(abs(est) / se)))


def rung2_summary(r2, *, y_sd_train):
    """Compact, selection-facing rung-2 record (estimate withheld unless identified)."""
    sup = r2.get("support") or {}
    est = r2.get("estimate") or {}
    sens = r2.get("sensitivity") or {}
    identified = r2.get("state") == ps3c.IDENTIFIED
    rec = {"raw_state": r2.get("state"), "reasons": list(r2.get("reasons", [])), "estimand": r2.get("estimand"),
           "contrast": r2.get("contrast"), "dag": r2.get("dag"), "adjustment": r2.get("adjustment"),
           "conditioning_set": r2.get("conditioning_set"), "excluded_from_adjustment": r2.get("excluded_from_adjustment"),
           "assumptions_declared": r2.get("assumptions_declared"), "assumptions_evidence": r2.get("assumptions_evidence"),
           "assumptions_unverified": r2.get("assumptions_unverified"), "assumption_strength": r2.get("assumption_strength"),
           "support": {k: sup.get(k) for k in ("state", "n_per_side", "n_population", "propensity_range", "propensity_bounds",
                                                "balance_max_smd", "balance_bound", "effective_sample_size")},
           "placebo": r2.get("placebo"), "population": r2.get("population"), "estimator": r2.get("estimator"),
           "sensitivity": {k: sens.get(k) for k in ("robustness_value_q1", "robustness_value_q1_alpha05", "att_aipw",
                                                    "att_propensity_matching_1nn_caliper02", "matching_unmatched_treated",
                                                    "ate_gcomputation_ols", "placebo_shift_share_abs_z_gt_1_96")},
           "estimate": {"value": est.get("value"), "interval": est.get("interval"), "unit": est.get("unit"),
                        "uncertainty": est.get("uncertainty")} if identified else None,
           "cate_by_vol_regime": est.get("heterogeneity") if identified else None,
           "y_sd_train": y_sd_train, "equivalence_margin": EQUIVALENCE_MARGIN_SD * y_sd_train if y_sd_train else None,
           "multiplicity": {"family": "rung2:(target) over all cells with an identified estimate",
                            "correction": "benjamini_hochberg", "q_level": FDR_Q}}
    if identified:
        lo, hi = est["interval"]
        rec["p_linear"] = p_from_interval(est["value"], lo, hi)
    else:
        rec["p_linear"] = None
    return rec


def assign_rung2_state(rec, q, nonlinear, rung1_sign):
    """Final rung-2 state once the family BH q and the nonlinear confirmation are known."""
    rec["q"] = q
    rec["nonlinear"] = nonlinear
    if rec["raw_state"] != ps3c.IDENTIFIED:
        rec["state"] = NOT_IDENTIFIED
        rec["abstention_reason"] = ";".join(rec["reasons"]) or "NO_ESTIMATE"
        return rec
    est = rec["estimate"]
    lo, hi = est["interval"]
    excl0 = lo > 0 or hi < 0
    sig = q is not None and q <= FDR_Q
    nl_ok = bool(nonlinear and nonlinear.get("state") == "ESTIMATED"
                 and (nonlinear.get("confirmation_identity") or {}).get("compatible", False))
    nl_sign = int(np.sign(nonlinear["estimate"])) if nl_ok else None
    nl_excl0 = bool(nl_ok and (nonlinear["interval"][0] > 0 or nonlinear["interval"][1] < 0))
    rv = (rec.get("sensitivity") or {}).get("robustness_value_q1")
    robust_base = bool(nl_ok and rv is not None and rv >= RV_FLOOR and (rec.get("placebo") or {}).get("state") == "PASSED")
    margin = rec.get("equivalence_margin")
    sign = int(np.sign(est["value"]))
    if sig and excl0 and nl_excl0 and nl_sign == sign:
        if rung1_sign not in (None, 0) and sign != rung1_sign:
            rec["state"] = CONTRADICTED
            rec["contradiction_kind"] = "IDENTIFIED_EFFECT_OPPOSITE_TO_STABLE_RUNG1_ASSOCIATION"
            rec["robust"] = robust_base
        else:
            rec["state"] = SUPPORTED
            rec["robust"] = robust_base
        rec["abstention_reason"] = None
        return rec
    if margin and -margin <= lo and hi <= margin and nl_ok and -margin <= nonlinear["interval"][0] \
            and nonlinear["interval"][1] <= margin:
        rec["state"] = CONTRADICTED
        rec["contradiction_kind"] = f"PRECISE_NULL_WITHIN_{EQUIVALENCE_MARGIN_SD}_SD_EQUIVALENCE_MARGIN"
        rec["robust"] = robust_base
        rec["abstention_reason"] = None
        return rec
    rec["state"] = NOT_IDENTIFIED
    why = []
    if not sig:
        why.append(f"EFFECT_NOT_SIGNIFICANT_AT_FAMILY_FDR_{FDR_Q}")
    if not excl0:
        why.append("INTERVAL_INCLUDES_ZERO_AND_NOT_PRECISE_NULL")
    if not nl_ok:
        why.append("NONLINEAR_CONFIRMATION_" + str((nonlinear or {}).get("state", "NOT_RUN")))
    elif not nl_excl0 or nl_sign != sign:
        why.append("NONLINEAR_CONFIRMATION_DISAGREES")
    rec["abstention_reason"] = ";".join(why) or "GATE_NOT_MET"
    return rec


# ------------------------------------------------------------------------------------------------- rung 3


def rung3_cell(ep, *, target, r2_state, w_cols, seed=SEED, mediator_target="Y_s_1h"):
    meds = ["M_first_hour"] if mediator_target and target != mediator_target and "M_first_hour" in ep else []
    pl = next((c for c in ep.columns if c.startswith("Ypre_")), None)
    r3, rows = ps3c.rung3_population(ep, treatment="A", outcome=target, adjustment_cols=w_cols, a0=0.0,
                                     rung2_state=r2_state, mediators=meds, placebo_outcome=pl,
                                     time_key="decision_time", seed=seed)
    return r3, len(rows)


def rung3_summary(r3, n_rows):
    sens = r3.get("sensitivity") or {}
    pred = r3.get("prediction") or {}
    return {"raw_state": r3.get("state"), "label": r3.get("label"), "reasons": list(r3.get("reasons", [])),
            "scm": r3.get("scm"), "abduction": r3.get("abduction"), "action": r3.get("action"),
            "prediction": pred or None, "episodes_n": int(n_rows),
            "sensitivity": {k: sens.get(k) for k in ("reconstruction_max_abs_error", "analog_pairs", "analog_gap_z",
                                                     "analog_state", "placebo_state", "null_action_max_abs_delta",
                                                     "placebo_z", "alternatives_worst_sign_agreement",
                                                     "alternatives_mean_abs_delta", "counterfactual_population_n")},
            "never_observed": "the counterfactual outcome is an estimate under the declared SCM, never an observation"}


def assign_rung3_state(rec, rung2_state, rung2_sign):
    if rec["raw_state"] != ps3c.CF_STATE or rung2_state not in (SUPPORTED, CONTRADICTED):
        rec["state"] = NOT_IDENTIFIED
        rec["abstention_reason"] = ";".join(rec["reasons"]) or ("RUNG2_" + str(rung2_state))
        return rec
    delta = (rec.get("prediction") or {}).get("delta")
    agree = (rec.get("sensitivity") or {}).get("alternatives_worst_sign_agreement") or 0.0
    analog = (rec.get("sensitivity") or {}).get("analog_state")
    if delta is None:
        rec["state"] = NOT_IDENTIFIED
        rec["abstention_reason"] = "NO_COUNTERFACTUAL_DELTA"
        return rec
    # delta = factual - counterfactual(A:=0) = the episode's own estimate of the effect of A=1
    sign = int(np.sign(delta))
    if agree >= 0.9 and analog == "CONSISTENT" and rung2_sign not in (None, 0):
        if sign == rung2_sign:
            rec["state"] = SUPPORTED
            rec["robust"] = True
        else:
            rec["state"] = CONTRADICTED
            rec["contradiction_kind"] = "COUNTERFACTUAL_SIGN_OPPOSES_IDENTIFIED_RUNG2_EFFECT"
            rec["robust"] = True
        rec["abstention_reason"] = None
    else:
        rec["state"] = NOT_IDENTIFIED
        rec["abstention_reason"] = "MODEL_DEPENDENT_OR_ANALOGS_NOT_CONSISTENT"
    return rec


# ------------------------------------------------------------------------------------------------- helpers


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=ps3c._json_default, separators=(",", ":")).encode()).hexdigest()


def feature_weight_against(cells):
    """Selection-facing aggregation for one feature: only robust CONTRADICTED weighs against it."""
    against = [c for c in cells if any(c[r]["state"] == CONTRADICTED and c[r].get("robust") for r in ("rung1", "rung2", "rung3"))]
    supported = [c for c in cells if any(c[r]["state"] == SUPPORTED for r in ("rung1", "rung2", "rung3"))]
    return {"robust_contradicted_cells": len(against), "supported_cells": len(supported), "cells": len(cells),
            "weighs_against": bool(against), "not_identified_never_eliminates": True}
