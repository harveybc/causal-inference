"""PS3-C batch runner over a lane-A EURUSD PS0/PS1 batch (canonical selection-first plan, 2026-10-03).

    python -m causal_inference_provider.ps3c_batch --lane-a-batch DIR --inputs DIR --lane-a-code DIR \
        --out DIR --batch batch_001 --revision <causal-inference commit>

Reads ONLY the lane-A TRAIN artifacts (features_train / targets_train / folds / contract /
admissible_features, verified against READY + digests.json) and, for economic-event episodes,
the calendar archive through lane A's own loader (its measured clock eras). Nothing at or after
the contract's TRAIN end is read; the external test is never touched.

Subjects:
* every admissible lane-A feature (role "feature"): rung 1 on the hourly decision rows with the
  contract's purged forward-chaining inner folds; rungs 2-3 on crossing episodes (first available
  crossing of the TRAIN q80 threshold vs comparable non-crossings);
* every high-impact USD/EUR release type with enough consensus rows: episode = one release,
  A = (actual - prior consensus) / TRAIN MAD scale of the type (A = 0 means "as expected").

Per (subject, target) one ``causal_dossier.v1`` document plus a diagnostics record; then a
summary, a report with denominators/states/digests/cost, digests.json and READY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import ps3c
from . import ps3c_stats as st

H_BASE = ["px.logret_24h", "px.logret_120h", "px.ewma_vol_24", "px.ewma_vol_168",
          "cal.hour_sin", "cal.hour_cos", "cal.dow_sin", "cal.dow_cos"]
PLACEBO_PRE = ["px.logret_1h", "px.logret_6h"]
TARGETS = ([(f"Y_s_{h}h", "Y_s", "short", 60 * h) for h in (1, 2, 3, 4, 5, 6)]
           + [(f"Y_l_{h}h", "Y_l", "long", 60 * h) for h in (24, 48, 72, 96, 120, 144)]
           + [("Y_b_s6", "Y_b", "barrier", 360), ("Y_b_l144", "Y_b", "barrier", 8640)])
DAG = {"nodes": ["W", "A", "M", "Y"],
       "edges": [["W", "A"], ["W", "Y"], ["W", "M"], ["A", "M"], ["M", "Y"], ["A", "Y"]]}
ASSUMPTIONS = {k: True for k in ps3c.REQUIRED_ASSUMPTIONS}
GROUPS = ("USD", "EUR")
HOUR = pd.Timedelta(hours=1)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=ps3c._json_default)


def _ns(values):
    """UTC instants as int64 nanoseconds, whatever the pandas resolution."""
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)).as_unit("ns").asi8


def slug(s):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s.lower())[:90].strip("_")


# --------------------------------------------------------------------------------------------- inputs


def verify_batch(bdir):
    ready = json.loads(open(os.path.join(bdir, "READY")).read())
    dig_path = os.path.join(bdir, "digests.json")
    if sha(dig_path) != ready["digests_sha256"]:
        raise SystemExit("REFUSED: digests.json does not match READY")
    dig = json.load(open(dig_path))
    checked = {}
    for name in ("features_train.parquet", "targets_train.parquet", "folds.json", "contract.json",
                 "admissible_features.json"):
        got = sha(os.path.join(bdir, name))
        if dig["artifacts_sha256"].get(name) != got:
            raise SystemExit(f"REFUSED: {name} digest mismatch")
        checked[name] = got
    return ready, dig, checked


def load_batch(bdir):
    X = pd.read_parquet(os.path.join(bdir, "features_train.parquet"))
    Y = pd.read_parquet(os.path.join(bdir, "targets_train.parquet"))
    X["t_decision_utc"] = pd.to_datetime(X["t_decision_utc"], utc=True)
    Y["t_decision_utc"] = pd.to_datetime(Y["t_decision_utc"], utc=True)
    if not (X["row_id"].to_numpy() == Y["row_id"].to_numpy()).all() or not (
            X["t_decision_utc"].to_numpy() == Y["t_decision_utc"].to_numpy()).all():
        raise SystemExit("REFUSED: features and targets rows differ")
    if not X["t_decision_utc"].is_monotonic_increasing:
        raise SystemExit("REFUSED: decision rows not in time order")
    contract = json.load(open(os.path.join(bdir, "contract.json")))
    folds_doc = json.load(open(os.path.join(bdir, "folds.json")))
    meta = json.load(open(os.path.join(bdir, "admissible_features.json")))["features"]
    train_end = pd.Timestamp(contract["periods"]["train"][1])
    if X["t_decision_utc"].max() >= train_end:
        raise SystemExit("REFUSED: a decision row at or after TRAIN end")
    folds = []
    for f in folds_doc["folds"]:
        if f.get("train_rows") and f.get("val_rows"):
            folds.append((np.arange(*f["train_rows"]), np.arange(*f["val_rows"])))
    return X, Y, contract, folds_doc, folds, meta, train_end


def asset_slot(contract, dig, bdir_digests):
    """The outcome price series binding: lane A's business contract (not a C127 census appearance)."""
    res = "eurusd_5m.parquet"
    return {"state": "CONTRACTED_BUSINESS_CONTRACT", "entity": contract.get("asset", "EURUSD"),
            "contract_schema": contract["schema"], "contract_sha256": contract["contract_sha256"],
            "resource": "lake:features/trading_asset_data/eurusd/5m.parquet (lane A input eurusd_5m.parquet)",
            "resource_sha256": dig["inputs_sha256"].get(res, "UNKNOWN"),
            "train_period": contract["periods"]["train"], "frequency": "1h",
            "lane_a_digests_sha256": bdir_digests}


# --------------------------------------------------------------------------------------------- episodes


def crossing_episodes(X, Y, fid, q=0.8, band_q=0.6, min_gap_h=24, stride_h=6, threshold=None, band=None):
    """A=1: first available crossing of the TRAIN q80 threshold (row t-1 below, row t at/above).
    A=0: rows that stayed below with the previous value inside [q60, q80). W from row t-1."""
    x = X[fid].to_numpy(float)
    ok = np.isfinite(x)
    if ok.sum() < 200 or np.nanstd(x) == 0:
        return None, {"reason": "TOO_FEW_FINITE_OR_CONSTANT"}
    thr = float(np.nanquantile(x, q)) if threshold is None else float(threshold)
    band = float(np.nanquantile(x, band_q)) if band is None else float(band)
    if not band < thr:
        return None, {"reason": "THRESHOLD_BAND_DEGENERATE", "threshold": thr}
    tn = _ns(X["t_decision_utc"])
    prev, now = x[:-1], x[1:]
    with np.errstate(invalid="ignore"):
        cross = np.where((prev < thr) & (now >= thr))[0] + 1
        ctrl = np.where((prev < thr) & (now < thr) & (prev >= band))[0] + 1
    H_NS = 3600 * 10**9
    gap = (tn[1:] - tn[:-1]) / H_NS
    treated, last = [], None
    for i in cross:
        if gap[i - 1] > 72:  # the previous row is not the previous market hour
            continue
        if last is None or (tn[i] - last) / H_NS >= min_gap_h:
            treated.append(i)
            last = tn[i]
    tt = tn[treated] if treated else np.array([], dtype=np.int64)
    controls, lastc = [], None
    for i in ctrl:
        if gap[i - 1] > 72:
            continue
        ti = tn[i]
        if lastc is not None and (ti - lastc) < stride_h * 3600 * 10**9:
            continue
        # only PAST crossings may exclude a control: excluding rows followed by a crossing would
        # select controls on their future path (a leak that manufactures an "effect")
        past = tt[tt <= ti]
        if len(past) and (ti - past[-1]) < min_gap_h * 3600 * 10**9:
            continue
        controls.append(i)
        lastc = ti
    rows = np.array(treated + controls, dtype=int)
    rows.sort()
    if len(rows) == 0:
        return None, {"reason": "NO_EPISODES", "threshold": thr}
    pre = rows - 1
    ep = pd.DataFrame({"episode_id": [f"{fid}|{pd.Timestamp(tn[i], tz='UTC').isoformat()}" for i in rows],
                       "decision_time": X["t_decision_utc"].to_numpy()[rows],
                       "A": np.isin(rows, treated).astype(float),
                       "W_x_prev": x[pre],
                       "W_x_trend_24": x[pre] - x[np.clip(pre - 24, 0, None)]})
    # For an endogenous transition the recent path is UPSTREAM of A (a confounder), so it enters W;
    # the placebo is a distant pre-period return (24h ending 144h before the pre-row), a negative-control
    # outcome that the transition cannot have caused.
    for c in [*H_BASE, *PLACEBO_PRE]:
        if c in X and c != fid:
            ep[f"W_{c}"] = X[c].to_numpy(float)[pre]
    if "px.logret_24h" in X:
        j = np.searchsorted(tn, tn[pre] - 144 * H_NS, side="right") - 1
        far = X["px.logret_24h"].to_numpy(float)[np.clip(j, 0, None)]
        ep["Ypre_distant_24h_ending_t_minus_145h"] = np.where(j >= 0, far, np.nan)
    for name, *_ in TARGETS:
        ep[name] = Y[name].to_numpy(float)[rows]
    ep["M_first_hour"] = Y["Y_s_1h"].to_numpy(float)[rows]
    info = {"threshold_q": q, "threshold": thr, "band_q": band_q, "band": band, "treated": len(treated),
            "controls": len(controls), "min_gap_h": min_gap_h, "control_stride_h": stride_h}
    return ep, info


def event_episodes(X, Y, inputs, lane_a_code, train_end, min_releases=60, top_k=8):
    """One row per high-impact USD/EUR release with consensus, anchored at its availability."""
    sys.path.insert(0, lane_a_code)
    from tools.eurusd_ps import features as LF  # noqa: E402  (lane A's loader, its measured clock eras)
    from tools.eurusd_ps import run_batch as LR  # noqa: E402
    from tools.eurusd_ps import sources as LS  # noqa: E402

    arch, arch_stats = LS.load_archive_calendar(os.path.join(inputs, "economic_calendar_2011_2021.csv"),
                                                LR.ARCHIVE_ERAS)
    ev = LF.archive_event_table(arch)
    ex = Counter()
    ev = ev[ev["avail_utc"] < train_end]
    hi = ev[(ev["group"].isin(GROUPS)) & (ev["volatility"] == LF.TIERS["high"])].copy()
    ex["NO_CONSENSUS"] = int(hi["forecast_v"].isna().sum())
    ex["NO_ACTUAL"] = int(hi["actual_v"].isna().sum())
    hi = hi[hi["forecast_v"].notna() & hi["actual_v"].notna()]
    keys = []
    for g in GROUPS:
        vc = hi[hi["group"] == g]["key"].value_counts()
        keys += [k for k, n in vc.head(top_k).items() if n >= min_releases]
    t_grid = X["t_decision_utc"]
    tg = _ns(t_grid)
    episodes, pseudo, scales = {}, {}, {}
    all_hi = hi.sort_values("avail_utc")
    av_all = _ns(all_hi["avail_utc"])
    for key in keys:
        sub = all_hi[all_hi["key"] == key].copy()
        raw = sub["surprise"].to_numpy(float)
        mad = float(np.median(np.abs(raw - np.median(raw)))) * 1.4826
        scale = mad if mad > 0 else float(np.std(raw, ddof=1))
        if not scale > 0:
            ex[f"SCALE_UNDEFINED:{key}"] += len(sub)
            continue
        scales[key] = scale
        sub["A"] = raw / scale
        rows, prow = [], []
        last_a, last_t = np.nan, None
        for _, r in sub.iterrows():
            av = np.int64(_ns([r["avail_utc"]])[0])
            i = int(np.searchsorted(tg, av, side="left"))  # first decision row with t >= availability
            if i >= len(tg) or i == 0 or (tg[i] - av) > 2 * 3600 * 10**9:
                ex["NO_ENTRY_ROW_WITHIN_2H"] += 1
                last_a, last_t = r["A"], av
                continue
            ip = i - 1
            if (av - tg[ip]) > 2 * 3600 * 10**9:
                ex["NO_PRE_EVENT_ROW_WITHIN_2H"] += 1
                last_a, last_t = r["A"], av
                continue
            if last_t is not None and (av - last_t) < 24 * 3600 * 10**9:
                ex["SAME_TYPE_RELEASE_WITHIN_24H_BEFORE"] += 1
                last_a, last_t = r["A"], av
                continue
            same_instant = (av_all == av) & (all_hi["key"].to_numpy() != key)
            co = all_hi.loc[same_instant, "surprise_z"].to_numpy(float)
            row = {"episode_id": hashlib.sha1(f"{key}|{av}".encode()).hexdigest()[:16],
                   "decision_time": t_grid.iloc[i], "avail_utc": r["avail_utc"], "A": float(r["A"]),
                   "raw_surprise": float(r["surprise"]), "W_lag_A_same_type": last_a,
                   "W_concurrent_releases": float(same_instant.sum()),
                   "W_concurrent_surprise_z_sum": float(np.nansum(np.clip(co, -10, 10)))}
            for c in H_BASE:
                row[f"W_{c}"] = float(X[c].iloc[ip])
            for g in GROUPS:
                for nm in ("count_24h", "sum_surprise_z_24h"):
                    c = f"ev.{g}.high.{nm}"
                    if c in X:
                        row[f"W_{c}"] = float(X[c].iloc[ip])
            for c in PLACEBO_PRE:
                row[f"Ypre_{c}"] = float(X[c].iloc[ip])
            for name, *_ in TARGETS:
                row[name] = float(Y[name].iloc[i])
            row["M_impact"] = float(Y["Y_s_1h"].iloc[ip])  # C(t_pre + 1h) / C(t_pre): the bar containing the release
            rows.append(row)
            # pseudo-event: same hour one week earlier, no release of this type within +-24h
            tp = tg[i] - 168 * 3600 * 10**9
            j = int(np.searchsorted(tg, tp, side="left"))
            if j < len(tg) and j > 0 and tg[j] == tp and not np.any(
                    np.abs(_ns(sub["avail_utc"]) - tp) <= 24 * 3600 * 10**9):
                pr = {"A": float(r["A"])}
                for k, v in row.items():
                    if k.startswith("W_") and k[2:] in X:
                        pr[k] = float(X[k[2:]].iloc[j - 1])
                pr["W_lag_A_same_type"] = last_a
                pr["W_concurrent_releases"] = 0.0
                pr["W_concurrent_surprise_z_sum"] = 0.0
                for name, *_ in TARGETS:
                    pr[name] = float(Y[name].iloc[j])
                pr["M_impact"] = float(Y["Y_s_1h"].iloc[j - 1])
                prow.append(pr)
            last_a, last_t = r["A"], av
        episodes[key] = pd.DataFrame(rows)
        pseudo[key] = pd.DataFrame(prow)
    info = {"archive_load": arch_stats, "keys": keys, "scales_train_mad": scales,
            "exclusions": dict(ex), "releases_high_usd_eur_with_consensus": int(len(hi))}
    return episodes, pseudo, info


# --------------------------------------------------------------------------------------------- per cell


def run_cell(ep, *, target, mediator, w_cols, context, pseudo=None, seed=1729, kind="AUTO"):
    cols_needed = ["A", target, *w_cols]
    r2 = ps3c.rung2_effect(ep, treatment="A", outcome=target, adjustment=["W"], contrast=(1.0, 0.0), dag=DAG,
                           node_columns={"W": w_cols}, treatment_node="A", outcome_node="Y", context=context,
                           assumptions=ASSUMPTIONS, placebo_outcomes=[c for c in ep.columns if c.startswith("Ypre_")],
                           placebo_episodes=pseudo, time_key="decision_time", treatment_kind=kind, seed=seed,
                           support={"min_episodes_per_side": 20})
    meds = [mediator] if mediator and mediator != target and mediator in ep else []
    if r2["state"] == ps3c.IDENTIFIED:
        pl = next((c for c in ep.columns if c.startswith("Ypre_")), None)
        r3, rows = ps3c.rung3_population(ep, treatment="A", outcome=target, adjustment_cols=w_cols, a0=0.0,
                                         rung2_state=r2["state"], mediators=meds, placebo_outcome=pl,
                                         time_key="decision_time", seed=seed)
    else:
        r3, rows = {"state": ps3c.NOT_IDENTIFIED, "label": "NONE", "reasons": ["RUNG2_NOT_IDENTIFIED"]}, []
    del cols_needed
    return r2, r3, rows


NOT_APPLICABLE_REASONS = {"KNOWN_IN_ADVANCE_CALENDAR_NOT_AN_OBSERVED_INTERVENTION"}
NO_VARIATION_REASONS = {"TOO_FEW_FINITE_OR_CONSTANT", "THRESHOLD_BAND_DEGENERATE", "NO_EPISODES"}


def summary_state(rung, raw, reasons=""):
    """Selection-facing state per rung: ESTIMATED / NOT_IDENTIFIED / NOT_APPLICABLE / FAILED (PENDING = not run).

    ESTIMATED on rung 1 is an association estimate (never causal); on rung 2 an effect identified
    conditional on the declared assumptions; on rung 3 a same-episode counterfactual under the declared SCM.
    """
    reasons = set(filter(None, str(reasons or "").split(";")))
    if any(r.startswith("EXECUTION_FAILED") for r in reasons):
        return "FAILED"
    if reasons & NOT_APPLICABLE_REASONS:
        return "NOT_APPLICABLE"
    if raw in ("ASSOCIATION_REPORTED", ps3c.IDENTIFIED, ps3c.CF_STATE):
        return "ESTIMATED"
    if raw in ("TOO_FEW_EVENTS", "ZERO_VARIANCE", ps3c.NOT_IDENTIFIED):
        return "NOT_IDENTIFIED"
    if raw == "NOT_EVALUATED" and reasons & NO_VARIATION_REASONS:
        return "NOT_IDENTIFIED"
    if raw == "NOT_EVALUATED" and reasons:
        return "FAILED" if any("MISSING" in r for r in reasons) else "NOT_IDENTIFIED"
    return "PENDING"


def safe_run_cell(ep, **kw):
    try:
        return run_cell(ep, **kw)
    except Exception as trouble:  # recorded as FAILED with the error, never replaced by a number
        why = f"EXECUTION_FAILED:{type(trouble).__name__}"
        return ({"state": "NOT_EVALUATED", "reasons": [why], "estimand": "not evaluated (execution failed)",
                 "rung1": {"state": "NOT_EVALUATED"}, "diagnostics": {"error": str(trouble)[:500]}},
                {"state": "NOT_EVALUATED", "label": "NONE", "reasons": [why]}, [])


def upstream_mechanism(ep, w_cols):
    """What produced the historical transition: cross-fitted logistic propensity of A on the pre-row W."""
    if ep is None or ep["A"].nunique() < 2:
        return {"abstention": "NO_VARIATION_IN_A"}
    d = ep.dropna(subset=["A", *w_cols])
    if len(d) < 50 or d["A"].nunique() < 2:
        return {"abstention": "TOO_FEW_COMPLETE_EPISODES"}
    w = d[w_cols].to_numpy(float)
    sd = w.std(0)
    sd[sd == 0] = 1.0
    wz = (w - w.mean(0)) / sd
    x = st.add_const(wz)
    t = d["A"].to_numpy(float)
    e = st.crossfit_predict(x, t, logistic_model=True)
    pos, neg = e[t == 1], e[t == 0]
    ranks = st.rankdata(np.r_[pos, neg])
    auc = float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))) if len(pos) and len(neg) else None
    beta = st.logistic(x, t)
    coefs = sorted(zip(w_cols, beta[1:]), key=lambda kv: -abs(kv[1]))
    return {"model": "logistic(A ~ standardized W at row t-1), cross-fitted", "auc_crossfit": auc,
            "standardized_coefficients": {k: float(v) for k, v in coefs}, "n": int(len(d))}


def candidate_card(*, question, treatment, target, history, r2, r3, w_cols, mediator, a0_note):
    """Minimum per prioritized candidate (canonical order section 6): every field filled or an explicit abstention."""
    def ab(reason):
        return {"abstention": reason}

    r2 = r2 or {}
    r3 = r3 or {}
    r2_why = ";".join(r2.get("reasons", [])) or r2.get("state", "NOT_EVALUATED")
    r3_why = ";".join(r3.get("reasons", [])) or r3.get("state", "NOT_EVALUATED")
    sens3 = r3.get("sensitivity") or {}
    has_r3 = r3.get("state") == ps3c.CF_STATE
    return {
        "question": question, "A": treatment, "Y": target, "H": history,
        "rung2": {
            "DAG": r2.get("dag") or ab(r2_why), "adjustment_set": r2.get("adjustment") or ab(r2_why),
            "excluded_from_adjustment": r2.get("excluded_from_adjustment", []),
            "support_overlap": r2.get("support") if (r2.get("support") or {}).get("state") not in (None, "NOT_EVALUATED") else ab(r2_why),
            "estimator": r2.get("estimator") or ab(r2_why),
            "diagnostics": {"placebo": r2.get("placebo") or ab(r2_why), "sensitivity": r2.get("sensitivity") or ab(r2_why)},
            "estimate": r2.get("estimate") if r2.get("estimate") is not None else ab(r2_why),
            "state": r2.get("state"), "reasons": r2.get("reasons", []),
        },
        "rung3": {
            "SCM": r3.get("scm") if has_r3 else ab(r3_why),
            "abduction": r3.get("abduction") if has_r3 else ab(r3_why),
            "historically_supported_alternative": a0_note if has_r3 else ab(r3_why),
            "propagation": {"mediators": [mediator] if mediator else [], "outcome": target} if has_r3 else ab(r3_why),
            "factual_reconstruction": sens3.get("reconstruction_max_abs_error") if has_r3 else ab(r3_why),
            "historical_analogs": {"state": sens3.get("analog_state"), "pairs": sens3.get("analog_pairs"),
                                   "z": sens3.get("analog_gap_z")} if has_r3 else ab(r3_why),
            "placebos": {"state": sens3.get("placebo_state"), "null_action_max_abs_delta": sens3.get("null_action_max_abs_delta"),
                         "placebo_z": sens3.get("placebo_z")} if has_r3 else ab(r3_why),
            "sensitivity": {"alternatives_worst_sign_agreement": sens3.get("alternatives_worst_sign_agreement"),
                            "alternatives_mean_abs_delta": sens3.get("alternatives_mean_abs_delta")} if has_r3 else ab(r3_why),
            "prediction": r3.get("prediction") if has_r3 else ab(r3_why),
            "state": r3.get("state"), "reasons": r3.get("reasons", []),
        },
        "adjustment_columns": w_cols,
        "not_an_input_adapter": "selector evidence only; feeding calendar/surprise/dossiers to predictor/core/RL/M5PHET is I11 (deferred)",
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane-a-batch", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--lane-a-code", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--only-features", default=None, help="comma list (pilot)")
    ap.add_argument("--max-event-types", type=int, default=None, help="pilot limit")
    ap.add_argument("--permutations", type=int, default=200)
    a = ap.parse_args(argv)
    t_all = time.time()
    out = a.out
    if os.path.exists(os.path.join(out, "READY")):
        raise SystemExit("REFUSED: output batch already READY")
    os.makedirs(os.path.join(out, "dossiers"), exist_ok=True)
    os.makedirs(os.path.join(out, "records"), exist_ok=True)
    cost = {}
    t0 = time.time()
    ready, dig, checked = verify_batch(a.lane_a_batch)
    X, Y, contract, folds_doc, folds, meta, train_end = load_batch(a.lane_a_batch)
    cost["load_s"] = time.time() - t0
    lane_a_dig = sha(os.path.join(a.lane_a_batch, "digests.json"))
    slot = asset_slot(contract, dig, lane_a_dig)
    produced = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    feats = [m for m in meta if m.get("role") == "feature" and str(m.get("admissibility", "")).startswith("ADMISSIBLE")
             and m["feature_id"] in X]
    if a.only_features:
        keep = set(a.only_features.split(","))
        feats = [m for m in feats if m["feature_id"] in keep]
    cells, pvals, docs = [], [], []

    def emit(subject_id, subject_kind, target, family, head, minutes, r1, r2, r3, treatment, manifest_extra, extra):
        dossier_id = slug(f"eurusd.{a.batch}.{subject_kind}.{subject_id}.{target}")[:120]
        r2c = {k: v for k, v in r2.items() if k not in ("rung1", "diagnostics")} if r2 else None
        doc = ps3c.dossier(
            dossier_id=dossier_id, producer_revision=a.revision, produced_at=produced,
            subject={"kind": "EPISODE_POPULATION", "population": extra["population"], "event_type": extra["event_type"],
                     "asset": "EURUSD", "head": head, "target": family, "horizon_minutes": int(minutes)},
            data_manifest={"sources": extra["sources"], "asset_appearance": slot,
                           "publication_clock": manifest_extra["publication_clock"],
                           "consensus_clock": manifest_extra["consensus_clock"],
                           "expectation_kind": manifest_extra["expectation_kind"],
                           "n_episodes": int(extra["n_episodes"]), "exclusions": extra["exclusions"],
                           "train_folds": [f["name"] for f in folds_doc["folds"]]},
            treatment=treatment, rung1=r1, rung2=r2c, rung3=r3,
            emission={"operational_use": "RETROSPECTIVE_ONLY",
                      "emittable_from": {"rung3.abduction": str(train_end.isoformat()).replace("+00:00", "Z")}},
            limitations=extra["limitations"])
        path = os.path.join(out, "dossiers", dossier_id + ".json")
        jdump(doc, path)
        docs.append(path)
        return doc

    # ---------------- features: rung 1 on hourly rows, rungs 2-3 on crossing episodes
    t0 = time.time()
    feat_records = []
    for m in feats:
        fid = m["feature_id"]
        hist = [c for c in H_BASE if c in X and c != fid]
        archive = "economic_calendar_2011_2021" in str(m.get("source", ""))
        calendar = fid.startswith("cal.")
        ctx = {"publication_clock": "ASSUMED_SCHEDULED_PUBLICATION"} if archive else {}
        ep, cinfo = (None, {"reason": "KNOWN_IN_ADVANCE_CALENDAR_NOT_AN_OBSERVED_INTERVENTION"}) if calendar else \
            crossing_episodes(X, Y, fid)
        if ep is not None:
            cinfo["upstream_mechanism"] = upstream_mechanism(ep, [c for c in ep.columns if c.startswith("W_")])
        base = {fid: X[fid].to_numpy(float), **{c: X[c].to_numpy(float) for c in hist}}
        for name, family, head, minutes in TARGETS:
            cols = dict(base)
            cols[name] = Y[name].to_numpy(float)
            r1 = ps3c.rung1_association(cols, treatment=fid, outcome=name, history=hist, explicit_folds=folds,
                                        permutations=a.permutations, min_events=200)
            if ep is not None:
                wc = [c for c in ep.columns if c.startswith("W_")]
                r2, r3, rows = safe_run_cell(ep, target=name, mediator=None if name == "Y_s_1h" else "M_first_hour",
                                             w_cols=wc, context=ctx, kind="BINARY")
            else:
                r2 = {"state": "NOT_EVALUATED", "reasons": [cinfo["reason"]],
                      "estimand": "ATE of a first available threshold crossing vs no crossing"}
                r3, rows = {"state": "NOT_EVALUATED", "label": "NONE", "reasons": [cinfo["reason"]]}, []
            treat = {"name": f"crossing({fid} >= TRAIN q80)",
                     "definition": "A=1 at the first decision row where the feature, available at t, crosses its TRAIN "
                                   "q80 threshold from below (row t-1 below); A=0 at rows that stayed below with row t-1 "
                                   "inside [q60, q80); W, history and placebo outcomes from row t-1",
                     "kind": "BINARY", "standardization": "threshold fitted on all TRAIN decision rows",
                     "dose_support": {"min": 0.0, "max": 1.0, "n": int(len(ep)) if ep is not None else 0},
                     "sequential_treatment_policy": "SEPARATED_EPISODES_ONLY"}
            src = [{"role": "covariates", "resource": "laneA:features_train.parquet",
                    "sha256": checked["features_train.parquet"], "rows": int(len(X)), "governed": False},
                   {"role": "bars", "resource": "laneA:targets_train.parquet",
                    "sha256": checked["targets_train.parquet"], "rows": int(len(Y)), "governed": False}]
            lim = ["rung 1 is association on hourly decision rows (overlapping outcomes; circular-shift null)",
                   "rungs 2-3 use crossing episodes; the crossing is an observed state change, not a market intervention",
                   "linear outcome/propensity models; identification is conditional on the declared DAG and W"]
            if archive:
                lim.append("archive calendar clock ASSUMED (scheduled+1min): rungs 2-3 cannot be identified")
            extra = {"population": f"crossing episodes of {fid}" if ep is not None else f"hourly rows ({fid})",
                     "event_type": f"feature:{fid}", "sources": src, "n_episodes": len(ep) if ep is not None else 0,
                     "exclusions": {}, "limitations": lim}
            mx = {"publication_clock": "ASSUMED_SCHEDULED_PUBLICATION" if archive else "OBSERVED_PUBLICATION_CLOCK",
                  "consensus_clock": "ASSUMED_BEFORE_RELEASE" if archive else "NONE",
                  "expectation_kind": "PUBLISHED_CONSENSUS" if archive else "NONE"}
            doc = emit(fid, "feature", name, family, head, minutes, r1, r2, r3, treat, mx, extra)
            pc = next((e for e in r1.get("evidence", []) if e["measure"] == "partial_corr_given_H" and e["regime"] is None), None)
            oof = next((e["value"] for e in r1.get("evidence", []) if e["measure"].startswith("oof_relative")), None)
            cells.append({"subject_kind": "feature", "subject": fid, "family": m.get("family"), "target": name,
                          "dossier_id": doc["dossier_id"], "rung1_raw": r1["state"], "rung2_raw": r2["state"],
                          "rung3_raw": r3["state"], "r1_partial_corr": pc["value"] if pc else None,
                          "r1_p": pc["p"] if pc else None, "r1_oof_gain": oof,
                          "r1_reasons": ";".join(filter(None, [(r1.get("diagnostics") or {}).get("reason", ""),
                                                              *[x for x in (r2 or {}).get("reasons", [])
                                                                if str(x).startswith("EXECUTION_FAILED")]])),
                          "r2_reasons": ";".join(r2.get("reasons", [])), "r3_reasons": ";".join(r3.get("reasons", [])),
                          "r2_estimand": r2.get("estimand"), "r2_support": (r2.get("support") or {}).get("state"),
                          "r2_n_per_side": json.dumps((r2.get("support") or {}).get("n_per_side")),
                          "r2_balance_max_smd": (r2.get("support") or {}).get("balance_max_smd"),
                          "r2_propensity_range": json.dumps((r2.get("support") or {}).get("propensity_range")),
                          "r2_placebo": (r2.get("placebo") or {}).get("state"),
                          "r2_estimate": (r2.get("estimate") or {}).get("value"),
                          "r2_interval": json.dumps((r2.get("estimate") or {}).get("interval")),
                          "r2_rv_q1": (r2.get("sensitivity") or {}).get("robustness_value_q1"),
                          "r3_delta": ((r3.get("prediction") or {}).get("delta")),
                          "r3_analog": (r3.get("sensitivity") or {}).get("analog_state"),
                          "r3_worst_sign_agreement": (r3.get("sensitivity") or {}).get("alternatives_worst_sign_agreement")})
            pvals.append(pc["p"] if pc else None)
            card = candidate_card(
                question=(f"What is the effect on {name} of the historical transition 'first available crossing of "
                          f"{fid} above its TRAIN q80' versus comparable hours that stayed below, given the pre-row "
                          f"history? (not do({fid}=value))"),
                treatment=treat, target=name, history=hist, r2=r2, r3=r3,
                w_cols=[c for c in ep.columns if c.startswith("W_")] if ep is not None else [],
                mediator=None if name == "Y_s_1h" else "M_first_hour",
                a0_note={"action": "A := 0 (no crossing)", "support": "controls with A=0 exist in the same pre-band"})
            rec = {"dossier_id": doc["dossier_id"], "rung1_diagnostics": r1.get("diagnostics"),
                   "rung2_diagnostics": (r2 or {}).get("diagnostics"), "rung2_sensitivity": (r2 or {}).get("sensitivity"),
                   "crossing": cinfo, "rung3_rows_n": len(rows), "candidate_card": card}
            jdump(rec, os.path.join(out, "records", doc["dossier_id"] + ".json"))
        feat_records.append({"feature_id": fid, "crossing": cinfo})
    cost["features_s"] = time.time() - t0

    # ---------------- economic events -> EURUSD
    t0 = time.time()
    try:
        evs, pseudo, einfo = event_episodes(X, Y, a.inputs, a.lane_a_code, train_end)
    except Exception as trouble:  # recorded, not hidden: the event study is then PENDING
        evs, pseudo, einfo = {}, {}, {"error": f"{type(trouble).__name__}: {trouble}"}
    keys = list(evs)[: a.max_event_types] if a.max_event_types else list(evs)
    for key in keys:
        ep = evs[key]
        wc = [c for c in ep.columns if c.startswith("W_")]
        ctx = {"publication_clock": "ASSUMED_SCHEDULED_PUBLICATION", "expectation_kind": "PUBLISHED_CONSENSUS"}
        for name, family, head, minutes in TARGETS + [("M_impact", "Y_s", "short", 60)]:
            r2, r3, rows = safe_run_cell(ep.dropna(subset=["W_lag_A_same_type"]) if ep["W_lag_A_same_type"].notna().sum() > 60
                                    else ep.drop(columns=["W_lag_A_same_type"]), target=name,
                                    mediator=None if name in ("M_impact",) else "M_impact",
                                    w_cols=[c for c in wc if c != "W_lag_A_same_type" or ep[c].notna().sum() > 60],
                                    context=ctx, pseudo=pseudo.get(key), kind="CONTINUOUS")
            r1 = r2["rung1"]
            a_vals = ep["A"].to_numpy(float)
            treat = {"name": "standardized consensus surprise",
                     "definition": "A = (actual_initial - prior consensus) / TRAIN MAD scale of the release type; "
                                   "A = 0 means published as expected; contrast +1 scale unit vs 0",
                     "kind": "CONTINUOUS", "standardization": "MAD*1.4826 of TRAIN surprises of this release type "
                                                              f"({einfo['scales_train_mad'][key]:.6g})",
                     "dose_support": {"min": float(np.min(a_vals)), "max": float(np.max(a_vals)), "n": int(len(a_vals)),
                                      "quantiles": {f"q{int(q * 100):02d}": float(np.quantile(a_vals, q))
                                                    for q in (0.01, 0.1, 0.5, 0.9, 0.99)}},
                     "sequential_treatment_policy": "SEPARATED_EPISODES_ONLY"}
            src = [{"role": "calendar", "resource": "lane A input economic_calendar_2011_2021.csv (feature-eng archive)",
                    "sha256": dig["inputs_sha256"].get("economic_calendar_2011_2021.csv", "0" * 64), "rows": int(len(ep)),
                    "governed": False},
                   {"role": "bars", "resource": "laneA:targets_train.parquet", "sha256": checked["targets_train.parquet"],
                    "rows": int(len(Y)), "governed": False}]
            lim = ["archive calendar: provenance UNKNOWN, clock measured by era, availability ASSUMED scheduled+1min",
                   "hourly outcomes start at the first decision row at/after availability; the impact bar is M_impact",
                   "co-released surprises at the same instant adjusted as W (compound release); same-type releases "
                   "within 24h before are excluded"]
            extra = {"population": f"releases of {key} with consensus, TRAIN", "event_type": key, "sources": src,
                     "n_episodes": len(ep), "exclusions": {k: int(v) for k, v in einfo["exclusions"].items()},
                     "limitations": lim}
            mx = {"publication_clock": "ASSUMED_SCHEDULED_PUBLICATION", "consensus_clock": "ASSUMED_BEFORE_RELEASE",
                  "expectation_kind": "PUBLISHED_CONSENSUS"}
            doc = emit(key, "event", name, family, head, minutes, r1, r2, r3, treat, mx, extra)
            pc = next((e for e in r1.get("evidence", []) if e["measure"] == "partial_corr_given_H" and e["regime"] is None), None)
            oof = next((e["value"] for e in r1.get("evidence", []) if e["measure"].startswith("oof_relative")), None)
            cells.append({"subject_kind": "event", "subject": key, "family": "economic_event", "target": name,
                          "dossier_id": doc["dossier_id"], "rung1_raw": r1["state"], "rung2_raw": r2["state"],
                          "rung3_raw": r3["state"], "r1_partial_corr": pc["value"] if pc else None,
                          "r1_p": pc["p"] if pc else None, "r1_oof_gain": oof,
                          "r1_reasons": ";".join(filter(None, [(r1.get("diagnostics") or {}).get("reason", ""),
                                                              *[x for x in (r2 or {}).get("reasons", [])
                                                                if str(x).startswith("EXECUTION_FAILED")]])),
                          "r1_placebo": (r1.get("diagnostics") or {}).get("placebo_state"),
                          "r2_reasons": ";".join(r2.get("reasons", [])), "r3_reasons": ";".join(r3.get("reasons", [])),
                          "r2_estimand": r2.get("estimand"), "r2_support": (r2.get("support") or {}).get("state"),
                          "r2_n_per_side": json.dumps((r2.get("support") or {}).get("n_per_side")),
                          "r2_balance_max_smd": (r2.get("support") or {}).get("balance_max_smd"),
                          "r2_placebo": (r2.get("placebo") or {}).get("state"),
                          "r2_estimate": (r2.get("estimate") or {}).get("value"),
                          "r2_interval": json.dumps((r2.get("estimate") or {}).get("interval"))})
            pvals.append(pc["p"] if pc else None)
            card = candidate_card(
                question=(f"How would {name} have responded had the published {key} surprise been +1 TRAIN-scale unit "
                          f"instead of 0 (as expected), for comparable releases? Same-episode: had it been 0?"),
                treatment=treat, target=name, history=wc, r2=r2, r3=r3, w_cols=wc,
                mediator=None if name == "M_impact" else "M_impact",
                a0_note={"action": "A := 0 (published as expected)", "support": treat["dose_support"]})
            jdump({"dossier_id": doc["dossier_id"], "rung1_diagnostics": r1.get("diagnostics"),
                   "rung2_diagnostics": r2.get("diagnostics"), "rung2_placebo": r2.get("placebo"),
                   "rung2_support": r2.get("support"), "candidate_card": card},
                  os.path.join(out, "records", doc["dossier_id"] + ".json"))
    cost["events_s"] = time.time() - t0

    # ---------------- multiplicity, summary states, report
    qs = st.bh_q(pvals)
    for c, q in zip(cells, qs):
        c["r1_q_bh_batch"] = q
        c["rung1"] = summary_state(1, c["rung1_raw"], c.get("r1_reasons", ""))
        c["r1_robust_association"] = bool(c["rung1"] == "ESTIMATED" and q is not None and q <= 0.05
                                          and c["r1_oof_gain"] is not None and c["r1_oof_gain"] > 0)
        c["rung2"] = summary_state(2, c["rung2_raw"], c["r2_reasons"])
        c["rung3"] = summary_state(3, c["rung3_raw"], c["r3_reasons"])
    # write q back into the dossiers' rung-1 evidence
    for c in cells:
        path = os.path.join(out, "dossiers", c["dossier_id"] + ".json")
        doc = json.load(open(path))
        for e in doc["rung1"].get("evidence", []):
            if e["measure"] == "partial_corr_given_H" and e["regime"] is None:
                e["q"] = c["r1_q_bh_batch"]
        if "multiplicity" in doc["rung1"]:
            doc["rung1"]["multiplicity"]["n_tests"] = int(sum(p is not None for p in pvals)) or 1
        errs = ps3c.validate_dossier(doc)
        c["schema_errors"] = len(errs)
        c["schema_first_error"] = errs[0] if errs else ""
        jdump(doc, path)
    summ = pd.DataFrame(cells)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    by_subject = defaultdict(lambda: {"rung1": Counter(), "rung2": Counter(), "rung3": Counter()})
    for c in cells:
        for r in ("rung1", "rung2", "rung3"):
            by_subject[(c["subject_kind"], c["subject"])][r][c[r]] += 1
    STATES = ("ESTIMATED", "NOT_IDENTIFIED", "NOT_APPLICABLE", "FAILED", "PENDING")
    names = {r: {s_: sorted({c["subject"] for c in cells if c[r] == s_}) for s_ in STATES} for r in ("rung1", "rung2", "rung3")}
    names["rung1_robust_association"] = sorted({c["subject"] for c in cells if c["r1_robust_association"]})
    subj_rows = [{"subject_kind": k[0], "subject": k[1], **{r: dict(v[r]) for r in ("rung1", "rung2", "rung3")}}
                 for k, v in sorted(by_subject.items())]
    jdump(subj_rows, os.path.join(out, "summary_by_subject.json"))
    cost["wall_s"] = time.time() - t_all
    cost["peak_rss_kb_self"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    schema_note = ("validated against the vendored causal_dossier.v1 schema plus ONE declared local extension: the "
                   "asset_appearance branch CONTRACTED_BUSINESS_CONTRACT (lane A EURUSD business contract). The "
                   "upstream predictor schema only knows the C127 census appearance, so a consumer of the unmodified "
                   "schema fails closed on these dossiers until the contract owner adds the branch")
    report = {
        "schema": "laneC_ps3c_batch_report.v1", "batch": a.batch, "producer_revision": a.revision,
        "lane_a": {"batch_dir": os.path.basename(os.path.normpath(a.lane_a_batch)), "ready": ready,
                   "digests_sha256": lane_a_dig, "verified_artifacts": checked, "code_commit": dig.get("code_commit")},
        "train_period": contract["periods"]["train"], "test_read": False,
        "denominators": {"features_admissible_in": int(sum(1 for m in meta if str(m.get("admissibility", "")).startswith("ADMISSIBLE"))),
                         "features_role_feature_evaluated": len(feats), "targets": len(TARGETS),
                         "event_types": len(keys), "event_episodes": {k: int(len(evs[k])) for k in keys},
                         "event_pseudo_episodes": {k: int(len(pseudo.get(k, []))) for k in keys},
                         "cells": len(cells), "hourly_rows": int(len(X)), "inner_folds": len(folds)},
        "events": {k: v for k, v in einfo.items() if k != "keys"},
        "crossing": {r["feature_id"]: r["crossing"] for r in feat_records},
        "per_rung_counts": {r: dict(Counter(c[r] for c in cells)) for r in ("rung1", "rung2", "rung3")},
        "per_rung_raw_counts": {r: dict(Counter(c[f"{r}_raw"] for c in cells)) for r in ("rung1", "rung2", "rung3")},
        "subject_names_per_state": names,
        "summary_state_rule": {"rung1": "ESTIMATED = association measured (ASSOCIATION_REPORTED; never causal); "
                                        "robust_association additionally needs batch BH q<=0.05 on the partial "
                                        "correlation given H AND mean OOF MSE gain > 0",
                               "rung2": "ESTIMATED = IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS",
                               "rung3": "ESTIMATED = COUNTERFACTUAL_UNDER_DECLARED_SCM",
                               "NOT_APPLICABLE": "known-in-advance calendar features are not observed interventions",
                               "FAILED": "execution error or missing inputs (error recorded)",
                               "PENDING": "not run", "note": "NOT_IDENTIFIED is never a rejection and never an effect"},
        "scope": "selector evidence only (PS3-C); no adapter feeds calendar, surprises or dossiers to the predictor, "
                 "core, RL or M5PHET (I11, deferred)",
        "endogenous_indicators": "treatment = predeclared historical transition (first available crossing of the TRAIN "
                                 "q80 threshold), never do(indicator=value); its upstream mechanism is the cross-fitted "
                                 "propensity of the transition on the pre-row W (records + report.crossing)",
        "schema_validation": {"dossiers": len(cells), "with_errors": int(sum(c["schema_errors"] > 0 for c in cells)),
                              "note": schema_note},
        "cost": cost, "host_role": "worker_a", "hardware": "CPU only (CUDA_VISIBLE_DEVICES empty)",
    }
    jdump(report, os.path.join(out, "batch_report.json"))
    arts = {}
    for root, _, files in os.walk(out):
        for n in sorted(files):
            if n in ("digests.json", "READY"):
                continue
            p = os.path.join(root, n)
            arts[os.path.relpath(p, out)] = sha(p)
    jdump({"artifacts_sha256": arts, "producer_revision": a.revision, "lane_a_digests_sha256": lane_a_dig},
          os.path.join(out, "digests.json"))
    with open(os.path.join(out, "READY"), "w") as f:
        f.write(json.dumps({"batch": a.batch, "digests_sha256": sha(os.path.join(out, "digests.json")),
                            "written_utc": pd.Timestamp.now(tz="UTC").isoformat()}) + "\n")
    print(json.dumps({"per_rung_counts": report["per_rung_counts"], "cells": len(cells), "cost": cost}, default=str))


if __name__ == "__main__":
    main()
