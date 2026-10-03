"""PS3-C review of a finished ladder batch: join to lane B's PS2 candidates, multiplicity, nonlinear check.

    python -m causal_inference_provider.ps3c_review --ps3c-batch DIR --ps2-batch DIR --lane-a-batch DIR \
        [--base-batch DIR] --out-join FILE --revision REV

1. Join: every PS2 candidate (tiers 1-3 + exploration, and the non-prioritised rest) is tagged per
   target/horizon with its three rung states, reasons, rung-2 estimand/support/estimate and the
   dossier's ``causal_evidence_level``. A candidate with no dossier is PENDING, never dropped.
2. Multiplicity: Benjamini-Hochberg over a DECLARED family = every rung-2 ESTIMATED cell of the batch
   (not only the cells whose interval excluded 0). p is the two-sided normal p of the estimate with
   se = (interval width) / 3.92 from the moving-block bootstrap (declared approximation).
3. Nonlinear adjustment check, TRAIN-only, for BH survivors: cross-fitted gradient-boosted propensity
   and per-arm outcome models (scikit-learn HistGradientBoosting) inside the same AIPW; overlap
   population [0.05, 0.95] declared; block-bootstrap interval of the influence values. A cell
   SURVIVES when its linear estimate is BH-significant AND the nonlinear interval excludes 0 with
   the same sign. Nothing here changes a NOT_IDENTIFIED cell, and nothing is a rejection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from collections import Counter, defaultdict
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import ps3c_batch as B
from . import ps3c_stats as st

CELL_MAP = {**{f"Y_s|h{h}": f"Y_s_{h}h" for h in (1, 2, 3, 4, 5, 6)},
            **{f"Y_l|h{h}": f"Y_l_{h}h" for h in (24, 48, 72, 96, 120, 144)},
            "Y_b_s6|h6": "Y_b_s6", "Y_b_l144|h144": "Y_b_l144"}


def p_from_interval(est, lo, hi):
    se = (hi - lo) / 3.92
    if not se > 0:
        return None
    return float(2 * (1 - NormalDist().cdf(abs(est) / se)))


def _crossfit(model_fn, x, y, k=5, classify=False, rows=None):
    n = len(y)
    out = np.empty(n)
    edges = np.linspace(0, n, k + 1).astype(int)
    for j in range(k):
        te = np.arange(edges[j], edges[j + 1])
        tr = np.setdiff1d(np.arange(n), te)
        if rows is not None:
            tr = tr[rows[tr]]
        m = model_fn()
        m.fit(x[tr], y[tr])
        out[te] = m.predict_proba(x[te])[:, 1] if classify else m.predict(x[te])
    return out


def nonlinear_aipw(ep, target, w_cols, seed=1729, n_boot=200, bounds=(0.05, 0.95), min_side=20):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    d = ep.dropna(subset=["A", target, *w_cols]).sort_values("decision_time")
    t = d["A"].to_numpy(float)
    y = d[target].to_numpy(float)
    w = d[w_cols].to_numpy(float)
    if min(t.sum(), (1 - t).sum()) < min_side:
        return {"state": "NO_COMMON_SUPPORT", "n_per_side": [int(t.sum()), int((1 - t).sum())]}
    kw = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=seed)
    e = np.clip(_crossfit(lambda: HistGradientBoostingClassifier(**kw), w, t, classify=True), 1e-6, 1 - 1e-6)
    inside = (e >= bounds[0]) & (e <= bounds[1])
    t, y, w, e = t[inside], y[inside], w[inside], e[inside]
    if min(t.sum(), (1 - t).sum()) < min_side:
        return {"state": "OVERLAP_SCREEN_FAILED", "dropped_outside_overlap": int((~inside).sum())}
    mu1 = _crossfit(lambda: HistGradientBoostingRegressor(**kw), w, y, rows=(t == 1))
    mu0 = _crossfit(lambda: HistGradientBoostingRegressor(**kw), w, y, rows=(t == 0))
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    rng = np.random.default_rng(seed)
    _, (lo, hi) = _boot_mean_se(psi, rng, n_boot)
    return {"state": "ESTIMATED", "estimate": float(np.mean(psi)), "interval": [lo, hi],
            "n_per_side": [int(t.sum()), int((1 - t).sum())], "dropped_outside_overlap": int((~inside).sum()),
            "propensity_range": [float(e.min()), float(e.max())],
            "estimator": "cross-fitted AIPW, HistGradientBoosting propensity and per-arm outcome models",
            "uncertainty": f"moving-block bootstrap of influence values, B={n_boot}"}


def _boot_mean_se(psi, rng, reps=200):
    n = len(psi)
    vals = [float(np.mean(psi[st.block_indices(n, rng)])) for _ in range(reps)]
    return float(np.std(vals)), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps3c-batch", required=True)
    ap.add_argument("--ps2-batch", required=True)
    ap.add_argument("--lane-a-batch", required=True)
    ap.add_argument("--base-batch", default=None)
    ap.add_argument("--out-join", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--q", type=float, default=0.05)
    a = ap.parse_args(argv)
    t_all = time.time()
    ready = json.loads(open(os.path.join(a.ps3c_batch, "READY")).read())
    if B.sha(os.path.join(a.ps3c_batch, "digests.json")) != ready["digests_sha256"]:
        raise SystemExit("REFUSED: ps3c batch digests do not match READY")
    ps2_ready = json.loads(open(os.path.join(a.ps2_batch, "READY")).read())
    summ = pd.read_csv(os.path.join(a.ps3c_batch, "summary.csv"))
    cand_doc = json.load(open(os.path.join(a.ps2_batch, "ps2_candidates_lane_c.json")))
    prio = json.load(open(os.path.join(a.ps2_batch, "ps2_extractor_priority.json")))
    tiers = defaultdict(list)
    for tier in ("tier_1", "tier_2", "tier_3", "exploration"):
        for f in prio.get(tier, []):
            tiers[f].append(tier)

    # ---- multiplicity over the declared family: every rung-2 ESTIMATED cell of this batch
    est = summ[summ.rung2 == "ESTIMATED"].copy()
    iv = est.r2_interval.map(json.loads)
    est["p_linear"] = [p_from_interval(e, v[0], v[1]) for e, v in zip(est.r2_estimate, iv)]
    est["q_bh_family"] = st.bh_q(list(est["p_linear"]))
    est["excludes_zero"] = [v[0] > 0 or v[1] < 0 for v in iv]
    est["bh_significant"] = [q is not None and q <= a.q for q in est["q_bh_family"]]

    # ---- nonlinear check on the BH survivors
    t0 = time.time()
    need = est[est.bh_significant]
    review = {}
    if len(need):
        if a.base_batch:
            Xb, Y, *_ = B.load_batch(a.base_batch)
            _, _, _, X, _ = B.load_extension_batch(a.lane_a_batch, a.base_batch, Xb)
        else:
            X, Y, *_ = B.load_batch(a.lane_a_batch)
        for fid, grp in need.groupby("subject"):
            ep, info = B.crossing_episodes(X, Y, fid)
            wc = [c for c in ep.columns if c.startswith("W_")]
            for _, r in grp.iterrows():
                nl = nonlinear_aipw(ep, r.target, wc)
                surv = (nl.get("state") == "ESTIMATED" and nl["interval"][0] is not None
                        and (nl["interval"][0] > 0 or nl["interval"][1] < 0)
                        and np.sign(nl["estimate"]) == np.sign(r.r2_estimate))
                nl["survives"] = bool(surv)
                review[(fid, r.target)] = nl
    cost_nl = time.time() - t0
    est["nonlinear"] = [review.get((s, t)) for s, t in zip(est.subject, est.target)]
    est["survives"] = [bool(x and x.get("survives")) for x in est["nonlinear"]]
    by_cell = {(r.subject, r.target): r for r in est.itertuples()}

    # ---- join
    out_c = []
    level_counts = defaultdict(Counter)
    state_counts = defaultdict(lambda: defaultdict(Counter))
    for c in cand_doc["candidates"]:
        fid = c["feature_id"]
        rows = summ[summ.subject == fid].set_index("target")
        cells = {}
        for key, ps2cell in c["cells"].items():
            tgt = CELL_MAP.get(key)
            if tgt is None or tgt not in rows.index:
                cell = {"rung1": "PENDING", "rung2": "PENDING", "rung3": "PENDING",
                        "causal_evidence_level": "NOT_EVALUATED", "reason": "NO_DOSSIER_FOR_CELL"}
            else:
                r = rows.loc[tgt]
                doc_level = ("COUNTERFACTUAL_SENSITIVITY" if r.rung3 == "ESTIMATED" else
                             "IDENTIFIED_EFFECT" if r.rung2 == "ESTIMATED" else
                             "ASSOCIATION" if r.rung1 == "ESTIMATED" else "NONE")
                cell = {"target": tgt, "dossier_id": r.dossier_id, "rung1": r.rung1, "rung2": r.rung2,
                        "rung3": r.rung3, "causal_evidence_level": doc_level,
                        "rung1_partial_corr": None if pd.isna(r.r1_partial_corr) else float(r.r1_partial_corr),
                        "rung1_q_bh_batch": None if pd.isna(r.r1_q_bh_batch) else float(r.r1_q_bh_batch),
                        "rung1_robust_association": bool(r.r1_robust_association),
                        "rung2_reasons": [] if pd.isna(r.r2_reasons) else str(r.r2_reasons).split(";"),
                        "rung3_reasons": [] if pd.isna(r.r3_reasons) else str(r.r3_reasons).split(";"),
                        "rung2_estimand": None if pd.isna(r.r2_estimand) else r.r2_estimand,
                        "rung2_support": None if pd.isna(r.r2_support) else r.r2_support,
                        "rung2_n_per_side": None if pd.isna(r.r2_n_per_side) else json.loads(r.r2_n_per_side),
                        "rung2_placebo": None if pd.isna(r.r2_placebo) else r.r2_placebo}
                e = by_cell.get((fid, tgt))
                if e is not None:
                    cell.update(rung2_estimate=float(e.r2_estimate), rung2_interval=json.loads(e.r2_interval),
                                rung2_p_linear=e.p_linear, rung2_q_bh_family=e.q_bh_family,
                                rung2_bh_significant=bool(e.bh_significant), rung2_nonlinear=e.nonlinear,
                                rung2_survives_multiplicity_and_nonlinear=bool(e.survives))
            cell["ps2_status"] = ps2cell.get("status")
            cell["ps2_reasons"] = ps2cell.get("reasons")
            cells[key] = cell
            level_counts[key][cell["causal_evidence_level"]] += 1
            for rung in ("rung1", "rung2", "rung3"):
                state_counts[key][rung][cell[rung]] += 1
        out_c.append({"feature_id": fid, "family": c.get("family"), "ps2_tiers": tiers.get(fid, []),
                      "prioritized": bool(tiers.get(fid)), "cells": cells})
    surv = est[est.survives]
    report = {
        "schema": "laneC_ps2_join.v1", "producer_revision": a.revision,
        "ps3c_batch": os.path.basename(os.path.normpath(a.ps3c_batch)), "ps3c_ready_digests_sha256": ready["digests_sha256"],
        "ps2_batch": cand_doc.get("batch_id"), "ps2_ready": ps2_ready,
        "ps2_candidates_sha256": B.sha(os.path.join(a.ps2_batch, "ps2_candidates_lane_c.json")),
        "ps2_priority_sha256": B.sha(os.path.join(a.ps2_batch, "ps2_extractor_priority.json")),
        "ps2_train_data_digest": cand_doc.get("train_data_digest"),
        "candidates": len(out_c), "prioritized": int(sum(x["prioritized"] for x in out_c)),
        "tier_counts": {t: len(prio.get(t, [])) for t in ("tier_1", "tier_2", "tier_3", "exploration")},
        "candidates_without_any_dossier": [x["feature_id"] for x in out_c
                                           if all(v.get("reason") == "NO_DOSSIER_FOR_CELL" for v in x["cells"].values())],
        "states_per_target": {k: {r: dict(v) for r, v in d.items()} for k, d in sorted(state_counts.items())},
        "causal_evidence_level_per_target": {k: dict(v) for k, v in sorted(level_counts.items())},
        "multiplicity": {"family": "all rung-2 ESTIMATED cells of this ps3c batch", "family_size": int(len(est)),
                         "method": "Benjamini-Hochberg", "q": a.q,
                         "p": "two-sided normal p, se = bootstrap 95% interval width / 3.92 (approximation)",
                         "excluding_zero_uncorrected": int(est.excludes_zero.sum()),
                         "bh_significant": int(est.bh_significant.sum()),
                         "nonlinear_checked": len(review), "survivors": int(len(surv))},
        "survivors": [{"feature_id": r.subject, "target": r.target, "estimand": r.r2_estimand,
                       "linear_estimate": float(r.r2_estimate), "linear_interval": json.loads(r.r2_interval),
                       "q_bh_family": r.q_bh_family, "support": r.r2_support,
                       "n_per_side": json.loads(r.r2_n_per_side) if isinstance(r.r2_n_per_side, str) else None,
                       "balance_max_smd": None if pd.isna(r.r2_balance_max_smd) else float(r.r2_balance_max_smd),
                       "propensity_range": json.loads(r.r2_propensity_range) if isinstance(r.r2_propensity_range, str) else None,
                       "placebo": r.r2_placebo, "robustness_value_q1": None if pd.isna(r.r2_rv_q1) else float(r.r2_rv_q1),
                       "rung3": r.rung3, "nonlinear": r.nonlinear, "dossier_id": r.dossier_id}
                      for r in surv.itertuples()],
        "not_rejection": "NOT_IDENTIFIED and non-surviving cells are not rejections; selector evidence only (I11 deferred)",
        "cost": {"wall_s": time.time() - t_all, "nonlinear_s": cost_nl,
                 "peak_rss_kb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "join": out_c,
    }
    with open(a.out_join, "w") as f:
        json.dump(report, f, indent=1, default=B.ps3c._json_default)
    print(json.dumps({k: report[k] for k in ("candidates", "prioritized", "multiplicity", "cost")}, default=str))


if __name__ == "__main__":
    main()
