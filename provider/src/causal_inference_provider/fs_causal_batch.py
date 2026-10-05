"""FS-CAUSAL batch runner: one unattended, resumable pass over ALL lane-A candidates x targets x three rungs.

    python -m causal_inference_provider.fs_causal_batch plan     --inputs PS1_ROOT --out OUT --revision REV [--block 8]
    python -m causal_inference_provider.fs_causal_batch run      --out OUT [--max-chunks N] [--permutations 200]
    python -m causal_inference_provider.fs_causal_batch finalize --out OUT
    python -m causal_inference_provider.fs_causal_batch progress --out OUT

``plan`` enumerates every role=feature ADMISSIBLE candidate of batch_001 (base) and the extension batches
002/003 (minus the SELECTOR_EPISODE_SOURCE overlay, which are episode locators and never candidates),
groups them into feature blocks per lane-A batch and freezes ``plan.json`` (denominators, digests, seed).
``run`` processes every chunk without a READY marker, in plan order, one at a time; each chunk writes
``cells.jsonl`` (one record per feature x target with the three raw rungs and the discovery screens),
``features.json`` (episode sets, thresholds, upstream mechanism), ``digests.json`` and READY, then rewrites
``progress.json`` (done/total, per-state provisional counts, failures, ETA from the observed median chunk time).
Killing the process at any point loses at most the current chunk. ``finalize`` applies Benjamini-Hochberg
per declared family (rung x target), assigns SUPPORTED / CONTRADICTED / NOT_IDENTIFIED, writes
``causal_evidence.jsonl`` (one line per feature x target), the summary tables, digests and READY.

TRAIN only: every lane-A artifact is verified against READY + digests.json; nothing at or after the contract's
TRAIN end is read; the external validation and the sealed test are never touched. One seed (1729).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import fs_causal as fc
from . import ps3c
from . import ps3c_batch as B
from . import ps3c_review as R
from . import ps3c_stats as st

BASE_COLS = ["t_decision_utc", "row_id", *fc.H_BASE, *fc.PRE_RETURNS, *fc.CALENDAR_LOCATORS]
PROVISIONAL = ("rung1_raw", "rung2_raw", "rung3_raw")


def sha(path):
    return B.sha(path)


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=ps3c._json_default)
    os.replace(tmp, path)


def now_utc():
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def clock_of(m):
    return B.clock_of(m)


# ----------------------------------------------------------------------------------------------------- plan


def candidates_of(bdir, overlay_sources):
    meta = json.load(open(os.path.join(bdir, "admissible_features.json")))["features"]
    return [m for m in meta if m.get("role") == "feature" and str(m.get("admissibility", "")).startswith("ADMISSIBLE")
            and m["feature_id"] not in overlay_sources]


def plan(inputs, out, revision, block=8, batches=("batch_001", "batch_002", "batch_003"), base="batch_001"):
    os.makedirs(out, exist_ok=True)
    overlay = set()
    for b in batches:
        for n in os.listdir(os.path.join(inputs, b)):
            if n.startswith("role_overlay") and n.endswith(".json"):
                doc = json.load(open(os.path.join(inputs, b, n)))
                overlay |= set(doc.get("selector_episode_source_features", []))
    chunks, denominators, verified = [], {}, {}
    for b in batches:
        bdir = os.path.join(inputs, b)
        req = ("features_train.parquet", "targets_train.parquet", "folds.json", "contract.json", "admissible_features.json") \
            if b == base else ("features_train.parquet", "admissible_features.json")
        ready, dig, checked = B.verify_batch(bdir, required=req)
        verified[b] = {"ready": ready, "artifacts_sha256": checked, "digests_sha256": sha(os.path.join(bdir, "digests.json")),
                       "code_commit": dig.get("code_commit")}
        if b != base and dig.get("base_batch_digests_sha256") != verified[base]["digests_sha256"]:
            raise SystemExit(f"REFUSED: {b} is not bound to {base}")
        import pyarrow.parquet as pq
        cols = set(pq.read_schema(os.path.join(bdir, "features_train.parquet")).names)
        feats = [m for m in candidates_of(bdir, overlay) if m["feature_id"] in cols]
        denominators[b] = {"admissible_role_feature": len(candidates_of(bdir, set())),
                           "episode_sources_excluded": len([m for m in candidates_of(bdir, set()) if m["feature_id"] in overlay]),
                           "candidates": len(feats)}
        for i in range(0, len(feats), block):
            chunks.append({"id": f"chunk_{len(chunks):03d}", "batch": b, "base": base,
                           "features": [m["feature_id"] for m in feats[i:i + block]],
                           "meta": {m["feature_id"]: {"family": m.get("family"), "availability_time": m.get("availability_time"),
                                                      "clock": clock_of(m), "source": m.get("source")} for m in feats[i:i + block]}})
    total_cells = sum(len(c["features"]) for c in chunks) * len(fc.TARGETS)
    doc = {"schema": "fs_causal_plan.v1", "revision": revision, "created_utc": now_utc(), "inputs_root": os.path.abspath(inputs),
           "seed": fc.SEED, "block": block, "batches": list(batches), "base": base, "verified": verified,
           "overlay_episode_sources": sorted(overlay), "denominators": denominators,
           "candidates_total": sum(len(c["features"]) for c in chunks), "targets": [t for t, *_ in fc.TARGETS],
           "cells_total": total_cells, "rungs": 3, "chunks": chunks,
           "families": {"rung1": "per target: HAC partial-test p over all candidates",
                        "rung2": "per target: p_linear over all cells with an identified estimate",
                        "sypi_condition1": "per target: condition-1 p over all candidates"},
           "state_rule": fc.__doc__.split("State semantics")[1].split("Calendar columns")[0].strip()}
    jdump(doc, os.path.join(out, "plan.json"))
    write_progress(out)
    return doc


# ------------------------------------------------------------------------------------------------------ run


def load_base(inputs, base):
    X, Y, contract, folds_doc, folds, meta, train_end = B.load_batch(os.path.join(inputs, base))
    return X, Y, contract, folds_doc, folds, train_end


def load_extension_columns(inputs, b, X_base, features):
    bdir = os.path.join(inputs, b)
    Xn = pd.read_parquet(os.path.join(bdir, "features_train.parquet"), columns=["row_id", "t_decision_utc", *features])
    Xn["t_decision_utc"] = pd.to_datetime(Xn["t_decision_utc"], utc=True)
    if len(Xn) != len(X_base) or not (Xn["row_id"].to_numpy() == X_base["row_id"].to_numpy()).all() or not (
            Xn["t_decision_utc"].to_numpy() == X_base["t_decision_utc"].to_numpy()).all():
        raise SystemExit("REFUSED: extension batch rows differ from the base batch")
    keep = [c for c in BASE_COLS if c in X_base and c not in Xn]
    return pd.concat([Xn, X_base[keep]], axis=1)


def _episode_sets(X, Y, fid, clock, targets=None, history_columns=None, pre_return_columns=None,
                  calendar_locator_columns=None, mediator_target="Y_s_1h",
                  volatility_regime_column="px.ewma_vol_168", placebo_outcome_column="px.logret_24h"):
    """One episode frame per distinct horizon class (gap >= max(24, h))."""
    targets = fc.TARGETS if targets is None else targets
    sets = {}
    if clock == "KNOWN_IN_ADVANCE" or fid.startswith("cal."):
        return sets, {"reason": "KNOWN_IN_ADVANCE_CALENDAR_NOT_AN_OBSERVED_INTERVENTION"}
    for h in sorted({max(24, horizon) for _, _, _, horizon in targets}):
        ep, info = fc.crossing_episodes_h(
            X, Y, fid, h,
            locators=fc.CALENDAR_LOCATORS if calendar_locator_columns is None else calendar_locator_columns,
            targets=targets,
            history_columns=history_columns,
            pre_return_columns=pre_return_columns,
            mediator_target=mediator_target,
            volatility_regime_column=volatility_regime_column,
            placebo_outcome_column=placebo_outcome_column,
        )
        if ep is not None:
            info["upstream_mechanism"] = B.upstream_mechanism(ep, [c for c in ep.columns if c.startswith("W_")])
        sets[h] = (ep, info)
    return sets, {}


def run_feature(fid, meta, X, Y, folds, permutations, y_sd, *, targets=None,
                history_columns=None, pre_return_columns=None, calendar_locator_columns=None,
                mediator_target="Y_s_1h", volatility_regime_column="px.ewma_vol_168",
                placebo_outcome_column="px.logret_24h"):
    """Run the unchanged three-rung method for one feature.

    The optional declarations make the existing scientific implementation usable
    by a generic target pack. Omitting them preserves the historical EURUSD path
    byte-for-byte: its module constants remain the defaults.
    """
    t0 = time.time()
    targets = fc.TARGETS if targets is None else targets
    history_columns = fc.H_BASE if history_columns is None else history_columns
    pre_return_columns = fc.PRE_RETURNS if pre_return_columns is None else pre_return_columns
    clock = meta["clock"]
    hist = [c for c in history_columns if c in X and c != fid]
    H = X[hist].to_numpy(float)
    pool_names = [c for c in pre_return_columns if c in X and c != fid]
    pool = X[pool_names].to_numpy(float) if pool_names else None
    a = X[fid].to_numpy(float)
    sets, na = _episode_sets(
        X, Y, fid, clock, targets=targets, history_columns=history_columns,
        pre_return_columns=pre_return_columns, calendar_locator_columns=calendar_locator_columns,
        mediator_target=mediator_target, volatility_regime_column=volatility_regime_column,
        placebo_outcome_column=placebo_outcome_column,
    )
    cells, feature_rec = [], {"feature_id": fid, **meta, "episode_sets": {}}
    for h, (ep, info) in sets.items():
        feature_rec["episode_sets"][str(h)] = {k: v for k, v in info.items()}
    if na:
        feature_rec["episode_sets"]["not_applicable"] = na
    for name, family, head, horizon in targets:
        if name not in Y:
            continue
        y = Y[name].to_numpy(float)
        cell = {"feature_id": fid, "batch": meta.get("batch"), "family": meta.get("family"), "clock": clock,
                "target": name, "target_family": family, "head": head, "horizon_h": horizon, "seed": fc.SEED}
        try:
            cell["rung1"] = fc.rung1_cell(a, y, H, horizon_h=horizon, folds=folds, hist_names=hist, pool=pool,
                                          pool_names=pool_names, permutations=permutations)
            # the previous realised target value (available at t) joins the SyPI conditioning set
            yprev = fc._shift_rows(y, max(1, horizon))
            S = np.column_stack([H, yprev])
            cell["discovery"] = {"sypi": fc.sypi_conditions(a, y, S, horizon_h=horizon),
                                 "pcmci_plus": {"state": "PENDING", "stage": "fs_causal_discovery (separate stage)"},
                                 "arrow": {"state": "NOT_RUN", "reason": "accelerator only; no base method needed acceleration"}}
        except Exception as trouble:  # recorded, never replaced by a number
            cell["rung1"] = {"raw_state": "EXECUTION_FAILED", "error": f"{type(trouble).__name__}: {trouble}"[:500]}
            cell["discovery"] = {"sypi": {"state": "NOT_RUN", "reason": "RUNG1_FAILED"}}
        hk = max(24, horizon)
        ep, info = sets.get(hk, (None, na or {"reason": "NO_EPISODE_SET"}))
        if ep is None:
            r2 = {"state": "NOT_EVALUATED", "reasons": [info.get("reason", "NO_EPISODES")],
                  "estimand": "ATE of a first available TRAIN-q80 crossing vs staying below, both from the pre-row band [q60,q80) (not evaluated)"}
            cell["rung2"] = fc.rung2_summary(r2, y_sd_train=y_sd.get(name))
            cell["rung2"]["nonlinear"] = None
            cell["rung3"] = fc.rung3_summary({"state": "NOT_EVALUATED", "label": "NONE", "reasons": [info.get("reason", "NO_EPISODES")]}, 0)
        else:
            try:
                r2 = fc.rung2_cell(ep, fid=fid, target=name, horizon_h=horizon, clock=clock, info=info)
                cell["rung2"] = fc.rung2_summary(r2, y_sd_train=y_sd.get(name))
                w_cols = [c for c in ep.columns if c.startswith("W_")]
                if r2["state"] == ps3c.IDENTIFIED:
                    nl = R.nonlinear_aipw(ep, name, w_cols)
                    nl["confirmation_identity"] = R.confirmation_identity(
                        {k: (r2.get("population") or {}).get(k) for k in ("population_n", "population_sha256", "estimand_id")}, nl)
                    cell["rung2"]["nonlinear"] = nl
                    r3, n3 = fc.rung3_cell(ep, target=name, r2_state=r2["state"], w_cols=w_cols,
                                           mediator_target=mediator_target)
                else:
                    cell["rung2"]["nonlinear"] = None
                    r3, n3 = {"state": ps3c.NOT_IDENTIFIED, "label": "NONE", "reasons": ["RUNG2_NOT_IDENTIFIED"]}, 0
                cell["rung3"] = fc.rung3_summary(r3, n3)
            except Exception as trouble:
                err = f"EXECUTION_FAILED:{type(trouble).__name__}: {trouble}"[:500]
                cell["rung2"] = fc.rung2_summary({"state": "NOT_EVALUATED", "reasons": [err], "estimand": "not evaluated"}, y_sd_train=y_sd.get(name))
                cell["rung2"]["nonlinear"] = None
                cell["rung3"] = fc.rung3_summary({"state": "NOT_EVALUATED", "label": "NONE", "reasons": [err]}, 0)
                cell["traceback"] = traceback.format_exc()[-1500:]
        cell["rung1_raw"] = cell["rung1"].get("raw_state")
        cell["rung2_raw"] = cell["rung2"].get("raw_state")
        cell["rung3_raw"] = cell["rung3"].get("raw_state")
        cells.append(cell)
    feature_rec["cost_s"] = time.time() - t0
    return cells, feature_rec


def run(out, max_chunks=None, permutations=200, only_features=None):
    plan_doc = json.load(open(os.path.join(out, "plan.json")))
    inputs = plan_doc["inputs_root"]
    pending = [c for c in plan_doc["chunks"] if not os.path.exists(os.path.join(out, c["id"], "READY"))]
    if max_chunks:
        pending = pending[:max_chunks]
    if not pending:
        write_progress(out)
        return 0
    X_base, Y, contract, folds_doc, folds, train_end = load_base(inputs, plan_doc["base"])
    y_sd = {t: float(np.nanstd(Y[t].to_numpy(float))) for t, *_ in fc.TARGETS if t in Y}
    for chunk in pending:
        t0 = time.time()
        cdir = os.path.join(out, chunk["id"])
        os.makedirs(cdir, exist_ok=True)
        feats = chunk["features"] if not only_features else [f for f in chunk["features"] if f in only_features]
        if chunk["batch"] == plan_doc["base"]:
            X = X_base
        else:
            X = load_extension_columns(inputs, chunk["batch"], X_base, feats)
        cells, frecs, failures = [], [], []
        for fid in feats:
            meta = {**chunk["meta"][fid], "batch": chunk["batch"]}
            try:
                cs, fr = run_feature(fid, meta, X, Y, folds, permutations, y_sd)
            except Exception as trouble:
                failures.append({"feature_id": fid, "error": f"{type(trouble).__name__}: {trouble}"[:500],
                                 "traceback": traceback.format_exc()[-2000:]})
                continue
            cells += cs
            frecs.append(fr)
        with open(os.path.join(cdir, "cells.jsonl"), "w") as f:
            for c in cells:
                f.write(json.dumps(c, default=ps3c._json_default) + "\n")
        jdump({"chunk": chunk["id"], "batch": chunk["batch"], "features": frecs, "failures": failures,
               "cost": {"wall_s": time.time() - t0, "peak_rss_kb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
               "train_end": str(train_end), "test_read": False, "seed": fc.SEED, "permutations": permutations,
               "written_utc": now_utc()}, os.path.join(cdir, "features.json"))
        arts = {n: sha(os.path.join(cdir, n)) for n in sorted(os.listdir(cdir)) if n not in ("READY", "digests.json")}
        jdump({"artifacts_sha256": arts, "revision": plan_doc["revision"]}, os.path.join(cdir, "digests.json"))
        with open(os.path.join(cdir, "READY"), "w") as f:
            f.write(json.dumps({"chunk": chunk["id"], "digests_sha256": sha(os.path.join(cdir, "digests.json")),
                                "written_utc": now_utc()}) + "\n")
        write_progress(out)
    return 0


# ------------------------------------------------------------------------------------------------- progress


def _read_cells(out, chunk_ids):
    for cid in chunk_ids:
        p = os.path.join(out, cid, "cells.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    yield json.loads(line)


def write_progress(out):
    plan_doc = json.load(open(os.path.join(out, "plan.json")))
    chunks = plan_doc["chunks"]
    done = [c for c in chunks if os.path.exists(os.path.join(out, c["id"], "READY"))]
    done_ids = {c["id"] for c in done}
    counts = {r: Counter() for r in PROVISIONAL}
    n_cells, failures, durations = 0, [], []
    for c in done:
        fj = json.load(open(os.path.join(out, c["id"], "features.json")))
        durations.append(fj["cost"]["wall_s"])
        failures += [{"chunk": c["id"], **x} for x in fj.get("failures", [])]
    for cell in _read_cells(out, [c["id"] for c in done]):
        n_cells += 1
        for r in PROVISIONAL:
            counts[r][str(cell.get(r))] += 1
    total_cells = plan_doc["cells_total"]
    pend = [c for c in chunks if c["id"] not in done_ids]
    med = float(np.median(durations)) if durations else None
    per_feature = (sum(durations) / max(sum(len(c["features"]) for c in done), 1)) if durations else None
    eta_s = (sum(len(c["features"]) for c in pend) * per_feature) if per_feature is not None else None
    final = os.path.exists(os.path.join(out, "READY"))
    prog = {"schema": "fs_causal_progress.v1", "updated_utc": now_utc(), "stage": "FINALIZED" if final else ("RUNNING" if pend else "CHUNKS_DONE_AWAITING_FINALIZE"),
            "chunks": {"done": len(done), "total": len(chunks)},
            "cells": {"done": n_cells, "total": total_cells, "failed_features": len(failures)},
            "candidates": {"done": sum(len(c["features"]) for c in done), "total": plan_doc["candidates_total"]},
            "per_state_provisional_raw": {r: dict(v) for r, v in counts.items()},
            "failures": failures[:50],
            "median_chunk_s": med, "mean_s_per_candidate": per_feature,
            "eta_s": eta_s, "eta_utc": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=eta_s)).strftime("%Y-%m-%dT%H:%M:%SZ") if eta_s is not None else None,
            "host_role": "worker_b", "hardware": "CPU only, single BLAS thread, crispdm-run capped",
            "seed": fc.SEED, "test_read": False}
    if final:
        fin = json.load(open(os.path.join(out, "final_summary.json")))
        prog["per_state_final"] = fin["per_state"]
    jdump(prog, os.path.join(out, "progress.json"))
    return prog


# ------------------------------------------------------------------------------------------------- finalize


def finalize(out):
    plan_doc = json.load(open(os.path.join(out, "plan.json")))
    chunks = plan_doc["chunks"]
    missing = [c["id"] for c in chunks if not os.path.exists(os.path.join(out, c["id"], "READY"))]
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} chunks not READY ({missing[:5]}...)")
    cells = list(_read_cells(out, [c["id"] for c in chunks]))
    # discovery comparator (separate, optional stage): merge PCMCI+ records into the Y_s_1h cell of each candidate
    pcmci_path = os.path.join(out, "pcmci_plus.jsonl")
    pcmci = {}
    if os.path.exists(pcmci_path):
        with open(pcmci_path) as f:
            for line in f:
                rec = json.loads(line)
                pcmci[rec["feature_id"]] = rec
    for c in cells:
        d = c.setdefault("discovery", {})
        if c["target"] == "Y_s_1h" and c["feature_id"] in pcmci:
            d["pcmci_plus"] = {k: v for k, v in pcmci[c["feature_id"]].items() if k != "traceback"}
        elif c["target"] == "Y_s_1h":
            d["pcmci_plus"] = {"state": "PENDING", "stage": "fs_causal_discovery (separate stage, not yet run)"}
        else:
            d["pcmci_plus"] = {"state": "NOT_RUN", "reason": "PCMCI+ comparator is run on the 1h-return series only"}
    by_target = defaultdict(list)
    for c in cells:
        by_target[c["target"]].append(c)
    # ---- BH per declared family, then states
    for tgt, cs in by_target.items():
        q1 = st.bh_q([c["rung1"].get("p") if c["rung1"].get("raw_state") == "ASSOCIATION_REPORTED" else None for c in cs])
        q2 = st.bh_q([c["rung2"].get("p_linear") for c in cs])
        qs = st.bh_q([(c.get("discovery") or {}).get("sypi", {}).get("condition1_p") if (c.get("discovery") or {}).get("sypi", {}).get("state") == "RUN" else None for c in cs])
        for c, a, b, s_ in zip(cs, q1, q2, qs):
            fc.assign_rung1_state(c["rung1"], a)
            r1sign = c["rung1"].get("sign") if c["rung1"].get("state") == fc.SUPPORTED else None
            fc.assign_rung2_state(c["rung2"], b, c["rung2"].get("nonlinear"), r1sign)
            r2sign = int(np.sign(c["rung2"]["estimate"]["value"])) if c["rung2"].get("estimate") else None
            fc.assign_rung3_state(c["rung3"], c["rung2"]["state"], r2sign)
            if (c.get("discovery") or {}).get("sypi"):
                fc.assign_sypi_state(c["discovery"]["sypi"], s_)
            c["abstention_reason"] = {r: c[r].get("abstention_reason") for r in ("rung1", "rung2", "rung3")}
            c["multiplicity"] = {"rung1": {"family": f"rung1:{tgt}", "n_tests": int(sum(x is not None for x in q1)), "q": a},
                                 "rung2": {"family": f"rung2:{tgt}", "n_tests": int(sum(x is not None for x in q2)), "q": b},
                                 "sypi_condition1": {"family": f"sypi1:{tgt}", "n_tests": int(sum(x is not None for x in qs)), "q": s_}}
    # ---- outputs
    ev_path = os.path.join(out, "causal_evidence.jsonl")
    with open(ev_path, "w") as f:
        for c in sorted(cells, key=lambda c: (c["feature_id"], c["target"])):
            f.write(json.dumps(c, default=ps3c._json_default) + "\n")
    per_state = {r: {tgt: dict(Counter(c[r]["state"] for c in cs)) for tgt, cs in sorted(by_target.items())} for r in ("rung1", "rung2", "rung3")}
    per_state_total = {r: dict(Counter(c[r]["state"] for c in cells)) for r in ("rung1", "rung2", "rung3")}
    rows = []
    for c in cells:
        rows.append({"feature_id": c["feature_id"], "batch": c["batch"], "family": c["family"], "clock": c["clock"], "target": c["target"],
                     "horizon_h": c["horizon_h"],
                     **{f"{r}_state": c[r]["state"] for r in ("rung1", "rung2", "rung3")},
                     **{f"{r}_robust": c[r].get("robust") for r in ("rung1", "rung2", "rung3")},
                     **{f"{r}_reason": c[r].get("abstention_reason") or c[r].get("contradiction_kind") for r in ("rung1", "rung2", "rung3")},
                     "r1_coef": c["rung1"].get("coef"), "r1_t_hac": c["rung1"].get("t_hac"), "r1_p": c["rung1"].get("p"), "r1_q": c["rung1"].get("q"),
                     "r1_oof_gain": c["rung1"].get("oof_gain"), "r1_oof_positive_folds": c["rung1"].get("oof_positive_folds"),
                     "r1_best_extra_lag": c["rung1"].get("best_extra_lag"), "r1_mss": json.dumps((c["rung1"].get("minimal_separating_set") or {}).get("set")),
                     "r2_estimate": (c["rung2"].get("estimate") or {}).get("value"), "r2_interval": json.dumps((c["rung2"].get("estimate") or {}).get("interval")),
                     "r2_q": c["rung2"].get("q"), "r2_support": (c["rung2"].get("support") or {}).get("state"),
                     "r2_n_per_side": json.dumps((c["rung2"].get("support") or {}).get("n_per_side")),
                     "r2_balance_max_smd": (c["rung2"].get("support") or {}).get("balance_max_smd"),
                     "r2_placebo": (c["rung2"].get("placebo") or {}).get("state"), "r2_rv_q1": (c["rung2"].get("sensitivity") or {}).get("robustness_value_q1"),
                     "r2_nonlinear_state": (c["rung2"].get("nonlinear") or {}).get("state"), "r2_nonlinear_estimate": (c["rung2"].get("nonlinear") or {}).get("estimate"),
                     "r3_delta": (c["rung3"].get("prediction") or {}).get("delta"), "r3_analog": (c["rung3"].get("sensitivity") or {}).get("analog_state"),
                     "sypi": ((c.get("discovery") or {}).get("sypi") or {}).get("verdict")})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "cells_summary.csv"), index=False)
    feat_rows = []
    for fid, cs in sorted(defaultdict(list, {f: [c for c in cells if c["feature_id"] == f] for f in {c["feature_id"] for c in cells}}).items()):
        agg = fc.feature_weight_against(cs)
        feat_rows.append({"feature_id": fid, "batch": cs[0]["batch"], "family": cs[0]["family"], "clock": cs[0]["clock"], **agg,
                          **{f"{r}_{s_}": sum(1 for c in cs if c[r]["state"] == s_) for r in ("rung1", "rung2", "rung3") for s_ in fc.STATES}})
    pd.DataFrame(feat_rows).to_csv(os.path.join(out, "feature_summary.csv"), index=False)
    sup = tab[(tab.rung1_state != fc.NOT_IDENTIFIED) | (tab.rung2_state != fc.NOT_IDENTIFIED) | (tab.rung3_state != fc.NOT_IDENTIFIED)]
    sup.to_csv(os.path.join(out, "supported_contradicted_cells.csv"), index=False)
    reasons = {r: dict(Counter(x for c in cells for x in str(c[r].get("abstention_reason") or "").split(";") if x)) for r in ("rung1", "rung2", "rung3")}
    summary = {"schema": "fs_causal_final_summary.v1", "revision": plan_doc["revision"], "finalized_utc": now_utc(),
               "candidates": len(feat_rows), "cells": len(cells), "targets": len(by_target), "per_state": per_state_total,
               "per_state_per_target": per_state, "abstention_reasons": reasons,
               "robust_contradicted_features": sorted({c["feature_id"] for c in cells if any(c[r]["state"] == fc.CONTRADICTED and c[r].get("robust") for r in ("rung1", "rung2", "rung3"))}),
               "supported_features_any_rung": sorted({c["feature_id"] for c in cells if any(c[r]["state"] == fc.SUPPORTED for r in ("rung1", "rung2", "rung3"))}),
               "rung2_supported_features": sorted({c["feature_id"] for c in cells if c["rung2"]["state"] == fc.SUPPORTED}),
               "rung3_supported_features": sorted({c["feature_id"] for c in cells if c["rung3"]["state"] == fc.SUPPORTED}),
               "sypi": dict(Counter(((c.get("discovery") or {}).get("sypi") or {}).get("verdict") for c in cells)),
               "pcmci_plus": dict(Counter(((c.get("discovery") or {}).get("pcmci_plus") or {}).get("verdict", ((c.get("discovery") or {}).get("pcmci_plus") or {}).get("state")) for c in cells if c["target"] == "Y_s_1h")),
               "clock_distribution": dict(Counter(c["clock"] for c in cells if c["target"] == cells[0]["target"])),
               "families": plan_doc["families"], "fdr_q": fc.FDR_Q, "seed": fc.SEED, "test_read": False,
               "not_rejection": "NOT_IDENTIFIED never eliminates a feature; only robust CONTRADICTED weighs against one"}
    jdump(summary, os.path.join(out, "final_summary.json"))
    arts = {n: sha(os.path.join(out, n)) for n in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, n)) and n not in ("READY", "digests.json", "progress.json")}
    for c in chunks:
        arts[f"{c['id']}/READY"] = sha(os.path.join(out, c["id"], "READY"))
    jdump({"artifacts_sha256": arts, "revision": plan_doc["revision"]}, os.path.join(out, "digests.json"))
    with open(os.path.join(out, "READY"), "w") as f:
        f.write(json.dumps({"digests_sha256": sha(os.path.join(out, "digests.json")), "written_utc": now_utc(), "cells": len(cells)}) + "\n")
    write_progress(out)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--inputs", required=True); p.add_argument("--out", required=True)
    p.add_argument("--revision", required=True); p.add_argument("--block", type=int, default=8)
    p.add_argument("--batches", default="batch_001,batch_002,batch_003")
    r = sub.add_parser("run"); r.add_argument("--out", required=True); r.add_argument("--max-chunks", type=int, default=None)
    r.add_argument("--permutations", type=int, default=200); r.add_argument("--only-features", default=None)
    f = sub.add_parser("finalize"); f.add_argument("--out", required=True)
    g = sub.add_parser("progress"); g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        doc = plan(a.inputs, a.out, a.revision, block=a.block, batches=tuple(a.batches.split(",")))
        print(json.dumps({"chunks": len(doc["chunks"]), "candidates": doc["candidates_total"], "cells": doc["cells_total"]}))
    elif a.cmd == "run":
        run(a.out, max_chunks=a.max_chunks, permutations=a.permutations,
            only_features=set(a.only_features.split(",")) if a.only_features else None)
        print(json.dumps(write_progress(a.out)["cells"]))
    elif a.cmd == "finalize":
        s = finalize(a.out)
        print(json.dumps({"per_state": s["per_state"], "cells": s["cells"]}))
    elif a.cmd == "progress":
        print(json.dumps(write_progress(a.out), indent=1))


if __name__ == "__main__":
    main()
