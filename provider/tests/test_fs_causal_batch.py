"""FS-CAUSAL batch runner end to end on a synthetic lane-A base + extension batch: plan, resumable run, finalize.

Synthetic bytes only (no market data), CPU, a few thousand rows. Checks: denominators, one evidence line per
feature x target, three states per rung, progress.json fields, chunk-level resumability (a killed run loses at
most one chunk; rerun completes without redoing READY chunks), digests + READY, TEST never read.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from causal_inference_provider import fs_causal as fc
from causal_inference_provider import fs_causal_batch as FB
from test_ps3c_batch import make_lane_a_batch


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def make_lane_a_pair(root, n_hours=24 * 500, seed=5):
    """Base batch (from the PS3-C fixture, with a planted cause added) + an extension batch bound to it."""
    base = make_lane_a_batch(os.path.join(root, "batch_001"), n_hours=n_hours, seed=seed)
    X = pd.read_parquet(os.path.join(base, "features_train.parquet"))
    Y = pd.read_parquet(os.path.join(base, "targets_train.parquet"))
    rng = np.random.default_rng(seed + 1)
    n = len(X)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal()
    thr = np.quantile(x, 0.8)
    cross = np.r_[False, (x[:-1] < thr) & (x[1:] >= thr)]
    # the planted cause moves the next-hour return (and hence every cumulative target) after a crossing
    bump = np.zeros(n)
    bump[1:] = 3e-3 * cross[:-1]
    s_add = np.cumsum(bump)
    for name, fam, head, h in fc.TARGETS:
        if fam != "Y_b":
            Y[name] = Y[name] + (pd.Series(s_add).shift(-h) - pd.Series(s_add)).to_numpy()
    X["ta.cause"] = x
    X.to_parquet(os.path.join(base, "features_train.parquet"), index=False)
    Y.to_parquet(os.path.join(base, "targets_train.parquet"), index=False)
    meta = json.load(open(os.path.join(base, "admissible_features.json")))
    meta["features"].append({"feature_id": "ta.cause", "family": "technical", "source": "synthetic", "role": "feature",
                             "admissibility": "ADMISSIBLE", "availability_time": "t (bar end)"})
    for m in meta["features"]:
        m.setdefault("availability_time", "known in advance" if m["feature_id"].startswith("cal.") else "t (bar end)")
    json.dump(meta, open(os.path.join(base, "admissible_features.json"), "w"))
    # an episode-source overlay: one base column is a locator, never a candidate
    json.dump({"schema": "role_overlay.v1", "applies_to_batch": "batch_001",
               "selector_episode_source_features": ["px.logret_6h"]},
              open(os.path.join(base, "role_overlay_batch_001.json"), "w"))
    arts = {n_: _sha(os.path.join(base, n_)) for n_ in os.listdir(base) if n_ not in ("digests.json", "READY")}
    json.dump({"artifacts_sha256": arts, "inputs_sha256": {}, "code_commit": "synthetic"}, open(os.path.join(base, "digests.json"), "w"))
    open(os.path.join(base, "READY"), "w").write(json.dumps({"digests_sha256": _sha(os.path.join(base, "digests.json"))}))
    # extension batch: two new columns on the same rows, one with an ASSUMED clock
    ext = os.path.join(root, "batch_002")
    os.makedirs(ext)
    Xe = pd.DataFrame({"t_decision_utc": X["t_decision_utc"], "row_id": X["row_id"],
                       "yahoo.spx_change": pd.Series(rng.normal(size=n)).rolling(24, min_periods=1).mean().to_numpy(),
                       "fx.cross_a": pd.Series(rng.normal(size=n)).rolling(6, min_periods=1).mean().to_numpy()})
    Xe.to_parquet(os.path.join(ext, "features_train.parquet"), index=False)
    json.dump({"batch": "batch_002", "features": [
        {"feature_id": "yahoo.spx_change", "family": "yahoo_change", "role": "feature", "admissibility": "ADMISSIBLE",
         "availability_time": "(D+1) 00:00 UTC"},
        {"feature_id": "fx.cross_a", "family": "fx_cross", "role": "feature", "admissibility": "ADMISSIBLE",
         "availability_time": "bar end"}]}, open(os.path.join(ext, "admissible_features.json"), "w"))
    arts = {n_: _sha(os.path.join(ext, n_)) for n_ in os.listdir(ext)}
    json.dump({"artifacts_sha256": arts, "inputs_sha256": {}, "code_commit": "synthetic",
               "base_batch_digests_sha256": _sha(os.path.join(base, "digests.json"))}, open(os.path.join(ext, "digests.json"), "w"))
    open(os.path.join(ext, "READY"), "w").write(json.dumps({"digests_sha256": _sha(os.path.join(ext, "digests.json"))}))
    return root


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    return make_lane_a_pair(str(tmp_path_factory.mktemp("laneA")))


def test_plan_run_resume_finalize(lane, tmp_path):
    out = str(tmp_path / "out")
    doc = FB.plan(lane, out, "deadbeef", block=3, batches=("batch_001", "batch_002"))
    feats = [f for c in doc["chunks"] for f in c["features"]]
    assert "px.logret_6h" not in feats and "ta.cause" in feats and "yahoo.spx_change" in feats and "cal.hour_sin" in feats
    assert doc["cells_total"] == len(feats) * len(fc.TARGETS) and len(doc["chunks"]) >= 3
    prog0 = json.load(open(os.path.join(out, "progress.json")))
    assert prog0["chunks"]["done"] == 0 and prog0["stage"] == "RUNNING"
    # first pass: one chunk only (simulates a kill after the first READY)
    FB.run(out, max_chunks=1, permutations=20)
    prog1 = json.load(open(os.path.join(out, "progress.json")))
    assert prog1["chunks"]["done"] == 1 and prog1["eta_s"] is not None and prog1["cells"]["done"] == 3 * len(fc.TARGETS)
    first_ready = open(os.path.join(out, "chunk_000", "READY")).read()
    with pytest.raises(SystemExit, match="not READY"):
        FB.finalize(out)
    # second pass completes the rest without touching the READY chunk
    FB.run(out, permutations=20)
    assert open(os.path.join(out, "chunk_000", "READY")).read() == first_ready
    prog2 = json.load(open(os.path.join(out, "progress.json")))
    assert prog2["chunks"]["done"] == prog2["chunks"]["total"] and prog2["stage"] == "CHUNKS_DONE_AWAITING_FINALIZE"
    assert prog2["cells"]["done"] == doc["cells_total"] and prog2["test_read"] is False
    summ = FB.finalize(out)
    lines = [json.loads(l) for l in open(os.path.join(out, "causal_evidence.jsonl"))]
    assert len(lines) == doc["cells_total"]
    assert {(l["feature_id"], l["target"]) for l in lines} == {(f, t) for f in feats for t, *_ in fc.TARGETS}
    for l in lines:
        for r in ("rung1", "rung2", "rung3"):
            assert l[r]["state"] in fc.STATES
            assert l[r]["state"] != fc.NOT_IDENTIFIED or l[r]["abstention_reason"]
        assert "multiplicity" in l and l["rung2"]["assumptions_evidence"] is not None or l["rung2"]["raw_state"] == "NOT_EVALUATED"
    by = {(l["feature_id"], l["target"]): l for l in lines}
    cal = [l for l in lines if l["feature_id"] == "cal.hour_sin"]
    assert all(l["rung2"]["state"] == fc.NOT_IDENTIFIED and "KNOWN_IN_ADVANCE" in l["rung2"]["abstention_reason"] for l in cal)
    assumed = [l for l in lines if l["feature_id"] == "yahoo.spx_change"]
    assert all(l["clock"] == "ASSUMED" and l["rung2"]["state"] == fc.NOT_IDENTIFIED for l in assumed)
    assert all("ASSUMED_PUBLICATION_CLOCK" in l["rung2"]["abstention_reason"] for l in assumed if l["rung2"]["raw_state"] != "NOT_EVALUATED")
    cause = by[("ta.cause", "Y_s_1h")]
    assert cause["rung1"]["state"] == fc.SUPPORTED, cause["rung1"]
    assert cause["rung1"]["q"] <= fc.FDR_Q and cause["rung1"]["ci_test"].startswith("HAC(")
    # the rung-2 gate may abstain on this small world (overlap/balance), but it never fabricates: states are named
    assert cause["rung2"]["state"] in fc.STATES and cause["rung2"]["assumptions_unverified"] == []
    if cause["rung2"]["state"] == fc.SUPPORTED:
        assert cause["rung2"]["estimate"]["value"] > 0 and cause["rung2"]["nonlinear"]["state"] == "ESTIMATED"
    osc = [l for l in lines if l["feature_id"] == "ta.osc"]
    assert sum(l["rung1"]["state"] == fc.SUPPORTED for l in osc) <= 1  # a null feature is (almost) never supported
    # summary artifacts, digests, READY
    assert summ["cells"] == doc["cells_total"] and set(summ["per_state"]) == {"rung1", "rung2", "rung3"}
    tab = pd.read_csv(os.path.join(out, "cells_summary.csv"))
    assert len(tab) == doc["cells_total"]
    fs = pd.read_csv(os.path.join(out, "feature_summary.csv"))
    assert len(fs) == len(feats) and set(fs.feature_id) == set(feats) and fs.not_identified_never_eliminates.all()
    dig = json.load(open(os.path.join(out, "digests.json")))
    assert _sha(os.path.join(out, "causal_evidence.jsonl")) == dig["artifacts_sha256"]["causal_evidence.jsonl"]
    ready = json.loads(open(os.path.join(out, "READY")).read())
    assert ready["digests_sha256"] == _sha(os.path.join(out, "digests.json"))
    prog3 = json.load(open(os.path.join(out, "progress.json")))
    assert prog3["stage"] == "FINALIZED" and "per_state_final" in prog3
    # rerunning run() is a no-op once everything is READY
    FB.run(out)
    assert open(os.path.join(out, "READY")).read() == json.dumps(ready) + "\n"
