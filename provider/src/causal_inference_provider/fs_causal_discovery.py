"""FS-CAUSAL discovery comparator stage: PCMCI+ (tigramite, GPL-3.0-or-later, pinned at import) per candidate.

    python -m causal_inference_provider.fs_causal_discovery pcmci --out OUT [--tau-max 6] [--pc-alpha 0.05] [--alpha 0.01]

A screening comparator under declared assumptions, never extra identification evidence. For each candidate x
the system is [x, r, H4] where r = the next-hour log return (lane-A target Y_s_1h at the decision row, realised
strictly after x) and H4 = px.logret_24h, px.logret_120h, px.ewma_vol_24, px.ewma_vol_168 at the decision row.
PCMCI+ (Runge 2020) with the linear partial-correlation CI test (analytic p), tau 0..tau_max, pc_alpha for the
PC1 condition-selection phase, alpha_level for the reported links. Rows with any missing value are masked
through tigramite's missing_flag. Lag-0 links x o-o r are time-ordered by construction (x is a bar-end value, r
the return of the following hour) and are reported with the orientation PCMCI+ assigns, never re-oriented here.

Resumable per chunk (``pcmci.json`` + ``READY_PCMCI`` inside each chunk dir); ``pcmci_plus.jsonl`` (one line per
candidate) is rewritten from the chunk files at the end of every chunk; ``progress_discovery.json`` carries
done/total, ETA and failures. If tigramite cannot be imported the stage records NOT_RUN with the reason and exits
non-zero; it never fakes a graph. ARROW is not run (accelerator only, no base method needed acceleration).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
import traceback

import numpy as np
import pandas as pd

from . import fs_causal as fc
from . import fs_causal_batch as FB
from . import ps3c

H4 = ["px.logret_24h", "px.logret_120h", "px.ewma_vol_24", "px.ewma_vol_168"]
MISSING_FLAG = 999.0


def tigramite_pin():
    try:
        import importlib.metadata as md
        import tigramite  # noqa: F401
        ver = md.version("tigramite")
        lic = md.metadata("tigramite").get("License")
        return {"library": "tigramite", "version": ver, "license": lic, "method": "PCMCI+ (run_pcmciplus) with ParCorr",
                "reference": "Runge (2020) UAI; Runge et al. (2019) Sci. Adv."}
    except Exception as trouble:  # pragma: no cover - environment without tigramite
        return {"library": "tigramite", "error": f"{type(trouble).__name__}: {trouble}"}


def pcmci_feature(x, r, H, *, tau_max, pc_alpha, alpha, var_names):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    data = np.column_stack([x, r, H]).astype(float)
    bad = ~np.all(np.isfinite(data), axis=1)
    data[bad] = MISSING_FLAG
    df = pp.DataFrame(data, var_names=var_names, missing_flag=MISSING_FLAG)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    res = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    graph, val, pval = res["graph"], res["val_matrix"], res["p_matrix"]
    ix, ir = 0, 1
    links_x_r, links_r_x = [], []
    for tau in range(tau_max + 1):
        g = str(graph[ix, ir, tau])
        if g and g != "" and pval[ix, ir, tau] <= alpha:
            links_x_r.append({"lag": tau, "edge": g, "val": float(val[ix, ir, tau]), "p": float(pval[ix, ir, tau])})
        if tau > 0:
            g2 = str(graph[ir, ix, tau])
            if g2 and g2 != "" and pval[ir, ix, tau] <= alpha:
                links_r_x.append({"lag": tau, "edge": g2, "val": float(val[ir, ix, tau]), "p": float(pval[ir, ix, tau])})
    directed = [l for l in links_x_r if l["edge"] == "-->"]
    return {"state": "RUN", "n_rows": int((~bad).sum()), "n_masked": int(bad.sum()), "tau_max": tau_max, "pc_alpha": pc_alpha,
            "alpha_level": alpha, "variables": var_names, "links_x_to_r": links_x_r, "links_r_to_x": links_r_x,
            "verdict": ("PCMCI_PLUS_DIRECTED_LINK_X_TO_R" if directed else
                        "PCMCI_PLUS_UNDIRECTED_OR_CONFLICTING_LINK" if links_x_r else "PCMCI_PLUS_NO_LINK_X_TO_R"),
            "scope": "screening comparator on the 1h-return series with H4 only; linear ParCorr; not identification evidence"}


def run_pcmci(out, tau_max=6, pc_alpha=0.05, alpha=0.01, max_features=None):
    plan_doc = json.load(open(os.path.join(out, "plan.json")))
    inputs = plan_doc["inputs_root"]
    pin = tigramite_pin()
    if "error" in pin:
        FB.jdump({"stage": "pcmci_plus", "state": "NOT_RUN", "reason": pin["error"], "updated_utc": FB.now_utc()},
                 os.path.join(out, "progress_discovery.json"))
        raise SystemExit("REFUSED: tigramite not importable; PCMCI+ comparator NOT_RUN (recorded)")
    pending = [c for c in plan_doc["chunks"] if os.path.exists(os.path.join(out, c["id"], "READY"))
               and not os.path.exists(os.path.join(out, c["id"], "READY_PCMCI"))]
    X_base, Y, contract, folds_doc, folds, train_end = FB.load_base(inputs, plan_doc["base"])
    r = Y["Y_s_1h"].to_numpy(float)
    H = X_base[[c for c in H4 if c in X_base]].to_numpy(float)
    hn = [c for c in H4 if c in X_base]
    done_feats = 0
    for chunk in pending:
        t0 = time.time()
        feats = chunk["features"]
        X = X_base if chunk["batch"] == plan_doc["base"] else FB.load_extension_columns(inputs, chunk["batch"], X_base, feats)
        recs = []
        for fid in feats:
            if max_features and done_feats >= max_features:
                break
            t1 = time.time()
            try:
                hh = H if fid not in hn else X_base[[c for c in hn if c != fid]].to_numpy(float)
                names = [fid, "r_next_1h", *[c for c in hn if c != fid]]
                rec = pcmci_feature(X[fid].to_numpy(float), r, hh, tau_max=tau_max, pc_alpha=pc_alpha, alpha=alpha, var_names=names)
            except Exception as trouble:
                rec = {"state": "FAILED", "error": f"{type(trouble).__name__}: {trouble}"[:500], "traceback": traceback.format_exc()[-1500:]}
            rec.update(feature_id=fid, batch=chunk["batch"], pin=pin, cost_s=time.time() - t1, seed=fc.SEED)
            recs.append(rec)
            done_feats += 1
        if max_features and done_feats >= max_features and len(recs) < len(feats):
            break  # partial chunk: not marked READY_PCMCI
        FB.jdump({"chunk": chunk["id"], "records": recs, "cost": {"wall_s": time.time() - t0,
                  "peak_rss_kb_self": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, "written_utc": FB.now_utc()},
                 os.path.join(out, chunk["id"], "pcmci.json"))
        with open(os.path.join(out, chunk["id"], "READY_PCMCI"), "w") as f:
            f.write(json.dumps({"chunk": chunk["id"], "sha256": FB.sha(os.path.join(out, chunk["id"], "pcmci.json")),
                                "written_utc": FB.now_utc()}) + "\n")
        write_discovery_progress(out)
    return write_discovery_progress(out)


def write_discovery_progress(out):
    plan_doc = json.load(open(os.path.join(out, "plan.json")))
    chunks = plan_doc["chunks"]
    done = [c for c in chunks if os.path.exists(os.path.join(out, c["id"], "READY_PCMCI"))]
    recs, durations = [], []
    for c in done:
        d = json.load(open(os.path.join(out, c["id"], "pcmci.json")))
        recs += d["records"]
        durations.append(d["cost"]["wall_s"])
    with open(os.path.join(out, "pcmci_plus.jsonl"), "w") as f:
        for rec in recs:
            f.write(json.dumps(rec, default=ps3c._json_default) + "\n")
    n_done = sum(len(c["features"]) for c in done)
    per = (sum(durations) / n_done) if n_done else None
    rest = plan_doc["candidates_total"] - n_done
    prog = {"schema": "fs_causal_discovery_progress.v1", "stage": "pcmci_plus", "updated_utc": FB.now_utc(),
            "chunks": {"done": len(done), "total": len(chunks)}, "candidates": {"done": n_done, "total": plan_doc["candidates_total"]},
            "verdicts": dict(pd.Series([r.get("verdict", r.get("state")) for r in recs]).value_counts()) if recs else {},
            "failures": [{"feature_id": r["feature_id"], "error": r.get("error")} for r in recs if r.get("state") == "FAILED"][:50],
            "mean_s_per_candidate": per, "eta_s": (rest * per) if per else None,
            "eta_utc": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=rest * per)).strftime("%Y-%m-%dT%H:%M:%SZ") if per else None,
            "complete": len(done) == len(chunks), "pin": recs[0]["pin"] if recs else tigramite_pin()}
    FB.jdump(prog, os.path.join(out, "progress_discovery.json"))
    if prog["complete"]:
        with open(os.path.join(out, "READY_PCMCI"), "w") as f:
            f.write(json.dumps({"sha256": FB.sha(os.path.join(out, "pcmci_plus.jsonl")), "written_utc": FB.now_utc()}) + "\n")
    return prog


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pcmci"); p.add_argument("--out", required=True); p.add_argument("--tau-max", type=int, default=6)
    p.add_argument("--pc-alpha", type=float, default=0.05); p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--max-features", type=int, default=None)
    a = ap.parse_args(argv)
    prog = run_pcmci(a.out, tau_max=a.tau_max, pc_alpha=a.pc_alpha, alpha=a.alpha, max_features=a.max_features)
    print(json.dumps({k: prog[k] for k in ("chunks", "candidates", "verdicts", "eta_utc")}, default=str))


if __name__ == "__main__":
    main()
