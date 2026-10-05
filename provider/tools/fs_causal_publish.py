"""Publish FS-CAUSAL run artifacts (mirrored from worker_b) into the predictor evidence directory as small tables + REPORT.md.

    python tools/fs_causal_publish.py --run MIRROR_DIR --dest EVIDENCE_DIR/fs_closure/fs_causal [--interim]

Reads progress.json, plan.json and (when finalized) final_summary.json, cells_summary.csv, feature_summary.csv,
supported_contradicted_cells.csv, digests.json, READY, progress_discovery.json. Writes the same small files plus
REPORT.md with denominators, per-state counts per rung/target, SUPPORTED/CONTRADICTED cells with estimand, support
and diagnostics, abstention reasons, digests and cost. Never writes host names; the host is named by role only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil

import pandas as pd

SMALL = ["progress.json", "plan.json", "final_summary.json", "cells_summary.csv", "feature_summary.csv",
         "supported_contradicted_cells.csv", "digests.json", "READY", "progress_discovery.json", "READY_PCMCI"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def md_table(df, cols=None, n=60):
    if df is None or len(df) == 0:
        return "_none_\n"
    df = df if cols is None else df[cols]
    df = df.head(n)
    out = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |\n"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--interim", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.dest, exist_ok=True)
    copied = {}
    for n in SMALL:
        p = os.path.join(a.run, n)
        if os.path.exists(p):
            if n == "plan.json":  # strip the absolute inputs path (host-local), keep the frozen denominators
                doc = json.load(open(p))
                doc["inputs_root"] = "<worker_b>/.local/state/canonical_20261003/fs_causal/inputs/ps1"
                json.dump(doc, open(os.path.join(a.dest, n), "w"), indent=1)
            else:
                shutil.copy2(p, os.path.join(a.dest, n))
            copied[n] = sha(os.path.join(a.dest, n))
    prog = json.load(open(os.path.join(a.dest, "progress.json")))
    plan = json.load(open(os.path.join(a.dest, "plan.json")))
    final = json.load(open(os.path.join(a.dest, "final_summary.json"))) if "final_summary.json" in copied else None
    lines = [f"# FS-CAUSAL {'interim progress' if a.interim or final is None else 'closure'} report", "",
             f"Generated {prog['updated_utc']} from the worker_b run directory (CPU only, crispdm-run capped, seed {prog['seed']}, TEST never read).",
             f"Code: causal-inference `{plan['revision']}` (`fs_causal.py`, `fs_causal_batch.py`, `fs_causal_discovery.py`).", "",
             "## Denominators", "",
             f"- candidates: **{plan['candidates_total']}** = " + " + ".join(f"{b} {d['candidates']}" for b, d in plan["denominators"].items())
             + f" (episode-source overlay columns excluded: {sum(d['episode_sources_excluded'] for d in plan['denominators'].values())}; they are locators, never candidates)",
             f"- targets: {len(plan['targets'])} ({', '.join(plan['targets'])})",
             f"- cells: **{plan['cells_total']}** feature x target, three rungs each; chunks: {len(plan['chunks'])} (block {plan['block']})",
             f"- multiplicity families: {json.dumps(plan['families'])}", "",
             "## Progress", "",
             f"- stage: **{prog['stage']}**; chunks {prog['chunks']['done']}/{prog['chunks']['total']}; cells {prog['cells']['done']}/{prog['cells']['total']}; "
             f"candidates {prog['candidates']['done']}/{prog['candidates']['total']}; failed features {prog['cells']['failed_features']}",
             f"- median chunk {prog['median_chunk_s'] and round(prog['median_chunk_s'])} s; {prog['mean_s_per_candidate'] and round(prog['mean_s_per_candidate'], 1)} s per candidate; ETA {prog['eta_utc']}",
             f"- provisional raw states (before family BH): {json.dumps(prog['per_state_provisional_raw'])}", ""]
    if "progress_discovery.json" in copied:
        pdisc = json.load(open(os.path.join(a.dest, "progress_discovery.json")))
        lines += ["## Discovery comparator (PCMCI+, screening only)", "",
                  f"- {pdisc.get('stage')}: candidates {pdisc.get('candidates')}, verdicts {json.dumps(pdisc.get('verdicts'))}, ETA {pdisc.get('eta_utc')}, pin {json.dumps(pdisc.get('pin'))}", ""]
    if final:
        tab = pd.read_csv(os.path.join(a.dest, "cells_summary.csv"))
        lines += ["## Final states per rung (all targets)", "", "| rung | SUPPORTED | CONTRADICTED | NOT_IDENTIFIED |", "|---|---:|---:|---:|"]
        for r in ("rung1", "rung2", "rung3"):
            c = final["per_state"][r]
            lines.append(f"| {r} | {c.get('SUPPORTED', 0)} | {c.get('CONTRADICTED', 0)} | {c.get('NOT_IDENTIFIED', 0)} |")
        lines += ["", "## States per rung and target", "", "| target | rung1 S/C/N | rung2 S/C/N | rung3 S/C/N |", "|---|---|---|---|"]
        for tgt in plan["targets"]:
            cells = [final["per_state_per_target"][r].get(tgt, {}) for r in ("rung1", "rung2", "rung3")]
            lines.append(f"| {tgt} | " + " | ".join(f"{c.get('SUPPORTED', 0)}/{c.get('CONTRADICTED', 0)}/{c.get('NOT_IDENTIFIED', 0)}" for c in cells) + " |")
        lines += ["", "## SUPPORTED / CONTRADICTED cells", ""]
        sc = tab[(tab.rung1_state != "NOT_IDENTIFIED") | (tab.rung2_state != "NOT_IDENTIFIED") | (tab.rung3_state != "NOT_IDENTIFIED")]
        r2 = sc[sc.rung2_state != "NOT_IDENTIFIED"]
        lines += [f"Rung 2 (identified historical intervention; estimand: ATE of a first available TRAIN-q80 crossing vs staying below, both from the pre-row band [q60,q80), AIPW with declared W): {len(r2)} cells", "",
                  md_table(r2, ["feature_id", "target", "rung2_state", "rung2_robust", "rung2_reason", "r2_estimate", "r2_interval", "r2_q", "r2_n_per_side", "r2_balance_max_smd", "r2_placebo", "r2_rv_q1", "r2_nonlinear_estimate", "rung3_state", "r3_delta"]),
                  f"Rung 1 (association only, HAC partial test + OOF gain + BH per target): {int((sc.rung1_state != 'NOT_IDENTIFIED').sum())} cells; first 60 by |t|:", "",
                  md_table(sc[sc.rung1_state != "NOT_IDENTIFIED"].assign(abs_t=lambda d: d.r1_t_hac.abs()).sort_values("abs_t", ascending=False),
                           ["feature_id", "target", "rung1_state", "rung1_robust", "rung1_reason", "r1_coef", "r1_t_hac", "r1_q", "r1_oof_gain", "r1_oof_positive_folds", "r1_best_extra_lag", "r1_mss", "sypi"]),
                  "## Abstention reasons", "", "```json", json.dumps(final["abstention_reasons"], indent=1), "```", "",
                  f"- robust CONTRADICTED features (the only ones that may weigh against selection): {final['robust_contradicted_features']}",
                  f"- features SUPPORTED at rung 2: {final['rung2_supported_features']}",
                  f"- features SUPPORTED at rung 3: {final['rung3_supported_features']}",
                  f"- features SUPPORTED at any rung: {len(final['supported_features_any_rung'])}",
                  f"- SyPI screen verdicts: {json.dumps(final['sypi'])}; PCMCI+: {json.dumps(final.get('pcmci_plus'))}",
                  f"- clock distribution of candidates: {json.dumps(final['clock_distribution'])}", ""]
    lines += ["## Rules honoured", "",
              "- TRAIN only: thresholds, bands, regimes, folds, propensities, outcome models and SCMs fitted inside the lane-A TRAIN rows; external validation and sealed test never read.",
              "- Calendar columns are episode locators / W only; nothing feeds a predictor (I11 deferred).",
              "- Endogenous indicators: treatment = predeclared historical transition (first available crossing of the TRAIN q80 threshold from the pre-row band), never do(indicator=value); upstream mechanism = cross-fitted propensity on W recorded per episode set.",
              "- Repaired fail-closed gate (causal-inference 48ae17c) unchanged: declared DAG + back-door check, support, overlap without trimming, balance <= 0.1, placebo battery, four assumptions declared with evidence references (CAUSAL_SUFFICIENCY is DECLARED_WITH_SENSITIVITY_ONLY).",
              "- NOT_IDENTIFIED never eliminates; only robust CONTRADICTED weighs against a feature; the 1,076 historical rung-2 estimates were not reused.",
              "- One seed (1729); compute on worker_b CPU under crispdm-run with the cap bound to the measured pilot peak.", "",
              "## Digests", "", "```json", json.dumps(copied, indent=1), "```"]
    open(os.path.join(a.dest, "REPORT.md"), "w").write("\n".join(lines) + "\n")
    print(json.dumps({"copied": list(copied), "stage": prog["stage"]}))


if __name__ == "__main__":
    main()
