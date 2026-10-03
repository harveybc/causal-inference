"""Train-only feasibility report for the three-stage causal ladder.

Stages: (1) association candidates, (2) observed historical intervention
strata, (3) counterfactual-support overlap.  The report states support and
diagnostics only.  It never labels an effect causal: identification needs
assumptions this code cannot check.  Pure numpy/pandas; no training, no I/O
beyond the optional CLI reading one local CSV.

Required columns: decision_time, available_at, treatment, outcome, plus the
numeric covariates named by the caller.  Point-in-time rule: every row must
satisfy available_at <= decision_time <= train_end.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd


class FeasibilityError(ValueError):
    """Input violates the point-in-time or support preconditions."""


def _validate(df, train_end, covariates, levels):
    need = ["decision_time", "available_at", "treatment", "outcome", *covariates]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise FeasibilityError(f"missing columns: {missing}")
    if df["available_at"].isna().any():
        raise FeasibilityError("missing availability timestamps (available_at)")
    if df["decision_time"].isna().any():
        raise FeasibilityError("missing decision_time")
    out = df.copy()
    for c in ("decision_time", "available_at"):
        out[c] = pd.to_datetime(out[c], utc=True)
    cut = pd.Timestamp(train_end)
    cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
    if (out["decision_time"] > cut).any() or (out["available_at"] > cut).any():
        raise FeasibilityError("future timestamps beyond train_end")
    if (out["available_at"] > out["decision_time"]).any():
        raise FeasibilityError("available_at after decision_time (not point-in-time)")
    if out["treatment"].isna().any() or out["outcome"].isna().any():
        raise FeasibilityError("missing treatment or outcome values")
    observed = set(out["treatment"].unique())
    declared = list(levels) if levels is not None else sorted(observed, key=str)
    empty = [l for l in declared if l not in observed]
    if empty or not declared:
        raise FeasibilityError(f"empty intervention strata: {empty or 'none declared'}")
    if len(declared) < 2:
        raise FeasibilityError("need at least two intervention strata")
    unknown = sorted(observed - set(declared), key=str)
    if unknown:
        raise FeasibilityError(f"undeclared treatment levels: {unknown}")
    return out, declared


def feasibility_report(df, train_end, covariates, levels=None, min_stratum=30,
                       n_bins=4, min_cell=5, top_k=5):
    d, levels = _validate(df, train_end, list(covariates), levels)
    n = len(d)
    # Stage 1: association candidates (association only).
    assoc = []
    for c in covariates:
        x = pd.to_numeric(d[c], errors="coerce")
        y = pd.to_numeric(d["outcome"], errors="coerce")
        ok = x.notna() & y.notna()
        r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 else None
        assoc.append({"covariate": c, "n": int(ok.sum()), "pearson_r": r})
    assoc.sort(key=lambda a: -abs(a["pearson_r"]) if a["pearson_r"] is not None else 0)
    # Stage 2: observed intervention strata.
    counts = d["treatment"].value_counts()
    strata = {str(l): int(counts[l]) for l in levels}
    thin = [l for l, k in strata.items() if k < min_stratum]
    # Stage 3: overlap via train-quantile covariate cells.
    cells = pd.DataFrame(index=d.index)
    for c in covariates:
        cells[c] = pd.qcut(pd.to_numeric(d[c]), n_bins, labels=False, duplicates="drop")
    key = cells.astype(str).agg("|".join, axis=1) if len(covariates) else pd.Series("all", index=d.index)
    tab = pd.crosstab(key, d["treatment"]).reindex(columns=levels, fill_value=0)
    supported = (tab >= min_cell).all(axis=1)
    frac = float(tab[supported].to_numpy().sum() / n)
    overlap_ok = frac > 0 and not thin
    verdict = "support_present_assumptions_unverified" if overlap_ok else "insufficient_support"
    return {
        "scope": "train_only",
        "train_end": str(pd.Timestamp(train_end)),
        "n_rows": n,
        "causal_claim": "not_asserted",
        "verdict": verdict,
        "stage1_association_candidates": {"label": "association_only", "candidates": assoc[:top_k]},
        "stage2_intervention_strata": {"counts": strata, "min_stratum": min_stratum, "thin_strata": thin},
        "stage3_overlap": {"cells": int(len(tab)), "supported_cells": int(supported.sum()),
                           "supported_row_fraction": frac, "min_cell": min_cell},
        "notes": ["Support is necessary, not sufficient: ignorability and positivity "
                  "are assumptions, not checked here.",
                  "No effect estimate is produced or implied."],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", required=True)
    p.add_argument("--train_end", required=True)
    p.add_argument("--covariates", required=True, help="comma separated")
    p.add_argument("--levels", default=None, help="comma separated declared treatment levels")
    p.add_argument("--min_stratum", type=int, default=30)
    a = p.parse_args(argv)
    df = pd.read_csv(a.csv)
    lv = a.levels.split(",") if a.levels else None
    if lv is not None and pd.api.types.is_numeric_dtype(df["treatment"]):
        lv = [type(df["treatment"].iloc[0])(x) for x in lv]
    try:
        rep = feasibility_report(df, a.train_end, a.covariates.split(","), lv, a.min_stratum)
    except FeasibilityError as e:
        print(f"REJECTED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
