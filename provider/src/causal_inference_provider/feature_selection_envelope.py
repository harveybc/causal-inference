"""Producer for the warehouse-owned ``feature_selection_envelope.v1`` contract.

Field names and digest rules mirror data-warehouse commit 50bddf3. This module
only constructs documents; the warehouse remains the authoritative validator.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable


SCHEMA_VERSION = "feature_selection_envelope.v1"
ROW_FAMILIES = (
    "sampling_quality", "variable_profiles", "information_metrics",
    "pair_relations", "causal_evidence", "selection_decisions",
)
PROFILE_STATE = {
    "MEASURED": "MEASURED",
    "FAILED": "FAILED",
    "NOT_APPLICABLE": "NOT_APPLICABLE",
    "PENDING": "INCONCLUSIVE",
    "INCONCLUSIVE": "INCONCLUSIVE",
    "UNAVAILABLE": "UNAVAILABLE",
}
CAUSAL_STATE = {
    "IDENTIFIED": "IDENTIFIED",
    "SUPPORTED": "SUPPORTED",
    "CONTRADICTED": "NOT_SUPPORTED",
    "NOT_SUPPORTED": "NOT_SUPPORTED",
    "NOT_IDENTIFIED": "NOT_IDENTIFIED",
    "INCONCLUSIVE": "INCONCLUSIVE",
    "UNAVAILABLE": "UNAVAILABLE",
    "FAILED": "FAILED",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def seal_row(row: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(row)
    sealed["row_sha256"] = digest(sealed)
    return sealed


def build_envelope(run: dict[str, Any], rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    normalized = {family: [seal_row(row) for row in rows.get(family, [])] for family in ROW_FAMILIES}
    document = {"schema_version": SCHEMA_VERSION, "run": run, "rows": normalized}
    document["envelope_sha256"] = digest(document)
    return document


def _numeric_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, float | int]]:
    if isinstance(value, bool):
        yield prefix, int(value)
    elif isinstance(value, (int, float)) and math.isfinite(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in sorted(value.items()):
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _numeric_leaves(item, name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            name = f"{prefix}.{index}" if prefix else str(index)
            yield from _numeric_leaves(item, name)


def _unit(metric_name: str) -> str | None:
    lowered = metric_name.lower()
    if "fraction" in lowered or "share" in lowered or "eta2" in lowered:
        return "ratio"
    if any(token in lowered for token in ("count", "rows", "nobs", "n_unique", "bins", "bytes")):
        return "count"
    if "seconds" in lowered or lowered.endswith("_s"):
        return "seconds"
    return None


def profile_rows(profile: dict[str, Any], targets: list[tuple[str, int]], population_id: str) -> dict[str, list[dict[str, Any]]]:
    rows = {family: [] for family in ROW_FAMILIES}
    feature_id = profile.get("feature_id", "UNKNOWN")
    for cell in profile.get("cells", []):
        metric = str(cell["metric"])
        state = PROFILE_STATE.get(str(cell.get("state")), "FAILED")
        leaves = list(_numeric_leaves(cell.get("value")))
        if metric in {"missingness", "timestamp_gaps"}:
            family = "sampling_quality"
        elif metric == "information_entropy":
            family = "information_metrics"
        else:
            family = "variable_profiles"
        if family == "information_metrics":
            for target_id, horizon in targets:
                if not leaves:
                    row_state = "INCONCLUSIVE" if state == "MEASURED" else state
                    rows[family].append({"feature_id": feature_id, "target_id": target_id, "horizon": horizon,
                                         "split": "train", "metric_name": metric, "state": row_state,
                                         "population_id": population_id, "fold": None})
                for suffix, numeric in leaves:
                    rows[family].append({"feature_id": feature_id, "target_id": target_id, "horizon": horizon,
                                         "split": "train", "metric_name": f"{metric}.{suffix}",
                                         "metric_value": numeric, "state": state,
                                         "population_id": population_id, "fold": None})
        else:
            if not leaves:
                row_state = "INCONCLUSIVE" if state == "MEASURED" else state
                rows[family].append({"feature_id": feature_id, "split": "train", "metric_name": metric,
                                     "state": row_state, "unit": None,
                                     "population_id": population_id, "fold": None})
            for suffix, numeric in leaves:
                name = f"{metric}.{suffix}"
                rows[family].append({"feature_id": feature_id, "split": "train", "metric_name": name,
                                     "metric_value": numeric, "state": state, "unit": _unit(name),
                                     "population_id": population_id, "fold": None})
    return rows


def pair_relation_rows(feature_id: str, X, Y, target_definitions, frequency: str,
                       population_id: str) -> list[dict[str, Any]]:
    import numpy as np
    from scipy import stats
    import pandas as pd

    step = pd.Timedelta(frequency)
    output = []
    a = X[feature_id].to_numpy(float)
    for target in target_definitions:
        y = Y[target.name].to_numpy(float)
        lag_specs = {0: "0h", 1: frequency, target.horizon_hours: f"{target.horizon_hours}h"}
        unique_lags = {}
        for _, duration in lag_specs.items():
            ratio = pd.Timedelta(duration) / step
            if ratio >= 0 and math.isclose(ratio, round(ratio), abs_tol=1e-12):
                unique_lags[int(round(ratio))] = duration
        for lag, duration in sorted(unique_lags.items()):
            shifted = np.full(len(a), np.nan)
            shifted[lag:] = a[:len(a) - lag] if lag else a
            mask = np.isfinite(shifted) & np.isfinite(y)
            for method in ("pearson", "spearman"):
                row = {"feature_id": feature_id, "target_id": target.name, "horizon": target.horizon_hours,
                       "split": "train", "lag": lag, "metric_name": method,
                       "population_id": population_id, "fold": None}
                if mask.sum() < 3 or np.std(shifted[mask]) == 0 or np.std(y[mask]) == 0:
                    row["state"] = "NOT_APPLICABLE"
                else:
                    value = stats.pearsonr(shifted[mask], y[mask]).statistic if method == "pearson" \
                        else stats.spearmanr(shifted[mask], y[mask]).statistic
                    row.update(metric_value=float(value), state="MEASURED")
                output.append(row)
    return output


def causal_rows(cells: list[dict[str, Any]], population_id: str, *, final: bool) -> list[dict[str, Any]]:
    output = []
    for cell in cells:
        for rung_number, rung_name in enumerate(("rung1", "rung2", "rung3"), start=1):
            evidence = cell[rung_name]
            state = _causal_state(evidence, final=final)
            effect, lower, upper = _effect_interval(rung_number, evidence)
            assumptions = evidence.get("assumptions_evidence") or evidence.get("assumptions") or []
            if isinstance(assumptions, dict):
                assumptions = sorted(map(str, assumptions))
            adjustment = evidence.get("conditioning_set") or evidence.get("adjustment_set") or []
            support_n = evidence.get("effective_rows") or (evidence.get("support") or {}).get("population_n") \
                or evidence.get("episodes_n") or 0
            row = {
                "feature_id": cell["feature_id"], "target_id": cell["target"],
                "horizon": int(cell["horizon_h"]), "split": "train", "rung": rung_number,
                "estimand": str(evidence.get("estimand") or f"causal_ladder_rung_{rung_number}"),
                "estimator": _estimator(rung_number, evidence), "state": state,
                "effect": effect, "lower": lower, "upper": upper, "support_n": int(support_n),
                "assumptions": list(map(str, assumptions)), "adjustment_set": list(map(str, adjustment)),
                "evidence_sha256": digest(evidence), "population_id": population_id, "fold": None,
            }
            output.append(row)
    return output


def _causal_state(evidence: dict[str, Any], *, final: bool) -> str:
    candidate = evidence.get("state") if final else evidence.get("raw_state")
    if candidate in CAUSAL_STATE:
        return CAUSAL_STATE[candidate]
    text = " ".join(map(str, [candidate, evidence.get("abstention_reason"), evidence.get("reasons")]))
    if "FAILED" in text:
        return "FAILED"
    if "NOT_AVAILABLE" in text:
        return "UNAVAILABLE"
    if candidate == "IDENTIFIED":
        return "IDENTIFIED"
    return "NOT_IDENTIFIED"


def _effect_interval(rung: int, evidence: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    if rung == 1:
        return _finite(evidence.get("coef")), None, None
    if rung == 2:
        estimate = evidence.get("estimate") or {}
        interval = estimate.get("interval") or [None, None]
        return _finite(estimate.get("value")), _finite(interval[0]), _finite(interval[1])
    prediction = evidence.get("prediction") or {}
    return _finite(prediction.get("delta")), None, None


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _estimator(rung: int, evidence: dict[str, Any]) -> str:
    if rung == 1:
        return str(evidence.get("ci_test") or "HAC conditional association")
    if rung == 2:
        return "AIPW/g-computation/matching with nonlinear confirmation"
    return str(evidence.get("scm") or "additive-noise temporal SCM")


def selection_rows(cells: list[dict[str, Any]], population_id: str) -> list[dict[str, Any]]:
    output = []
    for cell in cells:
        rungs = [cell[name] for name in ("rung1", "rung2", "rung3")]
        supported = sum(item.get("state") == "SUPPORTED" for item in rungs)
        contradicted = sum(item.get("state") == "CONTRADICTED" and item.get("robust") for item in rungs)
        text = canonical_json(rungs)
        if "NOT_AVAILABLE" in text:
            decision, rule = "UNAVAILABLE", "required phase-1 input unavailable"
        elif contradicted:
            decision, rule = "REJECTED", "at least one robust CONTRADICTED causal rung"
        elif supported:
            decision, rule = "SELECTED", "at least one SUPPORTED rung and no robust contradiction"
        else:
            decision, rule = "NEUTRAL", "causal ladder did not identify support or robust contradiction"
        output.append({
            "feature_id": cell["feature_id"], "target_id": cell["target"],
            "horizon": int(cell["horizon_h"]), "method": "causal_ladder_global_fdr",
            "score": float(supported - contradicted), "rank": None, "decision": decision,
            "rule": rule, "evidence_sha256": digest(cell), "population_id": population_id,
        })
    return output
