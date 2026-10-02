"""Train-only descriptive diagnostics for the association/intervention/counterfactual ladder.

This module deliberately does not fit a model or identify an effect.  It only
summarizes evidence already available by a declared training cutoff.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timezone
from math import isfinite, sqrt
from statistics import fmean
from typing import Any, Mapping, Sequence


REPORT_KIND = "train_only_causal_ladder_feasibility.v1"


def _timestamp(value: Any, field: str, row_number: int) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"row {row_number} has an invalid {field} timestamp") from error
    else:
        raise ValueError(f"row {row_number} is missing {field} timestamp")
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _numeric(value: Any, field: str, row_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {row_number} has a non-numeric {field} value")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"row {row_number} has a non-finite {field} value")
    return value


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean, y_mean = fmean(xs), fmean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    denominator = sqrt(sum(value * value for value in x_centered) * sum(value * value for value in y_centered))
    return None if denominator == 0 else sum(x * y for x, y in zip(x_centered, y_centered)) / denominator


def _records(data: Sequence[Mapping[str, Any]] | Any) -> list[Mapping[str, Any]]:
    if hasattr(data, "to_dict") and callable(data.to_dict):
        data = data.to_dict("records")
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence) or not data:
        raise ValueError("data must be a non-empty sequence of records")
    if any(not isinstance(row, Mapping) for row in data):
        raise ValueError("every data row must be a mapping")
    return list(data)


def build_train_only_feasibility_report(
    data: Sequence[Mapping[str, Any]] | Any,
    *,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    event_time: str,
    available_at: str,
    train_end: datetime | date | str,
    intervention_values: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Describe train-only evidence without estimating or labelling an effect causal.

    Every row must have event and availability timestamps no later than
    ``train_end``.  ``intervention_values`` is optional, but when supplied it
    makes missing historical strata a hard error instead of silently ignoring
    them.  Covariate overlap is range-based observed support, not a propensity
    model and not proof of counterfactual support.
    """
    names = [treatment, outcome, event_time, available_at, *covariates]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("column names must be non-empty strings")
    if len(set([treatment, outcome, *covariates])) != len([treatment, outcome, *covariates]):
        raise ValueError("treatment, outcome, and covariates must be distinct")
    cutoff = _timestamp(train_end, "train_end", 0)
    rows = _records(data)
    required = set(names)
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"row {index} is missing required columns: {', '.join(sorted(missing))}")
        observed_at = _timestamp(row[event_time], event_time, index)
        availability_at = _timestamp(row[available_at], available_at, index)
        if observed_at > cutoff:
            raise ValueError(f"row {index} has a future {event_time} timestamp beyond train_end")
        if availability_at > cutoff:
            raise ValueError(f"row {index} was unavailable by train_end")
        prepared.append({
            "treatment": row[treatment],
            "outcome": _numeric(row[outcome], outcome, index),
            "covariates": {name: _numeric(row[name], name, index) for name in covariates},
            "event_time": observed_at,
            "available_at": availability_at,
        })

    observed_values = list(dict.fromkeys(row["treatment"] for row in prepared))
    expected_values = list(intervention_values) if intervention_values is not None else observed_values
    if len(expected_values) < 2:
        raise ValueError("at least two intervention strata are required")
    if len(set(expected_values)) != len(expected_values):
        raise ValueError("intervention_values must be unique")
    strata = {value: [row for row in prepared if row["treatment"] == value] for value in expected_values}
    empty = [value for value, members in strata.items() if not members]
    if empty:
        raise ValueError(f"empty intervention strata: {empty!r}")
    unexpected = [value for value in observed_values if value not in strata]
    if unexpected:
        raise ValueError(f"observed treatment values are not declared intervention strata: {unexpected!r}")

    reference = expected_values[0]
    reference_mean = fmean(row["outcome"] for row in strata[reference])
    historical_strata = []
    association_candidates = []
    for value in expected_values:
        members = strata[value]
        outcome_mean = fmean(row["outcome"] for row in members)
        historical_strata.append({
            "intervention": value,
            "n_observed": len(members),
            "outcome_mean": outcome_mean,
            "event_time_min": min(row["event_time"] for row in members).isoformat(),
            "event_time_max": max(row["event_time"] for row in members).isoformat(),
            "available_at_max": max(row["available_at"] for row in members).isoformat(),
        })
        if value != reference:
            association_candidates.append({
                "contrast": {"reference": reference, "comparison": value},
                "observed_mean_difference": outcome_mean - reference_mean,
                "interpretation": "descriptive association only; not a causal effect",
            })

    overlap: dict[str, Any] = {"method": "observed covariate-range intersection", "covariates": []}
    all_covariates_overlap = bool(covariates)
    for name in covariates:
        ranges = {
            value: {"min": min(row["covariates"][name] for row in members), "max": max(row["covariates"][name] for row in members)}
            for value, members in strata.items()
        }
        lower = max(value_range["min"] for value_range in ranges.values())
        upper = min(value_range["max"] for value_range in ranges.values())
        has_overlap = lower <= upper
        all_covariates_overlap = all_covariates_overlap and has_overlap
        overlap["covariates"].append({"name": name, "stratum_ranges": ranges, "common_range": [lower, upper] if has_overlap else None, "has_observed_range_overlap": has_overlap})
    overlap["status"] = "observed_range_overlap" if all_covariates_overlap else "no_observed_range_overlap"
    overlap["interpretation"] = "Observed support is a diagnostic, not evidence that counterfactual outcomes or identification assumptions hold."

    treatment_numbers = [row["treatment"] for row in prepared]
    numeric_treatment = all(
        not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(float(value))
        for value in treatment_numbers
    )
    association = _pearson([float(value) for value in treatment_numbers], [row["outcome"] for row in prepared]) if numeric_treatment else None
    return {
        "report_kind": REPORT_KIND,
        "conclusion": "feasibility diagnostics only; no causal effect was estimated",
        "association_candidates": {
            "treatment_outcome_pearson": association,
            "pearson_available": numeric_treatment,
            "historical_mean_contrasts": association_candidates,
        },
        "observed_historical_intervention_strata": historical_strata,
        "counterfactual_support_overlap": overlap,
        "diagnostics": {
            "n_train_rows": len(prepared),
            "train_end": cutoff.isoformat(),
            "event_time_column": event_time,
            "availability_time_column": available_at,
            "availability_checked": "all retained rows were available no later than train_end",
            "models_trained": 0,
        },
        "limitations": [
            "Association and observed support do not establish exchangeability, consistency, positivity, or a causal effect.",
            "This report uses only rows available by train_end and does not use future or unavailable observations.",
        ],
    }
