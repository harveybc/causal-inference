"""Tests for train-only causal-ladder feasibility diagnostics."""

from copy import deepcopy

import pytest

from causal_inference_provider import build_train_only_feasibility_report


@pytest.fixture
def train_rows():
    return [
        {"treatment": 0, "outcome": 1.0, "baseline": 0.2, "event_at": "2025-01-01T09:00:00", "available_at": "2025-01-01T09:01:00"},
        {"treatment": 0, "outcome": 1.5, "baseline": 0.8, "event_at": "2025-01-02T09:00:00", "available_at": "2025-01-02T09:01:00"},
        {"treatment": 1, "outcome": 2.0, "baseline": 0.3, "event_at": "2025-01-03T09:00:00", "available_at": "2025-01-03T09:01:00"},
        {"treatment": 1, "outcome": 2.5, "baseline": 0.7, "event_at": "2025-01-04T09:00:00", "available_at": "2025-01-04T09:01:00"},
    ]


def report(rows):
    return build_train_only_feasibility_report(
        rows,
        treatment="treatment",
        outcome="outcome",
        covariates=["baseline"],
        event_time="event_at",
        available_at="available_at",
        train_end="2025-01-05T00:00:00",
        intervention_values=[0, 1],
    )


def test_reports_the_ladder_without_a_causal_claim(train_rows):
    result = report(train_rows)

    assert result["diagnostics"]["models_trained"] == 0
    assert result["conclusion"] == "feasibility diagnostics only; no causal effect was estimated"
    assert len(result["association_candidates"]["historical_mean_contrasts"]) == 1
    assert [item["n_observed"] for item in result["observed_historical_intervention_strata"]] == [2, 2]
    assert result["counterfactual_support_overlap"]["status"] == "observed_range_overlap"
    assert result["counterfactual_support_overlap"]["covariates"][0]["common_range"] == [0.3, 0.7]


def test_rejects_future_event_timestamp(train_rows):
    rows = deepcopy(train_rows)
    rows[0]["event_at"] = "2025-01-06T00:00:00"

    with pytest.raises(ValueError, match="future event_at"):
        report(rows)


@pytest.mark.parametrize("availability", [None, "", "2025-01-06T00:00:00"])
def test_rejects_missing_or_unavailable_availability_timestamp(train_rows, availability):
    rows = deepcopy(train_rows)
    rows[0]["available_at"] = availability

    with pytest.raises(ValueError, match="available_at|unavailable"):
        report(rows)


def test_rejects_absent_availability_timestamp_column(train_rows):
    rows = deepcopy(train_rows)
    del rows[0]["available_at"]

    with pytest.raises(ValueError, match="missing required columns: available_at"):
        report(rows)


def test_rejects_empty_declared_intervention_stratum(train_rows):
    rows = deepcopy(train_rows)
    for row in rows:
        row["treatment"] = 0

    with pytest.raises(ValueError, match="empty intervention strata"):
        report(rows)
