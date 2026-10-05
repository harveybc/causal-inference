from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from causal_inference_provider import feature_selection_worker as worker


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_unit(tmp_path: Path, *, feature: str = "signal", missing_target: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T03:00:00Z"]
            ),
            feature: [1.0, 2.0, 4.0],
            "history": [0.1, 0.2, 0.3],
            "target_1h": [0.01, 0.02, 0.03],
        }
    )
    dataset = tmp_path / f"{feature}.csv"
    rows.to_csv(dataset, index=False)
    config = {
        "schema": "feature_selection_unit.v1",
        "unit_id": feature,
        "split": "TRAIN",
        "dataset": {
            "resource_id": f"fixture/{feature}",
            "path": str(dataset),
            "format": "csv",
            "timestamp_column": "time",
            "feature_column": feature,
            "frequency": "1h",
            "calendar": "24x7",
        },
        "feature": {"family": "fixture", "clock": "OBSERVED_AT_DECISION"},
        "target_pack": {
            "id": "ETH",
            "path": str(dataset),
            "format": "csv",
            "timestamp_column": "time",
            "definitions": [
                {
                    "name": "eth_return_1h",
                    "column": "absent" if missing_target else "target_1h",
                    "family": "return",
                    "head": "short",
                    "horizon_hours": 1,
                }
            ],
            "history_columns": ["history"],
            "pre_return_columns": [],
            "calendar_locator_columns": [],
        },
        "folds": [{"train_rows": [0, 2], "validation_rows": [2, 3]}],
        "permutations": 2,
        "output_dir": str(tmp_path / "output"),
    }
    path = tmp_path / f"{feature}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _raw_cells(fid, meta, X, Y, folds, permutations, y_sd, **kwargs):
    cells = []
    for name, family, head, horizon in kwargs["targets"]:
        cells.append(
            {
                "feature_id": fid,
                "batch": "per-column",
                "family": meta["family"],
                "clock": meta["clock"],
                "target": name,
                "target_family": family,
                "head": head,
                "horizon_h": horizon,
                "seed": 1729,
                "rung1": {"raw_state": "ASSOCIATION_REPORTED", "p": 0.5},
                "rung2": {"raw_state": "NOT_EVALUATED", "p_linear": None, "nonlinear": None},
                "rung3": {"raw_state": "NOT_EVALUATED"},
                "discovery": {"sypi": {"state": "NOT_RUN"}},
                "rung1_raw": "ASSOCIATION_REPORTED",
                "rung2_raw": "NOT_EVALUATED",
                "rung3_raw": "NOT_EVALUATED",
            }
        )
    return cells, {"feature_id": fid, **meta, "episode_sets": {}, "cost_s": 0.0}


def test_run_column_is_idempotent_and_profiles_missing_dates(tmp_path, monkeypatch):
    config = _write_unit(tmp_path)
    monkeypatch.setattr(worker.FB, "run_feature", _raw_cells)

    first = worker.run_column(config)
    terminal = Path(first["terminal_path"])
    envelope = Path(first["envelope_path"])
    first_bytes = (terminal.read_bytes(), envelope.read_bytes())
    assert first["status"] == "COMPLETED"
    assert first["global_state"] == "AWAITING_INVENTORY_FINALIZATION"
    assert first["profile"]["sampling"]["missing_timestamps"] == 1
    assert json.loads(envelope.read_text())["schema"] == "feature_selection_envelope.v1"

    second = worker.run_column(config)
    assert second["status"] == "REPLAY"
    assert (terminal.read_bytes(), envelope.read_bytes()) == first_bytes


def test_changed_input_bytes_create_a_new_scientific_identity(tmp_path, monkeypatch):
    config = _write_unit(tmp_path)
    monkeypatch.setattr(worker.FB, "run_feature", _raw_cells)
    first = worker.run_column(config)
    document = json.loads(config.read_text())
    frame = pd.read_csv(document["dataset"]["path"])
    frame.loc[0, "signal"] = 99.0
    frame.to_csv(document["dataset"]["path"], index=False)
    second = worker.run_column(config)
    assert second["status"] == "COMPLETED"
    assert first["unit_key"] != second["unit_key"]


def test_run_column_recovers_stale_staging_directory(tmp_path, monkeypatch):
    config = _write_unit(tmp_path)
    monkeypatch.setattr(worker.FB, "run_feature", _raw_cells)
    stale = tmp_path / "output" / ".staging-signal-dead"
    stale.mkdir(parents=True)
    (stale / "partial.json").write_text("broken", encoding="utf-8")

    result = worker.run_column(config)
    assert result["status"] == "COMPLETED"
    assert not stale.exists()


def test_missing_target_is_explicit_and_never_calls_scientific_runner(tmp_path, monkeypatch):
    config = _write_unit(tmp_path, missing_target=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("run_feature must not run without the declared target")

    monkeypatch.setattr(worker.FB, "run_feature", forbidden)
    result = worker.run_column(config)
    cells = [json.loads(line) for line in Path(result["raw_cells_path"]).read_text().splitlines()]
    assert result["status"] == "NOT_AVAILABLE"
    assert cells[0]["rung1"]["state"] == "NOT_IDENTIFIED"
    assert cells[0]["rung1"]["abstention_reason"] == "TARGET_COLUMN_NOT_AVAILABLE"


def test_missing_context_is_explicit(tmp_path, monkeypatch):
    config = _write_unit(tmp_path)
    doc = json.loads(config.read_text())
    doc["target_pack"]["history_columns"] = ["missing_history"]
    config.write_text(json.dumps(doc), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("run_feature must not run without declared context")

    monkeypatch.setattr(worker.FB, "run_feature", forbidden)
    result = worker.run_column(config)
    cells = [json.loads(line) for line in Path(result["raw_cells_path"]).read_text().splitlines()]
    assert result["status"] == "NOT_AVAILABLE"
    assert cells[0]["rung2"]["abstention_reason"] == "CONTEXT_COLUMN_NOT_AVAILABLE"


def test_inventory_finalizer_requires_every_terminal(tmp_path, monkeypatch):
    first_config = _write_unit(tmp_path / "first", feature="one")
    monkeypatch.setattr(worker.FB, "run_feature", _raw_cells)
    first = worker.run_column(first_config)
    inventory = {
        "schema": "feature_selection_inventory.v1",
        "inventory_id": "fixture",
        "units": [
            {"unit_id": "one", "terminal_path": first["terminal_path"]},
            {"unit_id": "two", "terminal_path": str(tmp_path / "missing" / "terminal.json")},
        ],
        "output_dir": str(tmp_path / "final"),
    }
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(worker.IncompleteInventory, match="two"):
        worker.finalize_inventory(manifest)


def test_inventory_finalizer_reuses_global_finalize_and_replays(tmp_path, monkeypatch):
    configs = [_write_unit(tmp_path / name, feature=name) for name in ("one", "two")]
    monkeypatch.setattr(worker.FB, "run_feature", _raw_cells)
    completed = [worker.run_column(path) for path in configs]
    inventory = {
        "schema": "feature_selection_inventory.v1",
        "inventory_id": "fixture",
        "units": [
            {"unit_id": item["unit_id"], "terminal_path": item["terminal_path"]} for item in completed
        ],
        "output_dir": str(tmp_path / "final"),
    }
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps(inventory), encoding="utf-8")
    calls = []

    def fake_finalize(out):
        calls.append(out)
        causal = Path(out) / "causal_evidence.jsonl"
        causal.write_text("{}\n", encoding="utf-8")
        summary = {"schema": "fs_causal_final_summary.v1", "cells": 2, "per_state": {}}
        (Path(out) / "final_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(worker.FB, "finalize", fake_finalize)
    first = worker.finalize_inventory(manifest)
    digest = _sha(Path(first["terminal_path"]))
    second = worker.finalize_inventory(manifest)
    assert first["status"] == "COMPLETED"
    assert second["status"] == "REPLAY"
    assert len(calls) == 1
    assert _sha(Path(first["terminal_path"])) == digest


def test_real_global_finalizer_handles_explicit_unavailable_cells(tmp_path):
    configs = [_write_unit(tmp_path / name, feature=name, missing_target=True) for name in ("one", "two")]
    completed = [worker.run_column(path) for path in configs]
    manifest = tmp_path / "inventory.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "feature_selection_inventory.v1",
                "inventory_id": "unavailable-fixture",
                "units": [
                    {"unit_id": item["unit_id"], "terminal_path": item["terminal_path"]} for item in completed
                ],
                "output_dir": str(tmp_path / "final"),
            }
        ),
        encoding="utf-8",
    )
    result = worker.finalize_inventory(manifest)
    evidence = [json.loads(line) for line in (tmp_path / "final" / "causal_evidence.jsonl").read_text().splitlines()]
    assert result["global_state"] == "FINALIZED_WITH_GLOBAL_BH_FDR"
    assert len(evidence) == 2
    assert all(cell["rung1"]["state"] == "NOT_IDENTIFIED" for cell in evidence)
    assert all(cell["rung1"]["abstention_reason"] == "TARGET_COLUMN_NOT_AVAILABLE" for cell in evidence)


def test_eth_and_eurusd_target_definitions_are_config_driven(tmp_path, monkeypatch):
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs["targets"])
        return _raw_cells(*args, **kwargs)

    monkeypatch.setattr(worker.FB, "run_feature", capture)
    eth = _write_unit(tmp_path / "eth", feature="eth_signal")
    eur = _write_unit(tmp_path / "eur", feature="eur_signal")
    eur_doc = json.loads(eur.read_text())
    eur_doc["target_pack"]["id"] = "EURUSD"
    eur_doc["target_pack"]["definitions"][0] |= {
        "name": "custom_eurusd_target",
        "family": "custom",
        "head": "custom_head",
        "horizon_hours": 7,
    }
    eur.write_text(json.dumps(eur_doc), encoding="utf-8")
    worker.run_column(eth)
    worker.run_column(eur)
    assert seen[0] == [("eth_return_1h", "return", "short", 1)]
    assert seen[1] == [("custom_eurusd_target", "custom", "custom_head", 7)]
