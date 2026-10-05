from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import numpy as np
import pytest

from causal_inference_provider import feature_selection_worker as worker


WAREHOUSE_CONTRACT = Path("/home/harveybc/Documents/GitHub/data-warehouse/data_warehouse_service/feature_selection.py")
WAREHOUSE_CONTRACT_SHA256 = "91fcb4fde495239a4e0a21d3a39f0b66d50bd0a5df4865db7bd720b454f5f75a"


def _warehouse_validator():
    assert _sha(WAREHOUSE_CONTRACT) == WAREHOUSE_CONTRACT_SHA256
    spec = importlib.util.spec_from_file_location("warehouse_feature_selection_50bddf3", WAREHOUSE_CONTRACT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.validate_envelope


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
        "run": {
            "run_id": "phase1-fixture",
            "campaign_sha256": "a" * 64,
            "inventory_sha256": "b" * 64,
            "created_at": "2026-10-05T12:00:00Z"
        },
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
    profile_cells = {cell["metric"]: cell for cell in first["profile"]["cells"]}
    assert profile_cells["timestamp_gaps"]["value"]["missing_timestamp_count"] == 1
    document = json.loads(envelope.read_text())
    validated = _warehouse_validator()(document)
    assert validated == document
    assert set(document["rows"]) == {
        "sampling_quality", "variable_profiles", "information_metrics", "pair_relations",
        "causal_evidence", "selection_decisions",
    }
    assert document["rows"]["sampling_quality"]
    assert document["rows"]["variable_profiles"]
    assert document["rows"]["causal_evidence"]
    assert document["rows"]["selection_decisions"] == []

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
    assert result["status"] == "UNAVAILABLE"
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
    assert result["status"] == "UNAVAILABLE"
    assert cells[0]["rung2"]["abstention_reason"] == "CONTEXT_COLUMN_NOT_AVAILABLE"


def test_ps1_has_established_families_frequency_aware_lags_and_no_silent_gap_fill():
    frame = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-01T04:00:00Z", "2024-01-01T12:00:00Z",
            "2024-01-01T16:00:00Z", "2024-01-01T20:00:00Z", "2024-01-02T00:00:00Z",
        ]),
        "signal": [1.0, 1.1, 20.0, 1.2, 1.3, 1.4],
    })
    profile = worker.profile_series(frame, "time", "signal", "4h", "24x7")
    by_name = {cell["metric"]: cell for cell in profile["cells"]}
    required = {
        "missingness", "constant", "scale_tails", "volatility", "acf", "pacf", "trend",
        "adf", "kpss", "seasonality", "spectrum", "information_entropy", "timestamp_gaps",
        "outliers", "cost",
    }
    assert required <= set(by_name)
    assert all(cell["state"] in {"MEASURED", "FAILED", "NOT_APPLICABLE", "PENDING"}
               for cell in profile["cells"])
    assert by_name["timestamp_gaps"]["value"]["missing_timestamp_count"] == 1
    assert by_name["acf"]["value"]["lags"]["24h"]["rows"] == 6
    assert by_name["acf"]["value"]["lags"]["1h"]["state"] == "NOT_APPLICABLE"
    assert "1 row = 1 market hour" not in json.dumps(profile)
    assert by_name["spectrum"]["value"]["diagnostics"]["zero_filled"] is False
    assert by_name["spectrum"]["value"]["diagnostics"]["gaps_compressed"] is False


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
        "run": {"run_id": "final-missing", "campaign_sha256": "a" * 64,
                "created_at": "2026-10-05T13:00:00Z"},
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
        "run": {"run_id": "final-fixture", "campaign_sha256": "a" * 64,
                "created_at": "2026-10-05T13:00:00Z"},
        "output_dir": str(tmp_path / "final"),
    }
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps(inventory), encoding="utf-8")
    calls = []

    def fake_finalize(out):
        calls.append(out)
        causal = Path(out) / "causal_evidence.jsonl"
        raw = json.loads((Path(out) / "unit_00000" / "cells.jsonl").read_text().splitlines()[0])
        for rung in ("rung1", "rung2", "rung3"):
            raw[rung]["state"] = "NOT_IDENTIFIED"
            raw[rung]["abstention_reason"] = raw[rung].get("abstention_reason") or "TEST_NOT_IDENTIFIED"
        causal.write_text(json.dumps(raw) + "\n", encoding="utf-8")
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
    final_envelope = json.loads(Path(first["envelope_path"]).read_text())
    _warehouse_validator()(final_envelope)
    assert final_envelope["rows"]["selection_decisions"]


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
                    {"unit_id": item["unit_id"], "terminal_path": item["terminal_path"],
                     "disposition": "UNAVAILABLE"} for item in completed
                    ],
                    "run": {"run_id": "final-unavailable", "campaign_sha256": "a" * 64,
                            "created_at": "2026-10-05T13:00:00Z"},
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
    _warehouse_validator()(json.loads(Path(result["envelope_path"]).read_text()))


@pytest.mark.parametrize("terminal_state", ["FAILED", "NOT_APPLICABLE"])
def test_finalizer_rejects_noncompleted_terminal_without_explicit_disposition(tmp_path, terminal_state):
    config = _write_unit(tmp_path / "one", feature="one", missing_target=True)
    item = worker.run_column(config)
    terminal_path = Path(item["terminal_path"])
    terminal = json.loads(terminal_path.read_text())
    terminal["status"] = terminal_state
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps({
        "schema": "feature_selection_inventory.v1",
        "inventory_id": "refuse-unavailable",
        "units": [{"unit_id": "one", "terminal_path": item["terminal_path"]}],
        "run": {"run_id": "final-refusal", "campaign_sha256": "a" * 64,
                "created_at": "2026-10-05T13:00:00Z"},
        "output_dir": str(tmp_path / "final"),
    }), encoding="utf-8")
    with pytest.raises(worker.IncompleteInventory, match="explicit UNAVAILABLE disposition"):
        worker.finalize_inventory(manifest)


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


def test_historical_causal_defaults_equal_their_explicit_declarations():
    """Generic target-pack parameters must not move the historical EURUSD path."""
    rng = np.random.default_rng(1729)
    count = 1200
    candidate = np.zeros(count)
    for index in range(1, count):
        candidate[index] = 0.85 * candidate[index - 1] + rng.normal()
    X = pd.DataFrame({
        "t_decision_utc": pd.date_range("2020-01-01", periods=count, freq="h", tz="UTC"),
        "candidate": candidate,
    })
    for name in set(worker.fc.H_BASE + worker.fc.PRE_RETURNS + worker.fc.CALENDAR_LOCATORS):
        X[name] = rng.normal(size=count)
    Y = pd.DataFrame({name: rng.normal(size=count) for name, *_ in worker.fc.TARGETS})

    historical, historical_info = worker.fc.crossing_episodes_h(X, Y, "candidate", 24)
    explicit, explicit_info = worker.fc.crossing_episodes_h(
        X, Y, "candidate", 24,
        locators=worker.fc.CALENDAR_LOCATORS,
        targets=worker.fc.TARGETS,
        history_columns=worker.fc.H_BASE,
        pre_return_columns=worker.fc.PRE_RETURNS,
        mediator_target="Y_s_1h",
        volatility_regime_column="px.ewma_vol_168",
        placebo_outcome_column="px.logret_24h",
    )
    pd.testing.assert_frame_equal(historical, explicit, check_exact=True)
    assert historical_info == explicit_info


def test_stdio_worker_resolves_host_local_deployment_and_emits_one_json(tmp_path):
    unit_config = json.loads(_write_unit(tmp_path / "data").read_text())
    source = pd.read_csv(unit_config["dataset"]["path"])
    unit_config["target_pack"]["definitions"] = []
    for horizon in range(1, 7):
        column = f"target_{horizon}h"
        source[column] = source["target_1h"] * horizon
        unit_config["target_pack"]["definitions"].append({
            "name": f"eth_return_{horizon}h", "column": column,
            "family": "return", "head": "short", "horizon_hours": horizon,
        })
    source.to_csv(unit_config["dataset"]["path"], index=False)
    deployment = {
        "schema": "phase1.column_worker_deployment.v1",
        "version": "fixture-1",
        "output_root": str(tmp_path / "worker-output"),
        "resources": {
            "fixture-signal": {
                "resource_id": unit_config["dataset"]["resource_id"],
                "path": unit_config["dataset"]["path"],
                "format": "csv",
                "timestamp_column": "time",
                "frequency": "1h",
                "calendar": "24x7",
            }
        },
        "populations": {
            "ETH": {
                "target_pack_id": "eth-short-long-v1",
                "resource_key_field": "resource_key",
                "feature_column_field": "feature_column",
                "feature_family_field": "family",
                "feature_clock_field": "clock",
                "target_pack": unit_config["target_pack"],
                "folds": unit_config["folds"],
                "permutations": 2,
                "campaign_sha256": "a" * 64,
                "created_at": "2026-10-05T12:00:00Z",
            }
        },
    }
    deployment["deployment_sha256"] = worker.EW.digest(deployment)
    deployment_path = tmp_path / "deployment.json"
    deployment_path.write_text(json.dumps(deployment), encoding="utf-8")
    inventory_row = {
        "feature_id": "signal",
        "feature_column": "signal",
        "resource_key": "fixture-signal",
        "family": "fixture",
        "clock": "OBSERVED_AT_DECISION",
    }
    request = {
        "schema": "phase1.column_request.v1",
        "feature_id": "signal",
        "feature_key": "c" * 64,
        "population_id": "ETH",
        "target_pack": "eth-short-long-v1",
        "plan_sha256": "d" * 64,
        "inventory_sha256": "e" * 64,
        "inventory_row_sha256": worker.EW.digest(inventory_row),
        "inventory_row": inventory_row,
        "attempt_number": 1,
    }
    request["request_sha256"] = worker.EW.digest(request)

    completed = subprocess.run(
        [sys.executable, "-m", "causal_inference_provider.feature_selection_worker",
         "--stdio", "--deployment-manifest", str(deployment_path)],
        input=json.dumps(request), text=True, capture_output=True, check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("\n") == 1
    result = json.loads(completed.stdout)
    assert result["schema"] == "phase1.column_result.v1"
    assert result["feature_id"] == "signal"
    assert result["state"] == "COMPLETED"
    assert result["request_sha256"] == request["request_sha256"]
    payload = result["finalization_payload"]
    assert payload["schema"] == "phase1.column_finalization_payload.v1"
    assert len(payload["raw_causal_cells"]) == 6
    assert payload["request"] == request
    assert payload["payload_sha256"] == worker.EW.digest(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    _warehouse_validator()(result["envelope"])
    assert "dataset_path" not in request["inventory_row"]

    replay = subprocess.run(
        completed.args, input=json.dumps(request), text=True, capture_output=True, check=False,
    )
    assert replay.returncode == 0
    assert replay.stdout == completed.stdout

    deployment.pop("deployment_sha256")
    deployment["populations"]["ETH"]["target_pack"]["path"] = str(tmp_path / "absent-target.csv")
    deployment["deployment_sha256"] = worker.EW.digest(deployment)
    deployment_path.write_text(json.dumps(deployment), encoding="utf-8")
    request.pop("request_sha256")
    request["feature_key"] = "f" * 64
    request["attempt_number"] = 2
    request["request_sha256"] = worker.EW.digest(request)
    unavailable = subprocess.run(
        completed.args, input=json.dumps(request), text=True, capture_output=True, check=False,
    )
    assert unavailable.returncode == 0
    unavailable_result = json.loads(unavailable.stdout)
    assert unavailable_result["state"] == "UNAVAILABLE"
    assert unavailable_result["state"] != "NOT_APPLICABLE"


def test_multiple_stdio_results_finalize_after_remote_filesystems_are_removed(tmp_path):
    unit_config = json.loads(_write_unit(tmp_path / "data").read_text())
    source = pd.read_csv(unit_config["dataset"]["path"])
    source["signal_two"] = source["signal"] * 1.5
    definitions = []
    for horizon in range(1, 7):
        column = f"target_{horizon}h"
        source[column] = source["target_1h"] * horizon
        definitions.append({"name": f"eth_return_{horizon}h", "column": column,
                            "family": "return", "head": "short", "horizon_hours": horizon})
    source.to_csv(unit_config["dataset"]["path"], index=False)
    unit_config["target_pack"]["definitions"] = definitions
    deployment = {
        "schema": "phase1.column_worker_deployment.v1",
        "version": "fixture-finalizer-1",
        "output_root": str(tmp_path / "remote-worker-output"),
        "resources": {"fixture": {
            "resource_id": "fixture/two-signals", "path": unit_config["dataset"]["path"],
            "format": "csv", "timestamp_column": "time", "frequency": "1h", "calendar": "24x7",
        }},
        "populations": {"ETH": {
            "target_pack_id": "eth-short-long-v1", "resource_key_field": "resource_key",
            "feature_column_field": "feature_column", "feature_family_field": "family",
            "feature_clock_field": "clock", "target_pack": unit_config["target_pack"],
            "folds": unit_config["folds"], "permutations": 2,
            "campaign_sha256": "a" * 64, "created_at": "2026-10-05T12:00:00Z",
        }},
    }
    deployment["deployment_sha256"] = worker.EW.digest(deployment)
    deployment_path = tmp_path / "deployment.json"
    deployment_path.write_text(json.dumps(deployment), encoding="utf-8")
    plan = {
        "schema": "phase1.inventory_plan.v2", "phase": "PHASE_1",
        "population_id": "ETH", "target_pack": "eth-short-long-v1",
        "inventory_total": 2, "inventory_sha256": "e" * 64,
        "config_sha256": "9" * 64, "items": [],
    }
    results = []
    for index, (feature_id, column) in enumerate((("signal", "signal"), ("signal_two", "signal_two")), 1):
        row = {"feature_id": feature_id, "feature_column": column, "resource_key": "fixture",
               "family": "fixture", "clock": "OBSERVED_AT_DECISION"}
        key = f"{index}" * 64
        item = {"feature_id": feature_id, "key": key, "row_count": len(source),
                "byte_count": 24, "estimated_cost": 24.0, "available": True,
                "inventory_row_sha256": worker.EW.digest(row), "host_id": f"remote-{index}",
                "inventory_file": f"inventory-{index}.csv"}
        plan["items"].append(item)
        results.append((item, row))
    plan["plan_sha256"] = worker.EW.digest(plan)
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    terminals = tmp_path / "terminals"
    terminals.mkdir()
    command = [sys.executable, "-m", "causal_inference_provider.feature_selection_worker",
               "--stdio", "--deployment-manifest", str(deployment_path)]
    for attempt, (item, row) in enumerate(results, 1):
        request = {"schema": "phase1.column_request.v1", "feature_id": item["feature_id"],
                   "feature_key": item["key"], "population_id": "ETH",
                   "target_pack": "eth-short-long-v1", "plan_sha256": plan["plan_sha256"],
                   "inventory_sha256": plan["inventory_sha256"],
                   "inventory_row_sha256": item["inventory_row_sha256"],
                   "inventory_row": row, "attempt_number": attempt}
        request["request_sha256"] = worker.EW.digest(request)
        process = subprocess.run(command, input=json.dumps(request), text=True,
                                 capture_output=True, check=False)
        assert process.returncode == 0, process.stderr
        result = json.loads(process.stdout)
        terminal = {"schema": "phase1.inventory_terminal.v2", "feature_id": item["feature_id"],
                    "key": item["key"], "state": "COMPLETED", "host_id": item["host_id"],
                    "plan_sha256": plan["plan_sha256"],
                    "inventory_sha256": plan["inventory_sha256"],
                    "estimated_cost": item["estimated_cost"], "duration_seconds": 0.1,
                    "return_code": 0, "attempt_sha256": f"{attempt + 2}" * 64, "result": result}
        terminal["terminal_sha256"] = worker.EW.digest(terminal)
        (terminals / f"{item['key']}.json").write_text(json.dumps(terminal), encoding="utf-8")

    first_terminal_path = terminals / f"{results[0][0]['key']}.json"
    original_terminal = first_terminal_path.read_text(encoding="utf-8")
    incomplete = json.loads(original_terminal)
    del incomplete["result"]["finalization_payload"]["raw_causal_cells"][0]["rung1"]["p"]
    payload = incomplete["result"]["finalization_payload"]
    payload["payload_sha256"] = worker.EW.digest(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    incomplete["terminal_sha256"] = worker.EW.digest(
        {key: value for key, value in incomplete.items() if key != "terminal_sha256"}
    )
    first_terminal_path.write_text(json.dumps(incomplete), encoding="utf-8")
    refused = subprocess.run(
        [sys.executable, "-m", "causal_inference_provider.feature_selection_worker",
         "finalize-terminals", "--plan", str(plan_path), "--terminals", str(terminals),
         "--output", str(tmp_path / "must-not-exist.json")],
        text=True, capture_output=True, check=False,
    )
    assert refused.returncode == 2
    assert "causal evidence is incomplete" in refused.stderr
    assert not (tmp_path / "must-not-exist.json").exists()
    first_terminal_path.write_text(original_terminal, encoding="utf-8")

    shutil.rmtree(tmp_path / "remote-worker-output")
    final_output = tmp_path / "finalizer-result.json"
    finalized = subprocess.run(
        [sys.executable, "-m", "causal_inference_provider.feature_selection_worker",
         "finalize-terminals", "--plan", str(plan_path), "--terminals", str(terminals),
         "--output", str(final_output)],
        text=True, capture_output=True, check=False,
    )
    assert finalized.returncode == 0, finalized.stderr
    document = json.loads(final_output.read_text())
    assert document["schema"] == "phase1.finalizer_result.v1"
    assert document["state"] == "PHASE_1_COMPLETE"
    assert document["terminal_count"] == 2
    assert len(document["envelope"]["rows"]["selection_decisions"]) == 12
    _warehouse_validator()(document["envelope"])
