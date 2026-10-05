"""Generic, CPU-only phase-1 worker for one inventory time series.

CAUSAL mode profiles one TRAIN column and delegates causal calculations to
``fs_causal_batch.run_feature``. PROFILE_ONLY mode publishes the established
noncausal metric families without constructing causal evidence. Causal states
become final only after complete-inventory multiplicity correction.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import fs_causal as fc
from . import fs_causal_batch as FB
from . import ps3c
from . import feature_selection_envelope as EW
from .feature_selection_profile import profile_series as _profile_series


UNIT_SCHEMA = "feature_selection_unit.v1"
INVENTORY_SCHEMA = "feature_selection_inventory.v1"
TERMINAL_SCHEMA = "feature_selection_terminal.v1"
FINAL_TERMINAL_SCHEMA = "feature_selection_inventory_terminal.v1"
TARGET_PACKS = {"EURUSD", "ETH"}
COLUMN_REQUEST_SCHEMA = "phase1.column_request.v1"
COLUMN_RESULT_SCHEMA = "phase1.column_result.v1"
DEPLOYMENT_SCHEMA = "phase1.column_worker_deployment.v1"
FINALIZATION_PAYLOAD_SCHEMA = "phase1.column_finalization_payload.v1"
ORCHESTRATOR_PLAN_SCHEMA = "phase1.inventory_plan.v2"
ORCHESTRATOR_TERMINAL_SCHEMA = "phase1.inventory_terminal.v2"
ORCHESTRATOR_FINALIZER_SCHEMA = "phase1.finalizer_result.v1"
TARGET_CELL_COUNTS = {"EURUSD": 14, "ETH": 6}
MAX_FINALIZATION_PAYLOAD_BYTES = 2_000_000
MAX_PROFILE_RESULT_BYTES = 2_000_000
MAX_PROFILE_MERGE_BYTES = 256_000_000
EXECUTION_MODES = {"CAUSAL", "PROFILE_ONLY"}
PROFILE_MERGE_SCHEMA = "phase1.profile_merge_result.v1"
EURUSD_FEATURE_COUNT = 366
EURUSD_TARGET_DENOMINATOR = {
    *((f"Y_s_{hours}h", hours) for hours in (1, 2, 3, 4, 5, 6)),
    *((f"Y_l_{hours}h", hours) for hours in (24, 48, 72, 96, 120, 144)),
    ("Y_b_s6", 6), ("Y_b_l144", 144),
}


class InvalidConfiguration(ValueError):
    """The requested unit cannot be identified without guessing."""


class IncompleteInventory(RuntimeError):
    """Global multiplicity correction was requested without every terminal."""


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    column: str
    family: str
    head: str
    horizon_hours: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TargetDefinition":
        required = ("name", "column", "family", "head", "horizon_hours")
        missing = [key for key in required if key not in value]
        if missing:
            raise InvalidConfiguration(f"target definition missing {missing}")
        horizon = value["horizon_hours"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise InvalidConfiguration("horizon_hours must be a positive integer")
        return cls(str(value["name"]), str(value["column"]), str(value["family"]),
                   str(value["head"]), horizon)

    def scientific_tuple(self) -> tuple[str, str, str, int]:
        return self.name, self.family, self.head, self.horizon_hours


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                       default=ps3c._json_default) + "\n").encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_file_identity(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"]).expanduser().resolve()
    return {"path": str(path), "state": "AVAILABLE" if path.is_file() else "NOT_AVAILABLE",
            "sha256": _digest_file(path) if path.is_file() else None}


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(path, _canonical_bytes(value))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InvalidConfiguration(f"{path} must contain a JSON object")
    return value


def _verify_self_digest(document: dict[str, Any], key: str, label: str) -> None:
    claimed = document.get(key)
    content = {name: value for name, value in document.items() if name != key}
    if not isinstance(claimed, str) or claimed != EW.digest(content):
        raise InvalidConfiguration(f"{label} {key} does not cover its content")


def _read_table(spec: dict[str, Any]) -> pd.DataFrame:
    path = Path(spec["path"]).expanduser().resolve()
    kind = str(spec.get("format") or path.suffix.lstrip(".")).lower()
    if kind == "csv":
        return pd.read_csv(path)
    if kind in {"parquet", "pq"}:
        return pd.read_parquet(path)
    raise InvalidConfiguration(f"unsupported table format: {kind}")


def profile_series(frame: pd.DataFrame, timestamp_column: str, feature_column: str,
                   frequency: str, calendar: str) -> dict[str, Any]:
    """Compatibility facade for the established PS1 profiler."""
    return _profile_series(feature_column, frame, timestamp_column, feature_column, frequency, calendar)


def _profile(config: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    supplied = config.get("ps1_profile")
    if supplied:
        payload = _load_json(Path(supplied["path"])) if "path" in supplied else supplied.get("payload")
        if not isinstance(payload, dict) or payload.get("schema") != "feature_selection_ps1_profile.v1":
            raise InvalidConfiguration("supplied PS1 profile has the wrong schema")
        return payload
    dataset = config["dataset"]
    return _profile_series(config["unit_id"], frame, dataset["timestamp_column"], dataset["feature_column"],
                           dataset["frequency"], dataset["calendar"])


def _target_definitions(target_pack: dict[str, Any]) -> list[TargetDefinition]:
    pack_id = str(target_pack.get("id", "")).upper()
    if pack_id not in TARGET_PACKS:
        raise InvalidConfiguration(f"target_pack.id must be one of {sorted(TARGET_PACKS)}")
    definitions = [TargetDefinition.from_dict(item) for item in target_pack.get("definitions", [])]
    if not definitions or len({item.name for item in definitions}) != len(definitions):
        raise InvalidConfiguration("target definitions must be nonempty and have unique names")
    return definitions


def _unavailable_cells(unit_id: str, config: dict[str, Any], definitions: list[TargetDefinition],
                       reason: str) -> list[dict[str, Any]]:
    meta = config.get("feature") or {}
    cells = []
    for target in definitions:
        cells.append({
            "feature_id": unit_id,
            "batch": "per-column",
            "family": meta.get("family", "UNDECLARED"),
            "clock": meta.get("clock", "UNDECLARED"),
            "target": target.name,
            "target_family": target.family,
            "head": target.head,
            "horizon_h": target.horizon_hours,
            "seed": fc.SEED,
            "rung1": {"raw_state": reason, "state": fc.NOT_IDENTIFIED, "p": None,
                      "abstention_reason": reason},
            "rung2": {"raw_state": "NOT_EVALUATED", "state": fc.NOT_IDENTIFIED,
                      "reasons": [reason], "p_linear": None, "nonlinear": None,
                      "estimate": None, "abstention_reason": reason},
            "rung3": {"raw_state": "NOT_EVALUATED", "state": fc.NOT_IDENTIFIED,
                      "reasons": [reason], "prediction": None, "abstention_reason": reason},
            "discovery": {"sypi": {"state": "NOT_RUN", "reason": reason}},
            "rung1_raw": reason,
            "rung2_raw": "NOT_EVALUATED",
            "rung3_raw": "NOT_EVALUATED",
        })
    return cells


def _unavailable_profile(unit_id: str, config: dict[str, Any], reason: str) -> dict[str, Any]:
    dataset = config["dataset"]
    return {
        "schema": "feature_selection_ps1_profile.v1",
        "feature_id": unit_id,
        "frequency": dataset["frequency"],
        "calendar_rule": dataset["calendar"],
        "state": "UNAVAILABLE",
        "reason": reason,
        "cells": [
            {"feature_id": unit_id, "metric": "missingness", "state": "UNAVAILABLE",
             "value": None, "reason": reason, "metrics_version": "generic_ps1_metrics.v1"},
            {"feature_id": unit_id, "metric": "timestamp_gaps", "state": "UNAVAILABLE",
             "value": None, "reason": reason, "metrics_version": "generic_ps1_metrics.v1"},
        ],
    }


def _folds(config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for fold in config.get("folds", []):
        train = fold.get("train_rows")
        validation = fold.get("validation_rows", fold.get("val_rows"))
        if train and validation:
            result.append((np.arange(*train), np.arange(*validation)))
    return result


def _prepare_scientific_frames(config: dict[str, Any], source: pd.DataFrame,
                               definitions: list[TargetDefinition]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dataset, pack = config["dataset"], config["target_pack"]
    timestamp = dataset["timestamp_column"]
    feature_column = dataset["feature_column"]
    declared_context = dict.fromkeys([
        *pack.get("history_columns", []), *pack.get("pre_return_columns", []),
        *pack.get("calendar_locator_columns", []),
    ])
    context = [column for column in declared_context if column not in {timestamp, feature_column}]
    missing_context = [column for column in context if column not in source]
    if missing_context:
        return pd.DataFrame(), pd.DataFrame(), missing_context
    feature_frame = source[[timestamp, feature_column, *context]].copy()
    feature_frame[timestamp] = pd.to_datetime(feature_frame[timestamp], utc=True, errors="coerce")
    feature_frame = feature_frame.dropna(subset=[timestamp]).sort_values(timestamp)
    feature_frame = feature_frame.rename(columns={timestamp: "t_decision_utc", feature_column: config["unit_id"]})

    target_source = _read_table(pack)
    target_timestamp = pack.get("timestamp_column", timestamp)
    missing_targets = [target.column for target in definitions if target.column not in target_source]
    if target_timestamp not in target_source:
        missing_targets.append(target_timestamp)
    if missing_targets:
        return pd.DataFrame(), pd.DataFrame(), missing_targets
    target_frame = target_source[[target_timestamp, *[target.column for target in definitions]]].copy()
    target_frame[target_timestamp] = pd.to_datetime(target_frame[target_timestamp], utc=True, errors="coerce")
    target_frame = target_frame.dropna(subset=[target_timestamp]).sort_values(target_timestamp)
    rename = {target_timestamp: "t_decision_utc", **{target.column: target.name for target in definitions}}
    target_frame = target_frame.rename(columns=rename)
    joined = feature_frame.merge(target_frame, on="t_decision_utc", how="inner", validate="one_to_one")
    start, end = (config.get("train_period") or [None, None])[:2]
    if start is not None:
        joined = joined[joined.t_decision_utc >= pd.Timestamp(start)]
    if end is not None:
        joined = joined[joined.t_decision_utc < pd.Timestamp(end)]
    joined = joined.reset_index(drop=True)
    joined.insert(1, "row_id", np.arange(len(joined), dtype=np.int64))
    x_columns = ["t_decision_utc", "row_id", config["unit_id"], *context]
    y_columns = ["t_decision_utc", "row_id", *[target.name for target in definitions]]
    return joined[x_columns], joined[y_columns], []


def _validate_unit(config: dict[str, Any]) -> list[TargetDefinition]:
    if config.get("schema") != UNIT_SCHEMA:
        raise InvalidConfiguration(f"schema must be {UNIT_SCHEMA}")
    if config.get("split") != "TRAIN":
        raise InvalidConfiguration("phase-1 workers read TRAIN only")
    if config.get("mode") not in EXECUTION_MODES:
        raise InvalidConfiguration(f"mode must be explicitly declared as one of {sorted(EXECUTION_MODES)}")
    for key in ("unit_id", "dataset", "target_pack", "run", "output_dir"):
        if key not in config:
            raise InvalidConfiguration(f"missing {key}")
    dataset = config["dataset"]
    for key in ("resource_id", "path", "timestamp_column", "feature_column", "frequency", "calendar"):
        if key not in dataset:
            raise InvalidConfiguration(f"dataset missing {key}")
    run = config["run"]
    for key in ("run_id", "campaign_sha256", "inventory_sha256", "created_at"):
        if key not in run:
            raise InvalidConfiguration(f"run missing {key}")
    return _target_definitions(config["target_pack"])


def _profile_only_column(config_path: Path, config: dict[str, Any],
                         definitions: list[TargetDefinition]) -> dict[str, Any]:
    """Publish PS1 and pair metrics without constructing causal evidence."""
    if config.get("ps1_profile") is not None:
        raise InvalidConfiguration("PROFILE_ONLY recomputation cannot reuse a supplied PS1 profile")
    dataset_identity = _declared_file_identity(config["dataset"])
    target_identity = _declared_file_identity(config["target_pack"])
    supplied_profile = config.get("ps1_profile") or {}
    profile_identity = _declared_file_identity(supplied_profile) if supplied_profile.get("path") else None
    identity = {
        "schema": UNIT_SCHEMA,
        "mode": "PROFILE_ONLY",
        "config": config,
        "config_sha256": _digest_file(config_path),
        "inputs": {"dataset": dataset_identity, "targets": target_identity, "ps1_profile": profile_identity},
        "metric_code_sha256": {
            "worker": _digest_file(Path(__file__)),
            "profile": _digest_file(Path(sys.modules[_profile_series.__module__].__file__)),
            "envelope": _digest_file(Path(EW.__file__)),
        },
    }
    identity_sha = _digest_bytes(_canonical_bytes(identity))
    root = Path(config["output_dir"]).expanduser().resolve()
    unit_id = str(config["unit_id"])
    unit_key = f"{unit_id}-{identity_sha[:16]}"
    unit_dir = root / "units" / unit_key
    terminal_path = unit_dir / "terminal.json"
    envelope_path = root / "outbox" / f"{unit_key}.json"
    if _terminal_is_valid(terminal_path, identity_sha) and envelope_path.exists():
        terminal = _load_json(terminal_path)
        if _digest_file(envelope_path) != terminal.get("outbox_file_sha256"):
            raise InvalidConfiguration(f"outbox integrity failed for {unit_id}")
        return {**terminal, "terminal_status": terminal["status"], "status": "REPLAY",
                "terminal_path": str(terminal_path), "envelope_path": str(envelope_path)}
    if terminal_path.exists():
        raise InvalidConfiguration(f"immutable terminal integrity failed for {unit_id}")
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.glob(f".staging-{unit_id}-*"):
        if stale.is_dir():
            shutil.rmtree(stale)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{unit_id}-", dir=root))
    try:
        X = Y = None
        dataset_path = Path(config["dataset"]["path"]).expanduser().resolve()
        if not dataset_path.is_file():
            profile = _unavailable_profile(unit_id, config, "DATASET_NOT_AVAILABLE")
            status = "UNAVAILABLE"
        else:
            source = _read_table(config["dataset"])
            required = [config["dataset"]["timestamp_column"], config["dataset"]["feature_column"]]
            absent = [column for column in required if column not in source]
            if absent:
                profile = _unavailable_profile(unit_id, config, "FEATURE_COLUMN_NOT_AVAILABLE")
                profile["columns"] = absent
                status = "UNAVAILABLE"
            else:
                profile = _profile(config, source)
                target_path = Path(config["target_pack"]["path"]).expanduser().resolve()
                if not target_path.is_file():
                    status = "UNAVAILABLE"
                else:
                    X, Y, missing = _prepare_scientific_frames(config, source, definitions)
                    status = "UNAVAILABLE" if missing or X.empty else "COMPLETED"
        profile_path = staging / "ps1_profile.json"
        _write_json_atomic(profile_path, profile)
        target_pairs = [(target.name, target.horizon_hours) for target in definitions]
        rows = EW.profile_rows(profile, target_pairs, identity_sha)
        if status == "COMPLETED" and X is not None and Y is not None:
            rows["pair_relations"] = EW.pair_relation_rows(
                unit_id, X, Y, definitions, config["dataset"]["frequency"], identity_sha,
            )
        rows["causal_evidence"] = []
        rows["selection_decisions"] = []
        envelope = EW.build_envelope({
            "run_id": f"{config['run']['run_id']}:{unit_key}",
            "campaign_sha256": config["run"]["campaign_sha256"],
            "code_sha256": EW.digest(identity["metric_code_sha256"]),
            "input_sha256": EW.digest(identity["inputs"]),
            "inventory_sha256": config["run"]["inventory_sha256"],
            "created_at": config["run"]["created_at"],
        }, rows)
        envelope_bytes = _canonical_bytes(envelope)
        terminal = {
            "schema": TERMINAL_SCHEMA, "mode": "PROFILE_ONLY", "unit_id": unit_id,
            "unit_key": unit_key, "identity_sha256": identity_sha, "status": status,
            "global_state": "PROFILE_ONLY_AWAITING_MERGE",
            "artifacts_sha256": {profile_path.name: _digest_file(profile_path)},
            "envelope_sha256": envelope["envelope_sha256"],
            "outbox_file_sha256": _digest_bytes(envelope_bytes), "profile": profile,
        }
        _write_json_atomic(staging / "terminal.json", terminal)
        _write_bytes_atomic(envelope_path, envelope_bytes)
        unit_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, unit_dir)
        return {**terminal, "terminal_path": str(terminal_path), "envelope_path": str(envelope_path)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _terminal_is_valid(path: Path, identity_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        terminal = _load_json(path)
        if terminal.get("identity_sha256") != identity_sha:
            return False
        for relative, expected in terminal.get("artifacts_sha256", {}).items():
            artifact = path.parent / relative
            if not artifact.is_file() or _digest_file(artifact) != expected:
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def run_column(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = _load_json(config_path)
    definitions = _validate_unit(config)
    if config["mode"] == "PROFILE_ONLY":
        return _profile_only_column(config_path, config, definitions)
    dataset_identity = _declared_file_identity(config["dataset"])
    target_identity = _declared_file_identity(config["target_pack"])
    supplied_profile = config.get("ps1_profile") or {}
    profile_identity = _declared_file_identity(supplied_profile) if supplied_profile.get("path") else None
    identity = {
        "schema": UNIT_SCHEMA,
        "config": config,
        "config_sha256": _digest_file(config_path),
        "inputs": {"dataset": dataset_identity, "targets": target_identity, "ps1_profile": profile_identity},
        "scientific_code_sha256": {
            "worker": _digest_file(Path(__file__)),
            "batch": _digest_file(Path(FB.__file__)),
            "ladder": _digest_file(Path(fc.__file__)),
        },
    }
    identity_sha = _digest_bytes(_canonical_bytes(identity))
    root = Path(config["output_dir"]).expanduser().resolve()
    unit_id = str(config["unit_id"])
    unit_key = f"{unit_id}-{identity_sha[:16]}"
    unit_dir = root / "units" / unit_key
    terminal_path = unit_dir / "terminal.json"
    envelope_path = root / "outbox" / f"{unit_key}.json"
    if _terminal_is_valid(terminal_path, identity_sha) and envelope_path.exists():
        terminal = _load_json(terminal_path)
        if _digest_file(envelope_path) != terminal.get("outbox_file_sha256"):
            raise InvalidConfiguration(f"outbox integrity failed for {unit_id}")
        return {**terminal, "terminal_status": terminal["status"], "status": "REPLAY",
                "terminal_path": str(terminal_path), "envelope_path": str(envelope_path)}
    if terminal_path.exists():
        raise InvalidConfiguration(f"immutable terminal integrity failed for {unit_id}")
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.glob(f".staging-{unit_id}-*"):
        if stale.is_dir():
            shutil.rmtree(stale)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{unit_id}-", dir=root))
    try:
        X = Y = None
        dataset_path = Path(config["dataset"]["path"]).expanduser().resolve()
        if not dataset_path.is_file():
            profile = _unavailable_profile(unit_id, config, "DATASET_NOT_AVAILABLE")
            cells = _unavailable_cells(unit_id, config, definitions, "DATASET_NOT_AVAILABLE")
            feature_record, status = {"feature_id": unit_id, "state": "UNAVAILABLE"}, "UNAVAILABLE"
        else:
            source = _read_table(config["dataset"])
            required = [config["dataset"]["timestamp_column"], config["dataset"]["feature_column"]]
            absent = [column for column in required if column not in source]
            if absent:
                profile = _unavailable_profile(unit_id, config, "FEATURE_COLUMN_NOT_AVAILABLE")
                profile["columns"] = absent
                cells = _unavailable_cells(unit_id, config, definitions, "FEATURE_COLUMN_NOT_AVAILABLE")
                feature_record, status = {"feature_id": unit_id, "state": "UNAVAILABLE"}, "UNAVAILABLE"
            else:
                profile = _profile(config, source)
                target_path = Path(config["target_pack"]["path"]).expanduser().resolve()
                if not target_path.is_file():
                    cells = _unavailable_cells(unit_id, config, definitions, "TARGET_RESOURCE_NOT_AVAILABLE")
                    feature_record, status = {"feature_id": unit_id, "state": "UNAVAILABLE",
                                              "missing_resource": str(target_path)}, "UNAVAILABLE"
                else:
                    X, Y, missing = _prepare_scientific_frames(config, source, definitions)
                if target_path.is_file() and missing:
                    target_columns = {target.column for target in definitions}
                    reason = "TARGET_COLUMN_NOT_AVAILABLE" if any(item in target_columns for item in missing) else "CONTEXT_COLUMN_NOT_AVAILABLE"
                    cells = _unavailable_cells(unit_id, config, definitions, reason)
                    feature_record, status = {"feature_id": unit_id, "state": "UNAVAILABLE",
                                              "missing_columns": missing}, "UNAVAILABLE"
                elif target_path.is_file() and not _folds(config):
                    cells = _unavailable_cells(unit_id, config, definitions, "FOLDS_NOT_DECLARED")
                    feature_record, status = {"feature_id": unit_id, "state": "NOT_APPLICABLE"}, "UNAVAILABLE"
                elif target_path.is_file():
                    targets = [target.scientific_tuple() for target in definitions]
                    y_sd = {target.name: float(np.nanstd(Y[target.name].to_numpy(float))) for target in definitions}
                    meta = {"family": (config.get("feature") or {}).get("family", "UNDECLARED"),
                            "clock": (config.get("feature") or {}).get("clock", "UNDECLARED"),
                            "source": config["dataset"]["resource_id"], "batch": "per-column"}
                    cells, feature_record = FB.run_feature(
                        unit_id, meta, X, Y, _folds(config), int(config.get("permutations", 200)), y_sd,
                        targets=targets,
                        history_columns=config["target_pack"].get("history_columns", []),
                        pre_return_columns=config["target_pack"].get("pre_return_columns", []),
                        calendar_locator_columns=config["target_pack"].get("calendar_locator_columns", []),
                        mediator_target=config["target_pack"].get("mediator_target"),
                        volatility_regime_column=config["target_pack"].get("volatility_regime_column"),
                        placebo_outcome_column=config["target_pack"].get("placebo_outcome_column"),
                    )
                    status = "COMPLETED"
        profile_path = staging / "ps1_profile.json"
        cells_path = staging / "raw_causal_cells.jsonl"
        feature_path = staging / "feature_record.json"
        _write_json_atomic(profile_path, profile)
        _write_bytes_atomic(cells_path, b"".join(_canonical_bytes(cell) for cell in cells))
        _write_json_atomic(feature_path, feature_record)
        artifacts = {path.name: _digest_file(path) for path in (profile_path, cells_path, feature_path)}
        target_pairs = [(target.name, target.horizon_hours) for target in definitions]
        population_id = identity_sha
        rows = EW.profile_rows(profile, target_pairs, population_id)
        if X is not None and Y is not None and len(X):
            rows["pair_relations"] = EW.pair_relation_rows(
                unit_id, X, Y, definitions, config["dataset"]["frequency"], population_id,
            )
        rows["causal_evidence"] = EW.causal_rows(cells, population_id, final=False)
        run = {
            "run_id": f"{config['run']['run_id']}:{unit_key}",
            "campaign_sha256": config["run"]["campaign_sha256"],
            "code_sha256": EW.digest(identity["scientific_code_sha256"]),
            "input_sha256": EW.digest(identity["inputs"]),
            "inventory_sha256": config["run"]["inventory_sha256"],
            "created_at": config["run"]["created_at"],
        }
        envelope = EW.build_envelope(run, rows)
        envelope_bytes = _canonical_bytes(envelope)
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "mode": "CAUSAL",
            "unit_id": unit_id,
            "unit_key": unit_key,
            "identity_sha256": identity_sha,
            "status": status,
            "global_state": "AWAITING_INVENTORY_FINALIZATION",
            "artifacts_sha256": artifacts,
            "envelope_sha256": envelope["envelope_sha256"],
            "outbox_file_sha256": _digest_bytes(envelope_bytes),
            "profile": profile,
            "raw_cells_path": str(unit_dir / cells_path.name),
        }
        _write_json_atomic(staging / "terminal.json", terminal)
        _write_bytes_atomic(envelope_path, envelope_bytes)
        unit_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, unit_dir)
        return {**terminal, "terminal_path": str(terminal_path), "envelope_path": str(envelope_path)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _verify_terminal(path: Path, unit_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise IncompleteInventory(f"missing terminal for {unit_id}: {path}")
    terminal = _load_json(path)
    if terminal.get("schema") != TERMINAL_SCHEMA or terminal.get("unit_id") != unit_id:
        raise IncompleteInventory(f"invalid terminal for {unit_id}")
    if not _terminal_is_valid(path, terminal["identity_sha256"]):
        raise IncompleteInventory(f"terminal integrity failed for {unit_id}")
    return terminal


def finalize_inventory(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != INVENTORY_SCHEMA:
        raise InvalidConfiguration(f"schema must be {INVENTORY_SCHEMA}")
    units = manifest.get("units") or []
    if not units or len({unit.get("unit_id") for unit in units}) != len(units):
        raise InvalidConfiguration("inventory units must be nonempty and unique")
    run_config = manifest.get("run") or {}
    for key in ("run_id", "campaign_sha256", "created_at"):
        if key not in run_config:
            raise InvalidConfiguration(f"inventory run missing {key}")
    verified = [(unit, _verify_terminal(Path(unit["terminal_path"]), unit["unit_id"])) for unit in units]
    for unit, terminal in verified:
        if terminal.get("status") != "COMPLETED" and unit.get("disposition") != "UNAVAILABLE":
            raise IncompleteInventory(
                f"{unit['unit_id']} has terminal state {terminal.get('status')}; "
                "an explicit UNAVAILABLE disposition is required"
            )
        if unit.get("disposition") not in (None, "UNAVAILABLE"):
            raise InvalidConfiguration(f"unsupported disposition for {unit['unit_id']}: {unit.get('disposition')}")
    inventory_identity = {
        "manifest_sha256": _digest_file(manifest_path),
        "units": [{"unit_id": unit["unit_id"], "terminal_sha256": _digest_file(Path(unit["terminal_path"]))}
                  for unit, _ in verified],
    }
    inventory_sha = _digest_bytes(_canonical_bytes(inventory_identity))
    output = Path(manifest["output_dir"]).expanduser().resolve()
    terminal_path = output / "inventory_terminal.json"
    if terminal_path.exists():
        previous = _load_json(terminal_path)
        if previous.get("inventory_sha256") == inventory_sha:
            envelope_path = output / "feature_selection_envelope.json"
            if not envelope_path.is_file() or _digest_file(envelope_path) != previous.get("outbox_file_sha256"):
                raise InvalidConfiguration("final envelope integrity failed during replay")
            return {**previous, "status": "REPLAY", "terminal_path": str(terminal_path),
                    "envelope_path": str(envelope_path)}
        raise InvalidConfiguration("output already belongs to another inventory identity")
    output.mkdir(parents=True, exist_ok=True)
    chunks = []
    all_targets = set()
    for index, (unit, terminal) in enumerate(verified):
        chunk_id = f"unit_{index:05d}"
        chunk_dir = output / chunk_id
        chunk_dir.mkdir()
        source_dir = Path(unit["terminal_path"]).parent
        shutil.copyfile(source_dir / "raw_causal_cells.jsonl", chunk_dir / "cells.jsonl")
        feature_record = _load_json(source_dir / "feature_record.json")
        _write_json_atomic(chunk_dir / "features.json", {"features": [feature_record], "failures": [],
                                                          "cost": {"wall_s": feature_record.get("cost_s", 0.0)}})
        _write_json_atomic(chunk_dir / "READY", {"unit_id": unit["unit_id"],
                                                  "terminal_sha256": _digest_file(Path(unit["terminal_path"]))})
        cells = [json.loads(line) for line in (chunk_dir / "cells.jsonl").read_text().splitlines() if line]
        all_targets.update(cell["target"] for cell in cells)
        chunks.append({"id": chunk_id, "features": [unit["unit_id"]]})
    plan = {
        "schema": "fs_causal_plan.v1",
        "revision": inventory_sha,
        "chunks": chunks,
        "candidates_total": len(chunks),
        "cells_total": sum(1 for chunk in chunks for _ in (output / chunk["id"] / "cells.jsonl").open()),
        "targets": sorted(all_targets),
        "families": {"rung1": "per target", "rung2": "per target", "sypi_condition1": "per target"},
        "seed": fc.SEED,
    }
    _write_json_atomic(output / "plan.json", plan)
    summary = FB.finalize(str(output))
    final_cells = [json.loads(line) for line in (output / "causal_evidence.jsonl").read_text().splitlines() if line]
    rows = {family: [] for family in EW.ROW_FAMILIES}
    for unit, terminal in verified:
        unit_envelope = _load_json(Path(unit["terminal_path"]).parents[2] / "outbox" / f"{terminal['unit_key']}.json")
        for family in ("sampling_quality", "variable_profiles", "information_metrics", "pair_relations"):
            rows[family].extend({key: value for key, value in row.items() if key != "row_sha256"}
                                for row in unit_envelope["rows"][family])
    rows["causal_evidence"] = EW.causal_rows(final_cells, inventory_sha, final=True)
    rows["selection_decisions"] = EW.selection_rows(final_cells, inventory_sha)
    final_envelope = EW.build_envelope({
        "run_id": run_config["run_id"],
        "campaign_sha256": run_config["campaign_sha256"],
        "code_sha256": EW.digest({"worker": _digest_file(Path(__file__)), "batch": _digest_file(Path(FB.__file__)),
                                   "ladder": _digest_file(Path(fc.__file__))}),
        "input_sha256": EW.digest(inventory_identity["units"]),
        "inventory_sha256": inventory_sha,
        "created_at": run_config["created_at"],
    }, rows)
    envelope_path = output / "feature_selection_envelope.json"
    _write_json_atomic(envelope_path, final_envelope)
    terminal = {
        "schema": FINAL_TERMINAL_SCHEMA,
        "inventory_id": manifest["inventory_id"],
        "inventory_sha256": inventory_sha,
        "status": "COMPLETED",
        "global_state": "FINALIZED_WITH_GLOBAL_BH_FDR",
        "units": len(units),
        "cells": summary["cells"],
        "causal_evidence_sha256": _digest_file(output / "causal_evidence.jsonl"),
        "envelope_sha256": final_envelope["envelope_sha256"],
        "outbox_file_sha256": _digest_file(envelope_path),
    }
    _write_json_atomic(terminal_path, terminal)
    return {**terminal, "terminal_path": str(terminal_path), "envelope_path": str(envelope_path)}


def _column_config(request: dict[str, Any], deployment: dict[str, Any]) -> dict[str, Any]:
    if request.get("schema") != COLUMN_REQUEST_SCHEMA:
        raise InvalidConfiguration(f"request schema must be {COLUMN_REQUEST_SCHEMA}")
    _verify_self_digest(request, "request_sha256", "request")
    row = request.get("inventory_row")
    if not isinstance(row, dict) or EW.digest(row) != request.get("inventory_row_sha256"):
        raise InvalidConfiguration("inventory_row_sha256 does not cover inventory_row")
    if row.get("feature_id") != request.get("feature_id"):
        raise InvalidConfiguration("request feature_id contradicts inventory row")
    if deployment.get("schema") != DEPLOYMENT_SCHEMA or not isinstance(deployment.get("version"), str):
        raise InvalidConfiguration(f"deployment schema must be {DEPLOYMENT_SCHEMA} with a version")
    _verify_self_digest(deployment, "deployment_sha256", "deployment")
    population_id = request.get("population_id")
    population = (deployment.get("populations") or {}).get(population_id)
    if not isinstance(population, dict):
        raise InvalidConfiguration(f"deployment has no population {population_id!r}")
    if population.get("target_pack_id") != request.get("target_pack"):
        raise InvalidConfiguration("request target_pack contradicts deployment")
    if population.get("mode") not in EXECUTION_MODES:
        raise InvalidConfiguration(f"deployment population must explicitly declare one of {sorted(EXECUTION_MODES)}")
    resource_key = row.get(population.get("resource_key_field", "resource_key"))
    resource = (deployment.get("resources") or {}).get(resource_key)
    if not isinstance(resource, dict):
        raise InvalidConfiguration(f"deployment has no resource {resource_key!r}")
    feature_column = row.get(population.get("feature_column_field", "feature_column"))
    if not isinstance(feature_column, str) or not feature_column:
        raise InvalidConfiguration("inventory row has no feature column")
    feature_id = str(request["feature_id"])
    output_root = Path(deployment["output_root"]).expanduser().resolve() / str(population_id)
    config = {
        "schema": UNIT_SCHEMA,
        "mode": population["mode"],
        "unit_id": feature_id,
        "split": "TRAIN",
        "dataset": {**resource, "feature_column": feature_column},
        "feature": {
            "family": row.get(population.get("feature_family_field", "family"), "UNDECLARED"),
            "clock": row.get(population.get("feature_clock_field", "clock"), "UNDECLARED"),
        },
        "target_pack": population["target_pack"],
        "folds": population.get("folds", []),
        "permutations": population.get("permutations", 200),
        "run": {
            "run_id": f"{population_id}:{request['plan_sha256'][:16]}",
            "campaign_sha256": population["campaign_sha256"],
            "inventory_sha256": request["inventory_sha256"],
            "created_at": population["created_at"],
        },
        "output_dir": str(output_root / "evidence"),
    }
    for optional in ("train_period", "ps1_profile"):
        if population.get(optional) is not None:
            config[optional] = population[optional]
    return config


def _validate_warehouse_envelope(envelope: dict[str, Any]) -> None:
    if envelope.get("schema_version") != EW.SCHEMA_VERSION or set(envelope.get("rows", {})) != set(EW.ROW_FAMILIES):
        raise InvalidConfiguration("worker envelope does not match the warehouse contract")
    claimed = envelope.get("envelope_sha256")
    unsigned = {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    if claimed != EW.digest(unsigned):
        raise InvalidConfiguration("worker envelope digest is invalid")
    for family in EW.ROW_FAMILIES:
        for row in envelope["rows"][family]:
            row_unsigned = {key: value for key, value in row.items() if key != "row_sha256"}
            if row.get("row_sha256") != EW.digest(row_unsigned):
                raise InvalidConfiguration(f"worker envelope has an invalid {family} row digest")


def _finalization_payload(request: dict[str, Any], terminal: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    unit_dir = Path(terminal["terminal_path"]).parent
    cells = [json.loads(line) for line in (unit_dir / "raw_causal_cells.jsonl").read_text().splitlines() if line]
    feature_record = _load_json(unit_dir / "feature_record.json")
    population_id = str(request["population_id"])
    expected = TARGET_CELL_COUNTS.get(population_id)
    if expected is None or len(cells) != expected:
        raise InvalidConfiguration(
            f"{population_id} finalization payload must contain exactly {expected} causal cells, got {len(cells)}"
        )
    targets = set()
    for cell in cells:
        if cell.get("feature_id") != request["feature_id"]:
            raise InvalidConfiguration("raw causal cell belongs to another feature")
        target = (cell.get("target"), cell.get("horizon_h"))
        if target in targets:
            raise InvalidConfiguration("raw causal cells contain a duplicate target/horizon")
        targets.add(target)
        if "p" not in cell.get("rung1", {}) or "p_linear" not in cell.get("rung2", {}):
            raise InvalidConfiguration("raw causal cell omits a p-value required for global FDR")
        sypi = (cell.get("discovery") or {}).get("sypi") or {}
        if sypi.get("state") == "RUN" and "condition1_p" not in sypi:
            raise InvalidConfiguration("raw causal cell omits the SyPI p-value required for global FDR")
    if feature_record.get("feature_id") != request["feature_id"]:
        raise InvalidConfiguration("feature record belongs to another feature")
    payload = {
        "schema": FINALIZATION_PAYLOAD_SCHEMA,
        "feature_id": request["feature_id"],
        "population_id": population_id,
        "target_pack": request["target_pack"],
        "plan_sha256": request["plan_sha256"],
        "inventory_sha256": request["inventory_sha256"],
        "request": request,
        "unit_envelope_sha256": envelope["envelope_sha256"],
        "raw_causal_cells": cells,
        "feature_record": feature_record,
    }
    payload["payload_sha256"] = EW.digest(payload)
    if len(_canonical_bytes(payload)) > MAX_FINALIZATION_PAYLOAD_BYTES:
        raise InvalidConfiguration("finalization payload exceeds its 2000000-byte bound")
    return payload


def run_stdio(deployment_path: str | Path, input_text: str) -> tuple[dict[str, Any], int]:
    """Execute exactly one orchestrator request and return its sole stdout object."""
    request: dict[str, Any] = {}
    try:
        parsed = json.loads(input_text)
        if not isinstance(parsed, dict):
            raise InvalidConfiguration("stdin request must be one JSON object")
        request = parsed
        deployment = _load_json(Path(deployment_path).expanduser().resolve())
        config = _column_config(request, deployment)
        config_root = Path(config["output_dir"]).parent / "requests"
        config_path = config_root / f"{request['feature_key']}-{request['request_sha256']}.json"
        payload = _canonical_bytes(config)
        if config_path.exists() and config_path.read_bytes() != payload:
            raise InvalidConfiguration("request config path contains different bytes")
        if not config_path.exists():
            _write_bytes_atomic(config_path, payload)
        with contextlib.redirect_stdout(sys.stderr):
            terminal = run_column(config_path)
        envelope = _load_json(Path(terminal["envelope_path"]))
        _validate_warehouse_envelope(envelope)
        terminal_state = terminal.get("terminal_status", terminal["status"])
        state = "COMPLETED" if terminal_state == "COMPLETED" else "UNAVAILABLE"
        if config["mode"] == "PROFILE_ONLY":
            if envelope["rows"]["causal_evidence"] or envelope["rows"]["selection_decisions"]:
                raise InvalidConfiguration("PROFILE_ONLY envelope contains causal rows")
            result = {
                "schema": COLUMN_RESULT_SCHEMA, "mode": "PROFILE_ONLY",
                "feature_id": request["feature_id"], "state": state,
                "request_sha256": request["request_sha256"], "request": request,
                "envelope": envelope,
            }
            if len(_canonical_bytes(result)) > MAX_PROFILE_RESULT_BYTES:
                raise InvalidConfiguration("PROFILE_ONLY result exceeds its 2000000-byte bound")
            return result, 0
        payload = _finalization_payload(request, terminal, envelope)
        return {
            "schema": COLUMN_RESULT_SCHEMA,
            "mode": "CAUSAL",
            "feature_id": request["feature_id"],
            "state": state,
            "request_sha256": request["request_sha256"],
            "envelope": envelope,
            "finalization_payload": payload,
        }, 0
    except Exception as trouble:  # one typed result is the transport contract
        return {
            "schema": COLUMN_RESULT_SCHEMA,
            "feature_id": request.get("feature_id", "UNKNOWN"),
            "state": "FAILED",
            "failure_class": type(trouble).__name__,
            "reason": str(trouble)[:1000],
        }, 1


def _verified_plan(path: Path) -> dict[str, Any]:
    plan = _load_json(path)
    if plan.get("schema") != ORCHESTRATOR_PLAN_SCHEMA:
        raise InvalidConfiguration(f"plan schema must be {ORCHESTRATOR_PLAN_SCHEMA}")
    claimed = plan.get("plan_sha256")
    if claimed != EW.digest({key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise InvalidConfiguration("plan_sha256 does not cover PLAN.json")
    items = plan.get("items") or []
    if len(items) != plan.get("inventory_total") or len({item.get("feature_id") for item in items}) != len(items):
        raise IncompleteInventory("PLAN.json inventory denominator is inconsistent")
    if plan.get("mode", "CAUSAL") not in EXECUTION_MODES:
        raise InvalidConfiguration(f"PLAN.json mode must be one of {sorted(EXECUTION_MODES)}")
    return plan


def _verified_orchestrator_terminal(path: Path, item: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    terminal = _load_json(path)
    claimed = terminal.get("terminal_sha256")
    if terminal.get("schema") != ORCHESTRATOR_TERMINAL_SCHEMA or claimed != EW.digest(
            {key: value for key, value in terminal.items() if key != "terminal_sha256"}):
        raise IncompleteInventory(f"invalid orchestrator terminal for {item['feature_id']}")
    bindings = {
        "feature_id": item["feature_id"], "key": item["key"],
        "plan_sha256": plan["plan_sha256"], "inventory_sha256": plan["inventory_sha256"],
    }
    for key, value in bindings.items():
        if terminal.get(key) != value:
            raise IncompleteInventory(f"terminal {key} contradicts PLAN.json for {item['feature_id']}")
    if terminal.get("state") not in {"COMPLETED", "UNAVAILABLE"}:
        raise IncompleteInventory(f"terminal state is not closure-eligible for {item['feature_id']}")
    result = terminal.get("result")
    if not isinstance(result, dict) or result.get("schema") != COLUMN_RESULT_SCHEMA \
            or result.get("feature_id") != item["feature_id"] or result.get("state") != terminal["state"]:
        raise IncompleteInventory(f"terminal result identity is invalid for {item['feature_id']}")
    if result.get("mode", "CAUSAL") != "CAUSAL":
        raise IncompleteInventory(f"PROFILE_ONLY result cannot authorize causal closure for {item['feature_id']}")
    payload = result.get("finalization_payload")
    if payload is None:
        if terminal["state"] == "UNAVAILABLE" and item.get("available") is False:
            return terminal
        raise IncompleteInventory(f"terminal has no finalization payload for {item['feature_id']}")
    if not isinstance(payload, dict) or payload.get("schema") != FINALIZATION_PAYLOAD_SCHEMA:
        raise IncompleteInventory(f"unknown finalization payload for {item['feature_id']}")
    if len(_canonical_bytes(payload)) > MAX_FINALIZATION_PAYLOAD_BYTES:
        raise IncompleteInventory(f"finalization payload exceeds its bound for {item['feature_id']}")
    if payload.get("payload_sha256") != EW.digest(
            {key: value for key, value in payload.items() if key != "payload_sha256"}):
        raise IncompleteInventory(f"finalization payload digest is invalid for {item['feature_id']}")
    request = payload.get("request")
    if not isinstance(request, dict) or request.get("request_sha256") != EW.digest(
            {key: value for key, value in request.items() if key != "request_sha256"}):
        raise IncompleteInventory(f"worker request digest is invalid for {item['feature_id']}")
    request_bindings = {
        "schema": COLUMN_REQUEST_SCHEMA, "feature_id": item["feature_id"], "feature_key": item["key"],
        "population_id": plan["population_id"], "target_pack": plan["target_pack"],
        "plan_sha256": plan["plan_sha256"], "inventory_sha256": plan["inventory_sha256"],
        "inventory_row_sha256": item["inventory_row_sha256"],
    }
    for key, value in request_bindings.items():
        if request.get(key) != value:
            raise IncompleteInventory(f"worker request {key} contradicts PLAN.json for {item['feature_id']}")
    payload_bindings = {
        "feature_id": item["feature_id"], "population_id": plan["population_id"],
        "target_pack": plan["target_pack"], "plan_sha256": plan["plan_sha256"],
        "inventory_sha256": plan["inventory_sha256"],
    }
    for key, value in payload_bindings.items():
        if payload.get(key) != value:
            raise IncompleteInventory(f"finalization payload {key} contradicts PLAN.json for {item['feature_id']}")
    if EW.digest(request.get("inventory_row")) != item["inventory_row_sha256"]:
        raise IncompleteInventory(f"worker inventory row is invalid for {item['feature_id']}")
    if result.get("request_sha256") != request["request_sha256"]:
        raise IncompleteInventory(f"worker result request identity is invalid for {item['feature_id']}")
    envelope = result.get("envelope")
    if not isinstance(envelope, dict):
        raise IncompleteInventory(f"worker result has no envelope for {item['feature_id']}")
    _validate_warehouse_envelope(envelope)
    if payload.get("unit_envelope_sha256") != envelope["envelope_sha256"]:
        raise IncompleteInventory(f"payload/envelope identity mismatch for {item['feature_id']}")
    expected = TARGET_CELL_COUNTS.get(plan["population_id"])
    cells = payload.get("raw_causal_cells")
    if not isinstance(cells, list) or len(cells) != expected:
        raise IncompleteInventory(f"causal cell denominator is invalid for {item['feature_id']}")
    targets = set()
    for cell in cells:
        target = (cell.get("target"), cell.get("horizon_h"))
        if target in targets:
            raise IncompleteInventory(f"causal evidence repeats a target for {item['feature_id']}")
        targets.add(target)
        if cell.get("feature_id") != item["feature_id"] or "p" not in cell.get("rung1", {}) \
                or "p_linear" not in cell.get("rung2", {}):
            raise IncompleteInventory(f"causal evidence is incomplete for {item['feature_id']}")
        sypi = (cell.get("discovery") or {}).get("sypi") or {}
        if sypi.get("state") == "RUN" and "condition1_p" not in sypi:
            raise IncompleteInventory(f"SyPI evidence is incomplete for {item['feature_id']}")
    if (payload.get("feature_record") or {}).get("feature_id") != item["feature_id"]:
        raise IncompleteInventory(f"feature record is invalid for {item['feature_id']}")
    return terminal


def _verified_profile_terminal(path: Path, item: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    terminal = _load_json(path)
    claimed = terminal.get("terminal_sha256")
    unsigned = {key: value for key, value in terminal.items() if key != "terminal_sha256"}
    if terminal.get("schema") != ORCHESTRATOR_TERMINAL_SCHEMA or claimed != EW.digest(unsigned):
        raise IncompleteInventory(f"invalid profile terminal for {item['feature_id']}")
    bindings = {
        "feature_id": item["feature_id"], "key": item["key"],
        "plan_sha256": plan["plan_sha256"], "inventory_sha256": plan["inventory_sha256"],
    }
    for key, value in bindings.items():
        if terminal.get(key) != value:
            raise IncompleteInventory(f"profile terminal {key} contradicts PLAN.json for {item['feature_id']}")
    if terminal.get("state") != "COMPLETED":
        raise IncompleteInventory(
            f"profile terminal is not complete for {item['feature_id']}: {terminal.get('state')}"
        )
    result = terminal.get("result")
    if not isinstance(result, dict) or result.get("schema") != COLUMN_RESULT_SCHEMA \
            or result.get("mode") != "PROFILE_ONLY" or result.get("feature_id") != item["feature_id"] \
            or result.get("state") != terminal["state"]:
        raise IncompleteInventory(f"profile result identity is invalid for {item['feature_id']}")
    if "finalization_payload" in result:
        raise IncompleteInventory(f"profile result carries a causal finalization payload for {item['feature_id']}")
    if len(_canonical_bytes(result)) > MAX_PROFILE_RESULT_BYTES:
        raise IncompleteInventory(f"profile result exceeds its bound for {item['feature_id']}")
    request = result.get("request")
    if not isinstance(request, dict) or request.get("request_sha256") != EW.digest(
            {key: value for key, value in request.items() if key != "request_sha256"}):
        raise IncompleteInventory(f"profile worker request digest is invalid for {item['feature_id']}")
    request_bindings = {
        "schema": COLUMN_REQUEST_SCHEMA, "feature_id": item["feature_id"], "feature_key": item["key"],
        "population_id": plan["population_id"], "target_pack": plan["target_pack"],
        "plan_sha256": plan["plan_sha256"], "inventory_sha256": plan["inventory_sha256"],
        "inventory_row_sha256": item["inventory_row_sha256"],
    }
    for key, value in request_bindings.items():
        if request.get(key) != value:
            raise IncompleteInventory(f"profile request {key} contradicts PLAN.json for {item['feature_id']}")
    if EW.digest(request.get("inventory_row")) != item["inventory_row_sha256"]:
        raise IncompleteInventory(f"profile inventory row is invalid for {item['feature_id']}")
    if result.get("request_sha256") != request["request_sha256"]:
        raise IncompleteInventory(f"profile result request identity is invalid for {item['feature_id']}")
    envelope = result.get("envelope")
    if not isinstance(envelope, dict):
        raise IncompleteInventory(f"profile result has no envelope for {item['feature_id']}")
    _validate_warehouse_envelope(envelope)
    if envelope["rows"]["causal_evidence"] or envelope["rows"]["selection_decisions"]:
        raise IncompleteInventory(f"profile envelope contains causal closure rows for {item['feature_id']}")
    if envelope["run"].get("campaign_sha256") != plan.get("campaign_sha256") \
            or envelope["run"].get("inventory_sha256") != plan["inventory_sha256"]:
        raise IncompleteInventory(f"profile campaign identity is invalid for {item['feature_id']}")
    noncausal = ("sampling_quality", "variable_profiles", "information_metrics", "pair_relations")
    if terminal["state"] == "COMPLETED" and any(not envelope["rows"][family] for family in noncausal):
        raise IncompleteInventory(f"profile result omits a metric family for {item['feature_id']}")
    for family in noncausal:
        if any(row.get("feature_id") != item["feature_id"] for row in envelope["rows"][family]):
            raise IncompleteInventory(f"profile {family} rows belong to another feature")
    return terminal


def merge_profile_terminals(plan_path: str | Path, terminals_dir: str | Path,
                            output_path: str | Path) -> dict[str, Any]:
    """Merge a complete retained PROFILE_ONLY denominator without remote paths."""
    plan = _verified_plan(Path(plan_path).expanduser().resolve())
    if plan.get("mode") != "PROFILE_ONLY" or not isinstance(plan.get("campaign_sha256"), str):
        raise InvalidConfiguration("profile merge requires a distinct PROFILE_ONLY campaign")
    terminals_root = Path(terminals_dir).expanduser().resolve()
    expected_names = {f"{item['key']}.json" for item in plan["items"]}
    actual_names = {path.name for path in terminals_root.glob("*.json")}
    if actual_names != expected_names:
        missing, unexpected = sorted(expected_names - actual_names), sorted(actual_names - expected_names)
        raise IncompleteInventory(f"profile denominator mismatch; missing={missing[:3]}, unexpected={unexpected[:3]}")
    verified = []
    for item in plan["items"]:
        path = terminals_root / f"{item['key']}.json"
        verified.append((item, _verified_profile_terminal(path, item, plan)))
    rows = {family: [] for family in EW.ROW_FAMILIES}
    created_times = set()
    for _, terminal in verified:
        envelope = terminal["result"]["envelope"]
        created_times.add(envelope["run"]["created_at"])
        for family in ("sampling_quality", "variable_profiles", "information_metrics", "pair_relations"):
            for sealed in envelope["rows"][family]:
                row = {key: value for key, value in sealed.items() if key != "row_sha256"}
                row["population_id"] = plan["inventory_sha256"]
                rows[family].append(row)
    if len(created_times) != 1:
        raise IncompleteInventory("profile unit envelopes disagree on campaign clock")
    for family in rows:
        rows[family].sort(key=EW.canonical_json)
    envelope = EW.build_envelope({
        "run_id": f"phase1-profile:{plan['population_id']}:{plan['plan_sha256'][:16]}",
        "campaign_sha256": plan["campaign_sha256"],
        "code_sha256": EW.digest({
            "worker": _digest_file(Path(__file__)),
            "profile": _digest_file(Path(sys.modules[_profile_series.__module__].__file__)),
            "envelope": _digest_file(Path(EW.__file__)),
        }),
        "input_sha256": EW.digest(sorted(terminal["terminal_sha256"] for _, terminal in verified)),
        "inventory_sha256": plan["inventory_sha256"],
        "created_at": next(iter(created_times)),
    }, rows)
    result = {
        "schema": PROFILE_MERGE_SCHEMA, "mode": "PROFILE_ONLY", "state": "PROFILE_METRICS_COMPLETE",
        "terminal_count": len(verified),
        "completed_count": sum(terminal["state"] == "COMPLETED" for _, terminal in verified),
        "unavailable_count": sum(terminal["state"] == "UNAVAILABLE" for _, terminal in verified),
        "plan_sha256": plan["plan_sha256"], "inventory_sha256": plan["inventory_sha256"],
        "envelope": envelope,
    }
    result["result_sha256"] = EW.digest(result)
    if len(_canonical_bytes(result)) > MAX_PROFILE_MERGE_BYTES:
        raise IncompleteInventory("merged profile result exceeds its 256000000-byte bound")
    _write_json_atomic(Path(output_path).expanduser().resolve(), result)
    return result


def _authenticated_envelope(document: Any, label: str) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise IncompleteInventory(f"{label} is not an envelope object")
    try:
        _validate_warehouse_envelope(document)
    except (InvalidConfiguration, KeyError, TypeError, ValueError) as trouble:
        raise IncompleteInventory(f"{label} envelope or row digest is invalid: {trouble}") from trouble
    return document


def _safe_adoption_path(bundle_root: Path, report_dir: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise IncompleteInventory("adoption envelope path is invalid")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise IncompleteInventory("adoption envelope path escapes the adoption directory")
    root = bundle_root.resolve()
    candidates = [(bundle_root / candidate).resolve(), (report_dir / candidate).resolve()]
    for resolved in candidates:
        if root in resolved.parents and resolved.is_file():
            return resolved
    raise IncompleteInventory(f"adopted envelope is missing or escapes the bundle: {relative}")


def _verified_bundle_members(bundle_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = bundle_root / "BUNDLE_MANIFEST.json"
    if not manifest_path.is_file():
        raise IncompleteInventory("predictor adoption bundle has no BUNDLE_MANIFEST.json")
    manifest = _load_json(manifest_path)
    claimed = manifest.get("bundle_sha256")
    if manifest.get("schema") != "phase1.deployment_bundle.v1" or claimed != EW.digest(
            {key: value for key, value in manifest.items() if key != "bundle_sha256"}):
        raise IncompleteInventory("predictor bundle manifest digest is invalid")
    members = manifest.get("files")
    if not isinstance(members, list):
        raise IncompleteInventory("predictor bundle manifest has no file denominator")
    by_path = {}
    for item in members:
        relative = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative, str) or relative in by_path:
            raise IncompleteInventory("predictor bundle manifest has an invalid or duplicate member")
        candidate = Path(relative)
        resolved = (bundle_root / candidate).resolve()
        if candidate.is_absolute() or ".." in candidate.parts or bundle_root.resolve() not in resolved.parents:
            raise IncompleteInventory("predictor bundle manifest member escapes the bundle")
        by_path[relative] = item
    return by_path


def _verify_bundle_member(bundle_root: Path, path: Path, members: dict[str, dict[str, Any]]) -> None:
    relative = str(path.resolve().relative_to(bundle_root.resolve()))
    item = members.get(relative)
    if item is None or item.get("bytes") != path.stat().st_size or item.get("sha256") != _digest_file(path):
        raise IncompleteInventory(f"predictor bundle member digest mismatch: {relative}")


def _verify_profile_merge_result(path: Path) -> tuple[dict[str, Any], set[str]]:
    result = _load_json(path)
    claimed = result.get("result_sha256")
    if claimed != EW.digest({key: value for key, value in result.items() if key != "result_sha256"}):
        raise IncompleteInventory("profile merge result digest is invalid")
    expected = {
        "schema": PROFILE_MERGE_SCHEMA, "mode": "PROFILE_ONLY",
        "state": "PROFILE_METRICS_COMPLETE", "terminal_count": EURUSD_FEATURE_COUNT,
        "completed_count": EURUSD_FEATURE_COUNT, "unavailable_count": 0,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise IncompleteInventory(f"profile denominator must be EURUSD 366/366: {key}")
    inventory = result.get("inventory_sha256")
    if not isinstance(inventory, str) or len(inventory) != 64:
        raise IncompleteInventory("profile inventory identity is invalid")
    envelope = _authenticated_envelope(result.get("envelope"), "profile merge")
    if envelope["run"].get("inventory_sha256") != inventory:
        raise IncompleteInventory("profile result and envelope inventory identities differ")
    if envelope["rows"]["causal_evidence"] or envelope["rows"]["selection_decisions"]:
        raise IncompleteInventory("profile merge contains causal rows")
    families = ("sampling_quality", "variable_profiles", "information_metrics", "pair_relations")
    populations = set()
    feature_sets = {}
    for family in families:
        rows = envelope["rows"][family]
        identities = set()
        for row in rows:
            identity = EW.canonical_json({key: value for key, value in row.items() if key != "row_sha256"})
            if identity in identities:
                raise IncompleteInventory(f"profile merge contains a duplicate {family} row")
            identities.add(identity)
            populations.add(row.get("population_id"))
        feature_sets[family] = {row["feature_id"] for row in rows}
    denominator = feature_sets["sampling_quality"]
    if len(denominator) != EURUSD_FEATURE_COUNT or any(features != denominator for features in feature_sets.values()):
        raise IncompleteInventory("profile metric families do not share the exact 366-feature denominator")
    if populations != {inventory}:
        raise IncompleteInventory("profile metric rows do not share the inventory population identity")
    return result, denominator


def _verify_adopted_evidence(adoption_dir: Path,
                             expected_features: set[str]) -> tuple[dict[str, Any], str,
                                                                   list[dict[str, Any]],
                                                                   list[dict[str, Any]], list[str]]:
    if (adoption_dir / "ADOPTION.json").is_file():
        report_dir, bundle_root = adoption_dir, adoption_dir.parent
    elif (adoption_dir / "adoption" / "ADOPTION.json").is_file():
        report_dir, bundle_root = adoption_dir / "adoption", adoption_dir
    else:
        raise IncompleteInventory("predictor bundle has no adoption/ADOPTION.json")
    bundle_members = _verified_bundle_members(bundle_root)
    report_path = report_dir / "ADOPTION.json"
    _verify_bundle_member(bundle_root, report_path, bundle_members)
    report = _load_json(report_path)
    claimed = report.get("adoption_sha256")
    if claimed != EW.digest({key: value for key, value in report.items() if key != "adoption_sha256"}):
        raise IncompleteInventory("adoption report digest is invalid")
    if report.get("schema") != "phase1.evidence_adoption.v1" \
            or report.get("state") != "ADOPTED_VERIFIED_EVIDENCE" \
            or report.get("population_id") != "EURUSD":
        raise IncompleteInventory("adoption report is not retained verified EURUSD evidence")
    paths = report.get("envelopes")
    if report.get("feature_count") != EURUSD_FEATURE_COUNT or not isinstance(paths, list) \
            or len(paths) != EURUSD_FEATURE_COUNT or len(set(paths)) != len(paths):
        raise IncompleteInventory("adoption feature denominator is not exactly 366")
    if report.get("causal_rows") != EURUSD_FEATURE_COUNT * len(EURUSD_TARGET_DENOMINATOR):
        raise IncompleteInventory("adoption causal target-cell denominator is invalid")
    causal, decisions, envelope_hashes = [], [], []
    all_features = set()
    adopted_inventories = set()
    for relative in paths:
        path = _safe_adoption_path(bundle_root, report_dir, relative)
        _verify_bundle_member(bundle_root, path, bundle_members)
        envelope = _authenticated_envelope(_load_json(path), f"adopted {relative}")
        adopted_inventory = envelope["run"].get("inventory_sha256")
        if not isinstance(adopted_inventory, str) or len(adopted_inventory) != 64:
            raise IncompleteInventory(f"adopted envelope inventory identity is invalid: {relative}")
        adopted_inventories.add(adopted_inventory)
        expected_population = EW.digest({
            "population": "EURUSD", "inventory_sha256": adopted_inventory,
        })
        if any(row.get("population_id") != expected_population
               for family in EW.ROW_FAMILIES for row in envelope["rows"][family]):
            raise IncompleteInventory(f"adopted rows have the wrong population identity: {relative}")
        envelope_hashes.append(envelope["envelope_sha256"])
        envelope_features = {
            row["feature_id"] for family in EW.ROW_FAMILIES for row in envelope["rows"][family]
        }
        if len(envelope_features) != 1:
            raise IncompleteInventory(f"adopted envelope does not belong to exactly one feature: {relative}")
        all_features.update(envelope_features)
        causal.extend(envelope["rows"]["causal_evidence"])
        decisions.extend(envelope["rows"]["selection_decisions"])
    if all_features != expected_features:
        raise IncompleteInventory("adopted and profile feature populations have a missing or foreign feature")
    if len(adopted_inventories) != 1:
        raise IncompleteInventory("adopted envelopes do not share a single inventory identity")
    adopted_inventory = next(iter(adopted_inventories))
    causal_keys, decision_keys = set(), set()
    rungs_by_cell: dict[tuple[str, str, int], set[int]] = {}
    decisions_by_cell: set[tuple[str, str, int]] = set()
    for row in causal:
        key = (row["feature_id"], row["target_id"], int(row["horizon"]), int(row["rung"]))
        if key in causal_keys:
            raise IncompleteInventory("adopted causal evidence contains a duplicate row")
        causal_keys.add(key)
        cell = key[:3]
        rungs_by_cell.setdefault(cell, set()).add(key[3])
    for row in decisions:
        key = (row["feature_id"], row["target_id"], int(row["horizon"]), row["method"])
        if key in decision_keys:
            raise IncompleteInventory("adopted selection decisions contain a duplicate row")
        decision_keys.add(key)
        decisions_by_cell.add(key[:3])
    expected_cells = {
        (feature, target, horizon)
        for feature in expected_features for target, horizon in EURUSD_TARGET_DENOMINATOR
    }
    if set(rungs_by_cell) != expected_cells or decisions_by_cell != expected_cells:
        raise IncompleteInventory("adopted causal evidence has a missing or foreign target cell")
    if any(rungs != {1, 2, 3} for rungs in rungs_by_cell.values()):
        raise IncompleteInventory("adopted causal target cell does not contain exactly rungs 1, 2 and 3")
    return report, adopted_inventory, causal, decisions, envelope_hashes


def combine_adopted_eurusd(profile_result_path: str | Path, adoption_dir: str | Path,
                            output_path: str | Path) -> dict[str, Any]:
    """Compose fresh profile metrics with retained adopted causal decisions."""
    profile_path = Path(profile_result_path).expanduser().resolve()
    adoption_root = Path(adoption_dir).expanduser().resolve()
    profile, features = _verify_profile_merge_result(profile_path)
    profile_inventory = profile["inventory_sha256"]
    report, adopted_inventory, causal, decisions, adopted_hashes = _verify_adopted_evidence(
        adoption_root, features,
    )
    population = EW.digest({
        "population": "EURUSD",
        "profile_inventory_sha256": profile_inventory,
        "adopted_inventory_sha256": adopted_inventory,
    })
    rows = {family: [] for family in EW.ROW_FAMILIES}
    for family in ("sampling_quality", "variable_profiles", "information_metrics", "pair_relations"):
        for sealed in profile["envelope"]["rows"][family]:
            row = {key: value for key, value in sealed.items() if key != "row_sha256"}
            row["population_id"] = population
            rows[family].append(row)
    rows["causal_evidence"] = [
        {**{key: value for key, value in row.items() if key not in {"row_sha256", "population_id"}},
         "population_id": population}
        for row in causal
    ]
    rows["selection_decisions"] = [
        {**{key: value for key, value in row.items() if key not in {"row_sha256", "population_id"}},
         "population_id": population}
        for row in decisions
    ]
    for family in rows:
        rows[family].sort(key=EW.canonical_json)
    source_identity = {
        "profile_inventory_sha256": profile_inventory,
        "adopted_inventory_sha256": adopted_inventory,
        "profile_result_sha256": profile["result_sha256"],
        "adoption_sha256": report["adoption_sha256"],
        "adopted_envelope_sha256": sorted(adopted_hashes),
    }
    envelope = EW.build_envelope({
        "run_id": f"phase1-eurusd-final:{EW.digest(source_identity)[:16]}",
        "campaign_sha256": EW.digest({
            "profile_campaign_sha256": profile["envelope"]["run"]["campaign_sha256"],
            "adoption_sha256": report["adoption_sha256"],
        }),
        "code_sha256": EW.digest({
            "combiner": _digest_file(Path(__file__)), "envelope": _digest_file(Path(EW.__file__)),
        }),
        "input_sha256": EW.digest(source_identity), "inventory_sha256": population,
        "created_at": profile["envelope"]["run"]["created_at"],
    }, rows)
    _authenticated_envelope(envelope, "combined")
    payload = _canonical_bytes(envelope)
    if len(payload) > MAX_PROFILE_MERGE_BYTES:
        raise IncompleteInventory("combined feature-selection envelope exceeds its 256000000-byte bound")
    _write_bytes_atomic(Path(output_path).expanduser().resolve(), payload)
    return envelope


def finalize_orchestrator_terminals(plan_path: str | Path, terminals_dir: str | Path,
                                    output_path: str | Path) -> dict[str, Any]:
    """Apply global correction using only payloads retained by the coordinator."""
    plan = _verified_plan(Path(plan_path).expanduser().resolve())
    if plan.get("mode", "CAUSAL") != "CAUSAL":
        raise InvalidConfiguration("PROFILE_ONLY campaigns cannot enter causal finalization")
    terminals_root = Path(terminals_dir).expanduser().resolve()
    terminals = []
    for item in plan["items"]:
        terminal_path = terminals_root / f"{item['key']}.json"
        if not terminal_path.is_file():
            raise IncompleteInventory(f"missing terminal for {item['feature_id']}")
        terminals.append((item, _verified_orchestrator_terminal(terminal_path, item, plan)))
    unexpected = sorted(path.name for path in terminals_root.glob("*.json")
                        if path.stem not in {item["key"] for item in plan["items"]})
    if unexpected:
        raise IncompleteInventory(f"terminal directory contains unplanned files: {unexpected[:3]}")
    payloads = [terminal["result"]["finalization_payload"] for _, terminal in terminals
                if terminal["result"].get("finalization_payload") is not None]
    with tempfile.TemporaryDirectory(prefix="phase1-finalizer-") as temporary:
        working = Path(temporary)
        chunks = []
        for index, payload in enumerate(payloads):
            chunk_id = f"unit_{index:05d}"
            chunk = working / chunk_id
            chunk.mkdir()
            _write_bytes_atomic(chunk / "cells.jsonl", b"".join(
                _canonical_bytes(cell) for cell in payload["raw_causal_cells"]
            ))
            record = payload["feature_record"]
            _write_json_atomic(chunk / "features.json", {
                "features": [record], "failures": [], "cost": {"wall_s": record.get("cost_s", 0.0)},
            })
            _write_json_atomic(chunk / "READY", {"payload_sha256": payload["payload_sha256"]})
            chunks.append({"id": chunk_id, "features": [payload["feature_id"]]})
        final_cells = []
        summary = {"cells": 0, "per_state": {}}
        if chunks:
            raw_cells = [cell for payload in payloads for cell in payload["raw_causal_cells"]]
            _write_json_atomic(working / "plan.json", {
                "schema": "fs_causal_plan.v1", "revision": plan["plan_sha256"], "chunks": chunks,
                "candidates_total": len(chunks), "cells_total": len(raw_cells),
                "targets": sorted({cell["target"] for cell in raw_cells}), "seed": fc.SEED,
                "families": {"rung1": "per target", "rung2": "per target",
                             "sypi_condition1": "per target"},
            })
            summary = FB.finalize(str(working))
            final_cells = [json.loads(line) for line in (working / "causal_evidence.jsonl").read_text().splitlines() if line]
    rows = {family: [] for family in EW.ROW_FAMILIES}
    campaign_hashes, created_times = set(), set()
    for _, terminal in terminals:
        envelope = terminal["result"].get("envelope")
        if not isinstance(envelope, dict):
            continue
        campaign_hashes.add(envelope["run"]["campaign_sha256"])
        created_times.add(envelope["run"]["created_at"])
        for family in ("sampling_quality", "variable_profiles", "information_metrics", "pair_relations"):
            rows[family].extend({key: value for key, value in row.items() if key != "row_sha256"}
                                for row in envelope["rows"][family])
    if len(campaign_hashes) > 1 or len(created_times) > 1:
        raise IncompleteInventory("unit envelopes disagree on campaign identity")
    population_identity = plan["inventory_sha256"]
    rows["causal_evidence"] = EW.causal_rows(final_cells, population_identity, final=True)
    rows["selection_decisions"] = EW.selection_rows(final_cells, population_identity)
    envelope = EW.build_envelope({
        "run_id": f"phase1-final:{plan['population_id']}:{plan['plan_sha256'][:16]}",
        "campaign_sha256": next(iter(campaign_hashes), plan["config_sha256"]),
        "code_sha256": EW.digest({"worker": _digest_file(Path(__file__)), "batch": _digest_file(Path(FB.__file__))}),
        "input_sha256": EW.digest(sorted(terminal["terminal_sha256"] for _, terminal in terminals)),
        "inventory_sha256": population_identity,
        "created_at": next(iter(created_times), "1970-01-01T00:00:00Z"),
    }, rows)
    result = {
        "schema": ORCHESTRATOR_FINALIZER_SCHEMA, "state": "PHASE_1_COMPLETE",
        "terminal_count": len(terminals), "completed_count": sum(t["state"] == "COMPLETED" for _, t in terminals),
        "unavailable_count": sum(t["state"] == "UNAVAILABLE" for _, t in terminals),
        "plan_sha256": plan["plan_sha256"], "inventory_sha256": plan["inventory_sha256"],
        "cell_count": summary["cells"], "envelope": envelope,
    }
    result["result_sha256"] = EW.digest(result)
    _write_json_atomic(Path(output_path).expanduser().resolve(), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdio", action="store_true", help="Read one phase1.column_request.v1 from stdin")
    parser.add_argument("--deployment-manifest", type=Path)
    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser("run-column", help="Profile and run raw causal evidence for one TRAIN column")
    run_parser.add_argument("--config", type=Path, required=True)
    finalize_parser = commands.add_parser("finalize-inventory", help="Verify all units and apply global BH/FDR")
    finalize_parser.add_argument("--manifest", type=Path, required=True)
    terminal_parser = commands.add_parser("finalize-terminals", help="Finalize retained orchestrator terminals")
    terminal_parser.add_argument("--plan", type=Path, required=True)
    terminal_parser.add_argument("--terminals", type=Path, required=True)
    terminal_parser.add_argument("--output", type=Path, required=True)
    profile_parser = commands.add_parser("merge-profile-terminals",
                                         help="Merge retained PROFILE_ONLY terminals")
    profile_parser.add_argument("--plan", type=Path, required=True)
    profile_parser.add_argument("--terminals", type=Path, required=True)
    profile_parser.add_argument("--output", type=Path, required=True)
    combine_parser = commands.add_parser(
        "combine-adopted-eurusd", help="Combine complete EURUSD profiles with adopted causal evidence",
    )
    combine_parser.add_argument("--profile-result", type=Path, required=True)
    combine_parser.add_argument("--adoption-dir", type=Path, required=True)
    combine_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stdio:
        manifest = args.deployment_manifest or os.environ.get("PHASE1_COLUMN_WORKER_DEPLOYMENT")
        if not manifest:
            result, code = ({"schema": COLUMN_RESULT_SCHEMA, "feature_id": "UNKNOWN", "state": "FAILED",
                             "failure_class": "InvalidConfiguration",
                             "reason": "--deployment-manifest or PHASE1_COLUMN_WORKER_DEPLOYMENT is required"}, 1)
        else:
            result, code = run_stdio(manifest, sys.stdin.read())
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
        return code
    if args.command is None:
        parser.error("a command or --stdio is required")
    try:
        if args.command == "run-column":
            result = run_column(args.config)
        elif args.command == "finalize-inventory":
            result = finalize_inventory(args.manifest)
        elif args.command == "finalize-terminals":
            result = finalize_orchestrator_terminals(args.plan, args.terminals, args.output)
        elif args.command == "merge-profile-terminals":
            result = merge_profile_terminals(args.plan, args.terminals, args.output)
        else:
            envelope = combine_adopted_eurusd(args.profile_result, args.adoption_dir, args.output)
            result = {
                "schema": "phase1.combined_envelope_receipt.v1", "state": "COMPLETED",
                "output": str(args.output), "envelope_sha256": envelope["envelope_sha256"],
                "run_id": envelope["run"]["run_id"],
            }
    except (InvalidConfiguration, IncompleteInventory, OSError, ValueError) as trouble:
        print(json.dumps({"status": "REFUSED", "reason": str(trouble)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, default=ps3c._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
