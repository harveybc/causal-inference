"""Generic, CPU-only phase-1 worker for one inventory time series.

The worker profiles one TRAIN column and delegates all causal calculations to
``fs_causal_batch.run_feature``. Per-column output is deliberately raw: causal
states become final only when ``finalize-inventory`` verifies the complete
inventory and delegates multiplicity correction to ``fs_causal_batch.finalize``.
"""

from __future__ import annotations

import argparse
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


UNIT_SCHEMA = "feature_selection_unit.v1"
INVENTORY_SCHEMA = "feature_selection_inventory.v1"
ENVELOPE_SCHEMA = "feature_selection_envelope.v1"
TERMINAL_SCHEMA = "feature_selection_terminal.v1"
FINAL_TERMINAL_SCHEMA = "feature_selection_inventory_terminal.v1"
TARGET_PACKS = {"EURUSD", "ETH"}


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


def _read_table(spec: dict[str, Any]) -> pd.DataFrame:
    path = Path(spec["path"]).expanduser().resolve()
    kind = str(spec.get("format") or path.suffix.lstrip(".")).lower()
    if kind == "csv":
        return pd.read_csv(path)
    if kind in {"parquet", "pq"}:
        return pd.read_parquet(path)
    raise InvalidConfiguration(f"unsupported table format: {kind}")


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def profile_series(frame: pd.DataFrame, timestamp_column: str, feature_column: str,
                   frequency: str, calendar: str) -> dict[str, Any]:
    """Return a typed PS1 profile without imputing or reading outside this frame."""
    times = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    values = pd.to_numeric(frame[feature_column], errors="coerce").to_numpy(float)
    valid_times = times.dropna().sort_values()
    duplicate_timestamps = int(valid_times.duplicated().sum())
    missing_timestamps = None
    expected_timestamps = None
    if len(valid_times):
        expected = pd.date_range(valid_times.iloc[0], valid_times.iloc[-1], freq=frequency)
        expected_timestamps = int(len(expected))
        missing_timestamps = int(len(expected.difference(pd.DatetimeIndex(valid_times.unique()))))
    finite = np.isfinite(values)
    finite_values = values[finite]
    quantiles = {}
    if len(finite_values):
        quantiles = {str(q): float(np.quantile(finite_values, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
        q1, median, q3 = (quantiles["0.25"], quantiles["0.5"], quantiles["0.75"])
        iqr = q3 - q1
        mad = float(np.median(np.abs(finite_values - median)))
        outliers_iqr = int(np.sum((finite_values < q1 - 1.5 * iqr) | (finite_values > q3 + 1.5 * iqr)))
        outliers_mad = int(np.sum(np.abs(finite_values - median) > 3.5 * 1.4826 * mad)) if mad else 0
        mean, std = float(np.mean(finite_values)), float(np.std(finite_values))
        centered = finite_values - mean
        skewness = float(np.mean(centered ** 3) / std ** 3) if std else None
        kurtosis = float(np.mean(centered ** 4) / std ** 4 - 3.0) if std else None
    else:
        iqr = mad = mean = std = skewness = kurtosis = None
        outliers_iqr = outliers_mad = 0
    finite_pairs = finite[1:] & finite[:-1]
    differences = np.diff(values)[finite_pairs]
    return {
        "schema": "feature_selection_ps1_profile.v1",
        "sampling": {
            "calendar": calendar,
            "frequency": frequency,
            "rows": int(len(frame)),
            "valid_timestamps": int(times.notna().sum()),
            "first_timestamp": None if not len(valid_times) else valid_times.iloc[0].isoformat(),
            "last_timestamp": None if not len(valid_times) else valid_times.iloc[-1].isoformat(),
            "expected_timestamps": expected_timestamps,
            "missing_timestamps": missing_timestamps,
            "duplicate_timestamps": duplicate_timestamps,
            "monotonic_non_decreasing": bool(times.dropna().is_monotonic_increasing),
        },
        "quality": {
            "finite": int(finite.sum()),
            "missing_or_nonfinite": int((~finite).sum()),
            "longest_missing_run": _longest_run(~finite),
            "constant": bool(len(finite_values) and np.ptp(finite_values) == 0),
            "outliers_iqr_1_5": outliers_iqr,
            "outliers_robust_z_3_5": outliers_mad,
        },
        "distribution": {
            "mean": mean,
            "std_population": std,
            "min": None if not len(finite_values) else float(np.min(finite_values)),
            "max": None if not len(finite_values) else float(np.max(finite_values)),
            "quantiles": quantiles,
            "iqr": iqr,
            "mad": mad,
            "skewness": skewness,
            "excess_kurtosis": kurtosis,
        },
        "dynamics": {
            "finite_differences": int(len(differences)),
            "difference_std": None if not len(differences) else float(np.std(differences)),
            "zero_difference_fraction": None if not len(differences) else float(np.mean(differences == 0)),
        },
    }


def _profile(config: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    supplied = config.get("ps1_profile")
    if supplied:
        payload = _load_json(Path(supplied["path"])) if "path" in supplied else supplied.get("payload")
        if not isinstance(payload, dict) or payload.get("schema") != "feature_selection_ps1_profile.v1":
            raise InvalidConfiguration("supplied PS1 profile has the wrong schema")
        return payload
    dataset = config["dataset"]
    return profile_series(frame, dataset["timestamp_column"], dataset["feature_column"],
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
    context = list(dict.fromkeys([*pack.get("history_columns", []), *pack.get("pre_return_columns", []),
                                  *pack.get("calendar_locator_columns", [])]))
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
    for key in ("unit_id", "dataset", "target_pack", "output_dir"):
        if key not in config:
            raise InvalidConfiguration(f"missing {key}")
    dataset = config["dataset"]
    for key in ("resource_id", "path", "timestamp_column", "feature_column", "frequency", "calendar"):
        if key not in dataset:
            raise InvalidConfiguration(f"dataset missing {key}")
    return _target_definitions(config["target_pack"])


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
        if _digest_file(envelope_path) != terminal.get("envelope_sha256"):
            raise InvalidConfiguration(f"outbox integrity failed for {unit_id}")
        return {**terminal, "status": "REPLAY", "terminal_path": str(terminal_path), "envelope_path": str(envelope_path)}
    if terminal_path.exists():
        raise InvalidConfiguration(f"immutable terminal integrity failed for {unit_id}")
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.glob(f".staging-{unit_id}-*"):
        if stale.is_dir():
            shutil.rmtree(stale)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{unit_id}-", dir=root))
    try:
        dataset_path = Path(config["dataset"]["path"]).expanduser().resolve()
        if not dataset_path.is_file():
            profile = {"schema": "feature_selection_ps1_profile.v1", "state": "NOT_AVAILABLE",
                       "reason": "DATASET_NOT_AVAILABLE"}
            cells = _unavailable_cells(unit_id, config, definitions, "DATASET_NOT_AVAILABLE")
            feature_record, status = {"feature_id": unit_id, "state": "NOT_AVAILABLE"}, "NOT_AVAILABLE"
            input_sha = None
        else:
            source = _read_table(config["dataset"])
            required = [config["dataset"]["timestamp_column"], config["dataset"]["feature_column"]]
            absent = [column for column in required if column not in source]
            if absent:
                profile = {"schema": "feature_selection_ps1_profile.v1", "state": "NOT_AVAILABLE",
                           "reason": "FEATURE_COLUMN_NOT_AVAILABLE", "columns": absent}
                cells = _unavailable_cells(unit_id, config, definitions, "FEATURE_COLUMN_NOT_AVAILABLE")
                feature_record, status = {"feature_id": unit_id, "state": "NOT_AVAILABLE"}, "NOT_AVAILABLE"
            else:
                profile = _profile(config, source)
                target_path = Path(config["target_pack"]["path"]).expanduser().resolve()
                if not target_path.is_file():
                    cells = _unavailable_cells(unit_id, config, definitions, "TARGET_RESOURCE_NOT_AVAILABLE")
                    feature_record, status = {"feature_id": unit_id, "state": "NOT_AVAILABLE",
                                              "missing_resource": str(target_path)}, "NOT_AVAILABLE"
                else:
                    X, Y, missing = _prepare_scientific_frames(config, source, definitions)
                if target_path.is_file() and missing:
                    target_columns = {target.column for target in definitions}
                    reason = "TARGET_COLUMN_NOT_AVAILABLE" if any(item in target_columns for item in missing) else "CONTEXT_COLUMN_NOT_AVAILABLE"
                    cells = _unavailable_cells(unit_id, config, definitions, reason)
                    feature_record, status = {"feature_id": unit_id, "state": "NOT_AVAILABLE",
                                              "missing_columns": missing}, "NOT_AVAILABLE"
                elif target_path.is_file() and not _folds(config):
                    cells = _unavailable_cells(unit_id, config, definitions, "FOLDS_NOT_DECLARED")
                    feature_record, status = {"feature_id": unit_id, "state": "NOT_APPLICABLE"}, "NOT_APPLICABLE"
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
            input_sha = _digest_file(dataset_path)
        profile_path = staging / "ps1_profile.json"
        cells_path = staging / "raw_causal_cells.jsonl"
        feature_path = staging / "feature_record.json"
        _write_json_atomic(profile_path, profile)
        _write_bytes_atomic(cells_path, b"".join(_canonical_bytes(cell) for cell in cells))
        _write_json_atomic(feature_path, feature_record)
        artifacts = {path.name: _digest_file(path) for path in (profile_path, cells_path, feature_path)}
        envelope = {
            "schema": ENVELOPE_SCHEMA,
            "unit_id": unit_id,
            "unit_key": unit_key,
            "identity_sha256": identity_sha,
            "split": "TRAIN",
            "target_pack": config["target_pack"]["id"].upper(),
            "dataset": {"resource_id": config["dataset"]["resource_id"], "sha256": input_sha,
                        "feature_column": config["dataset"]["feature_column"]},
            "status": status,
            "causal_state": "RAW_AWAITING_GLOBAL_BH_FDR",
            "artifacts_sha256": artifacts,
            "metrics": {"ps1_profile": profile, "raw_causal_cells": len(cells)},
        }
        envelope_bytes = _canonical_bytes(envelope)
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "unit_id": unit_id,
            "unit_key": unit_key,
            "identity_sha256": identity_sha,
            "status": status,
            "global_state": "AWAITING_INVENTORY_FINALIZATION",
            "artifacts_sha256": artifacts,
            "envelope_sha256": _digest_bytes(envelope_bytes),
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
    verified = [(unit, _verify_terminal(Path(unit["terminal_path"]), unit["unit_id"])) for unit in units]
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
            return {**previous, "status": "REPLAY", "terminal_path": str(terminal_path)}
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
    terminal = {
        "schema": FINAL_TERMINAL_SCHEMA,
        "inventory_id": manifest["inventory_id"],
        "inventory_sha256": inventory_sha,
        "status": "COMPLETED",
        "global_state": "FINALIZED_WITH_GLOBAL_BH_FDR",
        "units": len(units),
        "cells": summary["cells"],
        "causal_evidence_sha256": _digest_file(output / "causal_evidence.jsonl"),
    }
    _write_json_atomic(terminal_path, terminal)
    return {**terminal, "terminal_path": str(terminal_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run-column", help="Profile and run raw causal evidence for one TRAIN column")
    run_parser.add_argument("--config", type=Path, required=True)
    finalize_parser = commands.add_parser("finalize-inventory", help="Verify all units and apply global BH/FDR")
    finalize_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_column(args.config) if args.command == "run-column" else finalize_inventory(args.manifest)
    except (InvalidConfiguration, IncompleteInventory, OSError, ValueError) as trouble:
        print(json.dumps({"status": "REFUSED", "reason": str(trouble)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, default=ps3c._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
