from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from causal_inference_provider import feature_selection_envelope as EW
from causal_inference_provider import feature_selection_worker as worker


FEATURE_COUNT = 366
TARGETS = [
    *[(f"Y_s_{h}h", h) for h in (1, 2, 3, 4, 5, 6)],
    *[(f"Y_l_{h}h", h) for h in (24, 48, 72, 96, 120, 144)],
    ("Y_b_s6", 6), ("Y_b_l144", 144),
]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_bundle_member(root: Path, relative: str) -> None:
    manifest_path = root / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    member = root / relative
    for item in manifest["files"]:
        if item["path"] == relative:
            item.update(bytes=member.stat().st_size, sha256=_sha(member))
            break
    else:
        raise AssertionError(f"bundle member not found: {relative}")
    manifest.pop("bundle_sha256")
    manifest["bundle_sha256"] = EW.digest(manifest)
    _write(manifest_path, manifest)


def _metric_rows(feature: str, population: str) -> dict[str, list[dict]]:
    base = {"feature_id": feature, "split": "train", "metric_value": 0.0,
            "state": "MEASURED", "population_id": population, "fold": None}
    target, horizon = TARGETS[0]
    return {
        "sampling_quality": [{**base, "metric_name": "missingness.fraction", "unit": "ratio"}],
        "variable_profiles": [{**base, "metric_name": "scale_tails.mean", "unit": None}],
        "information_metrics": [{**base, "target_id": target, "horizon": horizon,
                                  "metric_name": "information_entropy.value"}],
        "pair_relations": [{**base, "target_id": target, "horizon": horizon, "lag": 0,
                             "metric_name": "pearson"}],
        "causal_evidence": [], "selection_decisions": [],
    }


def _causal_rows(feature: str, population: str) -> tuple[list[dict], list[dict]]:
    evidence, decisions = [], []
    for target, horizon in TARGETS:
        evidence_id = EW.digest({"feature": feature, "target": target})
        for rung in (1, 2, 3):
            evidence.append({
                "feature_id": feature, "target_id": target, "horizon": horizon,
                "split": "train", "rung": rung, "estimand": f"rung_{rung}",
                "estimator": "retained_adopted_evidence", "state": "NOT_IDENTIFIED",
                "effect": None, "lower": None, "upper": None, "support_n": 10,
                "assumptions": [], "adjustment_set": [], "evidence_sha256": evidence_id,
                "population_id": population, "fold": None,
            })
        decisions.append({
            "feature_id": feature, "target_id": target, "horizon": horizon,
            "method": "causal_ladder_global_fdr", "score": 0.0, "rank": None,
            "decision": "NEUTRAL", "rule": "retained adopted decision",
            "evidence_sha256": evidence_id, "population_id": population,
        })
    return evidence, decisions


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    inventory = "8" * 64
    profile_population = inventory
    adopted_population = EW.digest({"population": "EURUSD", "inventory_sha256": inventory})
    features = [f"feature_{index:03d}" for index in range(FEATURE_COUNT)]
    rows = {family: [] for family in EW.ROW_FAMILIES}
    for feature in features:
        unit = _metric_rows(feature, profile_population)
        for family in ("sampling_quality", "variable_profiles", "information_metrics", "pair_relations"):
            rows[family].extend(unit[family])
    profile_envelope = EW.build_envelope({
        "run_id": "phase1-profile:EURUSD:fixture", "campaign_sha256": "7" * 64,
        "code_sha256": "6" * 64, "input_sha256": "5" * 64,
        "inventory_sha256": inventory, "created_at": "2026-10-05T12:00:00Z",
    }, rows)
    profile = {
        "schema": "phase1.profile_merge_result.v1", "mode": "PROFILE_ONLY",
        "state": "PROFILE_METRICS_COMPLETE", "terminal_count": FEATURE_COUNT,
        "completed_count": FEATURE_COUNT, "unavailable_count": 0,
        "plan_sha256": "4" * 64, "inventory_sha256": inventory,
        "envelope": profile_envelope,
    }
    profile["result_sha256"] = EW.digest(profile)
    profile_path = root / "profile-result.json"
    _write(profile_path, profile)

    adoption = root / "adoption"
    paths = []
    for feature in features:
        causal, decisions = _causal_rows(feature, adopted_population)
        envelope = EW.build_envelope({
            "run_id": f"phase1-eurusd-adopted:{feature}", "campaign_sha256": "3" * 64,
            "code_sha256": "2" * 64, "input_sha256": "1" * 64,
            "inventory_sha256": inventory, "created_at": "2026-10-05T00:00:00Z",
        }, {"causal_evidence": causal, "selection_decisions": decisions})
        relative = Path("warehouse_envelopes") / f"{feature}.json"
        _write(adoption / relative, envelope)
        paths.append(str(Path("adoption") / relative))
    report = {
        "schema": "phase1.evidence_adoption.v1", "state": "ADOPTED_VERIFIED_EVIDENCE",
        "population_id": "EURUSD", "feature_count": FEATURE_COUNT,
        "causal_rows": FEATURE_COUNT * len(TARGETS), "source_sha256": "a" * 64,
        "source_digest_manifest_sha256": "b" * 64, "predictor_revision": "fixture",
        "causal_revision": "fixture", "envelopes": paths,
    }
    report["adoption_sha256"] = EW.digest(report)
    _write(adoption / "ADOPTION.json", report)
    files = []
    for path in sorted(adoption.rglob("*.json")):
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                      "sha256": _sha(path)})
    manifest = {"schema": "phase1.deployment_bundle.v1", "files": files,
                "standard_install_root": "<fixture>"}
    manifest["bundle_sha256"] = EW.digest(manifest)
    _write(root / "BUNDLE_MANIFEST.json", manifest)
    return profile_path, adoption, root / "combined-envelope.json"


def test_combine_adopted_eurusd_authenticates_mutations_denominators_and_366_positive(tmp_path):
    profile_path, adoption, output = _fixture(tmp_path)
    first_envelope = adoption / "warehouse_envelopes" / "feature_000.json"

    original = first_envelope.read_text(encoding="utf-8")
    mutated = json.loads(original)
    mutated["rows"]["causal_evidence"][0]["support_n"] = 999
    _write(first_envelope, mutated)
    with pytest.raises(worker.IncompleteInventory, match="envelope|digest"):
        worker.combine_adopted_eurusd(profile_path, adoption, output)
    first_envelope.write_text(original, encoding="utf-8")

    report_path = adoption / "ADOPTION.json"
    original_report = report_path.read_text(encoding="utf-8")
    report = json.loads(original_report)
    report["envelopes"].pop()
    report["feature_count"] -= 1
    report.pop("adoption_sha256")
    report["adoption_sha256"] = EW.digest(report)
    _write(report_path, report)
    _refresh_bundle_member(tmp_path, "adoption/ADOPTION.json")
    with pytest.raises(worker.IncompleteInventory, match="denominator|population"):
        worker.combine_adopted_eurusd(profile_path, adoption, output)
    report_path.write_text(original_report, encoding="utf-8")
    _refresh_bundle_member(tmp_path, "adoption/ADOPTION.json")

    duplicate = json.loads(original)
    duplicate["rows"]["causal_evidence"].append(duplicate["rows"]["causal_evidence"][0])
    duplicate.pop("envelope_sha256")
    duplicate["envelope_sha256"] = EW.digest(duplicate)
    _write(first_envelope, duplicate)
    _refresh_bundle_member(tmp_path, "adoption/warehouse_envelopes/feature_000.json")
    with pytest.raises(worker.IncompleteInventory, match="duplicate"):
        worker.combine_adopted_eurusd(profile_path, adoption, output)
    first_envelope.write_text(original, encoding="utf-8")
    _refresh_bundle_member(tmp_path, "adoption/warehouse_envelopes/feature_000.json")

    original_profile = profile_path.read_text(encoding="utf-8")
    profile = json.loads(original_profile)
    profile["terminal_count"] = FEATURE_COUNT - 1
    profile["completed_count"] = FEATURE_COUNT - 1
    profile.pop("result_sha256")
    profile["result_sha256"] = EW.digest(profile)
    _write(profile_path, profile)
    with pytest.raises(worker.IncompleteInventory, match="366|denominator"):
        worker.combine_adopted_eurusd(profile_path, adoption, output)
    profile_path.write_text(original_profile, encoding="utf-8")

    command = [
        sys.executable, "-m", "causal_inference_provider.feature_selection_worker",
        "combine-adopted-eurusd", "--profile-result", str(profile_path),
        "--adoption-dir", str(adoption.parent), "--output", str(output),
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr
    document = json.loads(output.read_text())
    assert document["schema_version"] == "feature_selection_envelope.v1"
    assert document["run"]["inventory_sha256"] == "8" * 64
    assert len(document["rows"]["sampling_quality"]) == FEATURE_COUNT
    assert len(document["rows"]["causal_evidence"]) == FEATURE_COUNT * len(TARGETS) * 3
    assert len(document["rows"]["selection_decisions"]) == FEATURE_COUNT * len(TARGETS)
    assert {row["feature_id"] for row in document["rows"]["pair_relations"]} == {
        f"feature_{index:03d}" for index in range(FEATURE_COUNT)
    }
    worker._validate_warehouse_envelope(document)
