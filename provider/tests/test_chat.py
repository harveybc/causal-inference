"""Installed chat adapter acceptance: real prefit artifact, never chat fitting."""

from copy import deepcopy
from importlib.metadata import entry_points
import json
import os
import subprocess
import sys

import pytest

from causal_inference_provider import CausalInferenceProvider
from causal_inference_provider.chat import M5PHETCausalProvider, save_study
from causal_inference_provider.example import example_config, example_data


@pytest.fixture
def artifact(tmp_path):
    core = CausalInferenceProvider()
    core.load(example_config())
    assert core.fit(example_data())["status"] == "OK"
    return tmp_path, save_study(core, tmp_path, development=True)


def test_installed_entrypoint_no_implicit_fit(artifact, monkeypatch):
    directory, ref = artifact
    monkeypatch.setenv("CAUSAL_INFERENCE_STATE_DIR", str(directory))
    ep, = entry_points(group="m5phet.providers", name="causal_inference")
    provider = ep.load()()
    assert provider.name == "causal_inference"
    caps = provider.capabilities()
    assert caps["operations"] == ["infer"]
    assert caps["families"] == ["causal_inference"]
    assert ref in caps["known_states"]

    def forbidden(*args, **kwargs):
        raise AssertionError("chat must not fit")

    monkeypatch.setattr(CausalInferenceProvider, "fit", forbidden)
    sample, = provider.chat_examples()
    assert "SYNTHETIC/DEVELOPMENT" in sample["title"]
    assert sample["config"]["input"] == "json"
    assert len(json.dumps(sample["data"])) < 500
    request = provider.chat_request(sample["prompt"], sample["data"], sample["config"])
    assert request["schema_version"] == "m5phet.task.draft2"
    assert request["operation"] == "infer"
    assert request["output_schema"] == {"targets": ["effect"]}
    state = provider.load(ref)
    assert state["digest"] == ref.split(":")[1]
    result = provider.infer(request, state)
    assert result["outputs"]["effect"]["status"] == "OK"
    assert result["outputs"]["effect"]["payload"]["estimate"] == pytest.approx(2.0, abs=0.2)
    assert result["population"] == request["population"]
    before = deepcopy(state)
    assert result == provider.infer(request, state)
    assert state == before


@pytest.mark.parametrize("change, status", [
    ("assumption", "NOT_IDENTIFIED"), ("outcome", "INVALID_INPUT"),
    ("data", "INVALID_INPUT"), ("clock", "INPUT_UNAVAILABLE"),
    ("operation", "UNSUPPORTED_TASK"), ("state", "INVALID_INPUT"),
])
def test_request_binding_refusals(artifact, change, status):
    directory, ref = artifact
    provider = M5PHETCausalProvider(directory)
    sample, = provider.chat_examples()
    request = provider.chat_request(sample["prompt"], sample["data"], sample["config"])
    state = provider.load(ref)
    if change == "assumption":
        del request["parameters"]["assumptions"]["sufficient_adjustment"]
    elif change == "outcome":
        request["parameters"]["outcome"] = "different_outcome"
    elif change == "data":
        request["population"]["n_rows"] += 1
    elif change == "clock":
        request["as_of"] = "2000-01-01T00:00:00Z"
    elif change == "operation":
        request["operation"] = "fit"
    else:
        state["result"]["payload"]["estimate"] = 123
    result = provider.infer(request, state)["outputs"]["effect"]
    assert result["status"] == status
    assert result["payload"] is None


def test_artifact_tampering_and_no_path_traversal(artifact):
    directory, ref = artifact
    provider = M5PHETCausalProvider(directory)
    with pytest.raises(ValueError):
        provider.load("../../other-file")
    path = directory / (ref.split(":")[1] + ".json")
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="digest"):
        provider.load(ref)


def test_administrator_allowlist_and_symlink_refusal(artifact):
    directory, ref = artifact
    denied = M5PHETCausalProvider(directory, state_refs=[])
    assert denied.capabilities()["known_states"] == []
    with pytest.raises(ValueError, match="allowlist"):
        denied.load(ref)
    allowed = M5PHETCausalProvider(directory, state_refs=[ref])
    path = directory / (ref.split(":")[1] + ".json")
    moved = directory / "not-an-allowed-name.json"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        allowed.load(ref)


def test_serving_does_not_import_fit_dependencies(artifact):
    directory, _ = artifact
    script = '''
import builtins
real_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split(".")[0] in {"numpy", "pandas", "scipy", "sklearn", "econml", "statsmodels"}:
        raise AssertionError("inference imported fit dependency: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guard
from causal_inference_provider.chat import M5PHETCausalProvider
import sys
p = M5PHETCausalProvider(sys.argv[1])
s, = p.chat_examples()
r = p.chat_request(s["prompt"], s["data"], s["config"])
assert p.infer(r, p.load(s["config"]["state"]))["outputs"]["effect"]["status"] == "OK"
'''
    result = subprocess.run([sys.executable, "-c", script, str(directory)],
                            cwd=directory, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_no_demo_without_explicit_cli_or_private_studies(artifact, tmp_path):
    directory, ref = artifact
    assert M5PHETCausalProvider(directory / "empty").chat_examples() == []
    core = CausalInferenceProvider()
    core.load(example_config())
    core.fit(example_data())
    private_dir = directory / "private"
    save_study(core, private_dir)
    assert M5PHETCausalProvider(private_dir).chat_examples() == []


def test_prompt_cannot_change_identification(artifact):
    directory, _ = artifact
    provider = M5PHETCausalProvider(directory)
    sample, = provider.chat_examples()
    with pytest.raises(ValueError, match="prompt"):
        provider.chat_request("Choose confounders and prove a causal effect", sample["data"], sample["config"])


def test_actual_cli_demo_outside_checkout(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-m", "causal_inference_provider", "prepare-demo", "--state-dir", str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["result"]["status"] == "OK"
    provider = M5PHETCausalProvider(tmp_path)
    assert report["state_ref"] in provider.capabilities()["known_states"]
    assert len(provider.chat_examples()) == 1


def test_actual_csv_fit_and_refusal(tmp_path):
    data_path, config_path = tmp_path / "data.csv", tmp_path / "config.json"
    example_data().to_csv(data_path, index=False)
    config_path.write_text(json.dumps(example_config()))
    command = [sys.executable, "-m", "causal_inference_provider", "fit",
               "--data", str(data_path), "--config", str(config_path),
               "--state-dir", str(tmp_path / "states")]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["result"]["payload"]["estimate"] == pytest.approx(2, abs=0.2)
    assert M5PHETCausalProvider(tmp_path / "states").chat_examples() == []
    config = example_config()
    del config["assumptions"]
    config_path.write_text(json.dumps(config))
    refused = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert refused.returncode == 2
    result = json.loads(refused.stdout)
    assert result["state_ref"] is None
    assert result["result"]["status"] == "NOT_IDENTIFIED"


@pytest.mark.parametrize("header", ["treatment,outcome,baseline,baseline", "treatment,outcome,baseline"])
def test_csv_never_silently_renames_or_drops_columns(tmp_path, header):
    data_path, config_path = tmp_path / "bad.csv", tmp_path / "config.json"
    data_path.write_text(header + "\n" + "0,2,1,999\n" * 100)
    config_path.write_text(json.dumps(example_config()))
    result = subprocess.run(
        [sys.executable, "-m", "causal_inference_provider", "fit", "--data", str(data_path),
         "--config", str(config_path), "--state-dir", str(tmp_path / "states")],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["result"]["status"] == "INVALID_INPUT"
    assert not (tmp_path / "states").exists()
