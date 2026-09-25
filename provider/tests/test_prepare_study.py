"""`prepare-study`: one spec in, one retained study out -- and the numbers it recovers.

The fit is the only place in this repository where data are touched, and it happens here, in an explicit command, in
the interpreter that holds EconML. What the command produces is checked against the truth the study's own manifest
publishes: the generating equations of the synthetic modifier data say the effect is 1 where `baseline == 0`, 3 where
`baseline == 1` and 2 on average, so the test compares the fitted intervals against those numbers and not against a
number a test author remembered.

The spec travels with the study. Its manifest carries the spec it was fitted from and the digests of the decision
records that chose it, so a study made by a chooser can always be traced back to the choices -- and a study made by
hand carries an empty decision list, which says exactly that.
"""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from causal_inference_provider.chat import M5PHETCausalProvider, state_directory
from causal_inference_provider.study_spec import SPEC_SCHEMA

TOLERANCE = 0.25
HAS_FIT_DEPENDENCIES = importlib.util.find_spec("econml") is not None
requires_fit = pytest.mark.skipif(not HAS_FIT_DEPENDENCIES,
                                  reason="fitting needs EconML; this interpreter serves and does not fit")

WP20_SPEC = {
    "schema": SPEC_SCHEMA,
    "dataset": {"id": "synthetic-modifier", "columns": ["baseline", "confounder", "treatment", "outcome"]},
    "roles": {"treatment": "treatment", "outcome": "outcome", "confounder": "confounder", "baseline": "modifier"},
    "estimator": "LinearDML",
    "nuisance": {"model_y": "lasso", "model_t": "lasso"},
    "confidence_level": "0.95",
    "identification": ["sufficient_adjustment", "pre_treatment_adjustment", "consistency", "no_interference",
                       "positivity", "iid_sampling", "effect_linear_in_modifiers", "nuisance_models_correct"],
    "decisions": [],
    "provenance": "DEVELOPMENT",
    "study_id": "wp20-spec-modifier-v1",
}


def run_cli(spec, state_dir, tmp_path, name="spec.json"):
    path = tmp_path / name
    path.write_text(json.dumps(spec), encoding="utf-8")
    finished = subprocess.run([sys.executable, "-m", "causal_inference_provider", "prepare-study",
                               "--spec", str(path), "--state-dir", str(state_dir)],
                              capture_output=True, text=True, timeout=900)
    return finished, json.loads(finished.stdout or "{}")


def files_in(directory):
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(Path(directory).glob("*.json"))}


@pytest.fixture(scope="module")
def fitted_study(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wp20-studies")
    finished, report = run_cli(WP20_SPEC, directory, tmp_path_factory.mktemp("wp20-spec"))
    assert finished.returncode == 0, finished.stderr
    return directory, report


@requires_fit
def test_prepare_study_fits_the_spec_it_was_given(fitted_study):
    directory, report = fitted_study
    assert report["result"]["status"] == "OK"
    assert report["study_id"] == "wp20-spec-modifier-v1"
    assert re.fullmatch(r"causal-ate:[a-f0-9]{64}", report["state_ref"])
    body = json.loads((directory / (report["state_ref"].split(":")[1] + ".json")).read_text(encoding="utf-8"))
    assert body["config"]["estimand"] == "CATE"
    assert body["config"]["estimator"] == "LinearDML"
    assert body["config"]["nuisance"] == {"model_y": "lasso", "model_t": "lasso"}
    assert body["config"]["effect_modifiers"] == ["baseline"]
    assert body["config"]["adjustments"] == ["confounder"]
    assert body["result"]["payload"]["diagnostics"]["confidence_level"] == pytest.approx(0.95)
    # the spec and its decision digests travel with the study
    assert body["manifest"]["spec"]["schema"] == SPEC_SCHEMA
    assert body["manifest"]["spec"]["estimator"] == "LinearDML"
    assert body["manifest"]["spec"]["confidence_level"] == pytest.approx(0.95)
    assert body["manifest"]["decisions"] == []
    assert body["manifest"]["provenance"] == "DEVELOPMENT"
    assert body["manifest"]["estimator"] == "econml.dml.LinearDML"


@requires_fit
def test_the_fitted_study_recovers_the_effects_its_own_manifest_declares(fitted_study):
    """1 where `baseline == 0`, 3 where `baseline == 1`, 2 on average -- each inside its own interval."""
    directory, report = fitted_study
    provider = M5PHETCausalProvider(directory)
    ref, = provider.capabilities()["known_states"]
    assert ref == report["state_ref"]
    known = provider.load(ref)["manifest"]["origin"]["known_effects"]
    assert known == {"baseline == 0": 1.0, "baseline == 1": 3.0, "ATE": 2.0}
    answers = provider.answer_questions(
        {"state_ref": ref},
        {"promedio": {"type": "ate"},
         "sin": {"type": "cate", "subgroup": "baseline == 0"},
         "con": {"type": "cate", "subgroup": "baseline == 1"}}, None, None)
    for question, subgroup in (("promedio", "ATE"), ("sin", "baseline == 0"), ("con", "baseline == 1")):
        answer = answers[question]
        assert answer["status"] == "OK", answer
        low, high = answer["confidence_interval"]
        assert low < known[subgroup] < high, (subgroup, answer)
        assert answer["effect_size"] == pytest.approx(known[subgroup], abs=TOLERANCE), subgroup
    assert answers["con"]["effect_size"] > answers["sin"]["effect_size"]


@requires_fit
def test_answering_changes_no_byte_of_the_study(fitted_study):
    directory, report = fitted_study
    before = files_in(directory)
    provider = M5PHETCausalProvider(directory)
    ref, = provider.capabilities()["known_states"]
    for _ in range(2):
        out = provider.answer_questions({"state_ref": ref},
                                        {"a": {"type": "ate"}, "b": {"type": "cate", "subgroup": "baseline == 1"}},
                                        None, None)
        assert out["a"]["status"] == "OK" and out["b"]["status"] == "OK"
    assert files_in(directory) == before


@requires_fit
def test_a_spec_that_states_no_assumption_is_refused_and_nothing_is_fitted(tmp_path):
    directory = tmp_path / "studies"
    directory.mkdir()
    finished, report = run_cli(WP20_SPEC | {"identification": []}, directory, tmp_path)
    assert finished.returncode == 2
    assert report["result"]["status"] == "NOT_IDENTIFIED"
    assert "no assumption stated" in report["result"]["reason"]
    assert report["state_ref"] is None
    assert files_in(directory) == {}


@requires_fit
@pytest.mark.parametrize("change, status", [
    ({"estimator": "MyOwnEstimator"}, "UNKNOWN_ESTIMATOR"),
    ({"nuisance": {"model_y": "neural_net", "model_t": "lasso"}}, "UNKNOWN_NUISANCE_MODEL"),
    ({"roles": {"treatment": "exclude", "outcome": "outcome", "confounder": "confounder", "baseline": "modifier"}},
     "NO_TREATMENT"),
    ({"dataset": {"id": "synthetic-modifier", "columns": ["treatment", "outcome"]}}, "UNKNOWN_COLUMN"),
    ({"dataset": {"id": "synthetic-modifier",
                  "columns": ["baseline", "confounder", "treatment", "outcome", "humidity"]}}, "UNASSIGNED_COLUMN"),
])
def test_a_refused_spec_never_reaches_the_data(change, status, tmp_path):
    directory = tmp_path / "studies"
    directory.mkdir()
    finished, report = run_cli(WP20_SPEC | change, directory, tmp_path)
    assert finished.returncode == 2
    assert report["result"]["status"] == status, report
    assert files_in(directory) == {}


@requires_fit
def test_a_spec_written_for_another_dataset_is_not_fitted_on_this_one(tmp_path):
    directory = tmp_path / "studies"
    directory.mkdir()
    spec = WP20_SPEC | {"dataset": {"id": "synthetic-modifier",
                                    "columns": ["baseline", "confounder", "treatment", "outcome", "rainfall"]},
                        "roles": dict(WP20_SPEC["roles"], rainfall="exclude")}
    finished, report = run_cli(spec, directory, tmp_path)
    assert finished.returncode == 2
    assert report["result"]["status"] == "INVALID_INPUT"
    assert "rainfall" in report["result"]["reason"]
    assert files_in(directory) == {}


# --- the study this machine actually retains, through the workbench's own envelope -----------------------------------


def retained_spec_studies():
    """Every retained study that was fitted from a spec, from the configured state directory."""
    for directory in (state_directory(), Path.home() / ".local/share/causal-inference-m5phet/studies"):
        found = [path for path in sorted(directory.glob("*.json")) if re.fullmatch(r"[a-f0-9]{64}", path.stem)]
        bodies = [json.loads(path.read_text(encoding="utf-8")) for path in found]
        carrying = [body for body in bodies if isinstance((body.get("manifest") or {}).get("spec"), dict)]
        if carrying:
            return directory, carrying
    return None, []


def test_the_retained_spec_study_answers_the_envelope():
    """`ate` and `cate` on the study `prepare-study` put in the configured state directory, through `run_task`."""
    questions = pytest.importorskip("m5phet.questions")
    runtime = pytest.importorskip("m5phet.runtime")
    directory, carrying = retained_spec_studies()
    if not carrying:
        pytest.skip("no spec-fitted study is retained; run `python -m causal_inference_provider prepare-study` first")
    body = carrying[0]
    registry = runtime.Registry()
    registry.register(M5PHETCausalProvider(directory))
    ref = "causal-ate:" + sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
    task = {"schema": questions.TASK_SCHEMA, "area": "causal", "state": {"state_ref": ref},
            "questions": {"promedio": {"type": "ate"},
                          "con": {"type": "cate", "subgroup": "baseline == 1"},
                          "sin": {"type": "cate", "subgroup": "baseline == 0"}}}
    out = questions.run_task(task, registry)
    assert out["answered"] == 3 and out["refused"] == 0, out["answers"]
    assert out["execution_authorized"] is False
    known = body["manifest"]["origin"]["known_effects"]
    for name, subgroup in (("promedio", "ATE"), ("con", "baseline == 1"), ("sin", "baseline == 0")):
        answer = out["answers"][name]
        low, high = answer["confidence_interval"]
        assert low < known[subgroup] < high, (name, answer)
    assert body["manifest"]["spec"]["schema"] == SPEC_SCHEMA
