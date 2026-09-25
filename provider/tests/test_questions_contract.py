"""The m5phet question envelope over this provider: every answer read from the retained artifact, every refusal typed.

The registry holds the real provider and this machine's real fitted demo study. `run_task` is the workbench's own
entry point, so what these tests check is what a caller gets: an `ate` answered with the artifact's estimate and
interval, a graph the study was not fitted with refused by the difference, a `cate` refused with the reason, and no
statistic the artifact does not carry."""

from datetime import datetime, timedelta, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

import pytest

from causal_inference_provider.chat import M5PHETCausalProvider, state_directory
from causal_inference_provider.questions import NO_P_VALUE, QUESTION_TYPES


def contract():
    """The envelope under test, `m5phet.questions`.

    It is imported from the installed workbench when that copy carries it. An installed copy that predates the envelope
    is not replaced here -- it may be serving a live session -- so the contract file is then loaded from the checkout
    named by M5PHET_QUESTIONS_PATH. The module has no intra-package imports, so loading it alone is loading the
    contract, not a stand-in for it."""
    try:
        return importlib.import_module("m5phet.questions")
    except ImportError:
        pass
    path = Path(os.environ.get("M5PHET_QUESTIONS_PATH",
                               Path.home() / "Documents/GitHub/.worktrees/m5phet-chat/src/m5phet/questions.py"))
    if not path.is_file():
        pytest.skip("m5phet.questions is not installed and M5PHET_QUESTIONS_PATH names no file")
    spec = importlib.util.spec_from_file_location("m5phet.questions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


questions = contract()
runtime = pytest.importorskip("m5phet.runtime")


def fitted_artifacts():
    """Study files this machine actually holds, the configured directory first."""
    for directory in (state_directory(), Path.home() / ".local/share/causal-inference-m5phet/studies"):
        found = sorted(path for path in directory.glob("*.json") if re.fullmatch(r"[a-f0-9]{64}", path.stem))
        if found:
            return found
    return []


def constant_effect_artifacts():
    """Only the studies fitted WITHOUT an effect modifier.

    This file is about the constant-effect study and its refusals, and this machine also retains a study with a
    modifier; picking whichever file sorted first would have swapped one study's contract for the other's."""
    return [path for path in fitted_artifacts()
            if not (json.loads(path.read_text(encoding="utf-8")).get("config") or {}).get("effect_modifiers")]


@pytest.fixture
def retained(tmp_path):
    """The real fitted demo study, copied so nothing here reads from or writes to the live state directory."""
    sources = constant_effect_artifacts()
    if not sources:
        pytest.skip("no constant-effect study is retained; run `python -m causal_inference_provider prepare-demo` first")
    shutil.copy(sources[0], tmp_path / sources[0].name)
    return tmp_path


@pytest.fixture
def provider(retained):
    return M5PHETCausalProvider(retained)


@pytest.fixture
def registry(provider):
    r = runtime.Registry()
    r.register(provider)
    return r


@pytest.fixture
def study(provider):
    ref, = provider.capabilities()["known_states"]
    return provider.load(ref)


def graph_of(study):
    config = study["config"]
    return {"treatment": config["treatment"], "outcome": config["outcome"], "confounders": list(config["adjustments"])}


def task(state, **asked):
    return {"schema": questions.TASK_SCHEMA, "area": "causal", "state": state, "questions": asked}


ATE = {"type": "ate", "instructions": "Report the average effect."}
CATE = {"type": "cate", "condition": "edad < 30", "instructions": "Report the effect among the young."}


# --- what the provider declares --------------------------------------------------------------------------------------

def test_the_catalog_offers_ate_and_cate_for_the_causal_area(registry):
    cat = questions.catalog(registry)
    assert cat["causal"]["provider"] == "causal_inference"
    assert cat["causal"]["question_types"] == QUESTION_TYPES
    assert set(cat["causal"]["question_types"]) == {"ate", "cate"}
    # neither spelling of the subgroup is required at the envelope layer: a cate question with no subgroup must reach
    # the provider, which alone knows the subgroups the named study carries and can name them in its refusal
    assert cat["causal"]["question_types"]["cate"]["required"] == []
    assert cat["causal"]["question_types"]["cate"]["optional"] == ["subgroup", "condition"]


# --- ate ---------------------------------------------------------------------------------------------------------------

def test_an_ate_question_with_the_studys_graph_is_answered_from_the_artifact(registry, study):
    out = questions.run_task(task({"causal_graph": graph_of(study)}, efecto=ATE), registry)
    answer = out["answers"]["efecto"]
    payload = study["result"]["payload"]
    assert answer["status"] == "OK" and answer["type"] == "ate"
    assert answer["effect_size"] == payload["estimate"]
    assert answer["confidence_interval"] == payload["interval"]
    assert answer["unit"] == payload["unit"]
    assert answer["confidence_level"] == payload["diagnostics"]["confidence_level"]
    assert answer["diagnostics"] == payload["diagnostics"]
    assert answer["population"] == study["population"]
    assert out["state_ref"] == answer["state_ref"] == study["state_ref"]
    assert out["answered"] == 1 and out["refused"] == 0 and out["execution_authorized"] is False


def test_the_conclusion_is_read_off_the_numbers_and_travels_with_the_assumptions(registry, study):
    out = questions.run_task(task({"causal_graph": graph_of(study)}, efecto=ATE), registry)
    answer = out["answers"]["efecto"]
    payload = study["result"]["payload"]
    low, high = payload["interval"]
    sign = "positive" if payload["estimate"] > 0 else "negative"
    assert sign in answer["conclusion"]
    assert ("excludes zero" in answer["conclusion"]) == (low > 0 or high < 0)
    assert f"{payload['estimate']:.4g}" in answer["conclusion"]
    assert answer["assumptions"] == payload["assumptions"]
    assert all(name in answer["conclusion"] for name in payload["assumptions"])


def test_no_p_value_is_computed_that_the_artifact_does_not_carry(registry, study):
    assert "p_value" not in study["result"]["payload"]
    out = questions.run_task(task({"causal_graph": graph_of(study)}, efecto=ATE), registry)
    answer = out["answers"]["efecto"]
    assert "p_value" not in answer
    assert answer["not_carried"]["p_value"] == NO_P_VALUE and "no p-value" in NO_P_VALUE
    assert "p_value" not in json.dumps({k: v for k, v in answer.items() if k != "not_carried"})


def test_the_state_may_name_the_study_by_reference(registry, study):
    out = questions.run_task(task({"state_ref": study["state_ref"]}, efecto=ATE), registry)
    assert out["answers"]["efecto"]["status"] == "OK"
    assert out["answers"]["efecto"]["effect_size"] == study["result"]["payload"]["estimate"]

    # the population digest is the one name the study knows its dataset by
    named = task({"dataset_id": study["population"]["data_sha256"], "causal_graph": graph_of(study)}, efecto=ATE)
    assert questions.run_task(named, registry)["answers"]["efecto"]["status"] == "OK"


# --- a graph the study was not fitted with -----------------------------------------------------------------------------

def test_a_graph_with_another_treatment_is_refused_naming_it(registry, study):
    graph = dict(graph_of(study), treatment="rainfall")
    out = questions.run_task(task({"causal_graph": graph}, efecto=ATE), registry)
    answer = out["answers"]["efecto"]
    assert answer["status"] == "REFUSED" and answer["refusal"] == questions.NOT_ESTIMABLE
    assert "rainfall" in answer["why"] and study["config"]["treatment"] in answer["why"]
    assert "treatment" in answer["why"]
    assert "effect_size" not in answer and out["state_ref"] is None


def test_a_graph_with_another_adjustment_set_is_refused_naming_it(registry, study):
    graph = dict(graph_of(study), confounders=graph_of(study)["confounders"] + ["region"])
    answer = questions.run_task(task({"causal_graph": graph}, efecto=ATE), registry)["answers"]["efecto"]
    assert answer["refusal"] == questions.NOT_ESTIMABLE and "region" in answer["why"] and "confounders" in answer["why"]


def test_a_reference_with_a_contradicting_graph_is_refused(registry, study):
    state = {"state_ref": study["state_ref"], "causal_graph": dict(graph_of(study), outcome="revenue")}
    answer = questions.run_task(task(state, efecto=ATE), registry)["answers"]["efecto"]
    assert answer["refusal"] == questions.NOT_ESTIMABLE and "revenue" in answer["why"]


def test_a_dataset_the_study_does_not_know_is_refused(registry, study):
    state = {"dataset_id": "ventas_2024", "causal_graph": graph_of(study)}
    answer = questions.run_task(task(state, efecto=ATE), registry)["answers"]["efecto"]
    assert answer["refusal"] == questions.NOT_ESTIMABLE and "ventas_2024" in answer["why"]


def test_a_state_naming_no_study_is_refused_by_name(registry, study):
    answer = questions.run_task(task({"dataset_id": "ds"}, efecto=ATE), registry)["answers"]["efecto"]
    assert answer["refusal"] == questions.STATE_REQUIRED and study["state_ref"] in answer["why"]


def test_nothing_retained_is_said_so(tmp_path):
    r = runtime.Registry()
    r.register(M5PHETCausalProvider(tmp_path / "empty"))
    out = questions.run_task(task({"causal_graph": {"treatment": "t", "outcome": "y", "confounders": []}},
                                  efecto=ATE, jovenes=CATE), r)
    assert {a["refusal"] for a in out["answers"].values()} == {questions.STATE_REQUIRED}
    assert all("no fitted study is retained" in a["why"] for a in out["answers"].values())


def test_an_attached_dataset_is_not_fitted(registry, study):
    rows = [{"treatment": 1, "outcome": 2.0, "baseline": 1}] * 100
    answer = questions.run_task(task({"causal_graph": graph_of(study)}, efecto=ATE), registry,
                                data=rows)["answers"]["efecto"]
    assert answer["refusal"] == questions.NOT_ESTIMABLE and "does not fit" in answer["why"]


def test_a_clock_before_the_study_was_available_is_refused(registry, study):
    early = (datetime.fromisoformat(study["available_at"]) - timedelta(days=1)).isoformat()
    payload = dict(task({"causal_graph": graph_of(study)}, efecto=ATE), as_of=early)
    answer = questions.run_task(payload, registry)["answers"]["efecto"]
    assert answer["refusal"] == questions.NOT_ESTIMABLE and "available" in answer["why"]
    later = dict(payload, as_of=datetime.now(timezone.utc).isoformat())
    assert questions.run_task(later, registry)["answers"]["efecto"]["status"] == "OK"


# --- cate ----------------------------------------------------------------------------------------------------------------

def test_cate_is_refused_with_the_modifier_explanation(registry, study):
    out = questions.run_task(task({"causal_graph": graph_of(study)}, jovenes=CATE), registry)
    answer = out["answers"]["jovenes"]
    assert answer == questions.refusal(questions.NOT_ESTIMABLE, answer["why"], "cate")
    assert "average effect" in answer["why"] and "effect modifier" in answer["why"]
    assert "edad < 30" in answer["why"]
    assert "fitted explicitly with that modifier" in answer["why"]
    assert "refitting" in answer["why"]
    assert not any(k in answer for k in ("effect_size", "confidence_interval", "p_value"))


def test_cate_without_a_subgroup_is_refused_by_the_study_that_would_answer_it(registry, study):
    """The refusal moved from the envelope to the provider when a study could answer: the provider can say which
    subgroups the named study carries, and a study that carries none says that instead."""
    answer = questions.run_task(task({"causal_graph": graph_of(study)}, jovenes={"type": "cate"}),
                                registry)["answers"]["jovenes"]
    assert answer["status"] == "REFUSED" and answer["refusal"] == questions.NOT_ESTIMABLE
    assert "effect modifier" in answer["why"] and "None" in answer["why"]
    assert "effect_size" not in answer


def test_ate_and_cate_asked_together_come_back_together(registry, study):
    out = questions.run_task(task({"causal_graph": graph_of(study)}, efecto=ATE, jovenes=CATE), registry)
    assert list(out["answers"]) == ["efecto", "jovenes"]
    assert out["answers"]["efecto"]["status"] == "OK"
    assert out["answers"]["efecto"]["effect_size"] == study["result"]["payload"]["estimate"]
    assert out["answers"]["jovenes"]["refusal"] == questions.NOT_ESTIMABLE
    assert out["answered"] == 1 and out["refused"] == 1
    assert out["state_ref"] == study["state_ref"]


def test_every_ok_answer_carries_the_assumptions(registry, study):
    out = questions.run_task(task({"state_ref": study["state_ref"]}, a=ATE, b=ATE, c=CATE), registry)
    ok = [a for a in out["answers"].values() if a["status"] == "OK"]
    assert len(ok) == 2
    assert all(a["assumptions"] == study["result"]["payload"]["assumptions"] for a in ok)


def test_an_undeclared_type_never_reaches_the_study(registry, study):
    answer = questions.run_task(task({"state_ref": study["state_ref"]}, r={"type": "refutation"}),
                                registry)["answers"]["r"]
    assert answer["refusal"] == questions.UNSUPPORTED_QUESTION_TYPE and "ate" in answer["why"]


def test_serving_questions_imports_no_fit_dependency(retained, study):
    """The envelope is inference: answering it must not pull EconML or its numeric stack."""
    import subprocess

    script = '''
import builtins, json, sys
real_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split(".")[0] in {"numpy", "pandas", "scipy", "sklearn", "econml", "statsmodels"}:
        raise AssertionError("inference imported fit dependency: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guard
from causal_inference_provider.chat import M5PHETCausalProvider
p = M5PHETCausalProvider(sys.argv[1])
ref, = p.capabilities()["known_states"]
out = p.answer_questions({"state_ref": ref}, {"e": {"type": "ate"}, "c": {"type": "cate", "condition": "x < 1"}},
                         None, None)
assert out["e"]["status"] == "OK" and out["c"]["status"] == "REFUSED", out
'''
    result = subprocess.run([sys.executable, "-c", script, str(retained)], cwd=retained,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
