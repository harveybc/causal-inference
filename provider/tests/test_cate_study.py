"""The conditional-effect study, read from the artifact this machine actually holds.

Two studies are retained by design: one fitted with a constant effect (the ATE study) and one fitted with an explicit
effect modifier (the CATE study). What these tests check is the boundary between them. The CATE study answers a subgroup
question because its artifact CARRIES the subgroup effect, computed when it was fitted; the ATE study keeps refusing the
same question with the same reason it always gave. A subgroup naming a variable no study declared is refused by name,
and answering never touches the artifacts -- their bytes are hashed before and after.

The truth these tests compare against is not written here: it is the data-generating equations the fit recorded in the
study's own manifest, so the tolerance is checked against the study's declared truth and not against a number a test
author remembered.
"""

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from causal_inference_provider.chat import M5PHETCausalProvider, state_directory
from causal_inference_provider.questions import (MALFORMED_QUESTION, NOT_ESTIMABLE, QUESTION_TYPES, STATE_REQUIRED,
                                                SUBGROUP_FORM)

TOLERANCE = 0.25


def artifacts():
    """Every study file this machine holds, the configured state directory first."""
    for directory in (state_directory(), Path.home() / ".local/share/causal-inference-m5phet/studies"):
        found = sorted(path for path in directory.glob("*.json") if re.fullmatch(r"[a-f0-9]{64}", path.stem))
        if found:
            return found
    return []


def _copy(paths, tmp_path, name):
    if not paths:
        pytest.skip(f"no {name} study is retained; run `python -m causal_inference_provider prepare-demo"
                    f"{' --with-modifier' if name == 'modifier' else ''}` first")
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy(path, directory / path.name)
    return directory


def _has_modifier(path):
    body = json.loads(path.read_text(encoding="utf-8"))
    return bool((body.get("config") or {}).get("effect_modifiers"))


def _study_id(path):
    body = json.loads(path.read_text(encoding="utf-8"))
    return ((body.get("manifest") or {}).get("study_id"))


#: The study these tests are about: the one `prepare-demo --with-modifier` fits, by its own identifier. This machine may
#: retain other studies fitted with a modifier -- a spec-fitted one, say -- and they are other studies with other
#: numbers, so a fixture that means this one names it instead of trusting that it is the only one here.
DEMO_MODIFIER = "demo-modifier-v1"


@pytest.fixture
def modifier_dir(tmp_path):
    """Only the demo study fitted with an effect modifier, copied out of the live state directory."""
    return _copy([p for p in artifacts() if _study_id(p) == DEMO_MODIFIER], tmp_path, "modifier")


@pytest.fixture
def ate_only_dir(tmp_path):
    """Only the constant-effect study, so its refusal is observed with nothing else present."""
    return _copy([p for p in artifacts() if not _has_modifier(p)], tmp_path, "ate_only")


@pytest.fixture
def both_dir(tmp_path):
    """The two studies `prepare-demo` fits, at once -- where a wrong match between them would show."""
    paths = [p for p in artifacts() if _study_id(p) == DEMO_MODIFIER or not _has_modifier(p)]
    if not any(_has_modifier(p) for p in paths) or not any(not _has_modifier(p) for p in paths):
        pytest.skip("this machine does not hold both an ATE-only and the demo effect-modifier study")
    return _copy(paths, tmp_path, "both")


@pytest.fixture
def modifier_study(modifier_dir):
    provider = M5PHETCausalProvider(modifier_dir)
    ref, = provider.capabilities()["known_states"]
    return provider, provider.load(ref)


def ask(provider, state, **asked):
    """Ask through the provider's own envelope entry point, exactly as m5phet.questions calls it."""
    return provider.answer_questions(state, asked, None, None)


def truth(study):
    """The effects the generating equations put there, as the fit recorded them in the manifest."""
    return study["manifest"]["origin"]["known_effects"]


# --- what the fit recorded, so the truth is auditable -------------------------------------------------------------------

def test_the_manifest_declares_the_modifier_the_estimator_and_the_generating_equations(modifier_study):
    _provider, study = modifier_study
    manifest = study["manifest"]
    assert manifest["study_id"] == "demo-modifier-v1"
    assert manifest["provenance"] == "DEVELOPMENT" and study["development"] is True
    assert manifest["effect_modifiers"] == ["baseline"] == study["config"]["effect_modifiers"]
    assert manifest["estimator"] == "econml.dml.LinearDML"
    assert manifest["n"] == study["population"]["n_rows"] == manifest["origin"]["n"]
    assert isinstance(manifest["seed"], int) and isinstance(manifest["origin"]["seed"], int)
    equations = manifest["origin"]["generating_equations"]
    assert [e for e in equations if e.startswith("outcome")], equations
    assert all(isinstance(e, str) and e.strip() for e in equations)
    assert set(truth(study)) == {"ATE", "baseline == 0", "baseline == 1"}
    assert study["config"]["estimand"] == "CATE"
    # a heterogeneous study may not claim the effect is constant; that is the assumption it replaces
    assert "constant_effect" not in study["config"]["assumptions"]
    assert study["config"]["assumptions"]["effect_linear_in_modifiers"] is True


def test_the_provider_declares_a_subgroup_field_for_cate():
    assert QUESTION_TYPES["cate"]["required"] == []
    assert QUESTION_TYPES["cate"]["optional"] == ["subgroup", "condition"]


# --- known-effect recovery, both subgroups ------------------------------------------------------------------------------

@pytest.mark.parametrize("level", [0, 1])
def test_each_subgroups_known_effect_is_recovered_from_the_fitted_study(modifier_study, level):
    provider, study = modifier_study
    expression = f"baseline == {level}"
    answer = ask(provider, {"state_ref": study["state_ref"]}, grupo={"type": "cate", "subgroup": expression})["grupo"]
    known = truth(study)[expression]
    assert answer["status"] == "OK" and answer["type"] == "cate"
    assert answer["execution_authorized"] is False
    assert answer["estimand"] == "CATE" and answer["subgroup"] == expression
    assert answer["effect_size"] == pytest.approx(known, abs=TOLERANCE)
    low, high = answer["confidence_interval"]
    assert low < known < high and 0 < high - low < 1.0
    assert answer["confidence_level"] == study["result"]["payload"]["diagnostics"]["confidence_level"]
    assert answer["n_subgroup"] > 20 and answer["n_treated"] + answer["n_control"] == answer["n_subgroup"]
    assert answer["effect_modifiers"] == ["baseline"]
    assert answer["unit"] == study["config"]["unit"]
    assert answer["assumptions"] == study["result"]["payload"]["assumptions"]
    assert expression in answer["conclusion"] and f"{answer['effect_size']:.4g}" in answer["conclusion"]
    assert "p_value" not in answer and "p_value" in answer["not_carried"]
    assert answer["state_ref"] == study["state_ref"] and answer["development"] is True


def test_the_two_subgroups_differ_and_cover_the_whole_fitted_population(modifier_study):
    provider, study = modifier_study
    out = ask(provider, {"state_ref": study["state_ref"]},
              bajos={"type": "cate", "subgroup": "baseline == 0"},
              altos={"type": "cate", "subgroup": "baseline == 1"},
              promedio={"type": "ate"})
    bajos, altos, promedio = out["bajos"], out["altos"], out["promedio"]
    assert bajos["n_subgroup"] + altos["n_subgroup"] == study["population"]["n_rows"]
    assert altos["effect_size"] > bajos["effect_size"], "the generating equations make the high subgroup stronger"
    assert promedio["status"] == "OK" and promedio["type"] == "ate"
    assert promedio["effect_size"] == pytest.approx(truth(study)["ATE"], abs=TOLERANCE)
    assert promedio["confidence_interval"][0] < truth(study)["ATE"] < promedio["confidence_interval"][1]
    # the average effect of a heterogeneous study is still an average effect, and says which study it came from
    assert promedio["estimand"] == "ATE" and promedio["study_estimand"] == "CATE"


def test_the_condition_spelling_is_accepted_too(modifier_study):
    """`condition` is the spelling the envelope harness already uses; both name the same subgroup."""
    provider, study = modifier_study
    state = {"state_ref": study["state_ref"]}
    with_subgroup = ask(provider, state, g={"type": "cate", "subgroup": "baseline == 1"})["g"]
    with_condition = ask(provider, state, g={"type": "cate", "condition": "baseline == 1"})["g"]
    assert with_condition == with_subgroup and with_condition["status"] == "OK"


# --- the ATE-only study keeps its refusal ------------------------------------------------------------------------------

def test_cate_is_still_refused_on_the_constant_effect_study(ate_only_dir):
    provider = M5PHETCausalProvider(ate_only_dir)
    ref, = provider.capabilities()["known_states"]
    answer = ask(provider, {"state_ref": ref}, jovenes={"type": "cate", "subgroup": "baseline == 1"})["jovenes"]
    assert answer["status"] == "REFUSED" and answer["refusal"] == NOT_ESTIMABLE
    assert "average effect" in answer["why"] and "effect modifier" in answer["why"]
    assert "'baseline == 1'" in answer["why"] and "refitting" in answer["why"]
    assert "fitted explicitly with that modifier" in answer["why"]
    assert not any(k in answer for k in ("effect_size", "confidence_interval", "n_subgroup", "p_value"))


def test_with_both_studies_retained_each_is_asked_on_its_own(both_dir):
    """The graph of the constant-effect study must not resolve to the study that has a modifier, or the refusal the
    workbench is verified on would silently turn into an answer about another population."""
    provider = M5PHETCausalProvider(both_dir)
    constant = [b for b in provider._retained_studies().values() if not b["config"].get("effect_modifiers")]
    study, = constant
    graph = {"treatment": study["config"]["treatment"], "outcome": study["config"]["outcome"],
             "confounders": sorted(study["config"]["adjustments"])}
    out = ask(provider, {"causal_graph": graph}, efecto={"type": "ate"},
              jovenes={"type": "cate", "condition": "baseline == 1"})
    assert out["__state_ref__"] == study["state_ref"]
    assert out["efecto"]["status"] == "OK"
    assert out["jovenes"]["refusal"] == NOT_ESTIMABLE and "effect modifier" in out["jovenes"]["why"]


# --- a subgroup the study cannot read ----------------------------------------------------------------------------------

def test_a_subgroup_naming_a_variable_that_is_not_a_declared_modifier_is_refused_by_name(modifier_study):
    provider, study = modifier_study
    answer = ask(provider, {"state_ref": study["state_ref"]},
                 jovenes={"type": "cate", "subgroup": "edad == 1"})["jovenes"]
    assert answer["status"] == "REFUSED" and answer["refusal"] == NOT_ESTIMABLE
    assert "edad" in answer["why"] and "baseline" in answer["why"]
    assert "refitting" in answer["why"] or "refit" in answer["why"]
    assert not any(k in answer for k in ("effect_size", "confidence_interval", "n_subgroup"))


@pytest.mark.parametrize("expression", ["baseline > 0.5", "baseline == 2", "baseline != 1", "baseline"])
def test_a_subgroup_form_the_study_does_not_carry_is_refused_with_the_forms_it_does(modifier_study, expression):
    provider, study = modifier_study
    answer = ask(provider, {"state_ref": study["state_ref"]},
                 g={"type": "cate", "subgroup": expression})["g"]
    assert answer["status"] == "REFUSED"
    assert answer["refusal"] in (NOT_ESTIMABLE, MALFORMED_QUESTION)
    assert "baseline == 0" in answer["why"] and "baseline == 1" in answer["why"]
    assert "effect_size" not in answer


def test_a_cate_question_that_names_no_subgroup_is_refused(modifier_study):
    provider, study = modifier_study
    answer = ask(provider, {"state_ref": study["state_ref"]}, g={"type": "cate"})["g"]
    assert answer["refusal"] == MALFORMED_QUESTION
    assert "subgroup" in answer["why"] and "condition" in answer["why"] and SUBGROUP_FORM in answer["why"]


def test_a_study_nobody_retains_is_not_answered_by_the_one_that_is_here(modifier_dir):
    provider = M5PHETCausalProvider(modifier_dir)
    answer = ask(provider, {"study": "demo-modifier-v2"}, g={"type": "cate", "subgroup": "baseline == 1"})["g"]
    assert answer["refusal"] == STATE_REQUIRED and "demo-modifier-v2" in answer["why"]
    assert "demo-modifier-v1" in answer["why"] and "effect_size" not in answer


# --- the study is chosen by words, and nothing is fitted while answering ------------------------------------------------

def test_the_modifier_study_is_named_by_its_own_words(modifier_dir):
    interpret = pytest.importorskip("m5phet.interpret")
    provider = M5PHETCausalProvider(modifier_dir)
    slot, = provider.chat_slots()
    assert slot["name"] == "study" and slot["allowed"] == ["demo-modifier-v1"]
    aliases = slot["aliases"]["demo-modifier-v1"]
    assert "modifier study" in aliases and "estudio con modificador" in aliases
    words = interpret.Interpreter(command="", environ={})
    for question in ("dame el efecto en el estudio con modificador",
                     "report the modifier study for baseline == 1",
                     "what does demo-modifier-v1 say?"):
        report = interpret.interpret(question, provider.chat_slots(), interpreter=words)
        assert report["status"] == interpret.STATUS_OK, report["why"]
        assert report["parameters"] == {"study": "demo-modifier-v1"}
        assert set(report["sources"].values()) == {"QUESTION_TEXT"}


def test_the_state_may_name_the_study_by_its_id(modifier_dir):
    provider = M5PHETCausalProvider(modifier_dir)
    out = ask(provider, {"study": "demo-modifier-v1"},
              g={"type": "cate", "subgroup": "baseline == 1"}, m={"type": "ate"})
    assert out["g"]["status"] == "OK" and out["m"]["status"] == "OK"
    assert out["__state_ref__"] == provider.capabilities()["known_states"][0]


def test_answering_never_refits_the_study(both_dir):
    """The artifacts are the only source of every number, so their bytes may not move while questions are answered."""
    provider = M5PHETCausalProvider(both_dir)
    before = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(both_dir.glob("*.json"))}
    assert len(before) == 2
    for ref in provider.capabilities()["known_states"]:
        out = ask(provider, {"state_ref": ref}, a={"type": "ate"},
                  b={"type": "cate", "subgroup": "baseline == 1"},
                  c={"type": "cate", "subgroup": "baseline == 0"},
                  d={"type": "cate", "subgroup": "edad == 1"})
        assert {name for name, answer in out.items() if name != "__state_ref__" and answer["status"] == "OK"}
    after = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(both_dir.glob("*.json"))}
    assert after == before


def test_answering_a_subgroup_imports_no_fit_dependency(modifier_dir):
    """A conditional effect read from the artifact must not pull EconML: that would be the door to a refit."""
    script = '''
import builtins, sys
real_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split(".")[0] in {"numpy", "pandas", "scipy", "sklearn", "econml", "statsmodels"}:
        raise AssertionError("inference imported fit dependency: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guard
from causal_inference_provider.chat import M5PHETCausalProvider
p = M5PHETCausalProvider(sys.argv[1])
ref, = p.capabilities()["known_states"]
out = p.answer_questions({"state_ref": ref}, {"a": {"type": "cate", "subgroup": "baseline == 1"},
                                             "b": {"type": "cate", "subgroup": "edad == 0"},
                                             "c": {"type": "ate"}}, None, None)
assert out["a"]["status"] == "OK" and out["b"]["status"] == "REFUSED" and out["c"]["status"] == "OK", out
assert out["a"]["n_subgroup"] > 20, out["a"]
'''
    result = subprocess.run([sys.executable, "-c", script, str(modifier_dir)], cwd=modifier_dir,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


# --- through the workbench's own envelope -----------------------------------------------------------------------------

def test_the_envelope_answers_both_subgroups_and_the_average(both_dir):
    questions = pytest.importorskip("m5phet.questions")
    runtime = pytest.importorskip("m5phet.runtime")
    registry = runtime.Registry()
    registry.register(M5PHETCausalProvider(both_dir))
    task = {"schema": questions.TASK_SCHEMA, "area": "causal", "state": {"study": "demo-modifier-v1"},
            "questions": {"jovenes": {"type": "cate", "subgroup": "baseline == 1"},
                          "otros": {"type": "cate", "subgroup": "baseline == 0"},
                          "promedio": {"type": "ate"}}}
    out = questions.run_task(task, registry)
    assert out["answered"] == 3 and out["refused"] == 0, out["answers"]
    assert out["execution_authorized"] is False
    assert all(a["status"] == "OK" for a in out["answers"].values())
    catalog = questions.catalog(registry)
    assert "demo-modifier-v1" in catalog["causal"]["parameters"]["study"]
