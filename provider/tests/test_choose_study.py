"""WP20: the study Laya chooses -- the plumbing of it, against a fake classification provider.

These tests establish that the right questions are asked with the right declared options, that an estimator which
cannot serve the estimand the roles imply is never offered, that a composition which does not validate is refused by
the validator's own name with the decisions kept, and that a spec these choices produce is one `prepare-study`
accepts. They establish NOTHING about whether a choice is good: the fake here answers by a rule a test author wrote,
and even the real checkpoint's probabilities are uncalibrated head outputs. The recovery table is the judge.

The fake is the one from M5PHET's `tests/test_decide.py`, kept in Laya's answer shape, because `m5phet.decide` decides
whether a decision may exist by reading `backend` -- so a provider that is not Laya cannot produce one here either.
"""

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

pandas = pytest.importorskip("pandas")
decide = pytest.importorskip("m5phet.decide")
m5phet_questions = pytest.importorskip("m5phet.questions")
m5phet_runtime = pytest.importorskip("m5phet.runtime")

from causal_inference_provider import choose_study as chooser          # noqa: E402
from causal_inference_provider import study_space as space_module      # noqa: E402
from causal_inference_provider import study_spec as spec_module        # noqa: E402

#: The interpreter that FITS. No interpreter on this host holds both m5phet (which asks Laya) and EconML (which
#: fits), and neither is going to be installed into the other's venv by a test: the two are deliberately separate.
#: So the chooser runs here and the fit runs in the interpreter this variable names, exactly as the real path does.
FIT_PYTHON_VARIABLE = "CAUSAL_INFERENCE_FIT_PYTHON"


def fit_python():
    """The python that can fit: this one when it holds EconML, else the one `CAUSAL_INFERENCE_FIT_PYTHON` names."""
    if importlib.util.find_spec("econml") is not None:
        return sys.executable
    named = os.environ.get(FIT_PYTHON_VARIABLE)
    if named and Path(named).exists():
        return named
    return None


requires_fit = pytest.mark.skipif(fit_python() is None,
                                  reason=f"fitting needs EconML; this interpreter serves and does not fit, and "
                                         f"{FIT_PYTHON_VARIABLE} names no interpreter that does")

PROBLEM = ("Estimate the effect of treatment on outcome; baseline may modify the effect; confounder is a common "
           "cause")

#: the roles that make the synthetic modifier study, used as the fake's answer when it is asked about a column
MODIFIER_ROLES = {"baseline": "modifier", "confounder": "confounder", "treatment": "treatment", "outcome": "outcome"}

SERVING_SPACE = space_module.study_space(probe=space_module.declared)


class FakeLaya:
    """A classification provider with Laya's answer shape, answering by a declared rule instead of by a model.

    `answers` maps a question name to the key it chooses. Every call is kept in `seen`, so a test can read exactly
    which questions were asked, in which order, with which options and against which state."""

    name, area = "laya_news", "classification"

    def __init__(self, answers, *, backend="laya", confidence=None):
        self.answers, self.backend, self.seen = dict(answers), backend, []
        #: question name -> the probability the chosen key carries. Without it every option shares the mass, which is
        #: what the zero-shot checkpoint actually does and is below any threshold a measurement could justify.
        self.confidence = dict(confidence or {})

    def capabilities(self):
        return {"provider": self.name, "operations": ["infer"], "families": ["classification"],
                "output_kinds": ["typed_questions"], "uncertainty_methods": ["UNCALIBRATED_CLASS_PROBABILITIES"],
                "supported": [{"operation": "infer", "family": "classification", "output_kind": "typed_questions"}],
                "known_states": ["laya-checkpoint:fake"], "backend": self.backend}

    def question_types(self):
        return {"choice": {"required": ["options"], "optional": ["instructions"]}}

    def answer_questions(self, state, questions, data, as_of):
        self.seen.append({"state": state, "questions": {name: dict(question) for name, question in questions.items()}})
        answers = {}
        for name, question in questions.items():
            keys = [key for key, _label in question["options"]]
            chosen = self.answers.get(name)
            if chosen is None or chosen not in keys:
                raise AssertionError(f"the fake was not told what to answer for {name!r} among {keys}")
            top = self.confidence.get(name)
            share = round((1.0 - top) / (len(keys) - 1), 4) if top is not None else round(1.0 / len(keys), 4)
            probabilities = {key: share for key in keys}
            probabilities[chosen] = top if top is not None else round(1.0 - share * (len(keys) - 1), 4)
            answers[name] = {"type": "choice", "status": "OK", "label": chosen, "backend": self.backend,
                             "instructions": question.get("instructions"),
                             "options": [list(pair) for pair in question["options"]],
                             "uncalibrated_probabilities": probabilities, "probability_decimals": 4,
                             "calibration": "UNCALIBRATED", "execution_authorized": False}
        answers["__state_ref__"] = "laya-checkpoint:fake"
        return answers


def registry_with(provider):
    registry = m5phet_runtime.Registry()
    registry.register(provider)
    return registry


def answering(roles, confidence=None, **rest):
    """A fake that gives every column the role the mapping names, and the declared answer to every other question."""
    return FakeLaya({**roles, "estimator": "LinearDML", "model_y": "lasso", "model_t": "lasso",
                     "confidence_level": "0.95", **rest}, confidence=confidence)


#: every question this chooser asks, answered at a confidence a measurement can justify. Used as the base, so a test
#: about abstention changes exactly the questions it is about and nothing else.
def confident(**overrides):
    base = {name: 0.93 for name in list(MODIFIER_ROLES) + ["estimator", "model_y", "model_t", "confidence_level"]}
    return {**base, **overrides}


@pytest.fixture(scope="module")
def modifier_csv(tmp_path_factory):
    """The synthetic modifier data as a CSV, regenerated from its own declared generator and seed."""
    pytest.importorskip("numpy")
    from causal_inference_provider.example import modifier_example_data
    path = tmp_path_factory.mktemp("wp20") / "synthetic_modifier.csv"
    modifier_example_data().to_csv(path, index=False)
    return path


# --- the profile ---------------------------------------------------------------------------------------------------

def test_profile_describes_every_column_and_carries_no_rows(modifier_csv):
    profile = chooser.profile_dataset(modifier_csv)
    assert profile["columns"] == ["baseline", "confounder", "treatment", "outcome"]
    assert profile["rows"] == 2400
    detail = profile["columns_detail"]
    assert set(detail) == set(profile["columns"])
    assert detail["baseline"]["binary"] is True and detail["baseline"]["coded_0_1"] is True
    assert detail["confounder"]["binary"] is False
    for column, summary in detail.items():
        assert set(summary) >= {"dtype", "numeric", "missing_fraction", "distinct_values", "binary"}
        assert set(summary) >= {"mean", "std", "min", "max"}, column
    # the whole profile is 5 scalars per column plus a name list: nothing in it is as long as the table
    assert len(json.dumps(profile)) < 2000


def test_profile_rounds_every_number_to_the_declared_decimals(modifier_csv):
    profile = chooser.profile_dataset(modifier_csv)
    for summary in profile["columns_detail"].values():
        for key, value in summary.items():
            if isinstance(value, float):
                assert value == round(value, chooser.SUMMARY_DECIMALS), key


def test_a_state_text_carries_no_row_and_no_undeclared_decimal(modifier_csv):
    """What the model is shown is the description, at the decimals the profile declared, and nothing else."""
    fake = answering(MODIFIER_ROLES)
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM)
    assert document["status"] == "OK"
    frame = pandas.read_csv(modifier_csv)
    seen_states = [call["state"]["news"] for call in fake.seen]
    for state in seen_states:
        for number in re.findall(r"-?\d+\.(\d+)", state):
            assert len(number) <= chooser.SUMMARY_DECIMALS
        # no row of the table appears: the first row's confounder value, at full precision, is nowhere
        assert repr(float(frame["confounder"].iloc[0])) not in state
        assert len(state) < 4000


# --- the questions -------------------------------------------------------------------------------------------------

def test_every_column_is_asked_exactly_once_with_the_five_roles(modifier_csv):
    fake = answering(MODIFIER_ROLES)
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM)
    assert document["status"] == "OK"
    role_calls = [call for call in fake.seen if set(call["questions"]) <= set(MODIFIER_ROLES)]
    assert [list(call["questions"])[0] for call in role_calls] == ["baseline", "confounder", "treatment", "outcome"]
    five = [list(pair) for pair in SERVING_SPACE["roles"]["options"]]
    assert [key for key, _ in five] == ["treatment", "outcome", "confounder", "modifier", "exclude"]
    for call in role_calls:
        question, = call["questions"].values()
        assert [list(pair) for pair in question["options"]] == five
        assert question["type"] == "choice" and question["instructions"]


def test_a_column_state_carries_the_roles_already_assigned(modifier_csv):
    fake = answering(MODIFIER_ROLES)
    chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM)
    role_calls = [call for call in fake.seen if set(call["questions"]) <= set(MODIFIER_ROLES)]
    first, last = role_calls[0]["state"]["news"], role_calls[-1]["state"]["news"]
    assert "roles_already_assigned: {}" in first
    assert "baseline: modifier" in last and "treatment: treatment" in last
    assert PROBLEM in first


def test_the_estimator_question_excludes_cate_only_estimators_when_no_modifier_was_chosen(modifier_csv):
    """`CausalForestDML` is fitted ON the modifiers and the space says it reports no ATE; it is never offered here."""
    no_modifier = dict(MODIFIER_ROLES, baseline="confounder")
    fake = answering(no_modifier)
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM)
    assert document["status"] == "OK"
    estimator_call, = [call for call in fake.seen if "estimator" in call["questions"]]
    offered = [key for key, _ in estimator_call["questions"]["estimator"]["options"]]
    detail = SERVING_SPACE["estimators"]["detail"]
    assert "CausalForestDML" not in offered
    assert offered == [key for key, _ in SERVING_SPACE["estimators"]["options"] if "ATE" in detail[key]["estimands"]]
    assert "estimand: ATE" in estimator_call["state"]["news"]


def test_the_estimator_question_offers_the_cate_estimators_when_a_modifier_was_chosen(modifier_csv):
    fake = answering(MODIFIER_ROLES)
    chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM)
    estimator_call, = [call for call in fake.seen if "estimator" in call["questions"]]
    offered = [key for key, _ in estimator_call["questions"]["estimator"]["options"]]
    detail = SERVING_SPACE["estimators"]["detail"]
    assert "CausalForestDML" in offered
    assert offered == [key for key, _ in SERVING_SPACE["estimators"]["options"] if "CATE" in detail[key]["estimands"]]
    assert "estimand: CATE" in estimator_call["state"]["news"]


def test_estimator_options_read_the_spaces_own_estimand_declaration():
    """The filter is not a list of names kept here: it reads what the space declares each estimator can report."""
    for estimand in (space_module.ATE, space_module.CATE):
        offered = chooser.estimator_options(estimand, SERVING_SPACE)
        detail = SERVING_SPACE["estimators"]["detail"]
        assert offered and all(estimand in detail[key]["estimands"] for key, _ in offered)


def test_the_nuisance_and_confidence_questions_offer_exactly_the_declared_options(modifier_csv):
    fake = answering(MODIFIER_ROLES)
    chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM)
    nuisance_call, = [call for call in fake.seen if "model_y" in call["questions"]]
    declared = [list(pair) for pair in SERVING_SPACE["nuisance_models"]["options"]]
    assert set(nuisance_call["questions"]) == {"model_y", "model_t"}
    for question in nuisance_call["questions"].values():
        assert [list(pair) for pair in question["options"]] == declared
    confidence_call, = [call for call in fake.seen if "confidence_level" in call["questions"]]
    assert ([list(pair) for pair in confidence_call["questions"]["confidence_level"]["options"]]
            == [list(pair) for pair in SERVING_SPACE["confidence_levels"]["options"]])


def test_exactly_one_decision_is_recorded_per_choice(modifier_csv, tmp_path):
    fake = answering(MODIFIER_ROLES)
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM, record_dir=tmp_path)
    questions = [entry["question"] for entry in document["decisions"]]
    assert questions == ["baseline", "confounder", "treatment", "outcome", "estimator", "model_y", "model_t",
                         "confidence_level"]
    assert all(entry["kind"] == chooser.DECISION_KIND for entry in document["decisions"])
    for entry in document["decisions"]:
        loaded = decide.load(entry["record_path"])
        assert loaded["chosen"] == entry["chosen"] and loaded["backend"] == "laya"
        assert loaded["execution_authorized"] is False
        assert decide.decision_sha256(loaded) == entry["digest"]
    assert document["spec"]["decisions"] == [entry["digest"] for entry in document["decisions"]]


# --- refusals ------------------------------------------------------------------------------------------------------

def test_two_outcomes_are_refused_by_the_validators_own_name_with_the_decisions_kept(modifier_csv):
    """A chooser that gives two columns the outcome role has described no study, and that is what is reported."""
    fake = answering(dict(MODIFIER_ROLES, confounder="outcome"))
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM)
    assert document["status"] == "REFUSED"
    assert document["refusal"] == spec_module.TWO_OUTCOMES
    assert "confounder" in document["why"] and "outcome" in document["why"]
    assert document["spec"] is None
    assert [entry["question"] for entry in document["decisions"]][:4] == ["baseline", "confounder", "treatment",
                                                                         "outcome"]
    assert document["decisions"][1]["chosen"] == "outcome"


def test_no_treatment_is_refused_by_name(modifier_csv):
    fake = answering(dict(MODIFIER_ROLES, treatment="exclude"))
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM)
    assert document["status"] == "REFUSED" and document["refusal"] == spec_module.NO_TREATMENT
    assert document["spec"] is None and len(document["decisions"]) == 8


def test_a_fixture_backend_produces_no_decision_and_no_spec(modifier_csv):
    fake = FakeLaya({**MODIFIER_ROLES, "estimator": "LinearDML", "model_y": "lasso", "model_t": "lasso",
                     "confidence_level": "0.95"}, backend="fixture")
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM)
    assert document["status"] == "REFUSED" and document["refusal"] == decide.NON_MODEL_FIXTURE
    assert document["decisions"] == [] and document["spec"] is None


def test_a_provider_refusal_keeps_its_own_name(modifier_csv):
    class Refusing(FakeLaya):
        def answer_questions(self, state, questions, data, as_of):
            self.seen.append({"state": state, "questions": questions})
            return {name: m5phet_questions.refusal("TOKEN_BUDGET_EXCEEDED", "the pinned SDK would truncate it",
                                                   "choice")
                    for name in questions} | {"__state_ref__": "laya-checkpoint:fake"}

    document = chooser.choose_study(registry_with(Refusing(MODIFIER_ROLES)), modifier_csv, PROBLEM)
    assert document["status"] == "REFUSED" and document["refusal"] == "TOKEN_BUDGET_EXCEEDED"


def test_an_unreadable_dataset_is_refused_before_anything_is_asked(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("a,a\n1,2\n", encoding="utf-8")
    fake = answering(MODIFIER_ROLES)
    document = chooser.choose_study(registry_with(fake), path, PROBLEM)
    assert document["status"] == "REFUSED" and document["refusal"] == chooser.DATASET_UNREADABLE
    assert fake.seen == []


def test_a_column_named_like_one_of_the_choosers_own_questions_is_refused(tmp_path):
    path = tmp_path / "collide.csv"
    path.write_text("treatment,outcome,estimator\n0,1.5,1\n1,2.5,0\n", encoding="utf-8")
    fake = answering({"treatment": "treatment", "outcome": "outcome", "estimator": "exclude"})
    document = chooser.choose_study(registry_with(fake), path, PROBLEM)
    assert document["status"] == "REFUSED" and document["refusal"] == chooser.COLUMN_NAME_UNUSABLE
    assert "estimator" in document["why"]


def test_an_empty_problem_is_refused_and_nothing_is_asked(modifier_csv):
    fake = answering(MODIFIER_ROLES)
    document = chooser.choose_study(registry_with(fake), modifier_csv, "   ")
    assert document["status"] == "REFUSED" and document["refusal"] == m5phet_questions.STATE_REQUIRED
    assert fake.seen == []


# --- the spec the choices produce ------------------------------------------------------------------------------------

def test_the_composed_spec_validates_and_says_what_was_chosen(modifier_csv):
    document = chooser.choose_study(registry_with(answering(MODIFIER_ROLES)), modifier_csv, PROBLEM,
                                    provenance="DEVELOPMENT", study_id="test-laya-chosen-v1")
    spec = document["spec"]
    assert spec_module.validate_spec(spec) == spec
    assert spec["roles"] == MODIFIER_ROLES
    assert spec["estimator"] == "LinearDML" and spec["nuisance"] == {"model_y": "lasso", "model_t": "lasso"}
    assert spec["confidence_level"] == 0.95
    assert spec["dataset"] == {"path": str(modifier_csv), "columns": ["baseline", "confounder", "treatment",
                                                                     "outcome"]}
    assert spec["identification"] == list(space_module.assumptions_for("CATE"))
    assert spec_module.config_from_spec(spec)["estimand"] == "CATE"


def test_the_assumptions_are_the_spaces_own_and_follow_the_estimand(modifier_csv):
    """The chooser does not decide what a study may assume about the world; it copies what the space requires."""
    conditional = chooser.choose_study(registry_with(answering(MODIFIER_ROLES)), modifier_csv, PROBLEM)
    average = chooser.choose_study(registry_with(answering(dict(MODIFIER_ROLES, baseline="confounder"))),
                                   modifier_csv, PROBLEM)
    assert conditional["spec"]["identification"] == list(space_module.assumptions_for("CATE"))
    assert average["spec"]["identification"] == list(space_module.assumptions_for("ATE"))
    assert "constant_effect" in average["spec"]["identification"]
    assert "constant_effect" not in conditional["spec"]["identification"]


@requires_fit
def test_prepare_study_accepts_the_spec_the_chooser_wrote(modifier_csv, tmp_path):
    """The end of the path: a spec these decisions produced is fitted by the explicit command without an edit."""
    document = chooser.choose_study(registry_with(answering(MODIFIER_ROLES)), modifier_csv, PROBLEM,
                                    provenance="DEVELOPMENT", study_id="test-laya-chosen-v1",
                                    record_dir=tmp_path / "decisions")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(document["spec"]), encoding="utf-8")
    state_dir = tmp_path / "studies"
    finished = subprocess.run([fit_python(), "-m", "causal_inference_provider", "prepare-study",
                               "--spec", str(spec_path), "--state-dir", str(state_dir)],
                              capture_output=True, text=True, timeout=900)
    report = json.loads(finished.stdout or "{}")
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert report["result"]["status"] == "OK"
    assert report["study_id"] == "test-laya-chosen-v1"
    retained = json.loads(next(Path(state_dir).glob("*.json")).read_text(encoding="utf-8"))
    assert retained["manifest"]["spec"]["decisions"] == [entry["digest"] for entry in document["decisions"]]


def cli_arguments(**overrides):
    import argparse
    return argparse.Namespace(**{"dataset": None, "problem": PROBLEM, "out": None, "record_dir": None,
                                 "study_id": None, "provenance": "DEVELOPMENT", "outcome_unit": None,
                                 "as_of": None, "min_confidence": None, "abstention_source": None, **overrides})


def test_the_cli_writes_the_spec_it_composed(tmp_path, modifier_csv, capsys):
    """The command's contract: `--out` receives the spec, stdout the whole choice document."""
    from causal_inference_provider import __main__ as cli
    out = tmp_path / "study_spec.json"
    args = cli_arguments(dataset=modifier_csv, out=out, record_dir=tmp_path / "decisions",
                         study_id="test-laya-chosen-v1", outcome_unit="synthetic outcome units")
    assert cli.choose_study_command(args, decider=registry_with(answering(MODIFIER_ROLES))) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert spec_module.validate_spec(written) == written
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "OK" and printed["spec_path"] == str(out)
    assert printed["spec_sha256"] == spec_module.spec_digest(written)
    assert len(printed["decisions"]) == 8


def test_the_cli_writes_nothing_when_the_composition_is_refused(tmp_path, modifier_csv, capsys):
    from causal_inference_provider import __main__ as cli
    out = tmp_path / "study_spec.json"
    args = cli_arguments(dataset=modifier_csv, out=out, record_dir=tmp_path / "decisions")
    refusing = registry_with(answering(dict(MODIFIER_ROLES, confounder="outcome")))
    assert cli.choose_study_command(args, decider=refusing) == 2
    assert not out.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "REFUSED" and printed["refusal"] == spec_module.TWO_OUTCOMES


# --- WP20: the declared abstention threshold ------------------------------------------------------------------------

#: the reliability bins WP09 measured on 450 independently labelled rows. The fixture below carries exactly these, so
#: a threshold this suite declares is one that report resolved and the tests cannot drift from the measurement.
MEASURED_BINS = [(0.0, 0.1, 0, None), (0.1, 0.2, 0, None), (0.2, 0.3, 0, None), (0.3, 0.4, 134, 0.3208955223880597),
                 (0.4, 0.5, 152, 0.35526315789473684), (0.5, 0.6, 58, 0.29310344827586204),
                 (0.6, 0.7, 22, 0.2727272727272727), (0.7, 0.8, 29, 0.3103448275862069),
                 (0.8, 0.9, 24, 0.7083333333333334), (0.9, 1.0, 31, 1.0)]

THRESHOLD = 0.8


@pytest.fixture(scope="module")
def quality_report(tmp_path_factory):
    """A `m5phet-evaluation-report/1` document with WP09's measured bins: the citation a threshold needs."""
    path = tmp_path_factory.mktemp("wp09") / "report_laya_zero_shot.json"
    path.write_text(json.dumps({
        "version": "m5phet-evaluation-report/1", "stage": "laya_zero_shot", "family": "classification",
        "protocol_digest": "b9aefb3c" + "0" * 56, "corpus_seal": "31257d47" + "0" * 56,
        "metric_sets": [{"name": "calibration", "values": {"reliability": {
            "bins": [{"bin": [low, high], "count": count, "accuracy": accuracy}
                     for low, high, count, accuracy in MEASURED_BINS]}}}]}), encoding="utf-8")
    return path


def test_a_role_answered_below_the_threshold_leaves_its_column_unassigned_and_refuses_the_composition(
        modifier_csv, quality_report, tmp_path):
    """The whole rule in one run: one column below the threshold, and no study is composed from the hole it leaves."""
    fake = answering(MODIFIER_ROLES, confidence=confident(baseline=0.41))
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM, space=SERVING_SPACE,
                                    record_dir=tmp_path / "decisions", provenance="DEVELOPMENT",
                                    min_confidence=THRESHOLD, abstention_source=quality_report)
    assert document["status"] == "REFUSED"
    assert document["refusal"] == chooser.ROLES_INCOMPLETE
    assert "baseline" in document["why"]
    assert document["spec"] is None
    assert document["abstained"] == ["baseline"]
    # the other three columns were answered above the threshold, and the run stopped before the estimator question
    assert document["counts"] == {"questions_asked": 4, "answered": 3, "abstained": 1,
                                  "columns_abstained": ["baseline"]}
    # the abstention is recorded like any other decision, with no choice in it and the threshold it fell below
    abstention, = [entry for entry in document["decisions"] if entry["chosen"] is None]
    assert abstention["question"] == "baseline"
    assert abstention["abstention"]["top_probability"] == 0.41
    assert abstention["abstention"]["threshold"]["path"] == str(quality_report)
    assert decide.load(Path(abstention["record_path"]))["chosen"] is None


def test_every_column_abstaining_names_every_column_and_composes_nothing(modifier_csv, quality_report):
    """The zero-shot checkpoint's own behaviour: mass spread over five roles, which is below any measured threshold."""
    fake = answering(MODIFIER_ROLES)                      # no confidence declared: every option shares the mass
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM, space=SERVING_SPACE,
                                    provenance="DEVELOPMENT", min_confidence=THRESHOLD,
                                    abstention_source=quality_report)
    assert document["refusal"] == chooser.ROLES_INCOMPLETE
    assert document["abstained"] == ["baseline", "confounder", "treatment", "outcome"]
    assert document["counts"]["answered"] == 0 and document["counts"]["abstained"] == 4
    assert all(entry["chosen"] is None for entry in document["decisions"])


def test_with_every_answer_above_the_threshold_the_study_is_composed_exactly_as_without_one(
        modifier_csv, quality_report):
    confident_fake = answering(MODIFIER_ROLES, confidence=confident())
    gated = chooser.choose_study(registry_with(confident_fake), modifier_csv, PROBLEM, space=SERVING_SPACE,
                                 provenance="DEVELOPMENT", min_confidence=THRESHOLD,
                                 abstention_source=quality_report)
    assert gated["status"] == "OK"
    assert gated["counts"] == {"questions_asked": 8, "answered": 8, "abstained": 0, "columns_abstained": []}
    assert spec_module.validate_spec(gated["spec"], SERVING_SPACE) == gated["spec"]
    assert gated["spec"]["roles"] == MODIFIER_ROLES
    # and the gate changed nothing about the study itself
    ungated = chooser.choose_study(registry_with(answering(MODIFIER_ROLES, confidence=confident())), modifier_csv,
                                   PROBLEM, space=SERVING_SPACE, provenance="DEVELOPMENT")
    assert ungated["spec"]["roles"] == gated["spec"]["roles"]
    assert ungated["spec"]["estimator"] == gated["spec"]["estimator"]


def test_a_threshold_with_no_cited_measurement_refuses_before_the_dataset_is_even_read(modifier_csv):
    fake = answering(MODIFIER_ROLES, confidence=confident())
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM, space=SERVING_SPACE,
                                    provenance="DEVELOPMENT", min_confidence=THRESHOLD)
    assert document["refusal"] == decide.UNCITED_THRESHOLD
    assert document["profile"] is None and document["decisions"] == []
    assert fake.seen == []


def test_a_threshold_the_cited_report_never_resolved_is_refused_by_name(modifier_csv, quality_report):
    fake = answering(MODIFIER_ROLES, confidence=confident())
    document = chooser.choose_study(registry_with(fake), modifier_csv, PROBLEM, space=SERVING_SPACE,
                                    provenance="DEVELOPMENT", min_confidence=0.85, abstention_source=quality_report)
    assert document["refusal"] == decide.THRESHOLD_NOT_MEASURED
    assert fake.seen == []


def test_compose_spec_refuses_a_role_map_with_a_hole_in_it(modifier_csv):
    """Directly, so the refusal belongs to the composition and not to the loop that happened to call it."""
    profile = chooser.profile_dataset(modifier_csv)
    partial = {name: role for name, role in MODIFIER_ROLES.items() if name != "baseline"}
    with pytest.raises(chooser.ChoiceRefused) as refused:
        chooser.compose_spec(profile, partial, "LinearDML", {"model_y": "lasso", "model_t": "lasso"}, "0.95", [],
                             space=SERVING_SPACE, provenance="DEVELOPMENT", abstained=["baseline"])
    assert refused.value.refusal == chooser.ROLES_INCOMPLETE
    assert "baseline" in refused.value.why


def test_the_cli_prints_the_abstained_columns_and_writes_no_spec(tmp_path, modifier_csv, quality_report, capsys):
    from causal_inference_provider import __main__ as cli
    out = tmp_path / "study_spec.json"
    args = cli_arguments(dataset=modifier_csv, out=out, record_dir=tmp_path / "decisions",
                         min_confidence=THRESHOLD, abstention_source=quality_report)
    fake = answering(MODIFIER_ROLES, confidence=confident(baseline=0.41, confounder=0.55))
    assert cli.choose_study_command(args, decider=registry_with(fake)) == 2
    assert not out.exists()
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["refusal"] == chooser.ROLES_INCOMPLETE
    assert printed["abstained"] == ["baseline", "confounder"]
    assert "2 abstained" in captured.err and "'baseline'" in captured.err and "'confounder'" in captured.err
