"""The event study: registration by digest, the three questions it answers, and the refusals in both directions.

An event study is not a treatment-effect study and this file is mostly about keeping the two apart. It registers a
projections document under an identifier, checks that the manifest carries the sha256 of BOTH documents and no
number of its own, and asks the three types the envelope declares for it. Then it asks the questions the wrong way
round -- `ate` of the event study, `impulse_response` of the demo treatment-effect study -- and checks that each is
refused `NOT_ESTIMABLE` by name with the kind of study that WOULD answer named, because a refusal that says only
"unsupported" teaches nobody which study to fit.

The projections document here is a small hand-written fixture with the shape `feature_eng_m5phet.local_projections`
writes, not a fitted study: nothing in this file fits anything, and the numbers in it are arbitrary and are never
compared to a market. The one test that goes through the real counterfactual engine builds a real planted world with
feature-eng and is skipped where feature-eng is not installed -- never replaced by a stand-in.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil

import pytest

from causal_inference_provider import event_study as _event
from causal_inference_provider.chat import M5PHETCausalProvider, state_directory
from causal_inference_provider.questions import QUESTION_TYPES, answer_questions

EVENTS = ("United States | Nonfarm Payrolls", "United States | CPI")
HORIZONS = (5, 30, 60)
OUTCOMES = ("log_return", "realized_vol")
CAVEAT = ("the surprise's publication instant is assumed equal to the scheduled instant; no receipt or publication "
          "timestamp was observed; results are DEVELOPMENT and not identified until a publication clock exists")


def _coefficients(beta, half):
    return {"const": {"value": 0.0, "std_error": 1e-5, "ci_lower": -2e-5, "ci_upper": 2e-5},
            "surprise": {"value": beta, "std_error": half / 2.0, "ci_lower": beta - half, "ci_upper": beta + half},
            "pre_event_realized_vol": {"value": 0.5, "std_error": 0.1, "ci_lower": 0.3, "ci_upper": 0.7},
            "other_surprises_in_window_negative_offsets": {"value": 1e-6, "std_error": 1e-6, "ci_lower": -1e-6,
                                                           "ci_upper": 3e-6}}


def projections_document():
    """The shape `m5phet.event_projections.v1` has, with arbitrary numbers. Nothing here was fitted."""
    projections = []
    for index, event in enumerate(EVENTS):
        for horizon in HORIZONS:
            for outcome in OUTCOMES:
                beta = (index + 1) * horizon * 1e-6
                half = 2e-6 * horizon
                projections.append({
                    "event_type": event, "horizon_minutes": horizon, "outcome": outcome,
                    "n_fit_events": 80 + index, "n_holdout_events": 20, "status": "OK", "n": 80 + index,
                    "r_squared": 0.1, "hac_maxlags": 3,
                    "columns": ["const", "surprise", "pre_event_realized_vol",
                                "other_surprises_in_window_negative_offsets"],
                    "coefficients": _coefficients(beta, half),
                    "beta": beta, "beta_std_error": half / 2.0, "beta_ci_95": [beta - half, beta + half],
                })
    projections.append({"event_type": EVENTS[0], "horizon_minutes": 240, "outcome": "log_return",
                        "n_fit_events": 3, "n_holdout_events": 0, "status": "TOO_FEW_FIT_EVENTS",
                        "why": "3 fitting event(s) and 20 are declared as the fewest"})
    tests = [{"horizon_minutes": horizon, "outcome": outcome,
              "verdict": "INTERACTIONS_IMPROVE" if (horizon, outcome) == (60, "realized_vol") else "ADDITIVE_HOLDS",
              "why": "fixture"}
             for horizon in (*HORIZONS, 240) for outcome in OUTCOMES]
    return {
        "schema": _event.PROJECTIONS_SCHEMA,
        "provenance": "DEVELOPMENT_ASSUMED_CLOCK",
        "publication_clock": {"mode": "ASSUMED_SCHEDULED_PUBLICATION", "tolerance_seconds": 60,
                              "declared_by": "operator", "identification_caveat": CAVEAT},
        "identification": "NOT_IDENTIFIED",
        "identification_reasons": ["ASSUMED_PUBLICATION_CLOCK: " + CAVEAT],
        "identification_reading": "NOT_IDENTIFIED is the verdict whenever the publication clock was ASSUMED",
        "rows_document": {"path": "rows.json", "schema": "m5phet.event_rows.v1", "parameters": {"window_hours": 24.0}},
        "estimator": {"name": "local projection (Jorda 2005)", "covariance": "HAC (Newey-West)"},
        "held_out": {"fraction": 0.2, "rule": "the last fifth of each event type's releases"},
        "event_types": list(EVENTS), "horizons_minutes": [*HORIZONS, 240], "outcomes": list(OUTCOMES),
        "projections": projections,
        "closure_table": [],
        "superposition": {"verdict": "INTERACTIONS_IMPROVE", "tests": tests},
        "placebo": {"status": "OK", "verdicts": []},
        "execution_authorized": False,
    }


@pytest.fixture
def registered(tmp_path):
    """One registered event study, in a state directory of this test's own."""
    projections = tmp_path / "projections.json"
    projections.write_text(json.dumps(projections_document()), encoding="utf-8")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"schema": "m5phet.event_rows.v1", "rows": []}), encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    reference = _event.register(projections, rows, "eurusd-events-fixture-v1", state)
    return {"state": state, "projections": projections, "rows": rows, "state_ref": reference}


@pytest.fixture
def provider(registered):
    return M5PHETCausalProvider(registered["state"], state_refs=[])


# ------------------------------------------------------------------------------------- registration and its digests

def test_registration_records_the_digest_of_both_documents_and_no_number(registered, provider):
    manifest = provider.event_studies()["eurusd-events-fixture-v1"]
    assert manifest["schema"] == _event.SCHEMA and manifest["kind"] == "event_study"
    assert manifest["state_ref"] == registered["state_ref"]
    assert manifest["documents"]["projections"]["sha256"] == _event.digest_of(registered["projections"])
    assert manifest["documents"]["rows"]["sha256"] == _event.digest_of(registered["rows"])
    assert manifest["identification"]["verdict"] == "NOT_IDENTIFIED"
    assert manifest["identification_caveat"] == CAVEAT
    assert manifest["event_types"] == list(EVENTS)
    assert manifest["superposition"]["by_horizon_and_outcome"]["h=60 realized_vol"] == "INTERACTIONS_IMPROVE"
    text = json.dumps(manifest)
    assert "beta" not in text, "the manifest carries a coefficient; it must point at the document, not copy it"


def test_a_projections_document_that_changed_is_refused_by_name(registered, provider):
    document = projections_document()
    document["projections"][0]["beta"] = 999.0
    registered["projections"].write_text(json.dumps(document), encoding="utf-8")
    answer = answer_questions(provider, {"study": "eurusd-events-fixture-v1"},
                              {"q": {"type": "impulse_response", "event": EVENTS[0], "outcome": "return"}}, None, None)
    assert answer["q"]["status"] == "REFUSED" and answer["q"]["refusal"] == "NOT_ESTIMABLE"
    assert "PROJECTIONS_DIGEST_MISMATCH" in answer["q"]["why"]


def test_a_document_of_another_schema_and_a_bad_identifier_are_refused(tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": "m5phet.event_rows.v1"}), encoding="utf-8")
    rows = tmp_path / "rows.json"
    rows.write_text("{}", encoding="utf-8")
    with pytest.raises(_event.EventStudyError):
        _event.register(other, rows, "fine-id", tmp_path)
    good = tmp_path / "p.json"
    good.write_text(json.dumps(projections_document()), encoding="utf-8")
    with pytest.raises(_event.EventStudyError):
        _event.register(good, rows, "NOT AN ID", tmp_path)
    with pytest.raises(_event.EventStudyError):
        _event.register(good, tmp_path / "missing.json", "fine-id", tmp_path)


# ------------------------------------------------------------------------------------------------- the three answers

def ask(provider, question, **state):
    return answer_questions(provider, {"study": "eurusd-events-fixture-v1", **state}, {"q": question}, None, None)["q"]


def test_the_impulse_response_is_the_beta_path_with_its_interval_and_n(provider):
    answer = ask(provider, {"type": "impulse_response", "event": EVENTS[0], "outcome": "return"})
    assert answer["status"] == "OK" and answer["type"] == "impulse_response"
    assert answer["execution_authorized"] is False
    assert [row["horizon_minutes"] for row in answer["table"]] == [*HORIZONS, 240]
    fitted = [row for row in answer["table"] if row["status"] == "OK"]
    assert len(fitted) == len(HORIZONS)
    for row in fitted:
        assert row["confidence_interval"] == [row["ci_lower"], row["ci_upper"]]
        assert row["ci_lower"] < row["beta"] < row["ci_upper"]
        assert row["n"] == 80
    unfitted = [row for row in answer["table"] if row["status"] != "OK"]
    assert unfitted and all("beta" not in row for row in unfitted)
    assert answer["identification"]["verdict"] == "NOT_IDENTIFIED"
    assert answer["identification_caveat"] == CAVEAT
    assert "NOT_IDENTIFIED" in answer["conclusion"]


def test_the_impulse_response_refuses_an_event_or_a_horizon_nobody_fitted(provider):
    answer = ask(provider, {"type": "impulse_response", "event": "Neverland | Nothing", "outcome": "return"})
    assert answer["refusal"] == "NOT_ESTIMABLE" and "Neverland" in answer["why"]
    answer = ask(provider, {"type": "impulse_response", "event": EVENTS[0], "outcome": "profit"})
    assert answer["refusal"] == "MALFORMED_QUESTION"
    answer = ask(provider, {"type": "impulse_response", "event": EVENTS[0], "outcome": "return", "horizons": [7]})
    assert answer["refusal"] == "NOT_ESTIMABLE" and "not interpolated" in answer["why"]


def test_the_sensitivity_puts_several_events_side_by_side_inside_a_horizon_span(provider):
    answer = ask(provider, {"type": "sensitivity", "events": list(EVENTS), "outcome": "volatility", "window": "4h"})
    assert answer["status"] == "OK" and answer["window_minutes"] == 240.0
    assert answer["horizons_minutes"] == [*HORIZONS, 240]
    assert {row["event"] for row in answer["table"]} == set(EVENTS)
    assert answer["most_sensitive"]["event"] == EVENTS[1], "the larger fitted beta is the second event's"
    narrow = ask(provider, {"type": "sensitivity", "events": [EVENTS[0]], "outcome": "return", "window": 30})
    assert narrow["horizons_minutes"] == [5, 30]
    assert narrow["window_minutes"] == 30.0


def test_the_sensitivity_refuses_a_window_it_cannot_read_and_a_span_shorter_than_every_horizon(provider):
    answer = ask(provider, {"type": "sensitivity", "events": list(EVENTS), "outcome": "return",
                            "window": "the next few hours"})
    assert answer["refusal"] == "MALFORMED_QUESTION" and "horizon span" in answer["why"]
    answer = ask(provider, {"type": "sensitivity", "events": list(EVENTS), "outcome": "return", "window": 1})
    assert answer["refusal"] == "NOT_ESTIMABLE" and "not interpolated" in answer["why"]
    answer = ask(provider, {"type": "sensitivity", "events": ["Neverland | Nothing"], "outcome": "return",
                            "window": "4h"})
    assert answer["refusal"] == "NOT_ESTIMABLE"


def test_the_counterfactual_refuses_an_unfitted_event_and_a_window_that_is_not_two_instants(provider):
    answer = ask(provider, {"type": "counterfactual_path", "window": ["2019-01-01T00:00:00+00:00",
                                                                     "2019-01-02T00:00:00+00:00"],
                            "zero_out": "Neverland | Nothing"})
    assert answer["refusal"] == "NOT_ESTIMABLE" and "subtraction from nothing" in answer["why"]
    answer = ask(provider, {"type": "counterfactual_path", "window": "last week", "zero_out": EVENTS[0]})
    assert answer["refusal"] == "MALFORMED_QUESTION" and "pair of ISO-8601 instants" in answer["why"]


# ------------------------------------------------------------------------------- neither study answers the other's

def test_an_event_study_refuses_ate_and_cate_by_name(provider):
    for kind in ("ate", "cate"):
        answer = ask(provider, {"type": kind})
        assert answer["status"] == "REFUSED" and answer["refusal"] == "NOT_ESTIMABLE"
        assert "EVENT STUDY" in answer["why"] and "no treatment arms" in answer["why"]
        assert "impulse_response" in answer["why"]
        assert "effect_size" not in answer


def retained_effect_study():
    for directory in (state_directory(), Path.home() / ".local/share/causal-inference-m5phet/studies"):
        for path in sorted(directory.glob("*.json")):
            body = json.loads(path.read_text(encoding="utf-8"))
            if body.get("schema") == "causal-inference.study.v1" and not (body.get("config") or {}).get(
                    "effect_modifiers"):
                return path
    return None


def test_a_treatment_effect_study_refuses_the_three_event_types_by_name(tmp_path):
    source = retained_effect_study()
    if source is None:
        pytest.skip("no treatment-effect study is retained; run `python -m causal_inference_provider prepare-demo`")
    shutil.copy(source, tmp_path / source.name)
    provider = M5PHETCausalProvider(tmp_path)
    state = {"state_ref": "causal-ate:" + source.stem}
    questions = {"a": {"type": "impulse_response", "event": EVENTS[0], "outcome": "return"},
                 "b": {"type": "sensitivity", "events": list(EVENTS), "outcome": "return", "window": "4h"},
                 "c": {"type": "counterfactual_path", "window": ["2019-01-01T00:00:00+00:00",
                                                                 "2019-01-02T00:00:00+00:00"],
                       "zero_out": EVENTS[0]}}
    answers = answer_questions(provider, state, questions, None, None)
    for name, question in questions.items():
        answer = answers[name]
        assert answer["status"] == "REFUSED" and answer["refusal"] == "NOT_ESTIMABLE", answer
        assert "event_study" in answer["why"] and "register-event-study" in answer["why"]
        assert answer["type"] == question["type"]


def test_the_five_declared_types_are_the_two_plus_the_three(provider):
    assert sorted(QUESTION_TYPES) == ["ate", "cate", "counterfactual_path", "impulse_response", "sensitivity"]
    assert sorted(provider.question_types()) == sorted(QUESTION_TYPES)


def test_the_event_slot_offers_the_event_types_the_registered_study_carries(provider):
    slots = {slot["name"]: slot for slot in provider.chat_slots()}
    assert "event" in slots
    assert slots["event"]["allowed"] == sorted(EVENTS)
    assert "Nonfarm Payrolls" in slots["event"]["aliases"][EVENTS[0]]
    assert provider.event_types() == sorted(EVENTS)


def test_a_name_nobody_registered_is_refused_with_the_names_that_are(provider):
    answers = answer_questions(provider, {"study": "not-a-study"},
                               {"q": {"type": "impulse_response", "event": EVENTS[0], "outcome": "return"}},
                               None, None)
    assert answers["q"]["status"] == "REFUSED"
    assert "eurusd-events-fixture-v1" in answers["q"]["why"]


def test_a_dataset_attached_to_an_event_study_is_refused_rather_than_fitted(provider):
    answers = answer_questions(provider, {"study": "eurusd-events-fixture-v1"},
                               {"q": {"type": "impulse_response", "event": EVENTS[0], "outcome": "return"}},
                               [{"a": 1}], None)
    assert answers["q"]["status"] == "REFUSED" and "does not fit during inference" in answers["q"]["why"]


# ----------------------------------------------------------------------------------- the envelope, end to end

def test_run_task_answers_the_envelope_from_the_registered_event_study(provider):
    runtime = pytest.importorskip("m5phet.runtime")
    questions = pytest.importorskip("m5phet.questions")
    registry = runtime.Registry()
    registry.register(provider)
    response = questions.run_task({
        "area": "causal",
        "state": {"study": "eurusd-events-fixture-v1"},
        "questions": {"nfp": {"type": "impulse_response", "event": EVENTS[0], "outcome": "return"},
                      "cpi_vol": {"type": "impulse_response", "event": EVENTS[1], "outcome": "volatility"},
                      "ate": {"type": "ate"}},
    }, registry)
    assert response["answered"] == 2 and response["refused"] == 1
    assert response["execution_authorized"] is False
    assert response["state_ref"].startswith(_event.REF_PREFIX)
    assert response["answers"]["nfp"]["table"][0]["horizon_minutes"] == 5
    assert response["answers"]["ate"]["refusal"] == "NOT_ESTIMABLE"


# ------------------------------------------------------------- the counterfactual, through the engine that owns it

def _planted_world(tmp_path):
    """A tiny synthetic market with a response planted by hand, built by feature-eng itself.

    Two event types alternate every four hours; each release moves the log price by `beta * standardized surprise`
    thirty minutes later and by nothing else. It exists so the counterfactual answer is checked against a real fitted
    model rather than against a fixture that agrees with it by construction."""
    import numpy as np
    from feature_eng_m5phet import events, local_projections as lp

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    spacing, first, count = 240, 600, 100
    releases = [("A" if i % 2 == 0 else "B", first + i * spacing,
                 float(value)) for i, value in enumerate(
                     np.round(np.random.default_rng(20260925).uniform(-2.0, 2.0, count), 4))]
    minutes = releases[-1][1] + 2000
    calendar = tmp_path / "calendar.csv"
    lines = ["event_type,event_time,published_at,consensus_published_at,actual,consensus,previous,"
             "historical_availability"]
    for kind, minute, raw in releases:
        moment = start + timedelta(minutes=minute)
        lines.append(",".join([kind, moment.isoformat(), moment.isoformat(),
                               (moment - timedelta(days=1)).isoformat(),
                               repr(100.0 + raw), repr(100.0), repr(100.0), "KNOWN"]))
    calendar.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_bars(path, log_price):
        rows = ["timestamp,close"]
        for i, value in enumerate(log_price):
            rows.append(f"{(start + timedelta(minutes=i)).isoformat()},{float(np.exp(value))!r}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def noise():
        steps = np.random.default_rng(4242).normal(0.0, 1e-5, minutes)
        steps[0] = 0.0
        return np.cumsum(steps)

    build = dict(horizons_minutes=(5, 30), window_hours=4.0, pre_event_minutes=60, min_prior_releases=8)
    flat = write_bars(tmp_path / "flat.csv", noise())
    seen = {row["published_at"]: row["surprise"]
            for row in events.build(str(flat), str(calendar), events.CalendarMapping(), **build)["rows"]}
    path = noise()
    for kind, minute, _ in releases:
        surprise = seen.get((start + timedelta(minutes=minute)).isoformat())
        if surprise is not None:
            path[minute + 30:] += (0.002 if kind == "A" else 0.0011) * surprise
    bars = write_bars(tmp_path / "bars.csv", path)
    document = events.build(str(bars), str(calendar), events.CalendarMapping(), **build)
    rows = tmp_path / "event_rows.json"
    rows.write_text(json.dumps(document), encoding="utf-8")
    projections = lp.estimate(str(rows), outcomes=("log_return",), bars_path=str(bars))
    written = tmp_path / "event_projections.json"
    written.write_text(json.dumps(projections), encoding="utf-8")
    instants = sorted({row["published_at"] for row in document["rows"]})
    return written, rows, (instants[30], instants[60])


def test_the_counterfactual_path_is_computed_by_the_engine_that_owns_it_and_is_labelled(tmp_path):
    pytest.importorskip("feature_eng_m5phet.counterfactual")
    projections, rows, window = _planted_world(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    _event.register(projections, rows, "planted-world-v1", state)
    provider = M5PHETCausalProvider(state, state_refs=[])
    answer = answer_questions(provider, {"study": "planted-world-v1"},
                              {"q": {"type": "counterfactual_path", "window": list(window), "zero_out": "A",
                                     "outcome": "return"}}, None, None)["q"]
    assert answer["status"] == "OK", answer
    assert answer["label"] == "MODEL_BASED_COUNTERFACTUAL"
    assert "MODEL_BASED_COUNTERFACTUAL" in answer["conclusion"]
    assert answer["interval_caveat"].startswith("COVARIANCE_OFF_DIAGONAL_NOT_AVAILABLE")
    answered = [row for row in answer["table"] if row["status"] == "OK"]
    assert answered
    for row in answered:
        assert row["attributed_transient"] == pytest.approx(
            row["predicted_observed"] - row["predicted_counterfactual"], rel=1e-9, abs=1e-15)
        assert row["confidence_interval"][0] <= row["attributed_transient"] <= row["confidence_interval"][1]
        assert row["observed_outcome"] is not None
    planted = [row for row in answered if row["event"] == "A" and row["horizon_minutes"] == 30 and row["zeroed"]]
    assert planted, "no zeroed release of the planted type reached an answer"


def test_registering_an_event_study_does_not_make_an_average_effect_sentence_need_an_event(provider):
    """The regression this file exists to prevent: the moment the first event study was registered on the owner's
    instance, every ATE sentence was refused with 'the question does not name a supported event'."""
    from causal_inference_provider.chat import EVENT_SLOT
    slots = {slot["name"]: slot for slot in provider.chat_slots()}
    assert EVENT_SLOT in slots and slots[EVENT_SLOT].get("required") is False
    assert slots[EVENT_SLOT]["allowed"], "the slot lists the registered study's own events"
