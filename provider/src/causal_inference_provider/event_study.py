"""A second kind of retained study: an event study, and the three questions only it can answer.

Everything this provider retained until now is a treatment-effect study -- one number, or one number per declared
binary subgroup, fitted by EconML over a population of rows. WP22 asks a different question of a different object:
how EUR/USD's return and volatility respond, over the minutes and hours after a calendar release, to the surprise in
that release. The estimator for it is a local projection (Jorda 2005), it lives in feature-eng, and it writes an
`m5phet.event_projections.v1` document with, per event type, horizon and outcome, a coefficient `beta` and its HAC
interval. This module registers such a document as a study of kind `event_study` and answers three named question
types from it -- and only from it.

**What registration is.** `register-event-study --projections P --rows R --id <id>` writes a manifest into
`CAUSAL_INFERENCE_STATE_DIR/event_studies/`. The manifest carries the sha256 of BOTH documents, their paths, and the
declarations a person needs to see before asking anything: the provenance, the publication clock and its
identification caveat, the identification verdict and its reasons, the event types, horizons and outcomes, the
estimator block and the superposition verdict per horizon and outcome. It carries **no number**: every coefficient in
an answer is read from the projections document at the moment the question is asked, and the document's digest is
checked against the manifest first. A file that changed under a registered study is refused by name rather than
answered from.

**The three questions.**

* `impulse_response{event, outcome, horizons?}` -- the response path of one event type: horizon, beta, interval, n.
* `sensitivity{events, outcome, window}` -- the same path for several event types side by side, restricted to the
  horizons at or inside `window`. **`window` here is a horizon span**, not a calendar window: the owner's sentence is
  "what sensitivity does EUR/USD volatility have to the NFP and CPI surprises **in the next four hours**", and four
  hours is how far after the release the answer looks. It is accepted as a number of minutes or as `4h` / `240m`, and
  the answer says in `window_minutes` exactly what it meant. A calendar window -- a past interval of dates -- is the
  `counterfactual_path` question's `window`, and that one is a pair of instants.
* `counterfactual_path{window, zero_out}` -- the fitted model evaluated twice over a past calendar window, once with
  the surprises as published and once with one event type's set to zero. It is computed by
  `feature_eng_m5phet.counterfactual`, the module that owns that arithmetic, and it is labelled
  `MODEL_BASED_COUNTERFACTUAL` everywhere it appears. Where feature-eng is not installed in the serving interpreter,
  the question is refused by name; nothing is approximated in its place.

**What travels with every answer, verbatim.** The identification block of the projections document -- its verdict,
its reasons and its reading -- and the publication clock's identification caveat, copied character for character.
The EUR/USD study registered on this machine was built under an ASSUMED publication clock, so its verdict is
`NOT_IDENTIFIED` and every answer says so. A response path from a document that says it is not identified is a
conditional association with a name, and the answer must not let a reader forget which it is.

**And what it refuses.** An event study cannot answer `ate` or `cate`: it has no treatment arms, no population of
units and no contrast. A treatment-effect study cannot answer the three types above: it has no horizon and no event.
Both directions are refused `NOT_ESTIMABLE`, by name, with the kind of study that would answer stated.
"""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

#: the manifest this module writes and reads
SCHEMA = "causal-inference.event_study.v1"

#: the document the manifest points at, written by `feature_eng_m5phet.local_projections`
PROJECTIONS_SCHEMA = "m5phet.event_projections.v1"

#: the reference an event study is named by, so it can never be mistaken for `causal-ate:<digest>`
REF_PREFIX = "causal-event-study:"
REF_PATTERN = re.compile(re.escape(REF_PREFIX) + r"([a-f0-9]{64})")

#: where the manifests live inside the state directory; the top-level `*.json` there are treatment-effect studies
DIRECTORY_NAME = "event_studies"

STUDY_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,63}")

#: the study kind, as the answers name it
KIND = "event_study"

#: the question types this study kind answers, and the fields each needs
QUESTION_TYPES = {
    "impulse_response": {"required": ["event", "outcome"], "optional": ["horizons"]},
    "sensitivity": {"required": ["events", "outcome", "window"], "optional": []},
    "counterfactual_path": {"required": ["window", "zero_out"], "optional": ["outcome", "horizons"]},
}

#: the two outcomes a person names, and the field each is called in the projections document
OUTCOMES = {"return": "log_return", "volatility": "realized_vol"}

#: what a coefficient of each outcome is measured in. The response is per unit of STANDARDIZED surprise, which is
#: what the projection was fitted on, and saying only "log return" would drop half the unit.
UNITS = {
    "log_return": "log return of the pair over the horizon, per unit of standardized surprise",
    "realized_vol": "realized variance over the horizon (the sum of squared log returns of the declared step), per "
                    "unit of standardized surprise",
}

#: the superposition verdict a counterfactual needs before one pulse may be subtracted from a window
ADDITIVE = "ADDITIVE_HOLDS"

#: how many path rows a counterfactual answer carries into an envelope before it says how many it left out
MAX_PATH_ROWS = 200

#: the largest manifest this module will read back
MAX_MANIFEST_BYTES = 1_000_000

#: the largest projections document it will read at question time
MAX_PROJECTIONS_BYTES = 64_000_000

# Refusal codes shared with `m5phet.questions`, repeated here by value exactly as `questions.py` repeats them, so
# serving keeps no dependency on the workbench package.
NOT_ESTIMABLE = "NOT_ESTIMABLE"
MALFORMED_QUESTION = "MALFORMED_QUESTION"

WINDOW_FORM = ("a horizon span: a number of minutes, or `4h` / `240m`; it says how far after the release the "
               "response is reported, and it is not a calendar interval -- that is `counterfactual_path`'s window")


class EventStudyError(ValueError):
    """A registration or a read this module will not perform, naming what is wrong with it."""


def digest_of(path):
    """The sha256 of a file, read in blocks: a rows document of three hundred megabytes is never held in memory."""
    hasher = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def directory_for(state_dir):
    """Where event-study manifests live under a state directory."""
    return Path(state_dir) / DIRECTORY_NAME


def _clock():
    return datetime.now(timezone.utc).isoformat()


def _read_projections(path, *, limit=MAX_PROJECTIONS_BYTES):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise EventStudyError(f"the projections document {str(path)!r} is not a regular file on disk")
    if path.stat().st_size > limit:
        raise EventStudyError(f"the projections document is larger than the {limit} bytes this provider reads")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != PROJECTIONS_SCHEMA:
        raise EventStudyError(f"schema is {(document or {}).get('schema')!r} and an event study is registered from a "
                              f"{PROJECTIONS_SCHEMA!r} document")
    return document


def superposition_index(document):
    """The superposition verdict per (horizon, outcome) -- the grain the test was run at."""
    return {(int(test["horizon_minutes"]), test["outcome"]): test.get("verdict")
            for test in ((document.get("superposition") or {}).get("tests") or [])}


def register(projections_path, rows_path, study_id, directory):
    """Write the manifest of one event study, with the digests of both documents it points at. Nothing is fitted.

    The manifest carries declarations and no coefficient: an answer reads the projections document itself, after
    checking that its digest is still the one recorded here."""
    if not isinstance(study_id, str) or not STUDY_ID_PATTERN.fullmatch(study_id):
        raise EventStudyError("A study identifier must be 3-64 lowercase characters from [a-z0-9._-].")
    projections_path, rows_path = Path(projections_path).resolve(), Path(rows_path).resolve()
    document = _read_projections(projections_path)
    if not Path(rows_path).is_file():
        raise EventStudyError(f"the rows document {str(rows_path)!r} is not a file; it is what a counterfactual is "
                              f"evaluated over and a study is not registered without it")
    clock = document.get("publication_clock") or {}
    manifest = {
        "schema": SCHEMA,
        "kind": KIND,
        "study_id": study_id,
        "available_at": _clock(),
        "provenance": document.get("provenance"),
        "publication_clock": deepcopy(clock),
        "identification": {
            "verdict": document.get("identification"),
            "reasons": list(document.get("identification_reasons") or []),
            "reading": document.get("identification_reading"),
        },
        "identification_caveat": clock.get("identification_caveat"),
        "event_types": list(document.get("event_types") or []),
        "horizons_minutes": [int(h) for h in document.get("horizons_minutes") or []],
        "outcomes": list(document.get("outcomes") or []),
        "estimator": deepcopy(document.get("estimator")),
        "held_out": deepcopy(document.get("held_out")),
        "superposition": {"verdict": (document.get("superposition") or {}).get("verdict"),
                          "by_horizon_and_outcome": {f"h={horizon} {outcome}": verdict
                                                     for (horizon, outcome), verdict
                                                     in sorted(superposition_index(document).items())},
                          "required_for_a_counterfactual": ADDITIVE},
        "documents": {
            "projections": {"path": str(projections_path), "sha256": digest_of(projections_path),
                            "schema": document.get("schema"), "bytes": projections_path.stat().st_size},
            "rows": {"path": str(rows_path), "sha256": digest_of(rows_path), "schema": (
                document.get("rows_document") or {}).get("schema"), "bytes": Path(rows_path).stat().st_size},
        },
        "reading": ("an event study: local projections of one price series on the standardized surprises of dated "
                    "calendar releases. It has no treatment arms and no population of units, so it answers no `ate` "
                    "and no `cate`; it answers `impulse_response`, `sensitivity` and `counterfactual_path`. Its "
                    f"identification verdict is {document.get('identification')!r} and travels with every answer"),
        "execution_authorized": False,
    }
    raw = json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    reference = REF_PREFIX + sha256(raw).hexdigest()
    directory = directory_for(Path(directory).parent if Path(directory).name == DIRECTORY_NAME else directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (reference[len(REF_PREFIX):] + ".json")).open("xb") as output:
        output.write(raw)
    return reference


def load(path, reference=None):
    """One manifest, read back and checked to be one. The digest in its name is its identity."""
    path = Path(path)
    if path.is_symlink():
        raise EventStudyError("Event-study manifests must not be symlinks.")
    with path.open("rb") as source:
        raw = source.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise EventStudyError("Event-study manifest exceeds the byte limit.")
    digest = sha256(raw).hexdigest()
    if path.stem != digest:
        raise EventStudyError("Event-study manifest digest mismatch.")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise EventStudyError("Unsupported event-study manifest schema.")
    if reference is not None and reference != REF_PREFIX + digest:
        raise EventStudyError("Event-study reference does not name this manifest.")
    return manifest | {"state_ref": REF_PREFIX + digest, "digest": digest}


def retained(state_dir):
    """Every event study a state directory holds right now, keyed by the identifier it was registered under."""
    found = {}
    for path in sorted(directory_for(state_dir).glob("*.json")):
        if not re.fullmatch(r"[a-f0-9]{64}", path.stem) or not path.is_file():
            continue
        try:
            manifest = load(path)
        except (OSError, ValueError):
            continue
        label = manifest.get("study_id") or manifest["digest"][:12]
        found[label if label not in found else f"{label} ({manifest['digest'][:12]})"] = manifest
    return found


def projections_of(manifest):
    """The projections document a manifest points at, refused by name when it is gone or is no longer the same file."""
    declared = (manifest.get("documents") or {}).get("projections") or {}
    path = Path(declared.get("path") or "")
    try:
        document = _read_projections(path)
    except (OSError, ValueError) as trouble:
        raise EventStudyError(f"PROJECTIONS_NOT_READABLE: the projections document this study points at "
                              f"({str(path)!r}) could not be read: {trouble}") from None
    found = digest_of(path)
    if declared.get("sha256") and found != declared["sha256"]:
        raise EventStudyError(f"PROJECTIONS_DIGEST_MISMATCH: the file at {str(path)!r} is not the one this study was "
                              f"registered from ({declared['sha256']} was recorded, {found} is there now); a response "
                              f"path read out of a different document is a different study's number")
    return document


# --------------------------------------------------------------------------------------- what an answer is made of

def _refusal(kind, why, question_type):
    """The same shape as `questions.refusal`: no number, and a reason a caller can read."""
    return {"status": "REFUSED", "refusal": kind, "why": why, "type": question_type}


def identification_block(manifest):
    """The identification verdict, its reasons and its reading, copied from the study exactly as it was registered."""
    return deepcopy(manifest.get("identification") or {})


def _outcome_field(outcome, question_type):
    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        return None, _refusal(MALFORMED_QUESTION, f"`outcome` names {sorted(OUTCOMES)} and this question asks for "
                                                  f"{outcome!r}", question_type)
    return OUTCOMES[outcome], None


def _horizons(asked, manifest, question_type):
    declared = list(manifest.get("horizons_minutes") or [])
    if asked is None:
        return declared, None
    if not isinstance(asked, (list, tuple)) or not asked:
        return None, _refusal(MALFORMED_QUESTION, f"`horizons` is a non-empty list of minutes; this study carries "
                                                  f"{declared}", question_type)
    wanted = []
    for value in asked:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or int(value) not in declared:
            return None, _refusal(NOT_ESTIMABLE, f"horizon {value!r} was not fitted by this study; it fitted "
                                                 f"{declared} minutes, and a horizon nobody projected is not "
                                                 f"interpolated here", question_type)
        wanted.append(int(value))
    return sorted(set(wanted)), None


def window_minutes(value, question_type):
    """The horizon span a `sensitivity` question names, in minutes. See WINDOW_FORM: it is not a calendar interval."""
    if isinstance(value, bool):
        return None, _refusal(MALFORMED_QUESTION, f"`window` is {WINDOW_FORM}", question_type)
    if isinstance(value, (int, float)):
        minutes = float(value)
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(m|min|mins|minutes|h|hr|hrs|hours)?\s*", value, re.IGNORECASE)
        if match is None:
            return None, _refusal(MALFORMED_QUESTION, f"the window {value!r} is not a form this provider reads; it "
                                                      f"reads {WINDOW_FORM}", question_type)
        minutes = float(match[1]) * (60.0 if match[2] and match[2].lower().startswith(("h",)) else 1.0)
    else:
        return None, _refusal(MALFORMED_QUESTION, f"`window` is {WINDOW_FORM}", question_type)
    if not minutes > 0:
        return None, _refusal(MALFORMED_QUESTION, f"the window must be a positive span; got {value!r}", question_type)
    return minutes, None


def response_path(document, event_type, outcome_field, horizons, verdicts):
    """The response-path table of one event type: one row per horizon, with the interval and the counts it rests on.

    This is the table the workbench draws. Every number in it is copied from the projections document; a horizon whose
    projection did not fit carries its status and no number, because a blank in a table is read as a zero."""
    entries = {(int(entry["horizon_minutes"])): entry for entry in document.get("projections") or []
               if entry.get("event_type") == event_type and entry.get("outcome") == outcome_field}
    rows = []
    for horizon in horizons:
        entry = entries.get(int(horizon))
        row = {"event": event_type, "outcome": outcome_field, "horizon_minutes": int(horizon),
               "superposition": verdicts.get((int(horizon), outcome_field))}
        if entry is None:
            row.update({"status": "NOT_FITTED", "why": "this study carries no projection for that event type, "
                                                       "horizon and outcome"})
        elif entry.get("status") != "OK":
            row.update({"status": entry.get("status"), "why": entry.get("why"),
                        "n_fit_events": entry.get("n_fit_events"),
                        "n_holdout_events": entry.get("n_holdout_events")})
        else:
            low, high = entry.get("beta_ci_95") or [None, None]
            row.update({"status": "OK", "beta": entry.get("beta"), "ci_lower": low, "ci_upper": high,
                        "confidence_interval": [low, high], "std_error": entry.get("beta_std_error"),
                        "n": entry.get("n"), "n_fit_events": entry.get("n_fit_events"),
                        "n_holdout_events": entry.get("n_holdout_events"),
                        "r_squared": entry.get("r_squared")})
        rows.append(row)
    return rows


def _answered(rows):
    return [row for row in rows if row.get("status") == "OK"]


def _path_conclusion(event_type, outcome, rows, manifest):
    """A sentence about a response path, read off its own numbers and off the study's identification verdict."""
    answered = _answered(rows)
    verdict = (manifest.get("identification") or {}).get("verdict")
    if not answered:
        return (f"No horizon of {event_type} on the {outcome} of this pair carries a fitted response in this study; "
                f"every row below says why, and none of them is a zero.")
    excluding = [row for row in answered if (row["ci_lower"] or 0) > 0 or (row["ci_upper"] or 0) < 0]
    largest = max(answered, key=lambda row: abs(row["beta"]))
    return (f"Over {len(answered)} fitted horizon(s), the response of the {outcome} to a one-unit standardized "
            f"surprise in {event_type} is largest at {largest['horizon_minutes']} minutes: {largest['beta']:.4g} "
            f"{UNITS[OUTCOMES.get(outcome, outcome)] if outcome in OUTCOMES else ''}"
            f" (95 % interval [{largest['ci_lower']:.4g}, {largest['ci_upper']:.4g}], n = {largest['n']}); "
            f"{len(excluding)} of them have an interval excluding zero. The study's identification verdict is "
            f"{verdict}, so this is a conditional association unless that verdict says otherwise.")


def _provenance_tail(manifest):
    """What every event-study answer carries after its numbers: where they came from and what they are worth.

    It is a TAIL and not a head on purpose. The workbench's deterministic rendering shows an answer's first fields,
    and a line that opens with a state reference and a digest teaches a reader nothing; the response path, the unit
    and the sentence about them come first, and the provenance -- which is long, and which nobody reads at a glance
    -- comes after. Nothing is dropped: every field below is in the answer and in the JSON pane."""
    return {
        "study_kind": KIND, "study_id": manifest.get("study_id"), "state_ref": manifest["state_ref"],
        "provenance": manifest.get("provenance"),
        "identification": identification_block(manifest),
        "publication_clock": deepcopy(manifest.get("publication_clock")),
        "estimator": deepcopy(manifest.get("estimator")),
        "development": str(manifest.get("provenance") or "").startswith("DEVELOPMENT"),
        "execution_authorized": False,
    }


def impulse_response_answer(manifest, document, question):
    """One event type's response path, horizon by horizon, with the interval and n of each."""
    event_type = question.get("event")
    if not isinstance(event_type, str) or event_type not in (manifest.get("event_types") or []):
        return _refusal(NOT_ESTIMABLE, f"{event_type!r} is not an event type this study fitted; it fitted "
                                       f"{list(manifest.get('event_types') or [])}, and a response to a release no "
                                       f"projection was fitted on is not read out of another one", "impulse_response")
    field, trouble = _outcome_field(question.get("outcome"), "impulse_response")
    if trouble:
        return trouble
    horizons, trouble = _horizons(question.get("horizons"), manifest, "impulse_response")
    if trouble:
        return trouble
    rows = response_path(document, event_type, field, horizons, superposition_index(document))
    return {
        "type": "impulse_response", "status": "OK",
        "estimand": "impulse_response",
        "event": event_type, "outcome": question["outcome"], "outcome_field": field,
        "unit": UNITS[field],
        "n_fitted_horizons": len(_answered(rows)),
        "conclusion": _path_conclusion(event_type, question["outcome"], rows, manifest),
        "table": rows,
        "horizons_minutes": horizons,
        "confidence_level": 0.95,
        "identification_caveat": manifest.get("identification_caveat"),
        "reading": ("horizon x beta x 95 % interval x n, from a local projection of the outcome over the horizon on "
                    "the standardized surprise of this release, conditional on the declared controls"),
        **_provenance_tail(manifest),
    }


def sensitivity_answer(manifest, document, question):
    """Several event types' response paths side by side, within a declared horizon span."""
    events = question.get("events")
    if not isinstance(events, (list, tuple)) or not events or any(not isinstance(name, str) for name in events):
        return _refusal(MALFORMED_QUESTION, "`events` is a non-empty list of event type names; this study carries "
                                            f"{list(manifest.get('event_types') or [])}", "sensitivity")
    declared = list(manifest.get("event_types") or [])
    unknown = [name for name in events if name not in declared]
    if unknown:
        return _refusal(NOT_ESTIMABLE, f"{unknown} is not among the event types this study fitted; it fitted "
                                       f"{declared}, and a sensitivity to a release nobody projected is not "
                                       f"estimated during inference", "sensitivity")
    field, trouble = _outcome_field(question.get("outcome"), "sensitivity")
    if trouble:
        return trouble
    span, trouble = window_minutes(question.get("window"), "sensitivity")
    if trouble:
        return trouble
    declared_horizons = list(manifest.get("horizons_minutes") or [])
    horizons = [h for h in declared_horizons if h <= span]
    if not horizons:
        return _refusal(NOT_ESTIMABLE, f"this study fitted the horizons {declared_horizons} minutes and none of them "
                                       f"falls inside a window of {span:g} minutes; a response inside a span shorter "
                                       f"than the shortest horizon fitted is not interpolated here", "sensitivity")
    verdicts = superposition_index(document)
    rows, by_event = [], {}
    for name in events:
        path = response_path(document, name, field, horizons, verdicts)
        by_event[name] = path
        rows.extend(path)
    ranked = sorted((row for row in rows if row.get("status") == "OK"), key=lambda row: -abs(row["beta"]))
    return {
        "type": "sensitivity", "status": "OK",
        "estimand": "sensitivity",
        "events": list(events), "outcome": question["outcome"], "outcome_field": field,
        "unit": UNITS[field],
        "most_sensitive": ({"event": ranked[0]["event"], "horizon_minutes": ranked[0]["horizon_minutes"],
                            "beta": ranked[0]["beta"],
                            "confidence_interval": ranked[0]["confidence_interval"], "n": ranked[0]["n"]}
                           if ranked else None),
        "conclusion": _sensitivity_conclusion(events, question["outcome"], ranked, span, manifest),
        "table": rows,
        "by_event": by_event,
        "window": question["window"], "window_minutes": span, "window_reading": WINDOW_FORM,
        "horizons_minutes": horizons,
        "confidence_level": 0.95,
        "identification_caveat": manifest.get("identification_caveat"),
        "reading": ("one response path per event type, restricted to the horizons at or inside the declared span; "
                    "the paths are read side by side and are NOT added together here -- whether they add is the "
                    f"superposition test, and this study reports {manifest.get('superposition', {}).get('verdict')!r}"),
        **_provenance_tail(manifest),
    }


def _sensitivity_conclusion(events, outcome, ranked, span, manifest):
    verdict = (manifest.get("identification") or {}).get("verdict")
    if not ranked:
        return (f"No fitted response to {list(events)} on the {outcome} of this pair falls inside {span:g} minutes; "
                f"every row says why, and none of them is a zero.")
    first = ranked[0]
    return (f"Within {span:g} minutes of the release, the largest fitted response among {list(events)} on the "
            f"{outcome} is {first['event']} at {first['horizon_minutes']} minutes: {first['beta']:.4g} per unit of "
            f"standardized surprise (95 % interval [{first['ci_lower']:.4g}, {first['ci_upper']:.4g}], "
            f"n = {first['n']}). The study's identification verdict is {verdict}.")


def counterfactual_answer(manifest, document, question):
    """The fitted model evaluated twice over a past calendar window, with one event type's surprise set to zero."""
    zero_out = question.get("zero_out")
    declared = list(manifest.get("event_types") or [])
    if not isinstance(zero_out, str) or zero_out not in declared:
        return _refusal(NOT_ESTIMABLE, f"{zero_out!r} is not an event type this study fitted; it fitted {declared}, "
                                       f"and a counterfactual that sets to zero a surprise no model was fitted on "
                                       f"would be a subtraction from nothing", "counterfactual_path")
    window = question.get("window")
    if (not isinstance(window, (list, tuple)) or len(window) != 2
            or any(not isinstance(value, str) for value in window)):
        return _refusal(MALFORMED_QUESTION, "`window` is a pair of ISO-8601 instants with time zones, [start, end]: "
                                            "the past calendar interval whose releases are set to zero",
                        "counterfactual_path")
    outcomes = None
    if question.get("outcome") is not None:
        field, trouble = _outcome_field(question.get("outcome"), "counterfactual_path")
        if trouble:
            return trouble
        outcomes = (field,)
    horizons, trouble = _horizons(question.get("horizons"), manifest, "counterfactual_path")
    if trouble:
        return trouble
    try:
        from feature_eng_m5phet import counterfactual as engine
    except ImportError as missing:
        return _refusal(NOT_ESTIMABLE,
                        f"COUNTERFACTUAL_ENGINE_NOT_INSTALLED: the counterfactual path is computed by "
                        f"`feature_eng_m5phet.counterfactual`, the module that owns that arithmetic, and this "
                        f"interpreter does not hold it ({missing}). Nothing is approximated in its place",
                        "counterfactual_path")
    documents = manifest.get("documents") or {}
    rows_declared = documents.get("rows") or {}
    rows_path = Path(rows_declared.get("path") or "")
    if not rows_path.is_file():
        return _refusal(NOT_ESTIMABLE, f"ROWS_NOT_READABLE: the event rows this study was registered with are not on "
                                       f"disk at the path the manifest names, and a counterfactual is evaluated over "
                                       f"the rows or not at all", "counterfactual_path")
    if rows_declared.get("sha256") and digest_of(rows_path) != rows_declared["sha256"]:
        return _refusal(NOT_ESTIMABLE, f"ROWS_DIGEST_MISMATCH: the rows file is not the one this study was registered "
                                       f"from, so a path evaluated over it would be another study's path",
                        "counterfactual_path")
    try:
        computed = engine.paths((documents.get("projections") or {}).get("path"), str(rows_path),
                                window=tuple(window), zero_out=zero_out, outcomes=outcomes, horizons=horizons)
    except Exception as trouble:                                                             # noqa: BLE001
        code = getattr(trouble, "code", None) or type(trouble).__name__
        return _refusal(NOT_ESTIMABLE, f"{code}: {getattr(trouble, 'why', trouble)}", "counterfactual_path")
    rows = []
    for entry in computed.get("paths") or []:
        row = {"event": entry["event_type"], "published_at": entry["published_at"],
               "horizon_minutes": entry["horizon_minutes"], "outcome": entry["outcome"],
               "zeroed": entry.get("zeroed"), "status": entry.get("status"),
               "refusal": entry.get("refusal"), "why": entry.get("why")}
        if entry.get("status") == "OK":
            row.update({"predicted_observed": entry["predicted_observed"],
                        "predicted_counterfactual": entry["predicted_counterfactual"],
                        "attributed_transient": entry["attributed_transient"],
                        "confidence_interval": entry["attributed_transient_ci_95"],
                        "observed_outcome": entry["observed_outcome"],
                        "n": entry.get("n_fit_events")})
        rows.append(row)
    answer = {
        "type": "counterfactual_path", "status": "OK",
        "estimand": "counterfactual_path",
        "label": computed["label"],
        "zero_out": zero_out, "window": list(window), "outcome": question.get("outcome"),
        "conclusion": _counterfactual_conclusion(computed, zero_out, manifest),
        "table": rows[:MAX_PATH_ROWS],
        "rows_not_shown": max(0, len(rows) - MAX_PATH_ROWS),
        "label_reading": computed["label_reading"],
        "window_summary": deepcopy(computed["window"]),
        "horizons_minutes": computed["horizons_minutes"],
        "interval_method": computed["interval_method"], "interval_caveat": computed["interval_caveat"],
        "flags": list(computed.get("flags") or []),
        "superposition": deepcopy(computed.get("superposition")),
        "counters": deepcopy(computed.get("counters")),
        "confidence_level": 0.95,
        "identification_caveat": manifest.get("identification_caveat"),
        "reading": computed["reading"],
        **_provenance_tail(manifest),
    }
    # the identification block of the computed document is the study's own, and it is checked rather than assumed
    answer["identification"] = dict(identification_block(manifest), computed_verdict=computed.get("identification"))
    return answer


def _counterfactual_conclusion(computed, zero_out, manifest):
    answered = [row for row in computed.get("paths") or [] if row.get("status") == "OK"]
    verdict = (manifest.get("identification") or {}).get("verdict")
    label = computed["label"]
    if not answered:
        return (f"{label}: no cell of this window could be evaluated; {computed['counters']['refused']} were refused "
                f"by name and none of them is a zero.")
    largest = max(answered, key=lambda row: abs(row["attributed_transient"]))
    low, high = largest["attributed_transient_ci_95"]
    return (f"{label}: over {computed['window']['start']} to {computed['window']['end']}, with the "
            f"{computed['window']['releases_of_the_zeroed_type_in_window']} release(s) of {zero_out} set to zero, "
            f"the largest transient the fitted model attributes to it is {largest['attributed_transient']:.4g} at "
            f"{largest['horizon_minutes']} minutes after {largest['published_at']} (interval [{low:.4g}, {high:.4g}]). "
            f"No world in which that release surprised nobody was observed; the study's identification verdict is "
            f"{verdict}.")


ANSWERS = {"impulse_response": impulse_response_answer,
           "sensitivity": sensitivity_answer,
           "counterfactual_path": counterfactual_answer}


def answer(manifest, question_type, question):
    """One event-study answer, or the typed refusal that says why there is none. The document is read here, once."""
    builder = ANSWERS.get(question_type)
    if builder is None:
        return _refusal(NOT_ESTIMABLE, f"an event study answers {sorted(ANSWERS)} and was asked {question_type!r}",
                        question_type)
    try:
        document = projections_of(manifest)
    except EventStudyError as trouble:
        return _refusal(NOT_ESTIMABLE, str(trouble), question_type)
    return builder(manifest, document, question)
