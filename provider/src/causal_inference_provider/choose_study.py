"""WP20: the study a dataset and a problem sentence get, chosen one declared option at a time.

`study_space.py` declares WHAT a study may be configured with and `study_spec.py` says what a valid configuration is.
This module is the third part: it profiles a dataset, asks Laya -- through `m5phet.decide`, so every answer is a
recorded decision bound to a state digest -- to pick one declared option per choice, and composes the spec those
choices describe. It chooses; it never fits, and it never reads a row into an answer.

**What Laya is shown is a description.** A profile: the columns, their dtypes, the row count, and per column a
summary -- missing fraction, how many distinct values, whether it is binary, and for a numeric column its mean, sd,
minimum and maximum, each at `SUMMARY_DECIMALS` declared decimals. No row is ever rendered, and no number appears in
a state text that was not computed into that summary first. `m5phet.decide` refuses a list of more than 128 entries
as `ROWS_IN_STATE`; here the profile never builds one.

**Every option is declared by this repository.** The roles come from `study_space()["roles"]`, the estimators from
`["estimators"]`, the nuisance models from `["nuisance_models"]`, the confidence levels from `["confidence_levels"]`.
Laya is never offered an estimator that cannot serve the estimand the roles imply: a study with no declared modifier
asks for one average effect, and `CausalForestDML` -- fitted on the modifiers, with no ATE to give -- is not on the
list it is shown. That filter reads the space's own `estimands` declaration, so an estimator whose capabilities
change in `study_space.py` changes here with it and nowhere else.

**A composition that does not validate is refused, never repaired.** The choices are composed into a spec and handed
to `study_spec.validate_spec`, whose refusal is returned by its own name -- `TWO_OUTCOMES`, `NO_TREATMENT`,
`ESTIMATOR_HAS_NO_ATE` -- with the decision records kept. That a chooser can produce an impossible study is exactly
what WP20 was written to find out; silently moving a second outcome to `exclude` would hide the finding and publish a
study nobody chose.

**A decision is a hypothesis.** The probabilities in these records are Laya's own uncalibrated head outputs for these
exact wordings. They are not accuracies, they are not evidence that the chosen study is the right one, and nothing
here says the recovered effect will be near the truth. The fit that follows and the table that compares it against
the hand-written spec are the judge.
"""

import csv
import math
from copy import deepcopy
from pathlib import Path

from . import study_space as _space
from . import study_spec as _spec

CHOICE_SCHEMA = "m5phet.causal_study_choice.v1"

#: what these decisions are recorded under, so a record says what kind of thing was being chosen
DECISION_KIND = "causal_study"

#: decimals every float in a profile and in a state text carries. Declared once: the profile is rounded to it, so a
#: number with more precision than this never exists to be rendered.
SUMMARY_DECIMALS = 6

#: the longest question name and option key the classification adapter accepts (`news_signal.question`). A column
#: whose name does not fit is refused by name rather than renamed behind the person's back.
MAX_NAME_CHARACTERS = 64

#: the question names the non-role choices are asked under. Column roles are asked under the column's own name, so a
#: dataset carrying one of these as a column name is refused instead of quietly overwriting a decision.
ESTIMATOR_QUESTION = "estimator"
MODEL_Y_QUESTION = "model_y"
MODEL_T_QUESTION = "model_t"
CONFIDENCE_QUESTION = "confidence_level"
RESERVED_QUESTIONS = (ESTIMATOR_QUESTION, MODEL_Y_QUESTION, MODEL_T_QUESTION, CONFIDENCE_QUESTION)

#: refusals this module gives on its own. Every other refusal it returns belongs to somebody else -- `m5phet.decide`,
#: the classification provider, `study_spec.validate_spec` -- and is passed through under that name.
DATASET_UNREADABLE = "DATASET_UNREADABLE"
COLUMN_NAME_UNUSABLE = "COLUMN_NAME_UNUSABLE"
NO_ESTIMATOR_FOR_ESTIMAND = "NO_ESTIMATOR_FOR_ESTIMAND"
#: one or more columns were left with no role -- because the chooser abstained on them under a declared confidence
#: threshold -- and a study is not composed from a role map with holes in it
ROLES_INCOMPLETE = "ROLES_INCOMPLETE"

#: the refusal `m5phet.decide` gives an answer below the declared threshold. Repeated here so this module can tell an
#: abstention (a question that was asked and not answered) from every other refusal, which stops the chooser.
LOW_CONFIDENCE_ABSTAINED = "LOW_CONFIDENCE_ABSTAINED"

ROLE_INSTRUCTIONS = ("Which role does the column {column!r} play in this causal study? The problem statement and this "
                     "column's summary are above, with the roles already given to the other columns. Choose exactly "
                     "one role.")
ESTIMATOR_INSTRUCTIONS = ("Which estimator should fit this study? Only estimators that report the estimand these "
                          "roles imply, together with an interval, are offered. Choose exactly one.")
MODEL_Y_INSTRUCTIONS = ("Which model should fit the outcome regression (model_y) of this study -- the nuisance model "
                        "that predicts the outcome from the adjusted variables? Choose exactly one.")
MODEL_T_INSTRUCTIONS = ("Which model should fit the treatment classification (model_t) of this study -- the nuisance "
                        "model that predicts the binary treatment from the adjusted variables? Choose exactly one.")
CONFIDENCE_INSTRUCTIONS = ("At which confidence level should this study report its intervals? Choose exactly one.")


class ChoiceRefused(ValueError):
    """A choice that will not be composed, with the name of the refusal and the reason in the same object."""

    def __init__(self, refusal, why):
        super().__init__(f"{refusal}: {why}")
        self.refusal = refusal
        self.why = why


# --- the profile ---------------------------------------------------------------------------------------------------

def _round(value, decimals):
    """A float at the declared decimals, or None when it is not a finite number. Nothing keeps more precision."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, decimals) if math.isfinite(number) else None


def read_frame(path):
    """Read the CSV whose profile is being taken. Read in full, because the row count is part of the description."""
    location = Path(str(path)).expanduser()
    try:
        import pandas as pd
        with location.open(encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), [])
            if not header or len(set(header)) != len(header) or any(not name.strip() for name in header):
                raise ChoiceRefused(DATASET_UNREADABLE,
                                    f"{location} needs unique, non-empty column labels on its first line.")
            source.seek(0)
            frame = pd.read_csv(source, index_col=False)
    except ChoiceRefused:
        raise
    except (OSError, ValueError) as trouble:
        raise ChoiceRefused(DATASET_UNREADABLE, f"{location} cannot be read as a CSV table: {trouble}") from None
    if frame.empty or not len(frame.columns):
        raise ChoiceRefused(DATASET_UNREADABLE, f"{location} carries no rows; a study is not chosen from an empty table.")
    return frame


def column_summary(series, *, decimals=SUMMARY_DECIMALS):
    """One column described: dtype, missing fraction, distinct count, binary, and the numeric moments when numeric.

    Every number here is a summary of the whole column, rounded to `decimals`. No value of a row is reported except
    the minimum and the maximum, which are the extremes a summary is made of and which this module declares it shows.
    """
    import pandas as pd
    rows = int(len(series))
    present = series.dropna()
    distinct = int(present.nunique())
    numeric = bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))
    summary = {
        "dtype": str(series.dtype),
        "numeric": numeric,
        "missing_fraction": _round((rows - int(len(present))) / rows if rows else 0.0, decimals),
        "distinct_values": distinct,
        "binary": distinct == 2,
    }
    if numeric and len(present):
        summary["mean"] = _round(present.mean(), decimals)
        summary["std"] = _round(present.std(ddof=1), decimals) if len(present) > 1 else None
        summary["min"] = _round(present.min(), decimals)
        summary["max"] = _round(present.max(), decimals)
        summary["coded_0_1"] = bool(distinct == 2 and set(float(v) for v in present.unique()) == {0.0, 1.0})
    return summary


def profile_dataset(path, *, decimals=SUMMARY_DECIMALS):
    """The dataset as a description: its columns, their dtypes, its row count, and one summary per column.

    The rows are read to compute the summaries and are then dropped. This mapping is the only thing that reaches a
    state text, so it is also the whole of what Laya ever sees of the data."""
    frame = read_frame(path)
    columns = [str(name) for name in frame.columns]
    return {
        "path": str(Path(str(path)).expanduser()),
        "rows": int(len(frame)),
        "columns": columns,
        "summary_decimals": decimals,
        "columns_detail": {name: column_summary(frame[original], decimals=decimals)
                           for name, original in zip(columns, frame.columns)},
    }


# --- the option sets, each read from the declared space ---------------------------------------------------------------

def estimand_for(roles):
    """The estimand the roles imply. Not chosen: a declared modifier IS the request for a conditional effect."""
    return _space.CATE if any(role == "modifier" for role in roles.values()) else _space.ATE


def estimator_options(estimand, space):
    """The estimators offered for this estimand: exactly those the space declares can report it WITH an interval.

    This is where `CausalForestDML` leaves the list of an average-effect study. It is fitted on the modifiers and the
    space says so (`"estimands": ("CATE",)`), so offering it to a study that declared no modifier would be offering a
    choice the fit would then refuse -- and would invite a decision record for a study that cannot exist."""
    detail = space["estimators"]["detail"] or {}
    return [[key, label] for key, label in space["estimators"]["options"]
            if estimand in (detail.get(key) or {}).get("estimands", ())]


def role_options(space):
    return [[key, label] for key, label in space["roles"]["options"]]


def nuisance_options(space):
    return [[key, label] for key, label in space["nuisance_models"]["options"]]


def confidence_options(space):
    return [[key, label] for key, label in space["confidence_levels"]["options"]]


# --- asking ------------------------------------------------------------------------------------------------------------

def _usable_name(name):
    return bool(name) and name == name.strip() and 0 < len(name) <= MAX_NAME_CHARACTERS


def _ask_one(decider, *, kind, state, questions, as_of, record_dir, gate=None):
    """One `decide.ask`, returning `{name: entry}`. The import is local so this module loads where m5phet does not.

    The fit interpreter holds EconML and no m5phet; the chat interpreter holds m5phet and no EconML. Profiling and
    composing must work in both, so `m5phet.decide` is imported only on the path that actually asks.

    `gate` carries the declared abstention threshold and the report it is cited from, or is `None` when no threshold
    was declared. It is passed through unchanged: this module never decides what a confidence is worth."""
    from m5phet import decide
    gate = gate or {}
    return decide.ask(decider, state, questions, kind=kind, as_of=as_of, record_dir=record_dir,
                      min_confidence=gate.get("min_confidence"), abstention_source=gate.get("abstention_source"))


def _decision_of(entry, question):
    """The decision an entry carries, or a `ChoiceRefused` under the refusal's own name."""
    if entry.get("status") != "OK":
        raise ChoiceRefused(entry.get("refusal") or "PROVIDER_ERROR",
                            f"the decision {question!r} was not made: {entry.get('why')}")
    return entry


def _record(made, entry, question):
    """Keep one made decision -- or one abstention -- with its digest, in the order it happened.

    An abstention is kept here exactly like a choice, with `chosen: null` and the threshold it fell below, because a
    question the checkpoint could not answer is a finding about the checkpoint and the run has to show it."""
    from m5phet import decide
    decision = entry["decision"]
    made.append({"question": question,
                 "kind": decision["kind"],
                 "chosen": decision["chosen"],
                 "chosen_by": decision.get("chosen_by", "LAYA"),
                 "abstention": deepcopy(decision.get("abstention")),
                 "options": deepcopy(decision["options"]),
                 "probabilities": deepcopy(decision["probabilities"]),
                 "probability_decimals": decision["probability_decimals"],
                 "state_sha256": decision["state_sha256"],
                 "checkpoint": decision["checkpoint"],
                 "as_of": decision["as_of"],
                 "digest": decide.decision_sha256(decision),
                 "record_path": entry.get("record_path")})
    return decision["chosen"]


def _state(payload):
    from m5phet import decide
    return decide.decision_state(DECISION_KIND, payload, decimals=SUMMARY_DECIMALS)


def choose_study(decider, dataset, problem, *, space=None, as_of=None, record_dir=None, study_id=None,
                 provenance="UNDECLARED", outcome_unit=None, min_confidence=None, abstention_source=None):
    """Profile the dataset, ask Laya for every declared choice, and compose the spec those choices describe.

    `decider` is an `m5phet.web.engine.Engine` -- the worker route, the real checkpoint -- or a bare `Registry`.
    Returns an `m5phet.causal_study_choice.v1` document: the problem, the profile, every decision that was made with
    its option set and its own uncalibrated probabilities, and either the composed spec (`status: OK`) or the refusal
    that stopped it (`status: REFUSED`), by the name whoever raised it gave it. The decisions made before a refusal
    are kept and returned, because they are what the refusal is about.

    `min_confidence` declares the abstention threshold and `abstention_source` the report that MEASURED this
    checkpoint at it (WP20; `m5phet.decide.abstention_threshold` checks the number against that report's own bins and
    refuses one nobody measured before a single question is asked). Under a threshold a role question that abstains
    leaves its column **unassigned**, and an unassigned column is not silently excluded: the composition refuses
    `ROLES_INCOMPLETE` and names the columns. Giving an abstained column `exclude` would be this module choosing the
    column's role and calling the result Laya's study; the honest output of a chooser that could not choose is a
    refusal that says which questions it could not answer.
    """
    space = space if space is not None else _space.study_space(probe=_space.declared)
    made, abstained = [], []
    document = {"schema": CHOICE_SCHEMA, "kind": DECISION_KIND, "problem": problem,
                "profile": None, "decisions": made, "abstained": abstained, "abstention": None,
                "counts": None, "spec": None, "status": "OK"}
    if not isinstance(problem, str) or not problem.strip():
        return document | {"status": "REFUSED", "refusal": "STATE_REQUIRED",
                           "why": "a study is chosen for a stated problem; none was given."}
    gate = None
    if min_confidence is not None:
        from m5phet import decide
        citation, unresolved = decide.abstention_threshold(min_confidence, abstention_source)
        if unresolved is not None:
            # refused before the dataset is even read: a threshold nobody measured decides nothing here
            return document | {"status": "REFUSED", "refusal": unresolved[0], "why": unresolved[1]}
        gate = {"min_confidence": min_confidence, "abstention_source": abstention_source}
        document["abstention"] = {"min_confidence": citation["min_confidence"], "source": citation}
    try:
        profile = profile_dataset(dataset)
        document["profile"] = profile
        roles = _choose_roles(decider, profile, problem, space, made, as_of, record_dir, gate, abstained)
        _require_complete_roles(profile, roles, abstained)
        estimand = estimand_for(roles)
        estimator = _choose_estimator(decider, profile, problem, roles, estimand, space, made, as_of, record_dir,
                                      gate)
        nuisance = _choose_nuisance(decider, profile, problem, roles, estimand, estimator, space, made, as_of,
                                    record_dir, gate)
        level = _choose_confidence(decider, problem, roles, estimand, estimator, nuisance, space, made, as_of,
                                   record_dir, gate)
        spec = compose_spec(profile, roles, estimator, nuisance, level, made, space=space, study_id=study_id,
                            provenance=provenance, outcome_unit=outcome_unit, abstained=abstained)
    except ChoiceRefused as refused:
        return document | {"counts": _counts(made, abstained), "status": "REFUSED",
                           "refusal": refused.refusal, "why": refused.why}
    except _spec.SpecRefused as refused:
        # the composition the decisions describe is not a study this engine will fit. It is returned by the
        # validator's own name, with every decision kept, and it is not repaired here.
        return document | {"counts": _counts(made, abstained), "status": "REFUSED",
                           "refusal": refused.refusal, "why": refused.why}
    return document | {"counts": _counts(made, abstained), "spec": spec}


def _counts(made, abstained):
    """What the run did, in the three numbers a report needs: asked, answered above the threshold, abstained."""
    return {"questions_asked": len(made),
            "answered": sum(1 for entry in made if entry["chosen"] is not None),
            "abstained": sum(1 for entry in made if entry["chosen"] is None),
            "columns_abstained": list(abstained)}


def _require_complete_roles(profile, roles, abstained=()):
    """Raise `ROLES_INCOMPLETE` when a column of the dataset was left without a role.

    This is the guard the whole threshold rests on. Once a question may abstain, the cheapest thing to do with an
    unanswered column is to drop it -- and dropping it is the role `exclude`, given by this module and then reported
    as part of a study Laya chose. So a hole in the role map stops the composition and names the columns, and the
    person who wants the study anyway writes their role down as a human decision (`m5phet.decide.human_choice`)
    instead of receiving it silently."""
    unassigned = [column for column in profile["columns"] if column not in roles]
    if not unassigned:
        return roles
    abstained = [column for column in abstained if column in unassigned]
    raise ChoiceRefused(ROLES_INCOMPLETE,
                        f"the columns {unassigned} were given no role"
                        + (f"; the chooser abstained on {abstained} because its top probability was below the "
                           f"declared threshold" if abstained else "")
                        + ". A study is not composed from a role map with holes in it, and an unassigned column is "
                          "not silently excluded: `exclude` is a role, and giving it here would be this module "
                          "choosing what the question left open.")


def _choose_roles(decider, profile, problem, space, made, as_of, record_dir, gate=None, abstained=None):
    """One decision per column, in the dataset's own order, each asked with what the earlier columns were given.

    Each column is asked exactly once, under its own name, with the five declared roles. The state carries the
    problem, this column's summary and the roles already assigned -- so the choice for the last column is made
    knowing a treatment has already been named -- and nothing else.

    A column whose answer abstains under the declared threshold is left OUT of the role map and named in `abstained`.
    The run goes on to the next column: an abstention is one question's finding, not a reason to stop asking the
    others, and the count of how many columns the checkpoint could answer is exactly what WP20 set out to measure."""
    options, roles = role_options(space), {}
    abstained = abstained if abstained is not None else []
    for column in profile["columns"]:
        if not _usable_name(column):
            raise ChoiceRefused(COLUMN_NAME_UNUSABLE,
                                f"the column name {column!r} cannot be a question name (non-empty, no surrounding "
                                f"blanks, at most {MAX_NAME_CHARACTERS} characters); rename the column in the data.")
        if column in RESERVED_QUESTIONS:
            raise ChoiceRefused(COLUMN_NAME_UNUSABLE,
                                f"the column name {column!r} is one of the names this chooser asks its own questions "
                                f"under ({list(RESERVED_QUESTIONS)}); rename the column in the data.")
        state = _state({"problem": problem,
                        "dataset": {"rows": profile["rows"], "columns": profile["columns"]},
                        "column": {"name": column, **profile["columns_detail"][column]},
                        "roles_already_assigned": dict(roles)})
        asked = _ask_one(decider, kind=DECISION_KIND, as_of=as_of, record_dir=record_dir, state=state, gate=gate,
                         questions={column: {"options": options,
                                             "instructions": ROLE_INSTRUCTIONS.format(column=column)}})
        entry = asked[column]
        if entry.get("status") != "OK" and entry.get("refusal") == LOW_CONFIDENCE_ABSTAINED:
            _record(made, entry, column)            # `chosen: null`, with the threshold it fell below
            abstained.append(column)
            continue
        roles[column] = _record(made, _decision_of(entry, column), column)
    return roles


def _choose_estimator(decider, profile, problem, roles, estimand, space, made, as_of, record_dir, gate=None):
    options = estimator_options(estimand, space)
    if len(options) < 2:
        raise ChoiceRefused(NO_ESTIMATOR_FOR_ESTIMAND,
                            f"these roles ask for a {estimand} and this space offers "
                            f"{[key for key, _ in options]} for it; a decision needs at least two candidates, and "
                            f"which estimator a study is fitted with is not decided here by default.")
    state = _state({"problem": problem,
                    "dataset": {"rows": profile["rows"], "columns": profile["columns"]},
                    "roles": dict(roles),
                    "estimand": estimand,
                    "modifier_declared": estimand == _space.CATE})
    asked = _ask_one(decider, kind=DECISION_KIND, as_of=as_of, record_dir=record_dir, state=state, gate=gate,
                     questions={ESTIMATOR_QUESTION: {"options": options, "instructions": ESTIMATOR_INSTRUCTIONS}})
    return _record(made, _decision_of(asked[ESTIMATOR_QUESTION], ESTIMATOR_QUESTION), ESTIMATOR_QUESTION)


def _choose_nuisance(decider, profile, problem, roles, estimand, estimator, space, made, as_of, record_dir,
                     gate=None):
    """Both nuisance models, asked in one envelope from the same state: they are two roles of one declared list."""
    options = nuisance_options(space)
    state = _state({"problem": problem,
                    "dataset": {"rows": profile["rows"], "columns": profile["columns"]},
                    "roles": dict(roles), "estimand": estimand, "estimator": estimator})
    asked = _ask_one(decider, kind=DECISION_KIND, as_of=as_of, record_dir=record_dir, state=state, gate=gate,
                     questions={MODEL_Y_QUESTION: {"options": options, "instructions": MODEL_Y_INSTRUCTIONS},
                                MODEL_T_QUESTION: {"options": options, "instructions": MODEL_T_INSTRUCTIONS}})
    return {role: _record(made, _decision_of(asked[role], role), role)
            for role in (MODEL_Y_QUESTION, MODEL_T_QUESTION)}


def _choose_confidence(decider, problem, roles, estimand, estimator, nuisance, space, made, as_of, record_dir,
                       gate=None):
    state = _state({"problem": problem, "roles": dict(roles), "estimand": estimand, "estimator": estimator,
                    "nuisance": dict(nuisance)})
    asked = _ask_one(decider, kind=DECISION_KIND, as_of=as_of, record_dir=record_dir, state=state, gate=gate,
                     questions={CONFIDENCE_QUESTION: {"options": confidence_options(space),
                                                      "instructions": CONFIDENCE_INSTRUCTIONS}})
    return _record(made, _decision_of(asked[CONFIDENCE_QUESTION], CONFIDENCE_QUESTION), CONFIDENCE_QUESTION)


# --- composing -----------------------------------------------------------------------------------------------------

def compose_spec(profile, roles, estimator, nuisance, confidence_level, decisions, *, space=None, study_id=None,
                 provenance="UNDECLARED", outcome_unit=None, abstained=()):
    """The spec these choices describe, validated. Raises `study_spec.SpecRefused` when they describe no study.

    The identifying assumptions are NOT chosen: they are the ones this space declares a study of this estimand must
    state, copied in the space's own words. A chooser that could also decide which assumptions to claim would be
    choosing what the study is allowed to assume about the world, which is the person's claim to make and the
    validator's to check.

    A role map that does not cover every column of the profile raises `ROLES_INCOMPLETE` here, before anything is
    composed -- so the refusal belongs to the composition, not to a later validator that would have seen only a
    shorter column list and never known a question went unanswered."""
    _require_complete_roles(profile, roles, abstained)
    space = space if space is not None else _space.study_space(probe=_space.declared)
    estimand = estimand_for(roles)
    spec = {
        "schema": _spec.SPEC_SCHEMA,
        "dataset": {"path": profile["path"], "columns": list(profile["columns"])},
        "roles": dict(roles),
        "estimator": estimator,
        "nuisance": {"model_y": nuisance["model_y"], "model_t": nuisance["model_t"]},
        "confidence_level": confidence_level,
        "identification": list(space["identification_assumptions"]["required_for"][estimand]),
        "decisions": [entry["digest"] for entry in decisions],
        "provenance": provenance,
    }
    if study_id is not None:
        spec["study_id"] = study_id
    if outcome_unit is not None:
        spec["outcome_unit"] = outcome_unit
    return _spec.validate_spec(spec, space)
