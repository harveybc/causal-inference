"""Named, typed questions about a retained causal study, answered from its artifact and from nothing else.

This is the causal side of the m5phet question envelope. A caller describes the STATE it is asking about -- a fitted
study by reference, or the population and causal graph that identify one -- and asks named questions of declared types.
Every question is answered on its own, and a refusal is typed and carries its reason.

The engine estimates a treatment effect, fitted explicitly beforehand: constant across units, or varying with one
binary effect modifier the study declared. So `ate` is read from the retained artifact once the caller's graph is
verified to be the study's own graph, and `cate` is answered only from a subgroup effect the artifact ALREADY CARRIES --
computed when the study was fitted, for a level of a declared modifier. A subgroup a study does not carry is refused by
name: not left undeclared, which would say only "unknown type"; not answered from the average, which would put a number
where the study has none; and not obtained by subsetting the population and refitting, which no question does here.
Nothing in this file fits, subsets, or derives a statistic the artifact does not carry.

Since WP22 a second kind of study is retained beside those: an **event study** -- local projections of a price series
on the standardized surprises of dated calendar releases, registered from a `m5phet.event_projections.v1` document.
It answers `impulse_response`, `sensitivity` and `counterfactual_path`, and it answers no `ate` and no `cate`,
because it has no treatment arms, no population of units and no contrast. A treatment-effect study answers the
mirror image: no horizon, no release, no surprise, so none of the three. Both directions are refused `NOT_ESTIMABLE`
by name, with the kind of study that would answer stated, and neither study ever stands in for the other. The
answers themselves are built in `event_study.py`; this file routes to them and owns the refusals.
"""

from copy import deepcopy
from datetime import datetime, timezone
import re

from . import event_study as _event

# Refusal codes shared with m5phet.questions. They are repeated here by value so serving keeps no dependency on the
# workbench package; a caller matches on the strings, and the strings are fixed there.
NOT_ESTIMABLE = "NOT_ESTIMABLE"
STATE_REQUIRED = "STATE_REQUIRED"
MALFORMED_QUESTION = "MALFORMED_QUESTION"

GRAPH_FIELDS = ("treatment", "outcome", "confounders")

#: The two types a treatment-effect study answers. An event study answers neither, and says so by name.
EFFECT_QUESTION_TYPES = {
    # An average effect carries no field of its own: the study named by the state is the whole question.
    "ate": {"required": [], "optional": []},
    # A conditional effect names the subgroup it is about. Both spellings are accepted and mean the same thing:
    # `subgroup` is the word this envelope uses, `condition` the one callers wrote first. Neither is required at this
    # layer, so a cate question with no subgroup at all reaches the provider and is refused by it with the forms the
    # named study actually carries -- which is more than "a field is missing" can say.
    "cate": {"required": [], "optional": ["subgroup", "condition"]},
}

#: Every type this provider declares: the two above, plus the three an EVENT study answers (WP22). All five stay
#: declared whatever is retained, because a declared type reaches the provider and comes back with the reason the
#: study it was asked of cannot answer it, where an undeclared one would come back only as "unknown type".
QUESTION_TYPES = {**EFFECT_QUESTION_TYPES, **_event.QUESTION_TYPES}

EVENT_QUESTION_TYPES = tuple(_event.QUESTION_TYPES)

SUBGROUP_FIELDS = ("subgroup", "condition")
#: The only subgroup form a study can answer: a declared effect modifier at one of its two declared levels. Anything
#: else -- a cut of a continuous variable, an inequality, a conjunction -- is a different estimand, and a different
#: estimand is a different study.
SUBGROUP_FORM = "<effect modifier> == 0 or <effect modifier> == 1"
SUBGROUP_PATTERN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|=|!=|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*")

NO_P_VALUE = ("the retained artifact carries an estimate and an HC1 normal interval but no p-value; one is not derived "
              "from the interval here, because that would be a number the study never reported")


def refusal(kind, why, question_type):
    """The same shape as m5phet.questions.refusal: no number, and a reason a caller can read."""
    return {"status": "REFUSED", "refusal": kind, "why": why, "type": question_type}


def study_graph(config):
    """The causal graph a retained study was fitted with, in the caller's vocabulary.

    A declared effect modifier belongs in the confounder set: the estimator conditions on it exactly as it conditions on
    an adjustment, and leaving it out would show a person a graph narrower than the one their number came from."""
    return {"treatment": config["treatment"], "outcome": config["outcome"],
            "confounders": sorted([*config["adjustments"], *(config.get("effect_modifiers") or [])])}


def effect_modifiers(config):
    """The effect modifiers a study was fitted with; empty for a constant-effect study."""
    return list(config.get("effect_modifiers") or [])


def graph_differences(graph, config):
    """Every way the caller's graph departs from the study's own, each named; empty when they agree."""
    theirs = {"treatment": graph.get("treatment"), "outcome": graph.get("outcome"),
              "confounders": graph.get("confounders")}
    ours = study_graph(config)
    if isinstance(theirs["confounders"], (list, tuple)) and all(isinstance(c, str) for c in theirs["confounders"]):
        theirs["confounders"] = sorted(set(theirs["confounders"]))
    return [f"{field}: asked {theirs[field]!r}, fitted {ours[field]!r}"
            for field in GRAPH_FIELDS if theirs[field] != ours[field]]


def _names_population(dataset_id, body):
    """Whether an opaque dataset identifier names this study's fitted population or the study itself.

    A study knows its population by digest only; a caller who names it any other way is asking about something this
    provider cannot tell apart from a different dataset."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return False
    given = dataset_id.strip()
    if given.startswith("sha256:"):
        given = given[len("sha256:"):]
    return given in (body["population"]["data_sha256"], body["state_ref"], body["digest"])


def conclusion(config, payload):
    """A sentence about the sign of the estimate and whether its interval excludes zero, read off the numbers.

    It is only as good as the assumptions the study declared, which is why every answer also carries them."""
    low, high = payload["interval"]
    estimate = payload["estimate"]
    level = payload["diagnostics"]["confidence_level"]
    sign = "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
    excludes = low > 0 or high < 0
    verdict = "excludes zero" if excludes else "includes zero, so the sign is not established"
    return (f"The estimated average effect of {config['treatment']} on {config['outcome']} is {sign}: "
            f"{estimate:.4g} {payload['unit']} (contrast {payload['diagnostics']['contrast']['treated']} versus "
            f"{payload['diagnostics']['contrast']['control']}); the {level:.0%} interval "
            f"[{low:.4g}, {high:.4g}] {verdict}. This holds only under the declared assumptions: "
            + ", ".join(payload["assumptions"]) + ".")


def ate_answer(body):
    """The retained study's average effect, every value copied from the artifact.

    A study fitted with an effect modifier still has an average effect -- the mean of its conditional effects over the
    fitted population -- so this answers from it too, and says which estimand the study itself was fitted for."""
    payload = body["result"]["payload"]
    config = body["config"]
    answer = {
        "type": "ate", "status": "OK", "execution_authorized": False,
        "estimand": "ATE", "study_estimand": payload["estimand"],
        "effect_size": payload["estimate"], "unit": payload["unit"],
        "confidence_interval": list(payload["interval"]),
        "confidence_level": payload["diagnostics"]["confidence_level"],
        "conclusion": conclusion(config, payload),
        "assumptions": list(payload["assumptions"]),
        "causal_graph": study_graph(config),
        "effect_modifiers": effect_modifiers(config),
        "population": deepcopy(body["population"]),
        "diagnostics": deepcopy(payload["diagnostics"]),
        "state_ref": body["state_ref"],
        "development": body.get("development") is True,
    }
    if body.get("identification_caveat"):
        answer["identification_caveat"] = body["identification_caveat"]
    if "p_value" in payload:
        answer["p_value"] = payload["p_value"]
    else:
        answer["not_carried"] = {"p_value": NO_P_VALUE}
    return answer


def cate_refusal(body, condition):
    """Why a constant-effect study cannot answer a conditional question. Unchanged in substance: a study fitted without
    an effect modifier has no subgroup effect, and inference does not make one."""
    config = body["config"]
    why = (f"the retained study estimates an average effect ({config['estimand']} of {config['treatment']} on "
           f"{config['outcome']}, constant across units) and was not fitted with an effect modifier, so a conditional "
           f"effect under {condition!r} cannot be read from it, and it is not derived by subsetting the population and "
           f"refitting during inference; answering this needs a study fitted explicitly with that modifier declared "
           f"and a heterogeneous-effect estimand -- this provider serves one when it is retained (`python -m "
           f"causal_inference_provider prepare-demo --with-modifier` fits the synthetic development one), and it is "
           f"named as its own study, not read out of this one.")
    return refusal(NOT_ESTIMABLE, why, "cate")


def subgroup_asked(question):
    """The subgroup expression a cate question names, under either declared spelling, or None."""
    for field in SUBGROUP_FIELDS:
        value = question.get(field)
        if value is not None:
            return value
    return None


def carried_subgroups(payload):
    """The conditional effects a study's artifact carries, keyed by the subgroup each is about."""
    return {entry["subgroup"]: entry for entry in (payload.get("conditional_effects") or [])
            if isinstance(entry, dict) and isinstance(entry.get("subgroup"), str)}


def subgroup_conclusion(config, payload, entry):
    """A sentence about one subgroup's effect, read off the numbers the artifact carries for it."""
    low, high = entry["interval"]
    estimate = entry["estimate"]
    level = payload["diagnostics"]["confidence_level"]
    sign = "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
    verdict = "excludes zero" if low > 0 or high < 0 else "includes zero, so the sign is not established"
    return (f"Among the {entry['n_subgroup']} units with {entry['subgroup']}, the estimated effect of "
            f"{config['treatment']} on {config['outcome']} is {sign}: {estimate:.4g} {payload['unit']} "
            f"(contrast {payload['diagnostics']['contrast']['treated']} versus "
            f"{payload['diagnostics']['contrast']['control']}); the {level:.0%} interval "
            f"[{low:.4g}, {high:.4g}] {verdict}. This holds only under the declared assumptions: "
            + ", ".join(payload["assumptions"]) + ".")


def cate_answer(body, expression):
    """One subgroup effect, copied from the artifact that carries it, or the typed refusal that says why not.

    Four refusals, each about the study and not about the person: the study has no modifier at all; the question names no
    subgroup; the subgroup names a variable this study did not declare as a modifier; the subgroup is a form -- an
    inequality, another level -- the study does not carry. None of them is answered from the average, and none of them
    causes a fit."""
    config, payload = body["config"], body["result"]["payload"]
    declared = effect_modifiers(config)
    if not declared:
        return cate_refusal(body, expression)
    carried = carried_subgroups(payload)
    forms = sorted(carried)
    if not isinstance(expression, str) or not expression.strip():
        return refusal(MALFORMED_QUESTION, f"a conditional-effect question must name its subgroup in `subgroup` (or "
                                           f"`condition`), as {SUBGROUP_FORM}; study {body['state_ref']} carries "
                                           f"{forms}", "cate")
    match = SUBGROUP_PATTERN.fullmatch(expression)
    if match is None:
        return refusal(MALFORMED_QUESTION, f"the subgroup {expression!r} is not a form this provider reads; it reads "
                                           f"{SUBGROUP_FORM}, and study {body['state_ref']} carries {forms}", "cate")
    name, operator, level = match[1], match[2], match[3]
    if name not in declared:
        return refusal(NOT_ESTIMABLE, f"{name!r} is not an effect modifier this study declares; it was fitted with "
                                      f"{declared} and carries {forms}. A subgroup of a variable a study was not fitted "
                                      f"with cannot be read from it, and it is not obtained by subsetting the "
                                      f"population and refitting during inference; that is a new study, fitted "
                                      f"explicitly with {name!r} declared as a modifier", "cate")
    asked = f"{name} == {int(float(level))}" if operator == "==" and float(level).is_integer() else None
    if asked is None or asked not in carried:
        return refusal(NOT_ESTIMABLE, f"study {body['state_ref']} carries the conditional effects {forms} and the "
                                      f"subgroup {expression!r} is not one of them; a level or a comparison the study "
                                      f"was not fitted for is a different estimand, and this provider does not compute "
                                      f"one during inference", "cate")
    entry = carried[asked]
    answer = {
        "type": "cate", "status": "OK", "execution_authorized": False,
        "estimand": "CATE", "study_estimand": payload["estimand"],
        "subgroup": asked, "modifier": entry["modifier"], "level": entry["level"],
        "effect_size": entry["estimate"], "unit": payload["unit"],
        "confidence_interval": list(entry["interval"]),
        "confidence_level": payload["diagnostics"]["confidence_level"],
        "n_subgroup": entry["n_subgroup"], "n_treated": entry["n_treated"], "n_control": entry["n_control"],
        "conclusion": subgroup_conclusion(config, payload, entry),
        "assumptions": list(payload["assumptions"]),
        "causal_graph": study_graph(config),
        "effect_modifiers": declared,
        "population": deepcopy(body["population"]),
        "diagnostics": deepcopy(payload["diagnostics"]),
        "state_ref": body["state_ref"],
        "development": body.get("development") is True,
    }
    if body.get("identification_caveat"):
        answer["identification_caveat"] = body["identification_caveat"]
    answer["not_carried"] = {"p_value": NO_P_VALUE}
    return answer


def resolve_study(provider, state, data, as_of):
    """The one retained study the state names, or the typed refusal every question must carry.

    Returns (body, None) or (None, (kind, why)). A study that happens to be here never stands in for the one asked
    about: a graph naming another treatment, outcome or adjustment set is refused with the difference spelled out, and a
    name nobody retains is refused with the names that are retained.

    Three ways to name a study, in this order of authority: `state_ref` (its digest, which is exact), `study` (the name
    the provider's own slot declares -- a study identifier or its estimand and roles -- which is what a router resolves a
    sentence to), and `causal_graph` (the roles it was fitted with, which is what a person writing an envelope by hand
    knows). A `causal_graph` given alongside either of the others is still checked against the study they named."""
    studies = list(provider._retained_studies().values())
    if not studies:
        return None, (STATE_REQUIRED, "no fitted study is retained by this provider; explicitly fit one "
                                      "(`python -m causal_inference_provider fit` or `prepare-demo`) before asking. "
                                      "The event studies it retains, which answer other question types, are "
                                      + str(sorted(event_studies_of(provider))))
    if data is not None and not (isinstance(data, (str, list, tuple, dict)) and not data):
        return None, (NOT_ESTIMABLE, "a dataset was attached, but this provider does not fit during inference; "
                                     "questions are answered from studies fitted explicitly beforehand")
    graph = state.get("causal_graph")
    if graph is not None and (not isinstance(graph, dict) or set(graph) - set(GRAPH_FIELDS)):
        return None, (MALFORMED_QUESTION, f"`causal_graph` must be a mapping with {list(GRAPH_FIELDS)}")
    ref = state.get("state_ref")
    named = state.get("study")
    if ref is not None:
        candidates = [body for body in studies if body["state_ref"] == ref]
        if not candidates:
            return None, (STATE_REQUIRED, f"state_ref {ref!r} is not a study this provider retains; it holds "
                                          f"{[body['state_ref'] for body in studies]}")
    elif named is not None:
        candidates = provider.studies_named(named)
        if not candidates:
            return None, (STATE_REQUIRED, f"no retained study is named {named!r}; this provider holds "
                                          f"{provider.study_names()} and the event studies "
                                          f"{sorted(event_studies_of(provider))}, and the study that is here is not "
                                          f"an answer to a question about another one")
    elif graph is not None:
        candidates = [body for body in studies if not graph_differences(graph, body["config"])]
        if not candidates:
            fitted = [study_graph(body["config"]) for body in studies]
            differences = [graph_differences(graph, body["config"]) for body in studies]
            return None, (NOT_ESTIMABLE, f"no retained study was fitted with this causal graph; the retained graphs "
                                         f"are {fitted} and the differences are {differences}; a study for this "
                                         f"graph must be fitted explicitly before its effect can be reported")
    else:
        return None, (STATE_REQUIRED, "the state must name a study: `state_ref` (causal-ate:<digest>), `study` (one of "
                                      + str(provider.study_names()) + ") or `causal_graph` with treatment, outcome and "
                                      "confounders; this provider holds "
                                      + str([body["state_ref"] for body in studies]))
    dataset_id = state.get("dataset_id")
    if dataset_id is not None:
        candidates = [body for body in candidates if _names_population(dataset_id, body)]
        if not candidates:
            return None, (NOT_ESTIMABLE, f"dataset_id {dataset_id!r} does not name the fitted population of any "
                                         f"study this provider retains; a study knows its population by its data "
                                         f"sha256 only, so a dataset named otherwise is a different dataset, not fitted")
    if len(candidates) > 1:
        return None, (STATE_REQUIRED, "more than one retained study matches this state; name one by `state_ref`: "
                                      + str([body["state_ref"] for body in candidates]))
    body, = candidates
    if graph is not None:
        differences = graph_differences(graph, body["config"])
        if differences:
            return None, (NOT_ESTIMABLE, f"the causal graph asked about is not the graph study {body['state_ref']} "
                                         f"was fitted with: " + "; ".join(differences))
    if as_of is not None:
        try:
            asked = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None, (MALFORMED_QUESTION, "`as_of` must be an ISO-8601 clock")
        if asked.tzinfo is None:
            asked = asked.replace(tzinfo=timezone.utc)
        available = datetime.fromisoformat(body["available_at"])
        if asked < available:
            return None, (NOT_ESTIMABLE, f"study {body['state_ref']} became available at {body['available_at']}, "
                                         f"after the requested clock {as_of}")
    return body, None


def event_studies_of(provider):
    """Every event study the provider retains, or an empty mapping when it declares none (an older adapter)."""
    method = getattr(provider, "event_studies", None)
    return method() if callable(method) else {}


def resolve_event_study(provider, state):
    """The event study the state names, when it names one.

    Returns `(manifest, None)` when it does, `(None, trouble)` when it names one nobody retains, and `(None, None)`
    when it names no event study at all -- which is how a state about a treatment-effect study passes through here
    untouched. An event study is named the same three ways a treatment-effect study is, minus the causal graph: a
    graph has a treatment and an outcome, and an event study has releases and horizons instead."""
    studies = event_studies_of(provider)
    ref = state.get("state_ref")
    if isinstance(ref, str) and ref.startswith(_event.REF_PREFIX):
        for manifest in studies.values():
            if manifest["state_ref"] == ref:
                return manifest, None
        return None, (STATE_REQUIRED, f"state_ref {ref!r} is not an event study this provider retains; it holds "
                                      f"{[manifest['state_ref'] for manifest in studies.values()]}")
    named = state.get("study")
    if isinstance(named, str) and named.strip():
        wanted = named.strip().casefold()
        found = [manifest for label, manifest in studies.items()
                 if wanted in {label.casefold(), str(manifest.get("study_id") or "").casefold(),
                               manifest["state_ref"].casefold(), manifest["digest"].casefold()}]
        if len(found) > 1:
            return None, (STATE_REQUIRED, "more than one retained event study answers to that name; name one by "
                                          "`state_ref`: " + str([manifest["state_ref"] for manifest in found]))
        if found:
            return found[0], None
    return None, None


def event_study_refusal(manifest, kind):
    """Why an event study answers no `ate` and no `cate`. It is about the study, not about the person asking."""
    return refusal(NOT_ESTIMABLE,
                   f"study {manifest.get('study_id') or manifest['state_ref']!r} is an EVENT STUDY: local "
                   f"projections of a price series on the standardized surprises of dated calendar releases. It has "
                   f"no treatment arms, no population of units and no contrast, so it carries no {kind} and one is "
                   f"not derived from it during inference; it answers {sorted(_event.QUESTION_TYPES)}. An {kind} "
                   f"needs a study fitted explicitly with a binary treatment, and that is a different study",
                   kind)


def effect_study_refusal(body, kind):
    """Why a treatment-effect study answers none of the three event-study types."""
    config = body["config"]
    return refusal(NOT_ESTIMABLE,
                   f"study {body['state_ref']} estimates {config['estimand']} of {config['treatment']} on "
                   f"{config['outcome']} over a population of units; it has no calendar release, no horizon and no "
                   f"surprise, so {kind!r} cannot be read from it and it is not estimated during inference. That "
                   f"question needs a study of kind {_event.KIND!r}, registered with "
                   f"`python -m causal_inference_provider register-event-study --projections ... --rows ... --id ...` "
                   f"from a {_event.PROJECTIONS_SCHEMA} document",
                   kind)


def answer_questions(provider, state, questions, data, as_of):
    """Answer each declared question from the one study the state names, or refuse each by name.

    Two kinds of study are retained here and a state names exactly one of them. An event study is looked for first,
    because it is named by an identifier the treatment-effect resolver would not recognise; when the state names one,
    every question is answered from it and `ate`/`cate` are refused by name. Otherwise the treatment-effect resolver
    runs unchanged, and the three event-study types are refused by name there."""
    if not isinstance(state, dict):
        state = {}
    manifest, trouble = resolve_event_study(provider, state)
    if trouble is not None:
        kind, why = trouble
        return {name: refusal(kind, why, question["type"]) for name, question in questions.items()}
    if manifest is not None:
        if data is not None and not (isinstance(data, (str, list, tuple, dict)) and not data):
            why = ("a dataset was attached, but this provider does not fit during inference; an event study answers "
                   "from the projections document it was registered from")
            return {name: refusal(NOT_ESTIMABLE, why, question["type"]) for name, question in questions.items()}
        out = {"__state_ref__": manifest["state_ref"]}
        for name, question in questions.items():
            kind = question["type"]
            out[name] = (event_study_refusal(manifest, kind) if kind in EFFECT_QUESTION_TYPES
                         else _event.answer(manifest, kind, question))
        return out
    body, trouble = resolve_study(provider, state, data, as_of)
    if body is None:
        kind, why = trouble
        return {name: refusal(kind, why, question["type"]) for name, question in questions.items()}
    out = {"__state_ref__": body["state_ref"]}
    for name, question in questions.items():
        kind = question["type"]
        if kind == "ate":
            out[name] = ate_answer(body)
        elif kind == "cate":
            out[name] = cate_answer(body, subgroup_asked(question))
        elif kind in EVENT_QUESTION_TYPES:
            out[name] = effect_study_refusal(body, kind)
        else:
            out[name] = refusal(NOT_ESTIMABLE, f"this provider declares {sorted(QUESTION_TYPES)} and cannot answer "
                                               f"{kind!r}", kind)
    return out
