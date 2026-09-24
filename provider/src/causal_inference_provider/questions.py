"""Named, typed questions about a retained causal study, answered from its artifact and from nothing else.

This is the causal side of the m5phet question envelope. A caller describes the STATE it is asking about -- a fitted
study by reference, or the population and causal graph that identify one -- and asks named questions of declared types.
Every question is answered on its own, and a refusal is typed and carries its reason.

The engine estimates one thing: a constant, average treatment effect, fitted explicitly beforehand. So `ate` is read
from the retained artifact once the caller's graph is verified to be the study's own graph, and `cate` is declared so
that it can be refused precisely -- not left undeclared, which would say only "unknown type", and not answered from the
average, which would put a number where the study has none. Nothing here fits, subsets, or derives a statistic the
artifact does not carry.
"""

from copy import deepcopy
from datetime import datetime, timezone

# Refusal codes shared with m5phet.questions. They are repeated here by value so serving keeps no dependency on the
# workbench package; a caller matches on the strings, and the strings are fixed there.
NOT_ESTIMABLE = "NOT_ESTIMABLE"
STATE_REQUIRED = "STATE_REQUIRED"
MALFORMED_QUESTION = "MALFORMED_QUESTION"

GRAPH_FIELDS = ("treatment", "outcome", "confounders")

QUESTION_TYPES = {
    # An average effect carries no field of its own: the study named by the state is the whole question.
    "ate": {"required": [], "optional": []},
    # A conditional effect names its condition; it is declared so that the refusal below can say exactly why.
    "cate": {"required": ["condition"], "optional": []},
}

NO_P_VALUE = ("the retained artifact carries an estimate and an HC1 normal interval but no p-value; one is not derived "
              "from the interval here, because that would be a number the study never reported")


def refusal(kind, why, question_type):
    """The same shape as m5phet.questions.refusal: no number, and a reason a caller can read."""
    return {"status": "REFUSED", "refusal": kind, "why": why, "type": question_type}


def study_graph(config):
    """The causal graph a retained study was fitted with, in the caller's vocabulary."""
    return {"treatment": config["treatment"], "outcome": config["outcome"],
            "confounders": sorted(config["adjustments"])}


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
    """The retained study's average effect, every value copied from the artifact."""
    payload = body["result"]["payload"]
    config = body["config"]
    answer = {
        "type": "ate", "status": "OK", "execution_authorized": False,
        "estimand": payload["estimand"],
        "effect_size": payload["estimate"], "unit": payload["unit"],
        "confidence_interval": list(payload["interval"]),
        "confidence_level": payload["diagnostics"]["confidence_level"],
        "conclusion": conclusion(config, payload),
        "assumptions": list(payload["assumptions"]),
        "causal_graph": study_graph(config),
        "population": deepcopy(body["population"]),
        "diagnostics": deepcopy(payload["diagnostics"]),
        "state_ref": body["state_ref"],
        "development": body.get("development") is True,
    }
    if "p_value" in payload:
        answer["p_value"] = payload["p_value"]
    else:
        answer["not_carried"] = {"p_value": NO_P_VALUE}
    return answer


def cate_refusal(body, condition):
    config = body["config"]
    why = (f"the retained study estimates an average effect ({config['estimand']} of {config['treatment']} on "
           f"{config['outcome']}, constant across units) and was not fitted with an effect modifier, so a conditional "
           f"effect under {condition!r} cannot be read from it, and it is not derived by subsetting the population and "
           f"refitting during inference; answering this needs a study fitted explicitly with that modifier declared "
           f"(and with a heterogeneous-effect estimand), which this engine does not offer today.")
    return refusal(NOT_ESTIMABLE, why, "cate")


def resolve_study(provider, state, data, as_of):
    """The one retained study the state names, or the typed refusal every question must carry.

    Returns (body, None) or (None, (kind, why)). A study that happens to be here never stands in for the one asked
    about: a graph naming another treatment, outcome or adjustment set is refused with the difference spelled out."""
    studies = list(provider._retained_studies().values())
    if not studies:
        return None, (STATE_REQUIRED, "no fitted study is retained by this provider; explicitly fit one "
                                      "(`python -m causal_inference_provider fit` or `prepare-demo`) before asking")
    if data is not None and not (isinstance(data, (str, list, tuple, dict)) and not data):
        return None, (NOT_ESTIMABLE, "a dataset was attached, but this provider does not fit during inference; "
                                     "questions are answered from studies fitted explicitly beforehand")
    graph = state.get("causal_graph")
    if graph is not None and (not isinstance(graph, dict) or set(graph) - set(GRAPH_FIELDS)):
        return None, (MALFORMED_QUESTION, f"`causal_graph` must be a mapping with {list(GRAPH_FIELDS)}")
    ref = state.get("state_ref")
    if ref is not None:
        candidates = [body for body in studies if body["state_ref"] == ref]
        if not candidates:
            return None, (STATE_REQUIRED, f"state_ref {ref!r} is not a study this provider retains; it holds "
                                          f"{[body['state_ref'] for body in studies]}")
    elif graph is not None:
        candidates = [body for body in studies if not graph_differences(graph, body["config"])]
        if not candidates:
            fitted = [study_graph(body["config"]) for body in studies]
            differences = [graph_differences(graph, body["config"]) for body in studies]
            return None, (NOT_ESTIMABLE, f"no retained study was fitted with this causal graph; the retained graphs "
                                         f"are {fitted} and the differences are {differences}; a study for this "
                                         f"graph must be fitted explicitly before its effect can be reported")
    else:
        return None, (STATE_REQUIRED, "the state must name a study: `state_ref` (causal-ate:<digest>) or "
                                      "`causal_graph` with treatment, outcome and confounders; this provider holds "
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


def answer_questions(provider, state, questions, data, as_of):
    """Answer each declared question from the one study the state names, or refuse each by name."""
    if not isinstance(state, dict):
        state = {}
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
            out[name] = cate_refusal(body, question.get("condition"))
        else:
            out[name] = refusal(NOT_ESTIMABLE, f"this provider declares {sorted(QUESTION_TYPES)} and cannot answer "
                                               f"{kind!r}", kind)
    return out
