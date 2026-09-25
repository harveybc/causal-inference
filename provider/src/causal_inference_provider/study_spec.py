"""The study specification: one document that says exactly which study to fit, and what refuses it.

A spec is what a chooser produces and what the explicit fit consumes. It names the dataset and its columns, gives every
column a role, names the estimator and the two nuisance models from the declared space, states the confidence level,
lists the identifying assumptions in the engine's own words, and carries the digests of the decision records that
produced those choices (an empty list when a person wrote the spec by hand).

`validate_spec` refuses BY NAME. Every refusal below is a sentence about the spec, never about the person, and no
refusal is repaired silently: a spec with no assumption stated is `NOT_IDENTIFIED` and is never fitted, because a
causal estimate with no stated identification is a number without a claim.
"""

from copy import deepcopy
from hashlib import sha256
import json
import re

from . import study_space as _space

SPEC_SCHEMA = "m5phet.causal_study_spec.v1"

#: The provenance a fitted study is allowed to declare. `DEVELOPMENT` is synthetic or exploratory; `UNDECLARED` says
#: the provenance was not established, which is the only other thing this repository can honestly write.
PROVENANCE = ("DEVELOPMENT", "UNDECLARED")

#: The datasets this repository can fit from without leaving it: the two synthetic generators WP10 already ships. Any
#: other dataset arrives as a CSV `path`. A data-lake `id` this repository cannot resolve is refused by name.
SYNTHETIC_DATASETS = {
    "synthetic-constant": {
        "generator": "causal_inference_provider.example.example_data",
        "label": "SYNTHETIC/DEVELOPMENT: confounded treatment, constant effect 2",
        "columns": ("baseline", "treatment", "outcome"),
    },
    "synthetic-modifier": {
        "generator": "causal_inference_provider.example.modifier_example_data",
        "label": "SYNTHETIC/DEVELOPMENT: effect 1 where baseline == 0, 3 where baseline == 1, average 2",
        "columns": ("baseline", "confounder", "treatment", "outcome"),
    },
}

DIGEST_PATTERN = re.compile(r"[a-f0-9]{64}")

#: Every refusal this module can give, by name.
MALFORMED_SPEC = "MALFORMED_SPEC"
UNKNOWN_ESTIMATOR = "UNKNOWN_ESTIMATOR"
UNKNOWN_NUISANCE_MODEL = "UNKNOWN_NUISANCE_MODEL"
UNKNOWN_ROLE = "UNKNOWN_ROLE"
UNKNOWN_COLUMN = "UNKNOWN_COLUMN"
UNASSIGNED_COLUMN = "UNASSIGNED_COLUMN"
NO_TREATMENT = "NO_TREATMENT"
TWO_TREATMENTS = "TWO_TREATMENTS"
NO_OUTCOME = "NO_OUTCOME"
TWO_OUTCOMES = "TWO_OUTCOMES"
TWO_MODIFIERS = "TWO_MODIFIERS"
ESTIMATOR_HAS_NO_ATE = "ESTIMATOR_HAS_NO_ATE"
ESTIMATOR_HAS_NO_CATE = "ESTIMATOR_HAS_NO_CATE"
UNKNOWN_ASSUMPTION = "UNKNOWN_ASSUMPTION"
INAPPLICABLE_ASSUMPTION = "INAPPLICABLE_ASSUMPTION"
INCOMPLETE_IDENTIFICATION = "INCOMPLETE_IDENTIFICATION"
NOT_IDENTIFIED = "NOT_IDENTIFIED"
UNKNOWN_CONFIDENCE_LEVEL = "UNKNOWN_CONFIDENCE_LEVEL"
UNKNOWN_TREATMENT_KIND = "UNKNOWN_TREATMENT_KIND"
ESTIMATOR_HAS_NO_TREATMENT_KIND = "ESTIMATOR_HAS_NO_TREATMENT_KIND"
UNKNOWN_PROVENANCE = "UNKNOWN_PROVENANCE"
UNKNOWN_DATASET = "UNKNOWN_DATASET"
MALFORMED_DECISION_DIGEST = "MALFORMED_DECISION_DIGEST"

#: The fields a spec may carry. `study_id`, `outcome_unit` and `treatment_kind` are optional; everything else is
#: required. An absent `treatment_kind` is `binary`, which is what every spec written before that dimension existed
#: meant -- so adding the field changes no study that does not name it.
SPEC_FIELDS = ("schema", "dataset", "roles", "estimator", "nuisance", "confidence_level", "identification",
               "decisions", "provenance", "study_id", "outcome_unit", "treatment_kind", "identification_caveat")


class SpecRefused(ValueError):
    """A spec that will not be fitted, with the name of the refusal and the reason in the same object."""

    def __init__(self, refusal, why):
        super().__init__(f"{refusal}: {why}")
        self.refusal = refusal
        self.why = why


def _refuse(refusal, why):
    raise SpecRefused(refusal, why)


def spec_digest(spec):
    """The content address of a spec: the digest of its canonical JSON."""
    return sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()


def columns_by_role(roles):
    """The declared columns of each role, in the order the spec lists them."""
    found = {role: [] for role, _ in _space._ROLES}
    for column, role in roles.items():
        found.setdefault(role, []).append(column)
    return found


def validate_spec(spec, space=None):
    """Return the spec, normalised, or raise `SpecRefused` naming why it will not be fitted.

    The space is the one this interpreter declares unless the caller passes another; passing one is how a fit
    interpreter validates against what it can actually build."""
    space = space if space is not None else _space.study_space(probe=_space.declared)
    if not isinstance(spec, dict):
        _refuse(MALFORMED_SPEC, "a study spec is a JSON object.")
    unknown = sorted(str(field) for field in set(spec) - set(SPEC_FIELDS))
    if unknown:
        _refuse(MALFORMED_SPEC, f"unknown spec fields {unknown}; this schema declares {list(SPEC_FIELDS)}.")
    if spec.get("schema") != SPEC_SCHEMA:
        _refuse(MALFORMED_SPEC, f"a study spec declares schema {SPEC_SCHEMA!r}, not {spec.get('schema')!r}.")

    dataset = spec.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) - {"id", "path", "columns"}:
        _refuse(MALFORMED_SPEC, "`dataset` is an object with `id` or `path`, and `columns`.")
    columns = dataset.get("columns")
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(name, str) or not name.strip() for name in columns)
            or len(set(columns)) != len(columns)):
        _refuse(MALFORMED_SPEC, "`dataset.columns` is a nonempty list of distinct column names.")
    identifier, path = dataset.get("id"), dataset.get("path")
    if identifier is None and path is None:
        _refuse(UNKNOWN_DATASET, f"`dataset` names neither an `id` nor a `path`; the ids this repository can fit from "
                                 f"are {sorted(SYNTHETIC_DATASETS)}, and any other dataset arrives as a CSV `path`.")
    if identifier is not None and (not isinstance(identifier, str) or not identifier.strip()):
        _refuse(MALFORMED_SPEC, "`dataset.id` is a nonempty string.")
    if path is not None and (not isinstance(path, str) or not path.strip()):
        _refuse(MALFORMED_SPEC, "`dataset.path` is a nonempty string.")
    if identifier is not None and path is None and identifier not in SYNTHETIC_DATASETS:
        _refuse(UNKNOWN_DATASET, f"dataset id {identifier!r} is not one this repository can fit from; it holds "
                                 f"{sorted(SYNTHETIC_DATASETS)}, and every other dataset is given as a CSV `path`.")

    roles = spec.get("roles")
    if not isinstance(roles, dict) or not roles or any(not isinstance(name, str) for name in roles):
        _refuse(MALFORMED_SPEC, "`roles` maps every column name to one declared role.")
    declared_roles = [key for key, _ in space["roles"]["options"]]
    for column, role in roles.items():
        if column not in columns:
            _refuse(UNKNOWN_COLUMN, f"{column!r} is given a role but is not one of the dataset's columns {columns}.")
        if role not in declared_roles:
            _refuse(UNKNOWN_ROLE, f"{role!r} is not a declared role for column {column!r}; the declared roles are "
                                  f"{declared_roles}.")
    unassigned = [name for name in columns if name not in roles]
    if unassigned:
        _refuse(UNASSIGNED_COLUMN, f"the columns {unassigned} were given no role; every column of the dataset takes "
                                   f"one of {declared_roles}, `exclude` included.")
    grouped = columns_by_role(roles)
    if not grouped["treatment"]:
        _refuse(NO_TREATMENT, "no column is the treatment; a causal study is about an intervention, and this spec "
                              "names none.")
    if len(grouped["treatment"]) > 1:
        _refuse(TWO_TREATMENTS, f"the columns {sorted(grouped['treatment'])} are all given the treatment role; this "
                                f"engine estimates the effect of exactly one treatment column.")
    if not grouped["outcome"]:
        _refuse(NO_OUTCOME, "no column is the outcome; an effect is an effect on something, and this spec names "
                            "nothing.")
    if len(grouped["outcome"]) > 1:
        _refuse(TWO_OUTCOMES, f"the columns {sorted(grouped['outcome'])} are all given the outcome role; two outcomes "
                              f"are two studies, each fitted and reported on its own.")
    if len(grouped["modifier"]) > 1:
        _refuse(TWO_MODIFIERS, f"the columns {sorted(grouped['modifier'])} are all given the modifier role; this "
                               f"engine reports an effect per level of exactly one binary modifier.")

    kinds = space.get("treatment_kinds") or {"options": [], "default": "binary"}
    declared_kinds = [key for key, _ in kinds["options"]]
    treatment_kind = spec.get("treatment_kind", kinds["default"])
    if treatment_kind not in declared_kinds:
        _refuse(UNKNOWN_TREATMENT_KIND, f"{treatment_kind!r} is not a treatment kind this space declares; it declares "
                                        f"{declared_kinds}, and a kind it cannot read it cannot fit.")

    estimator = spec.get("estimator")
    detail = (space["estimators"]["detail"] or {}).get(estimator) if isinstance(estimator, str) else None
    if detail is None:
        _refuse(UNKNOWN_ESTIMATOR, f"{estimator!r} is not an estimator this space offers; it offers "
                                   f"{[key for key, _ in space['estimators']['options']]}.")
    served = list(detail.get("treatments") or [])
    if treatment_kind not in served:
        _refuse(ESTIMATOR_HAS_NO_TREATMENT_KIND,
                f"this spec declares a {treatment_kind} treatment and estimator {estimator!r} is declared for "
                f"{served}; the estimators this space offers for a {treatment_kind} treatment are "
                f"{(kinds.get('served_by') or {}).get(treatment_kind, [])}.")
    estimand = "CATE" if grouped["modifier"] else "ATE"
    if estimand == "CATE" and "CATE" not in detail["estimands"]:
        _refuse(ESTIMATOR_HAS_NO_CATE, f"column {grouped['modifier'][0]!r} is declared an effect modifier, but "
                                       f"estimator {estimator!r} serves {list(detail['estimands'])} only; a "
                                       f"conditional effect needs an estimator that reports one with its interval.")
    if estimand == "ATE" and "ATE" not in detail["estimands"]:
        _refuse(ESTIMATOR_HAS_NO_ATE, f"no column is declared an effect modifier, so this spec asks for one average "
                                      f"effect, and estimator {estimator!r} serves {list(detail['estimands'])} only; "
                                      f"it is fitted on the modifiers and has none here.")

    nuisance = spec.get("nuisance")
    if not isinstance(nuisance, dict) or set(nuisance) != {"model_y", "model_t"}:
        _refuse(MALFORMED_SPEC, "`nuisance` names exactly `model_y` (the outcome regression) and `model_t` (the "
                                "treatment classification).")
    offered = [key for key, _ in space["nuisance_models"]["options"]]
    for role, chosen in sorted(nuisance.items()):
        if chosen not in offered:
            _refuse(UNKNOWN_NUISANCE_MODEL, f"{chosen!r} is not a nuisance model this space offers for {role}; it "
                                            f"offers {offered}.")

    level = _space.confidence_level_value(spec.get("confidence_level"), space)
    if level is None:
        _refuse(UNKNOWN_CONFIDENCE_LEVEL, f"{spec.get('confidence_level')!r} is not a declared confidence level; the "
                                          f"declared ones are {sorted(space['confidence_levels']['values'])}.")

    identification = spec.get("identification")
    if not isinstance(identification, list) or any(not isinstance(name, str) for name in identification):
        _refuse(MALFORMED_SPEC, "`identification` is a list of declared assumption names.")
    if not identification:
        _refuse(NOT_IDENTIFIED, "no assumption stated; this engine identifies nothing on its own, so a study whose "
                                "identification is empty is never fitted.")
    if len(set(identification)) != len(identification):
        _refuse(MALFORMED_SPEC, "`identification` repeats an assumption.")
    declared_assumptions = [key for key, _ in space["identification_assumptions"]["options"]]
    for name in identification:
        if name not in declared_assumptions:
            _refuse(UNKNOWN_ASSUMPTION, f"{name!r} is not an assumption this provider declares; the declared ones are "
                                        f"{declared_assumptions}, and an assumption it cannot read it cannot check.")
    required = list(space["identification_assumptions"]["required_for"][estimand])
    inapplicable = sorted(set(identification) - set(required))
    if inapplicable:
        _refuse(INAPPLICABLE_ASSUMPTION, f"a {estimand} study does not declare {inapplicable}; it declares "
                                         f"{required}.")
    missing = sorted(set(required) - set(identification))
    if missing:
        _refuse(INCOMPLETE_IDENTIFICATION, f"a {estimand} study declares every one of {required}; {missing} "
                                           f"{'is' if len(missing) == 1 else 'are'} not stated, and an assumption "
                                           f"left unstated is not assumed here.")

    decisions = spec.get("decisions")
    if not isinstance(decisions, list):
        _refuse(MALFORMED_SPEC, "`decisions` is a list of decision-record digests, empty when the spec was written "
                                "by hand.")
    for entry in decisions:
        if not isinstance(entry, str) or DIGEST_PATTERN.fullmatch(entry) is None:
            _refuse(MALFORMED_DECISION_DIGEST, f"{entry!r} is not a decision-record digest; a decision is recorded by "
                                               f"its sha256, so a spec can be traced back to the choices that made it.")
    if len(set(decisions)) != len(decisions):
        _refuse(MALFORMED_SPEC, "`decisions` repeats a digest.")

    provenance = spec.get("provenance")
    if provenance not in PROVENANCE:
        _refuse(UNKNOWN_PROVENANCE, f"{provenance!r} is not a provenance this repository can declare; it declares "
                                    f"{list(PROVENANCE)}.")
    if identifier in SYNTHETIC_DATASETS and path is None and provenance != "DEVELOPMENT":
        _refuse(UNKNOWN_PROVENANCE, f"dataset {identifier!r} is synthetic, so its study is DEVELOPMENT; it cannot be "
                                    f"declared {provenance!r}.")

    study_id = spec.get("study_id")
    if study_id is not None:
        from .chat import STUDY_ID_PATTERN
        if not isinstance(study_id, str) or STUDY_ID_PATTERN.fullmatch(study_id) is None:
            _refuse(MALFORMED_SPEC, "`study_id` is 3-64 lowercase characters from [a-z0-9._-], starting with a letter "
                                    "or digit.")
    unit = spec.get("outcome_unit")
    if unit is not None and (not isinstance(unit, str) or not unit.strip() or len(unit) > 128):
        _refuse(MALFORMED_SPEC, "`outcome_unit` is a nonempty string of at most 128 characters.")
    # What the DATA cannot deliver, in the words of whoever built it, carried verbatim onto every answer this study
    # ever gives. `identification` above is the list of assumptions the engine checks it declares; this is the thing
    # no assumption list can express -- an observation that was never made, a clock nobody read. It is copied and
    # never summarised, because a caveat a reader has to go and look up is a caveat that gets dropped.
    caveat = spec.get("identification_caveat")
    if caveat is not None and (not isinstance(caveat, str) or not caveat.strip() or len(caveat) > 4096):
        _refuse(MALFORMED_SPEC, "`identification_caveat` is a nonempty string of at most 4096 characters: what the "
                                "data could not deliver, in the words of whoever built it.")

    normalised = deepcopy(spec)
    normalised["confidence_level"] = level
    if "treatment_kind" in normalised or treatment_kind != kinds["default"]:
        normalised["treatment_kind"] = treatment_kind
    return normalised


def study_identifier(spec):
    """The name the fitted study is retained under: the one the spec gives, else its own content address."""
    return spec.get("study_id") or ("study-" + spec_digest(spec)[:12])


def config_from_spec(spec, space=None):
    """The engine's identifying config for a valid spec: roles, estimand, assumptions, estimator and nuisance models.

    The estimand is not chosen: it FOLLOWS from the roles. A spec that gives a column the modifier role asks for an
    effect that varies with it, and that is a CATE study; a spec with no modifier asks for one number, and that is an
    ATE study."""
    spec = validate_spec(spec, space)
    grouped = columns_by_role(spec["roles"])
    treatment, = grouped["treatment"]
    outcome, = grouped["outcome"]
    modifiers = sorted(grouped["modifier"])
    config = {
        "estimand": "CATE" if modifiers else "ATE",
        "treatment": treatment,
        "outcome": outcome,
        "adjustments": sorted(grouped["confounder"]),
        "unit": spec.get("outcome_unit") or f"{outcome} units",
        "assumptions": {name: True for name in sorted(spec["identification"])},
        "alpha": round(1.0 - float(spec["confidence_level"]), 10),
        "estimator": spec["estimator"],
        "nuisance": {"model_y": spec["nuisance"]["model_y"], "model_t": spec["nuisance"]["model_t"]},
    }
    # the DEFAULT kind stays absent from the identifying config: its digest is the task identity of every study
    # already retained, and a field that means what its absence already meant would rename all of them
    kind = spec.get("treatment_kind")
    if kind is not None and kind != (space or _space.study_space(probe=_space.declared))["treatment_kinds"]["default"]:
        config["treatment_kind"] = kind
    if modifiers:
        config["effect_modifiers"] = modifiers
    return config
