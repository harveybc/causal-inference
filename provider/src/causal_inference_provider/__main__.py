"""Explicit local CSV + JSON analysis; no natural-language interpretation."""

import argparse
import csv
import json
from pathlib import Path
import warnings

import pandas as pd

from .provider import CausalInferenceProvider, MAX_ROWS, _response
from .chat import save_study, state_directory
from . import event_study as _event
from . import study_space as _space
from . import study_spec as _spec


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
        if not header or len(set(header)) != len(header) or any(not name.strip() for name in header):
            raise ValueError("CSV requires unique nonempty column labels.")
        source.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            return pd.read_csv(source, nrows=MAX_ROWS + 1, index_col=False)


def dataset_for(spec):
    """The rows a spec names: one of the declared synthetic generators, or the CSV at its path.

    A spec that names both takes the path, because a file on disk is the more specific of the two. The columns the data
    actually carry are checked against the ones the spec declared, so a spec written for one dataset is never fitted on
    another that happens to be there."""
    path = (spec.get("dataset") or {}).get("path")
    identifier = (spec.get("dataset") or {}).get("id")
    if path:
        data, origin = read_csv(Path(path).expanduser()), None
    else:
        declared = _spec.SYNTHETIC_DATASETS[identifier]
        module, _, name = declared["generator"].rpartition(".")
        from importlib import import_module
        data = getattr(import_module(module), name)()
        origin = None
        if identifier == "synthetic-modifier":
            from .example import modifier_origin
            origin = modifier_origin()
    declared_columns = sorted(spec["dataset"]["columns"])
    if sorted(map(str, data.columns)) != declared_columns:
        raise ValueError(f"The dataset carries the columns {sorted(map(str, data.columns))} and the spec declares "
                         f"{declared_columns}; a spec is fitted on the dataset it was written for.")
    return data, origin


def prepare_study(args):
    """Fit exactly the study a spec describes, and retain it with the spec and its decision digests in its manifest."""
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(json.dumps({"state_ref": None,
                          "result": _response("INVALID_INPUT", "Cannot read the study spec document.")}, indent=2))
        return 2
    try:
        # the space of THIS interpreter -- the one that will fit -- so an estimator it cannot build is refused by name
        # before anything is read, and never after a fit has already happened
        space = _space.study_space()
        spec = _spec.validate_spec(spec, space)
        config = _spec.config_from_spec(spec, space)
    except _spec.SpecRefused as refused:
        print(json.dumps({"state_ref": None, "result": _response(refused.refusal, refused.why)}, indent=2))
        return 2
    state_ref = None
    try:
        data, origin = dataset_for(spec)
        provider = CausalInferenceProvider()
        result = provider.load(config)
        if result["status"] == "OK":
            provider.fit(data)
            result = provider.infer()
            if result["status"] == "OK":
                state_ref = save_study(provider, args.state_dir, development=spec["provenance"] == "DEVELOPMENT",
                                       study_id=_spec.study_identifier(spec), origin=origin, spec=spec)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.ParserWarning) as trouble:
        result = _response("INVALID_INPUT", str(trouble))
    print(json.dumps({"state_ref": state_ref, "study_id": _spec.study_identifier(spec),
                      "spec_sha256": _spec.spec_digest(spec), "result": result}, indent=2, allow_nan=False))
    return 0 if result["status"] == "OK" else 2


def register_event_study(args):
    """Register a fitted event study: the projections document, the rows it was fitted from, and an identifier.

    Nothing is fitted or read into memory here beyond the projections document itself. The manifest records the
    sha256 of BOTH files, so a document that changes underneath a registered study is refused by name at question
    time rather than answered from."""
    try:
        state_ref = _event.register(args.projections, args.rows, args.id, args.state_dir)
    except (OSError, ValueError) as trouble:
        print(json.dumps({"state_ref": None, "study_id": args.id,
                          "result": _response("INVALID_INPUT", str(trouble))}, indent=2, allow_nan=False))
        return 2
    manifest = _event.load(_event.directory_for(args.state_dir) / (state_ref[len(_event.REF_PREFIX):] + ".json"),
                           reference=state_ref)
    print(json.dumps({
        "state_ref": state_ref, "study_id": manifest["study_id"], "kind": manifest["kind"],
        "projections_sha256": manifest["documents"]["projections"]["sha256"],
        "rows_sha256": manifest["documents"]["rows"]["sha256"],
        "event_types": manifest["event_types"], "horizons_minutes": manifest["horizons_minutes"],
        "outcomes": manifest["outcomes"], "identification": manifest["identification"]["verdict"],
        "identification_caveat": manifest["identification_caveat"],
        "superposition": manifest["superposition"]["verdict"],
        "result": _response("OK", "Registered only; nothing was fitted and no number was copied."),
    }, indent=2, allow_nan=False))
    return 0


def choose_study_command(args, decider=None):
    """Profile a CSV, ask Laya for every declared choice, and write the spec those decisions describe.

    Nothing is fitted here and nothing is fitted afterwards without being asked: the spec this writes is handed to
    `prepare-study` explicitly, by a person who has read it. The whole choice document -- profile, every decision with
    its option set and its own uncalibrated probabilities, and the spec or the refusal -- goes to stdout, so a run is
    auditable from its own output; `--out` receives the spec alone, and receives nothing at all when the composition
    was refused, because there is no spec to write.

    `decider` is the thing Laya is asked through; with none the command builds the workbench's own `Engine`, which
    takes the private worker route. A test passes a registry instead, and no test ever reaches the worker."""
    from . import choose_study as chooser
    if decider is None:
        try:
            from m5phet.web.engine import Engine
        except ImportError as missing:
            print(json.dumps({"status": "REFUSED", "refusal": "PROVIDER_ERROR",
                              "why": f"choosing a study asks Laya through m5phet, which this interpreter does not "
                                     f"hold: {missing}"}, indent=2))
            return 2
        decider = Engine()
    document = chooser.choose_study(decider, args.dataset, args.problem, as_of=args.as_of,
                                    record_dir=args.record_dir, study_id=args.study_id,
                                    provenance=args.provenance, outcome_unit=args.outcome_unit)
    if document["status"] == "OK" and args.out is not None:
        args.out.write_text(json.dumps(document["spec"], indent=2, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
        document = document | {"spec_path": str(args.out), "spec_sha256": _spec.spec_digest(document["spec"])}
    print(json.dumps(document, indent=2, allow_nan=False))
    return 0 if document["status"] == "OK" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Explicitly fit a local CSV study and save its result.")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--state-dir", type=Path, default=state_directory())
    demo = commands.add_parser("prepare-demo", help="Explicitly fit the SYNTHETIC/DEVELOPMENT study.")
    demo.add_argument("--state-dir", type=Path, default=state_directory())
    study = commands.add_parser("prepare-study", help="Explicitly fit the study one m5phet.causal_study_spec.v1 "
                                                     "document describes, and retain it beside its spec.")
    study.add_argument("--spec", type=Path, required=True,
                       help="The study spec: the dataset and its columns, every column's role, the estimator, the two "
                            "nuisance models, the confidence level, the identifying assumptions and the digests of "
                            "the decision records that chose them.")
    study.add_argument("--state-dir", type=Path, default=state_directory())
    demo.add_argument("--with-modifier", action="store_true",
                      help="Fit the SYNTHETIC/DEVELOPMENT study that declares an effect modifier (estimand CATE), so "
                           "conditional-effect questions about its declared subgroups can be answered. It is a second, "
                           "separate study: nothing already retained is read, changed or replaced.")
    event = commands.add_parser("register-event-study",
                                help="Register an already fitted EVENT study -- local projections of a price series "
                                     "on calendar surprises -- so the envelope can ask impulse_response, "
                                     "sensitivity and counterfactual_path of it. Nothing is fitted here.")
    event.add_argument("--projections", type=Path, required=True,
                       help="The m5phet.event_projections.v1 document written by feature_eng_m5phet.local_projections.")
    event.add_argument("--rows", type=Path, required=True,
                       help="The m5phet.event_rows.v1 document those projections were fitted from; a counterfactual "
                            "path is evaluated over it.")
    event.add_argument("--id", required=True, help="The identifier the study is retained and asked for under.")
    event.add_argument("--state-dir", type=Path, default=state_directory())
    choose = commands.add_parser("choose-study", help="Ask Laya to choose a study's configuration from a dataset "
                                                      "profile and a problem sentence, and write the spec.")
    choose.add_argument("--dataset", type=Path, required=True, help="The CSV the study would be fitted on. Only its "
                                                                    "profile is shown to the model; no row is.")
    choose.add_argument("--problem", required=True, help="One sentence saying what the study is about.")
    choose.add_argument("--out", type=Path, default=None, help="Where to write the composed spec. Nothing is written "
                                                               "when the composition is refused.")
    choose.add_argument("--record-dir", type=Path, default=state_directory() / "decisions",
                        help="Where the decision records are filed, content-addressed. Their digests go into the "
                             "spec, so a study can always be traced back to the choices that made it; by default "
                             "they are filed beside the studies this repository retains.")
    choose.add_argument("--study-id", default=None, help="The identifier the fitted study would be retained under.")
    choose.add_argument("--provenance", default="UNDECLARED", choices=list(_spec.PROVENANCE),
                        help="The provenance of the data. A chooser cannot establish it, so it is declared here.")
    choose.add_argument("--outcome-unit", default=None, help="The unit the outcome is measured in.")
    choose.add_argument("--as-of", default=None, help="The clock the decisions are stamped at; now by default.")
    args = parser.parse_args()
    state_ref = None
    extra = {}
    saved = {"development": args.command == "prepare-demo"}
    try:
        if args.command == "register-event-study":
            return register_event_study(args)
        if args.command == "choose-study":
            return choose_study_command(args)
        if args.command == "prepare-study":
            return prepare_study(args)
        if args.command == "prepare-demo":
            if args.with_modifier:
                from .example import (MODIFIER_STUDY_ID, modifier_example_config, modifier_example_data,
                                      modifier_origin)
                config, data = modifier_example_config(), modifier_example_data()
                saved |= {"study_id": MODIFIER_STUDY_ID, "origin": modifier_origin()}
            else:
                from .example import example_config, example_data
                config, data = example_config(), example_data()
        else:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            data = None
        provider = CausalInferenceProvider()
        result = provider.load(config)
        if result["status"] == "OK":
            if data is None:
                data = read_csv(args.data)
            provider.fit(data)
            result = provider.infer()
            if result["status"] == "OK":
                state_ref = save_study(provider, args.state_dir, **saved)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.ParserWarning):
        result = _response("INVALID_INPUT", "Cannot read the CSV/JSON input or write the study artifact.")
    print(json.dumps({"state_ref": state_ref, "result": result}, indent=2, allow_nan=False))
    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
