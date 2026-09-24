"""Explicit local CSV + JSON analysis; no natural-language interpretation."""

import argparse
import csv
import json
from pathlib import Path
import warnings

import pandas as pd

from .provider import CausalInferenceProvider, MAX_ROWS, _response
from .chat import save_study, state_directory


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
        if not header or len(set(header)) != len(header) or any(not name.strip() for name in header):
            raise ValueError("CSV requires unique nonempty column labels.")
        source.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            return pd.read_csv(source, nrows=MAX_ROWS + 1, index_col=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Explicitly fit a local CSV study and save its result.")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--state-dir", type=Path, default=state_directory())
    demo = commands.add_parser("prepare-demo", help="Explicitly fit the SYNTHETIC/DEVELOPMENT study.")
    demo.add_argument("--state-dir", type=Path, default=state_directory())
    args = parser.parse_args()
    state_ref = None
    try:
        if args.command == "prepare-demo":
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
                state_ref = save_study(provider, args.state_dir, development=args.command == "prepare-demo")
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.ParserWarning):
        result = _response("INVALID_INPUT", "Cannot read the CSV/JSON input or write the study artifact.")
    print(json.dumps({"state_ref": state_ref, "result": result}, indent=2, allow_nan=False))
    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
