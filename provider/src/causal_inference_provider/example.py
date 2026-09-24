"""A synthetic calibration example, not evidence about any real application."""

import json

import numpy as np
import pandas as pd

from .provider import CausalInferenceProvider


def example_config():
    # These declarations follow from this synthetic data-generating process only.
    return {
        "estimand": "ATE", "treatment": "treatment", "outcome": "outcome",
        "adjustments": ["baseline"], "unit": "synthetic outcome units",
        "assumptions": {
            "sufficient_adjustment": True, "pre_treatment_adjustment": True,
            "consistency": True, "no_interference": True, "positivity": True,
            "iid_sampling": True, "constant_effect": True, "nuisance_models_correct": True,
        },
    }


def example_data():
    rng = np.random.default_rng(17)
    z = rng.binomial(1, 0.5, 2400)
    t = rng.binomial(1, 0.2 + 0.6 * z)
    return pd.DataFrame({"baseline": z, "treatment": t,
                         "outcome": 2.0 * t + 4.0 * z + rng.normal(size=len(t))})


def main():
    provider = CausalInferenceProvider()
    data = example_data()
    loaded = provider.load(example_config())
    if loaded["status"] != "OK":
        raise RuntimeError(loaded["reason"])
    provider.fit(data)
    result = provider.infer()
    naive = float(data.groupby("treatment")["outcome"].mean().diff().iloc[-1])
    print(json.dumps({"synthetic_true_ate": 2.0, "unadjusted_difference": naive,
                      "result": result}, indent=2, allow_nan=False))
    if result["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
