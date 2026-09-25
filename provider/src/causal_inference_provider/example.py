"""Synthetic calibration examples, not evidence about any real application.

Two of them, and both state their own truth. The first has a constant effect of 2 and a confounded treatment, so an
unadjusted difference is visibly wrong and the adjusted estimate is visibly right. The second has an effect that DEPENDS
on a declared binary modifier -- 1 where `baseline == 0`, 3 where `baseline == 1`, averaging 2 -- which is what makes a
conditional-effect question answerable and what makes the answer checkable: the equations below are written into the
study's manifest when it is fitted, so anyone reading the artifact can see the number the estimate is supposed to find.
"""

import json

import numpy as np
import pandas as pd

from .provider import CausalInferenceProvider

MODIFIER_STUDY_ID = "demo-modifier-v1"
MODIFIER_SEED = 4373
MODIFIER_ROWS = 2400
#: The data-generating process of the modifier study, written exactly as the code below implements it. `sigmoid` is the
#: logistic function; `Normal(0, 1)` is standard normal; every draw comes from numpy's PCG64 seeded with MODIFIER_SEED.
MODIFIER_EQUATIONS = (
    "baseline ~ Bernoulli(0.5)",
    "confounder ~ Normal(0, 1)",
    "treatment ~ Bernoulli(sigmoid(-0.6 + 1.2 * baseline + 0.35 * confounder))",
    "outcome = 1.0 * treatment + 2.0 * treatment * baseline + 3.0 * baseline + 1.5 * confounder + Normal(0, 1)",
)
#: What the equations imply, so the recovery check compares against the process and not against a remembered number.
#: The conditional effects are the coefficients of `treatment` at each level of `baseline`; the average effect is their
#: mixture under Bernoulli(0.5): 1.0 * 0.5 + 3.0 * 0.5 = 2.0.
MODIFIER_KNOWN_EFFECTS = {"baseline == 0": 1.0, "baseline == 1": 3.0, "ATE": 2.0}


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


def modifier_example_config():
    """The identifying declaration of the conditional-effect study.

    `baseline` is the effect modifier: it is pre-treatment, binary, it moves the treatment odds and it changes the size
    of the effect. `confounder` is adjusted for as a control. The constant-effect assumption is absent by construction --
    the effect is not constant here -- and in its place the study declares that the effect is linear in `baseline`,
    which for a single binary modifier is saturated and so restricts nothing."""
    return {
        "estimand": "CATE", "treatment": "treatment", "outcome": "outcome",
        "adjustments": ["confounder"], "effect_modifiers": ["baseline"],
        "unit": "synthetic outcome units",
        "assumptions": {
            "sufficient_adjustment": True, "pre_treatment_adjustment": True,
            "consistency": True, "no_interference": True, "positivity": True,
            "iid_sampling": True, "effect_linear_in_modifiers": True, "nuisance_models_correct": True,
        },
    }


def modifier_example_data():
    """MODIFIER_EQUATIONS, implemented once. Any change here is a change to the truth the manifest publishes."""
    rng = np.random.default_rng(MODIFIER_SEED)
    z = rng.binomial(1, 0.5, MODIFIER_ROWS)
    c = rng.normal(size=MODIFIER_ROWS)
    t = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.6 + 1.2 * z + 0.35 * c))))
    y = 1.0 * t + 2.0 * t * z + 3.0 * z + 1.5 * c + rng.normal(size=MODIFIER_ROWS)
    return pd.DataFrame({"baseline": z, "confounder": c, "treatment": t, "outcome": y})


def modifier_origin():
    """Where the modifier study's data came from, for its manifest: synthetic, with the equations and their implied
    effects, so the artifact carries its own auditable truth."""
    return {
        "kind": "SYNTHETIC",
        "generator": f"causal_inference_provider.example.modifier_example_data (numpy default_rng({MODIFIER_SEED}))",
        "generating_equations": list(MODIFIER_EQUATIONS),
        "known_effects": dict(MODIFIER_KNOWN_EFFECTS),
        "n": MODIFIER_ROWS, "seed": MODIFIER_SEED,
        "note": "Synthetic calibration data. It is evidence about this estimator's recovery of a known effect, and "
                "about nothing in any market or application.",
    }


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
