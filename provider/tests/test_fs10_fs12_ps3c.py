"""FS10, FS11, FS12 of the progressive-selection subplan, written RED before any PS3-C code exists.

Subplan `FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md` section 9:

* FS10 -- filtering events does not certify do(); identify the adjustment and check the support.
* FS11 -- a counterfactual preserves the inferred perturbations, propagates descendants, and never uses the live future.
* FS12 -- causal inference under confounding or missing support keeps the NOT_IDENTIFIED state.

Every test here that needs the missing mechanism names it: the module `causal_inference_provider.ps3c` with the four
names `rung2_effect`, `AdditiveSCM`, `counterfactual_same_episode` and `dossier`, as specified in
`docs/PS3C_CAUSAL_SPEC_2026_10_01.md` section 6. Where the mechanism is missing the test FAILS with a message that
begins `MECHANISM_MISSING` -- it is never skipped, because a skipped acceptance test reads as green in a summary.

Two tests are green today on purpose: they pin the mechanisms that already exist (the event study refuses `ate`; the
provider refuses undeclared assumptions) so that a regression of either is visible beside the red ones.

The planted worlds are synthetic and tiny (hundreds of rows, numpy only, seeded). The numbers in them are arbitrary
and are never compared to a market. CPU only; no fit of any library model happens in this file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from causal_inference_provider import event_study as _event
from causal_inference_provider.provider import CausalInferenceProvider

SEED = 1729
MECHANISM = "causal_inference_provider.ps3c"


def _ps3c(name):
    """The PS3-C mechanism, or a failure that names it. Never a skip."""
    try:
        import importlib

        module = importlib.import_module(MECHANISM)
    except ImportError as trouble:
        pytest.fail(f"MECHANISM_MISSING: {MECHANISM} does not exist ({trouble}); "
                    f"docs/PS3C_CAUSAL_SPEC_2026_10_01.md section 6 specifies it. This test turns green when "
                    f"`{MECHANISM}.{name}` exists with the contract in that table.")
    attribute = getattr(module, name, None)
    if attribute is None:
        pytest.fail(f"MECHANISM_MISSING: {MECHANISM} exists but has no `{name}`; see the spec's section 6 table.")
    return attribute


# --------------------------------------------------------------------------------------------- the planted worlds


def confounded_episodes(n=400, effect=0.5, seed=SEED):
    """Episodes where the pre-event state W drives BOTH the surprise A and the outcome Y.

    A = 0.8 * W + eps_a;  Y = effect * A + 1.5 * W + eps_y.  The planted effect of A on Y is `effect`.  A contrast of
    filtered means (A > 0 against A <= 0) does not recover it, because W is larger where A is larger.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = 0.8 * w + rng.normal(scale=0.5, size=n)
    y = effect * a + 1.5 * w + rng.normal(scale=0.1, size=n)
    return [{"episode_id": f"e{i}", "event_type": "SYN | release", "W": float(w[i]), "A": float(a[i]),
             "Y_h": float(y[i])} for i in range(n)]


def filtered_mean_contrast(episodes, a_cut=0.0):
    treated = [e["Y_h"] for e in episodes if e["A"] > a_cut]
    control = [e["Y_h"] for e in episodes if e["A"] <= a_cut]
    a_t = np.mean([e["A"] for e in episodes if e["A"] > a_cut])
    a_c = np.mean([e["A"] for e in episodes if e["A"] <= a_cut])
    # per unit of A, so it is comparable to the planted slope
    return (np.mean(treated) - np.mean(control)) / (a_t - a_c)


DAG = {
    "nodes": ["W", "A", "M", "Y_h"],
    "edges": [["W", "A"], ["W", "Y_h"], ["A", "M"], ["M", "Y_h"], ["A", "Y_h"]],
}


# ------------------------------------------------------------------------------------------------------------ FS10


def test_fs10_filtered_event_contrast_is_not_do():
    """Filtering episodes by the treatment and differencing their means is association; do() needs the adjustment."""
    episodes = confounded_episodes()
    naive = filtered_mean_contrast(episodes)
    # the trap is real: the filtered contrast is far from the planted 0.5 (it absorbs 1.5 * W / 0.8 of confounding)
    assert abs(naive - 0.5) > 0.5, f"the planted world is not confounded enough to make the point: {naive}"

    rung2_effect = _ps3c("rung2_effect")
    undeclared = rung2_effect(episodes, treatment="A", outcome="Y_h", adjustment=None, contrast=(1.0, 0.0),
                              dag=DAG, support={"min_episodes_per_side": 20})
    assert undeclared["state"] == "NOT_IDENTIFIED"
    assert "ADJUSTMENT_SET_NOT_DECLARED" in undeclared["reasons"]
    assert undeclared["estimate"] is None, "an undeclared adjustment must not release a number"

    declared = rung2_effect(episodes, treatment="A", outcome="Y_h", adjustment=["W"], contrast=(1.0, 0.0),
                            dag=DAG, support={"min_episodes_per_side": 20},
                            assumptions={name: True for name in _ps3c("REQUIRED_ASSUMPTIONS")},
                            assumption_evidence={name: "planted synthetic SCM" for name in
                                                 _ps3c("REQUIRED_ASSUMPTIONS")})
    assert declared["state"] == "NOT_IDENTIFIED"
    assert "IMBALANCE" in declared["reasons"]
    assert declared["estimate"] is None
    assert declared["adjustment"] == ["W"]
    assert declared["support"]["state"] == "IMBALANCE"


def test_fs10_existing_event_study_refuses_do_questions():
    """GREEN TODAY. An event study carries no contrast and no population: `ate` is refused NOT_ESTIMABLE by name."""
    manifest = {"schema": _event.SCHEMA, "study_id": "fs10-fixture", "kind": _event.KIND,
                "identification": {"verdict": "NOT_IDENTIFIED", "reasons": ["ASSUMED_PUBLICATION_CLOCK"]}}
    refusal = _event.answer(manifest, "ate", {"treatment": "surprise"})
    assert refusal["status"] == "REFUSED"
    assert refusal["refusal"] == _event.NOT_ESTIMABLE
    assert "ate" in refusal["why"]
    assert "do" not in _event.QUESTION_TYPES and "ate" not in _event.QUESTION_TYPES


# ------------------------------------------------------------------------------------------------------------ FS11


def planted_scm():
    """W exogenous; A observed; M = 0.7 * A + 0.2 * W + U_M; Y = 0.5 * A + 1.5 * W + 0.4 * M + U_Y."""
    AdditiveSCM = _ps3c("AdditiveSCM")
    return AdditiveSCM(order=["W", "A", "M", "Y_h"],
                       mechanisms={"M": lambda A, W: 0.7 * A + 0.2 * W,
                                   "Y_h": lambda A, W, M: 0.5 * A + 1.5 * W + 0.4 * M})


def test_fs11_counterfactual_preserves_inferred_perturbation():
    """y_cf = f(a0, w, m_cf) + u_e, with u_e abducted from THIS episode; f(a0, w, m_cf) alone is the population model."""
    scm = planted_scm()
    counterfactual = _ps3c("counterfactual_same_episode")
    u_m, u_y = 0.3, -0.25
    w, a = 0.4, 1.2
    m = 0.7 * a + 0.2 * w + u_m
    y = 0.5 * a + 1.5 * w + 0.4 * m + u_y
    episode = {"episode_id": "e1", "W": w, "A": a, "M": m, "Y_h": y, "published_at": "2021-01-08T13:30:00+00:00",
               "observed_at": "2021-01-08T14:30:00+00:00"}
    answer = counterfactual(scm, episode, intervention={"A": 0.0}, mode="RETROSPECTIVE")

    assert math.isclose(answer["abduction"]["U_M"], u_m, abs_tol=1e-9)
    assert math.isclose(answer["abduction"]["U_Y_h"], u_y, abs_tol=1e-9)
    m_cf = 0.7 * 0.0 + 0.2 * w + u_m
    y_cf = 0.5 * 0.0 + 1.5 * w + 0.4 * m_cf + u_y
    assert math.isclose(answer["prediction"]["Y_h"], y_cf, abs_tol=1e-9)
    # the MODEL_BASED number is the one WITHOUT u_e; the test insists the two are different objects and both named
    assert math.isclose(answer["model_based"]["Y_h"], y_cf - u_y, abs_tol=1e-9)
    assert answer["label"] == "SAME_EPISODE_COUNTERFACTUAL_UNDER_DECLARED_SCM"


def test_fs11_descendants_are_propagated_not_frozen():
    """A mediator is recomputed under the action with its own perturbation kept, never held at its factual value."""
    scm = planted_scm()
    counterfactual = _ps3c("counterfactual_same_episode")
    w, a, u_m, u_y = -0.2, 2.0, 0.05, 0.0
    m = 0.7 * a + 0.2 * w + u_m
    y = 0.5 * a + 1.5 * w + 0.4 * m + u_y
    episode = {"episode_id": "e2", "W": w, "A": a, "M": m, "Y_h": y, "published_at": "2021-02-05T13:30:00+00:00",
               "observed_at": "2021-02-05T14:30:00+00:00"}
    answer = counterfactual(scm, episode, intervention={"A": 0.0}, mode="RETROSPECTIVE")
    assert math.isclose(answer["prediction"]["M"], 0.2 * w + u_m, abs_tol=1e-9), "M must move with A"
    assert not math.isclose(answer["prediction"]["M"], m, abs_tol=1e-9), "M was frozen at its factual value"
    assert answer["propagation"]["order"] == ["W", "A", "M", "Y_h"]


def test_fs11_operational_mode_refuses_live_future():
    """An operational call may not see a realized outcome; it must refuse by name instead of abducting from it."""
    scm = planted_scm()
    counterfactual = _ps3c("counterfactual_same_episode")
    episode = {"episode_id": "e3", "W": 0.1, "A": 0.9, "M": 0.8, "Y_h": 0.7,
               "published_at": "2021-03-05T13:30:00+00:00", "emitted_at": "2021-03-05T13:30:00+00:00"}
    with pytest.raises(Exception) as trouble:
        counterfactual(scm, episode, intervention={"A": 0.0}, mode="OPERATIONAL")
    assert "FUTURE_OUTCOME_IN_OPERATIONAL_CALL" in str(trouble.value)


# ------------------------------------------------------------------------------------------------------------ FS12


def test_fs12_provider_keeps_not_identified_without_declared_assumptions():
    """GREEN TODAY. The retained provider releases no number when the identifying assumptions are not declared true."""
    provider = CausalInferenceProvider()
    result = provider.load({"estimand": "ATE", "treatment": "A", "outcome": "Y_h", "adjustments": ["W"],
                            "unit": "synthetic", "alpha": 0.05, "assumptions": {}})
    assert result["status"] == "NOT_IDENTIFIED"
    assert result["payload"] is None


def test_fs12_no_common_support_keeps_not_identified():
    """A contrast outside the dose support of the event type stays NOT_IDENTIFIED, and rung-1 evidence survives."""
    episodes = confounded_episodes()
    rung2_effect = _ps3c("rung2_effect")
    a_max = max(e["A"] for e in episodes)
    out = rung2_effect(episodes, treatment="A", outcome="Y_h", adjustment=["W"], contrast=(a_max + 5.0, 0.0),
                       dag=DAG, support={"min_episodes_per_side": 20})
    assert out["state"] == "NOT_IDENTIFIED"
    assert "NO_COMMON_SUPPORT" in out["reasons"]
    assert out["estimate"] is None
    assert out["rung1"]["state"] == "ASSOCIATION_REPORTED", "rung-1 evidence must not be erased by a rung-2 refusal"


def test_fs12_confounded_world_without_adjustment_keeps_not_identified():
    """An adjustment set that does not block the back-door path on the declared DAG keeps NOT_IDENTIFIED."""
    episodes = confounded_episodes()
    rung2_effect = _ps3c("rung2_effect")
    # `M` is a descendant of A: adjusting on it alone leaves the W back-door open and conditions on a mediator
    out = rung2_effect(episodes, treatment="A", outcome="Y_h", adjustment=["M"], contrast=(1.0, 0.0),
                       dag=DAG, support={"min_episodes_per_side": 20})
    assert out["state"] == "NOT_IDENTIFIED"
    assert "BACKDOOR_NOT_SATISFIED" in out["reasons"]
    assert out["estimate"] is None
