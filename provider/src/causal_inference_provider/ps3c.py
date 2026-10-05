"""PS3-C: the three causal rungs over historical episodes (subplan section 5, master plan v3 section 5).

Unit: one historical episode anchored at a decision instant t. Treatment A is fixed
before any outcome is looked at; outcomes Y are realised after t; the history H and
the pre-event covariates W are available before t. Everything is fitted inside the
TRAIN rows the caller passes; this module never reads a test partition.

Rung 1 (association / predictive relevance) -- ``rung1_association``
    conditional dependence of Y on A given H, out-of-fold gain of A over H on
    chronological folds, temporal placebos and negative controls. Never causal words.
Rung 2 (observed historical interventions) -- ``rung2_effect``
    episodes with A=a against controls with A=a' sharing the declared pre-event
    context; back-door check on the declared DAG, support/overlap/balance,
    g-computation, cross-fitted AIPW / partialling-out and matching; placebo and
    refutation battery; sensitivity. ``NOT_IDENTIFIED`` with named reasons otherwise.
Rung 3 (same-episode counterfactual) -- ``AdditiveSCM``, ``fit_linear_scm``,
    ``counterfactual_same_episode``, ``rung3_population``: abduction of the
    episode's own perturbations, action on A only (inside historical support),
    propagation of descendants; factual reconstruction, historical analog pairs,
    placebos and model sensitivity. The individual counterfactual is an estimate
    under a declared SCM, never an observation.

``dossier`` assembles a ``causal_dossier.v1`` document (vendored schema in
``contracts/``) with the three rung states separate. ``NOT_IDENTIFIED`` is a valid
output and never a rejection of the feature. Intervening on a network (permuting a
latent, ablating a branch) is not handled here and is never called causal.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import ps3c_graph as graph
from . import ps3c_stats as st

IDENTIFIED = "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS"
NOT_IDENTIFIED = "NOT_IDENTIFIED"
NOT_EVALUATED = "NOT_EVALUATED"
CF_STATE = "COUNTERFACTUAL_UNDER_DECLARED_SCM"
CF_LABEL = "SAME_EPISODE_COUNTERFACTUAL_UNDER_DECLARED_SCM"
REQUIRED_ASSUMPTIONS = (
    "CONSISTENCY",
    "NO_INTERFERENCE_BETWEEN_EPISODES",
    "CAUSAL_SUFFICIENCY_OF_DECLARED_DAG",
    "TEMPORAL_ORDER_W_BEFORE_A_BEFORE_Y",
)
ESTIMATOR_LIBRARY = {"library": "causal_inference_provider.ps3c (numpy)", "version": "0.1.0"}
SCHEMA_PATH = Path(__file__).with_name("contracts") / "causal_dossier.v1.schema.json"
SCHEMA_SOURCE = {"repository": "predictor", "branch": "satoshi/c-contracts-20261001",
                 "revision": "f509955fb827b32c012382d7031fc7716b712004",
                 "upstream_sha256": "23f9705c2b44e1d08665a9d5da9e30162381febb0e82f788448c9338ab62a158",
                 "local_extension": "asset_appearance branch CONTRACTED_BUSINESS_CONTRACT (lane A EURUSD contract)"}


class CounterfactualRefusal(ValueError):
    """A rung-3 question this module refuses by name (the code is the message's first token)."""


# ------------------------------------------------------------------------------------------- data plumbing


def _columns(episodes):
    """Normalise records / DataFrame / dict-of-arrays into dict[str, np.ndarray] and n."""
    if hasattr(episodes, "to_dict") and hasattr(episodes, "columns"):
        cols = {str(c): episodes[c].to_numpy() for c in episodes.columns}
    elif isinstance(episodes, dict):
        cols = {k: np.asarray(v) for k, v in episodes.items()}
    else:
        rows = list(episodes)
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        cols = {k: np.array([r.get(k) for r in rows], dtype=object) for k in keys}
    n = len(next(iter(cols.values()))) if cols else 0
    return cols, n


def estimand_population_identity(episode_ids, *, treatment, outcome, contrast, treatment_kind):
    """Stable identity for a declared estimand on an exact ordered episode population."""
    ids = [str(x) for x in episode_ids]
    population_sha = hashlib.sha256(json.dumps(ids, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    estimand = {"treatment": str(treatment), "outcome": str(outcome),
                "contrast": [float(contrast[0]), float(contrast[1])],
                "treatment_kind": str(treatment_kind), "population_sha256": population_sha}
    estimand_sha = hashlib.sha256(json.dumps(estimand, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"population_n": len(ids), "population_sha256": population_sha, "estimand_id": estimand_sha}


def _num(cols, name):
    v = cols[name]
    try:
        return np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return np.array([np.nan if x is None else float(x) for x in v], dtype=float)


def _resolve(node_columns, nodes):
    out = []
    for n in nodes:
        out.extend(node_columns.get(n, [n]) if node_columns else [n])
    return out


def _complete(cols, names):
    mask = np.ones(len(next(iter(cols.values()))), dtype=bool)
    for name in names:
        mask &= np.isfinite(_num(cols, name))
    return mask


def _order(cols, time_key, n):
    if not time_key or time_key not in cols:
        return np.arange(n)
    raw = np.asarray(cols[time_key])
    if np.issubdtype(raw.dtype, np.number):
        return np.argsort(raw, kind="mergesort")
    keys = np.array([np.datetime64(str(x).replace("+00:00", "").replace("Z", ""), "ns") for x in raw])
    return np.argsort(keys, kind="mergesort")


def _treatment_kind(a, declared="AUTO"):
    if declared != "AUTO":
        return declared
    vals = set(np.unique(a[np.isfinite(a)]).tolist())
    return "BINARY" if vals <= {0.0, 1.0} else "CONTINUOUS"


# ------------------------------------------------------------------------------------------------- rung 1


def rung1_association(episodes, *, treatment, outcome, history=(), folds=5, purge=0, permutations=200,
                      seed=1729, min_events=30, placebo_outcomes=(), negative_controls=(), regime=None,
                      time_key=None, explicit_folds=None, stride=1):
    """Association and predictive relevance of A for Y given H (never a causal statement).

    Returns a ``rung1`` block of the dossier plus ``diagnostics`` (stripped by ``dossier``).
    ``explicit_folds``: list of (train_idx, test_idx) over the INPUT rows (e.g. the purged
    forward-chaining inner folds of the business contract); input rows must then already be in
    time order and ``time_key`` is ignored. ``stride`` thins the permutation statistic only.
    """
    cols, n = _columns(episodes)
    if explicit_folds is not None:
        time_key = None
    order = _order(cols, time_key, n)
    cols = {k: v[order] for k, v in cols.items()}
    history = list(history)
    need = [treatment, outcome, *history]
    missing = [c for c in need if c not in cols]
    if missing:
        return {"state": NOT_EVALUATED, "evidence": [], "conditioning_set": history,
                "diagnostics": {"reason": "MISSING_COLUMNS", "missing": missing}}
    mask = _complete(cols, need)
    a, y = _num(cols, treatment)[mask], _num(cols, outcome)[mask]
    h = st.as_matrix([_num(cols, c)[mask] for c in history], int(mask.sum()))
    m = len(a)
    pos = np.full(n, -1)
    pos[np.where(mask)[0]] = np.arange(m)
    block = {"estimators": ["pearson", "spearman", "partial_corr_given_H(circular-shift null)",
                            "oof_mse_gain_ols_ridge(chronological, purged)"],
             "effective_n": int(m), "conditioning_set": history,
             "multiplicity": {"family": "batch:(subject,target,horizon)", "n_tests": 1,
                              "permutations": int(permutations), "p_floor": 1.0 / (permutations + 1),
                              "correction": "benjamini_hochberg_at_batch_level"}}
    if m < min_events:
        block.update(state="TOO_FEW_EVENTS", evidence=[], diagnostics={"n": int(m), "min_events": min_events})
        return block
    if np.std(a) == 0 or np.std(y) == 0:
        block.update(state="ZERO_VARIANCE", evidence=[], diagnostics={"n": int(m)})
        return block
    rng = np.random.default_rng(seed)
    evidence = []
    r = st.corr(a, y)
    evidence.append({"measure": "pearson_r", "value": r, "n": int(m), "p": None, "q": None, "regime": None,
                     "signed_direction_stable": None})
    evidence.append({"measure": "spearman_rho", "value": st.spearman(a, y), "n": int(m), "p": None, "q": None,
                     "regime": None, "signed_direction_stable": None})

    def partial(av, yv, hv):
        ra, ry = st.residualize(av, hv), st.residualize(yv, hv)
        stat = st.corr(ra, ry)
        if stride > 1:
            ra, ry = ra[::stride], ry[::stride]
        if stat is None:
            return None, None
        k = len(ra)
        lo = max(1, k // 10)
        b = 0
        for _ in range(permutations):
            shift = int(rng.integers(lo, max(k - lo, lo + 1)))
            ps = st.corr(np.roll(ra, shift), ry)
            if ps is not None and abs(ps) >= abs(stat):
                b += 1
        return stat, (b + 1) / (permutations + 1)

    stat, p = partial(a, y, h)
    diagnostics = {"partial_corr": stat, "partial_p": p}
    # out-of-fold gain on chronological folds
    gains, signs = [], []
    if explicit_folds is not None:
        fold_list = []
        for tr0, te0 in explicit_folds:
            tr1, te1 = pos[np.asarray(tr0, dtype=int)], pos[np.asarray(te0, dtype=int)]
            tr1, te1 = tr1[tr1 >= 0], te1[te1 >= 0]
            if len(tr1) >= max(20, h.shape[1] + 5) and len(te1) >= 5:
                fold_list.append((tr1, te1))
    else:
        fold_list = st.chrono_folds(m, k=folds, purge=purge, min_train=max(20, h.shape[1] + 5))
    for tr, te in fold_list:
        xb = st.add_const(h)
        xf = st.add_const(np.column_stack([h, a]) if h.shape[1] else a[:, None])
        fb = st.ols(xb[tr], y[tr], ridge=1e-6)
        ff = st.ols(xf[tr], y[tr], ridge=1e-6)
        mse_b = float(np.mean((y[te] - xb[te] @ fb["beta"]) ** 2))
        mse_f = float(np.mean((y[te] - xf[te] @ ff["beta"]) ** 2))
        gains.append(0.0 if mse_b == 0 else 1.0 - mse_f / mse_b)
        signs.append(float(np.sign(ff["beta"][-1])))
    stable = (len(set(signs)) == 1) if signs else None
    if stat is not None:
        evidence.append({"measure": "partial_corr_given_H", "value": stat, "n": int(m), "p": p, "q": None,
                         "regime": None, "signed_direction_stable": stable})
    if gains:
        evidence.append({"measure": "oof_relative_mse_gain_of_A_over_H", "value": float(np.mean(gains)),
                         "n": int(m), "p": None, "q": None, "regime": None, "signed_direction_stable": stable})
    diagnostics.update(oof_gains=gains, oof_positive_folds=int(sum(g > 0 for g in gains)), oof_folds=len(gains),
                       oof_signs=signs)
    # temporal placebos and negative controls: A must not "explain" what was fixed before t
    plac = []
    for kind, names in (("temporal_placebo", placebo_outcomes), ("negative_control", negative_controls)):
        for name in names:
            if name not in cols:
                continue
            mk = _complete(cols, [treatment, name, *history])
            if mk.sum() < min_events:
                continue
            pa, py = _num(cols, treatment)[mk], _num(cols, name)[mk]
            ph = st.as_matrix([_num(cols, c)[mk] for c in history], int(mk.sum()))
            ps, pp = partial(pa, py, ph)
            if ps is None:
                continue
            evidence.append({"measure": f"{kind}:{name}:partial_corr_given_H", "value": ps, "n": int(mk.sum()),
                             "p": pp, "q": None, "regime": None, "signed_direction_stable": None})
            plac.append({"name": f"{kind}:{name}", "value": ps, "p": pp})
    n_pl = max(len(plac), 1)
    placebo_state = "NOT_RUN" if not plac else (
        "PASSED" if all(x["p"] > 0.05 / n_pl for x in plac) else "FAILED")
    diagnostics.update(placebos=plac, placebo_state=placebo_state)
    if regime and regime in cols:
        reg = np.asarray(cols[regime], dtype=object)[mask]
        for lab in sorted({str(x) for x in reg}):
            mk = np.array([str(x) == lab for x in reg])
            if mk.sum() < min_events:
                continue
            rs, rp = partial(a[mk], y[mk], h[mk])
            if rs is not None:
                evidence.append({"measure": "partial_corr_given_H", "value": rs, "n": int(mk.sum()), "p": rp,
                                 "q": None, "regime": lab, "signed_direction_stable": None})
    for e in evidence:
        if e["value"] is None or not math.isfinite(e["value"]):
            e["value"] = 0.0
            e["measure"] += ":undefined"
    block.update(state="ASSOCIATION_REPORTED", evidence=evidence, diagnostics=diagnostics)
    return block


# ------------------------------------------------------------------------------------------------- rung 2


def _gcomp_continuous(a, y, w, contrast):
    x = st.add_const(np.column_stack([a, w]) if w.shape[1] else a[:, None])
    fit = st.ols(x, y)
    beta_a, se_a = float(fit["beta"][1]), float(fit["se"][1])
    d = contrast[0] - contrast[1]
    return beta_a * d, se_a * abs(d), beta_a / se_a if se_a > 0 else 0.0, fit["df"]


def _dml_continuous(a, y, w, contrast):
    if w.shape[1] == 0:
        ra, ry = a - a.mean(), y - y.mean()
    else:
        x = st.add_const(w)
        ra = a - st.crossfit_predict(x, a)
        ry = y - st.crossfit_predict(x, y)
    den = float(np.sum(ra * ra))
    if den == 0:
        return None, ra
    return float(np.sum(ra * ry) / den) * (contrast[0] - contrast[1]), ra


def _crossfit_arm_means(t, y, w, k=5):
    n = len(y)
    x = st.add_const(st.standardize(w))  # the tiny ridge must not depend on the covariates' units
    mu1, mu0 = np.empty(n), np.empty(n)
    edges = np.linspace(0, n, k + 1).astype(int)
    for j in range(k):
        te = np.arange(edges[j], edges[j + 1])
        tr = np.setdiff1d(np.arange(n), te)
        for arm, out in ((1, mu1), (0, mu0)):
            rows = tr[t[tr] == arm]
            if len(rows) < x.shape[1] + 1:
                rows = np.where(t == arm)[0]
            out[te] = x[te] @ st.ols(x[rows], y[rows], ridge=1e-6)["beta"]
    return mu1, mu0


def _aipw(t, y, w, e):
    mu1, mu0 = _crossfit_arm_means(t, y, w)
    psi_ate = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    pt = t.mean()
    psi_att = (t * (y - mu0) - (1 - t) * e / (1 - e) * (y - mu0)) / pt
    return psi_ate, psi_att


def _match_att(t, y, e):
    lg = np.log(e / (1 - e))
    cal = 0.2 * float(np.std(lg)) if np.std(lg) > 0 else np.inf
    ctrl = np.where(t == 0)[0]
    diffs, unmatched = [], 0
    for i in np.where(t == 1)[0]:
        j = ctrl[np.argmin(np.abs(lg[ctrl] - lg[i]))]
        if abs(lg[j] - lg[i]) > cal:
            unmatched += 1
            continue
        diffs.append(y[i] - y[j])
    return (float(np.mean(diffs)) if diffs else None), unmatched


def _boot_mean_se(psi, rng, reps=200):
    n = len(psi)
    vals = [float(np.mean(psi[st.block_indices(n, rng)])) for _ in range(reps)]
    return float(np.std(vals)), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def rung2_effect(episodes, *, treatment, outcome, adjustment, contrast, dag, support=None, node_columns=None,
                 treatment_kind="AUTO", context=None, assumptions=None, assumption_evidence=None,
                 placebo_outcomes=(), negative_controls=(),
                 placebo_episodes=None, modifiers=(), time_key=None, history=None, n_boot=200, seed=1729,
                 estimand=None, treatment_node=None, outcome_node=None):
    """Effect of a historically observed intervention, identified only under declared assumptions.

    ``adjustment`` lists DAG nodes (resolved to columns through ``node_columns``). ``contrast`` is
    (a, a0). ``support`` keys: ``min_episodes_per_side`` (20), ``quantiles`` ((0.01, 0.99)),
    ``bandwidth_sd`` (0.5), ``residual_variance_floor`` (0.05), ``propensity_bounds`` ((0.05, 0.95)),
    ``restrict_to_overlap`` is forbidden (PS3-C does not trim), ``balance_bound`` (0.1).
    ``assumption_evidence`` names evidence references; a boolean declaration alone is not evidence.
    ``context`` may carry ``publication_clock`` and ``expectation_kind``.
    """
    support = dict(support or {})
    min_side = int(support.get("min_episodes_per_side", 20))
    q_lo, q_hi = support.get("quantiles", (0.01, 0.99))
    rng = np.random.default_rng(seed)
    cols, n = _columns(episodes)
    order = _order(cols, time_key, n)
    cols = {k: v[order] for k, v in cols.items()}
    rung1 = rung1_association(cols, treatment=treatment, outcome=outcome,
                              history=history if history is not None else (
                                  [c for c in _resolve(node_columns, adjustment or []) if c in cols]),
                              seed=seed, permutations=min(200, max(n_boot, 50)),
                              placebo_outcomes=placebo_outcomes, negative_controls=negative_controls)
    reasons = []
    balance_bound = float(support.get("balance_bound", 0.1))
    if not math.isfinite(balance_bound) or balance_bound < 0 or balance_bound > 0.1:
        reasons.append("BALANCE_BOUND_EXCEEDS_SPEC")
        balance_bound = min(max(balance_bound, 0.0), 0.1) if math.isfinite(balance_bound) else 0.1
    ctx = dict(context or {})
    if ctx.get("publication_clock") == "ASSUMED_SCHEDULED_PUBLICATION":
        reasons.append("ASSUMED_PUBLICATION_CLOCK")
    if ctx.get("expectation_kind") == "MODEL_BASED_EXPECTATION":
        reasons.append("EXPECTATION_IS_MODEL_BASED")
    if ctx.get("expectation_kind") == "NONE":
        reasons.append("NO_EXPECTATION")
    declared = {} if assumptions is None else dict(assumptions)
    evidence = {} if assumption_evidence is None else dict(assumption_evidence)
    unverified = []
    for name in REQUIRED_ASSUMPTIONS:
        ref = evidence.get(name)
        if declared.get(name) is not True or not isinstance(ref, str) or not ref.strip():
            unverified.append(name)
            reasons.append(f"ASSUMPTION_NOT_EVIDENCED_{name}")
    out = {"state": NOT_IDENTIFIED, "reasons": reasons, "contrast": [float(contrast[0]), float(contrast[1])],
           "adjustment": None if adjustment is None else list(adjustment), "assumptions_declared": declared,
           "assumptions_evidence": {k: v for k, v in evidence.items() if isinstance(v, str)},
           "assumptions_unverified": unverified,
           "support": {"state": NOT_EVALUATED}, "placebo": {"state": "NOT_RUN", "tests": []},
           "sensitivity": {}, "estimate": None, "rung1": rung1, "diagnostics": {}}
    if dag is not None:
        out["dag"] = {"nodes": list(map(str, dag.get("nodes", []))), "edges": [list(e) for e in dag.get("edges", [])]}
    if adjustment is None:
        reasons.append("ADJUSTMENT_SET_NOT_DECLARED")
    if dag is None:
        reasons.append("DAG_NOT_DECLARED")
    adj_cols = []
    if adjustment is not None and dag is not None:
        ok, bd_reasons, excluded = graph.backdoor_check(dag, treatment_node or treatment, outcome_node or outcome,
                                                        adjustment)
        reasons += bd_reasons
        out["excluded_from_adjustment"] = excluded
        adj_cols = _resolve(node_columns, adjustment)
        miss = [c for c in adj_cols if c not in cols]
        if miss:
            reasons.append("ADJUSTMENT_VARIABLE_NOT_OBSERVED")
            adj_cols = [c for c in adj_cols if c in cols]
    if treatment not in cols or outcome not in cols:
        reasons.append("MISSING_COLUMNS")
        out["reasons"] = sorted(set(reasons))
        out["estimand"] = estimand or f"E[{outcome}|do({treatment}=a)]-E[{outcome}|do({treatment}=a0)]"
        return out
    mask = _complete(cols, [treatment, outcome, *adj_cols])
    a_all, y_all = _num(cols, treatment), _num(cols, outcome)
    a, y = a_all[mask], y_all[mask]
    w = st.as_matrix([_num(cols, c)[mask] for c in adj_cols], int(mask.sum()))
    kind = _treatment_kind(a, treatment_kind)
    out["diagnostics"]["treatment_kind"] = kind
    out["diagnostics"]["rows_dropped_incomplete"] = int((~mask).sum())
    ids = cols.get("episode_id", np.arange(n).astype(str))[order][mask]
    out["population"] = estimand_population_identity(ids, treatment=treatment, outcome=outcome,
                                                     contrast=contrast, treatment_kind=kind)
    a0, a1 = float(contrast[1]), float(contrast[0])
    if kind == "BINARY":
        out["estimand"] = estimand or f"ATE: E[{outcome}|do({treatment}=1)]-E[{outcome}|do({treatment}=0)]"
    else:
        out["estimand"] = estimand or (f"E[{outcome}|do({treatment}={a1:g})]-E[{outcome}|do({treatment}={a0:g})]"
                                       f" averaged over the episode population")
    sup = {}
    est = None
    if len(a) == 0:
        reasons += ["EMPTY_TREATMENT_STRATUM", "NO_COMMON_SUPPORT"]
        sup = {"state": "NO_COMMON_SUPPORT", "n_per_side": [0, 0]}
    elif kind == "CONTINUOUS":
        lo, hi = np.quantile(a, [q_lo, q_hi])
        bw = float(support.get("bandwidth_sd", 0.5)) * float(np.std(a))
        n1 = int(np.sum(np.abs(a - a1) <= bw))
        n0 = int(np.sum(np.abs(a - a0) <= bw))
        sup = {"n_per_side": [n1, n0]}
        in_range = lo <= a1 <= hi and lo <= a0 <= hi
        if not in_range or min(n1, n0) < min_side:
            sup["state"] = "NO_COMMON_SUPPORT"
            reasons.append("NO_COMMON_SUPPORT")
            if min(n1, n0) == 0:
                reasons.append("EMPTY_TREATMENT_STRATUM")
        else:
            share = 1.0
            if w.shape[1]:
                ra = a - st.crossfit_predict(st.add_const(w), a)
                share = float(np.clip(np.var(ra) / np.var(a), 0, 1)) if np.var(a) > 0 else 0.0
            sup["residual_variance_share"] = share
            if share < float(support.get("residual_variance_floor", 0.05)):
                sup["state"] = "TREATMENT_PREDICTED_BY_CONTROLS"
                reasons.append("TREATMENT_PREDICTED_BY_CONTROLS")
            else:
                sup["state"] = "SUPPORTED"
            side = a >= (a1 + a0) / 2.0
            smds = [st.smd(w[:, j], side) for j in range(w.shape[1])]
            smds = [s for s in smds if s is not None]
            if smds:
                sup["balance_max_smd"] = float(max(smds))
        if sup["state"] == "SUPPORTED":
            g, g_se, t_a, df = _gcomp_continuous(a, y, w, (a1, a0))
            dml, _ = _dml_continuous(a, y, w, (a1, a0))
            boots = []
            for _ in range(n_boot):
                idx = st.block_indices(len(a), rng)
                boots.append(_gcomp_continuous(a[idx], y[idx], w[idx], (a1, a0))[0])
            interval = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            est = {"value": float(g), "interval": interval, "unit": f"{outcome} per contrast {a1:g} vs {a0:g}",
                   "uncertainty": f"moving-block bootstrap over time-ordered episodes, B={n_boot}, 95% percentile"}
            out["estimator"] = {"name": "g-computation (OLS outcome model on A and the adjustment set)",
                                **ESTIMATOR_LIBRARY}
            out["sensitivity"].update({
                "estimate_partialling_out_dml": dml,
                "gcomp_hc1_se": g_se,
                "robustness_value_q1": st.robustness_value(t_a, df, 1.0),
                "robustness_value_q1_alpha05": st.robustness_value(t_a, df, 1.0, alpha=0.05),
            })
    else:  # BINARY
        if contrast[0] != 1 or contrast[1] != 0:
            reasons.append("BINARY_CONTRAST_MUST_BE_1_VS_0")
        t = (a == 1).astype(float)
        n1, n0 = int(t.sum()), int((1 - t).sum())
        sup = {"n_per_side": [n1, n0]}
        if min(n1, n0) == 0:
            reasons += ["EMPTY_TREATMENT_STRATUM", "NO_COMMON_SUPPORT"]
            sup["state"] = "NO_COMMON_SUPPORT"
        elif min(n1, n0) < min_side:
            reasons.append("NO_COMMON_SUPPORT")
            sup["state"] = "NO_COMMON_SUPPORT"
        else:
            # the propensity logistic is ridge-penalised: fit it on standardised W so a covariate measured in
            # small units (a log return ~1e-3) is not shrunk to zero by the penalty (scale invariance, tested)
            x = st.add_const(st.standardize(w))
            e = st.crossfit_predict(x, t, logistic_model=True) if w.shape[1] else np.full(len(t), t.mean())
            requested_pb = tuple(map(float, support.get("propensity_bounds", (0.05, 0.95))))
            if len(requested_pb) != 2 or requested_pb[0] >= requested_pb[1]:
                reasons.append("INVALID_PROPENSITY_BOUNDS")
                requested_pb = (0.05, 0.95)
            if requested_pb[0] < 0.05 or requested_pb[1] > 0.95:
                reasons.append("PROPENSITY_BOUNDS_EXCEED_SPEC")
            pb = (max(0.05, requested_pb[0]), min(0.95, requested_pb[1]))
            sup["propensity_bounds"] = [float(pb[0]), float(pb[1])]
            sup["propensity_range"] = [float(e.min()), float(e.max())]
            inside = (e >= pb[0]) & (e <= pb[1])
            restrict = bool(support.get("restrict_to_overlap", False))
            sup["n_population"] = int(len(t))
            if not inside.all():
                sup["state"] = "OVERLAP_SCREEN_FAILED"
                reasons.append("OVERLAP_SCREEN_FAILED")
            elif restrict:
                sup["state"] = "OVERLAP_SCREEN_FAILED"
                reasons.append("TRIMMING_FORBIDDEN")
            else:
                sup["state"] = "SUPPORTED"
            if sup["state"] == "SUPPORTED":
                ipw = np.where(t == 1, 1 / e, 1 / (1 - e))
                smds = [st.smd(w[:, j], t, ipw) for j in range(w.shape[1])]
                smds = [s for s in smds if s is not None]
                if smds:
                    sup["balance_max_smd"] = float(max(smds))
                sup["effective_sample_size"] = float(ipw.sum() ** 2 / np.sum(ipw ** 2))
                psi_ate, psi_att = _aipw(t, y, w, e)
                se, interval = _boot_mean_se(psi_ate, rng, n_boot)
                xg = st.add_const(np.column_stack([t, w]) if w.shape[1] else t[:, None])
                gfit = st.ols(xg, y)
                tg = float(gfit["beta"][1] / gfit["se"][1]) if gfit["se"][1] > 0 else 0.0
                att_match, unmatched = _match_att(t, y, e)
                est = {"value": float(np.mean(psi_ate)), "interval": [interval[0], interval[1]],
                       "unit": f"{outcome}, treated (A=1) minus control (A=0)",
                       "uncertainty": f"moving-block bootstrap of cross-fitted AIPW influence values, B={n_boot}"}
                out["estimator"] = {"name": "cross-fitted AIPW (logistic propensity, per-arm OLS outcome)",
                                    **ESTIMATOR_LIBRARY}
                out["sensitivity"].update({
                    "att_aipw": float(np.mean(psi_att)),
                    "att_propensity_matching_1nn_caliper02": att_match,
                    "matching_unmatched_treated": unmatched,
                    "ate_gcomputation_ols": float(gfit["beta"][1]),
                    "robustness_value_q1": st.robustness_value(tg, gfit["df"], 1.0),
                    "robustness_value_q1_alpha05": st.robustness_value(tg, gfit["df"], 1.0, alpha=0.05),
                })
    sup["balance_bound"] = balance_bound
    if sup.get("balance_max_smd") is not None and sup["balance_max_smd"] > balance_bound:
        reasons.append("IMBALANCE")
        sup["state"] = "IMBALANCE"
        out["sensitivity"]["balance_note"] = "IMBALANCE: max SMD above declared identification bound"
    out["support"] = sup

    # CATE by declared pre-event modifiers (within support only)
    if est is not None and modifiers:
        het = []
        for mod in modifiers:
            if mod not in cols:
                continue
            mv = _num(cols, mod)[mask]
            if kind == "BINARY" and "population_restricted_to_overlap_dropped" in out["sensitivity"]:
                continue
            cuts = np.nanquantile(mv, [1 / 3, 2 / 3])
            for lab, mk in (("low", mv <= cuts[0]), ("mid", (mv > cuts[0]) & (mv <= cuts[1])), ("high", mv > cuts[1])):
                if kind == "CONTINUOUS":
                    bw = 0.5 * float(np.std(a))
                    if min(np.sum(np.abs(a[mk] - a1) <= bw), np.sum(np.abs(a[mk] - a0) <= bw)) < min_side:
                        het.append({"modifier": mod, "stratum": lab, "state": "NO_COMMON_SUPPORT"})
                        continue
                    v = _gcomp_continuous(a[mk], y[mk], w[mk], (a1, a0))
                    het.append({"modifier": mod, "stratum": lab, "cut_points": [float(c) for c in cuts],
                                "value": float(v[0]), "hc1_se": float(v[1]), "n": int(mk.sum())})
                else:
                    tt = t[mk]
                    if min(tt.sum(), (1 - tt).sum()) < min_side:
                        het.append({"modifier": mod, "stratum": lab, "state": "NO_COMMON_SUPPORT"})
                        continue
                    xg = st.add_const(np.column_stack([tt, w[mk]]) if w.shape[1] else tt[:, None])
                    gf = st.ols(xg, y[mk])
                    het.append({"modifier": mod, "stratum": lab, "cut_points": [float(c) for c in cuts],
                                "value": float(gf["beta"][1]), "hc1_se": float(gf["se"][1]), "n": int(mk.sum())})
        est["heterogeneity"] = het

    # placebo / refutation battery (can refute assumptions, cannot prove them)
    if est is not None:
        tests = []

        def effect_on(yv, av, wv):
            if kind == "CONTINUOUS":
                g, se, _, _ = _gcomp_continuous(av, yv, wv, (a1, a0))
                return g, se
            xg = st.add_const(np.column_stack([av, wv]) if wv.shape[1] else av[:, None])
            f = st.ols(xg, yv)
            return float(f["beta"][1]), float(f["se"][1])

        base, base_se = effect_on(y, a, w)
        n_named = len(placebo_outcomes) + len(negative_controls) + (1 if placebo_episodes is not None else 0)
        from statistics import NormalDist

        zc = NormalDist().inv_cdf(1 - 0.025 / max(n_named, 1))
        # placebo treatment: A circularly shifted (keeps its autocorrelation, breaks its link to Y);
        # under a valid design about 5% of shifts exceed |z|>1.96; FAIL when more than 25% do.
        zs = []
        for _ in range(20):
            shift = int(rng.integers(max(1, len(a) // 10), max(2, 9 * len(a) // 10)))
            pv, pse = effect_on(y, np.roll(a, shift), w)
            zs.append(abs(pv) / pse if pse > 0 else 0.0)
        share = float(np.mean(np.array(zs) > 1.96))
        out["sensitivity"]["placebo_shift_share_abs_z_gt_1_96"] = share
        tests.append({"name": "placebo_treatment_circular_shift_x20", "verdict": "PASS" if share <= 0.25 else "FAIL",
                      "n": int(len(a))})
        rc = rng.standard_normal(len(a))
        rv, _ = effect_on(y, a, np.column_stack([w, rc]) if w.shape[1] else rc[:, None])
        tests.append({"name": "random_common_cause", "verdict": "PASS" if abs(rv - base) <= base_se else "FAIL",
                      "n": int(len(a))})
        subs = []
        for _ in range(20):
            k = int(0.8 * len(a))
            s0 = int(rng.integers(0, len(a) - k + 1))
            idx = np.arange(s0, s0 + k)
            subs.append(effect_on(y[idx], a[idx], w[idx])[0])
        tests.append({"name": "data_subset_contiguous_80pct", "verdict":
                      "PASS" if abs(float(np.mean(subs)) - base) <= max(base_se, 1e-12) else "FAIL", "n": 20})
        for kind_name, names in (("pre_event_outcome", placebo_outcomes), ("negative_control_outcome", negative_controls)):
            for name in names:
                if name not in cols:
                    continue
                pm = _num(cols, name)[mask]
                if kind == "BINARY" and "population_restricted_to_overlap_dropped" in out["sensitivity"]:
                    pm = pm[inside]
                ok = np.isfinite(pm)
                if ok.sum() < 2 * min_side:
                    tests.append({"name": f"{kind_name}:{name}", "verdict": "NOT_RUN", "n": int(ok.sum())})
                    continue
                v, s = effect_on(pm[ok], (t if kind == "BINARY" else a)[ok], w[ok])
                tests.append({"name": f"{kind_name}:{name}", "verdict": "PASS" if abs(v) <= zc * s else "FAIL",
                              "n": int(ok.sum())})
        if placebo_episodes is not None:
            pc, pn = _columns(placebo_episodes)
            if all(c in pc for c in [treatment, outcome, *adj_cols]) and pn:
                pmk = _complete(pc, [treatment, outcome, *adj_cols])
                if pmk.sum() >= 2 * min_side:
                    pw = st.as_matrix([_num(pc, c)[pmk] for c in adj_cols], int(pmk.sum()))
                    v, s = effect_on(_num(pc, outcome)[pmk], _num(pc, treatment)[pmk], pw)
                    tests.append({"name": "pseudo_event_placebo", "verdict": "PASS" if abs(v) <= zc * s else "FAIL",
                                  "n": int(pmk.sum())})
                else:
                    tests.append({"name": "pseudo_event_placebo", "verdict": "NOT_RUN", "n": int(pmk.sum())})
        verdicts = [x["verdict"] for x in tests if x["verdict"] != "NOT_RUN"]
        state = "PASSED" if verdicts and all(v == "PASS" for v in verdicts) else ("FAILED" if verdicts else "NOT_RUN")
        out["placebo"] = {"state": state, "tests": tests}
        if state == "FAILED":
            reasons.append("PLACEBO_FAILED")
        elif state == "NOT_RUN":
            reasons.append("PLACEBO_NOT_RUN")
    reasons = sorted(set(reasons))
    if not reasons and est is not None and sup.get("state") == "SUPPORTED":
        out["state"] = IDENTIFIED
        out["estimate"] = est
        out["reasons"] = []
    else:
        out["state"] = NOT_IDENTIFIED
        out["estimate"] = None
        out["reasons"] = reasons or ["NO_ESTIMATE"]
        if est is not None:
            # The number is withheld: a NOT_IDENTIFIED rung releases no effect estimate.
            out["diagnostics"]["estimate_withheld"] = True
    return out


# ------------------------------------------------------------------------------------------------- rung 3


class AdditiveSCM:
    """Declared SCM with additive noise: node = f(parents) + U_node, in an explicit topological order.

    ``mechanisms`` maps a node to a callable whose parameter names are its parents.
    Nodes in ``order`` without a mechanism are exogenous (taken from the episode).
    """

    def __init__(self, order, mechanisms, residual_sd=None, params=None, equations=None, support=None):
        self.order = list(order)
        self.mechanisms = dict(mechanisms)
        self.residual_sd = dict(residual_sd or {})
        self.params = params
        self.support = dict(support or {})
        self.parents = {}
        for node, fn in self.mechanisms.items():
            if node not in self.order:
                raise ValueError(f"mechanism for {node} not in order")
            ps = list(inspect.signature(fn).parameters)
            for p in ps:
                if p not in self.order or self.order.index(p) >= self.order.index(node):
                    raise ValueError(f"parent {p} of {node} is not earlier in the declared order")
            self.parents[node] = ps
        self.equations = equations or {n: f"{n} = f({', '.join(ps)}) + U_{n}" for n, ps in self.parents.items()}

    def dag(self):
        return {"nodes": self.order, "edges": [[p, n] for n, ps in self.parents.items() for p in ps]}

    def descendants(self, nodes):
        nodes = [nodes] if isinstance(nodes, str) else list(nodes)
        out = set()
        for x in nodes:
            out |= graph.descendants(self.dag(), x)
        return out

    def f(self, node, values):
        return float(self.mechanisms[node](**{p: values[p] for p in self.parents[node]}))

    def fit_digest(self):
        payload = {"order": self.order, "equations": self.equations,
                   "params": self.params, "residual_sd": self.residual_sd}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=float).encode()).hexdigest()


def _linear_mechanism(beta, ps):
    """A linear mechanism whose introspected signature names its parents (no eval)."""
    for p in ps:
        if not str(p).isidentifier():
            raise ValueError(f"node name {p!r} must be an identifier")

    def mechanism(**kw):
        return beta[0] + sum(b * kw[p] for b, p in zip(beta[1:], ps))

    mechanism.__signature__ = inspect.Signature(
        [inspect.Parameter(p, inspect.Parameter.KEYWORD_ONLY) for p in ps])
    return mechanism


def fit_linear_scm(episodes, *, order, parents, ridge=1e-8):
    """Fit linear additive-noise mechanisms on TRAIN episodes; parameters are JSON-serialisable."""
    cols, _ = _columns(episodes)
    mechanisms, params, sds, eqs = {}, {}, {}, {}
    for node, ps in parents.items():
        ps = list(ps)
        mk = _complete(cols, [node, *ps])
        x = st.add_const(st.as_matrix([_num(cols, p)[mk] for p in ps], int(mk.sum())))
        fit = st.ols(x, _num(cols, node)[mk], ridge=ridge)
        beta = [float(b) for b in fit["beta"]]
        params[node] = {"const": beta[0], "coef": dict(zip(ps, beta[1:])), "n": int(mk.sum())}
        sds[node] = float(np.std(fit["resid"]))
        eqs[node] = f"{node} = {beta[0]:.6g} " + " ".join(f"+ {b:.6g}*{p}" for p, b in zip(ps, beta[1:])) + f" + U_{node}"
        mechanisms[node] = _linear_mechanism(beta, ps)
    return AdditiveSCM(order, mechanisms, residual_sd=sds, params=params, equations=eqs)


def counterfactual_same_episode(scm, episode, *, intervention, mode="RETROSPECTIVE", support=None):
    """Abduction -> action -> prediction for ONE observed episode under the declared SCM.

    RETROSPECTIVE: may read the realised outcomes (after they happened) to abduct U.
    OPERATIONAL: refuses ``FUTURE_OUTCOME_IN_OPERATIONAL_CALL`` if the episode carries any
    descendant of the intervened node; otherwise returns the predictive distribution only.
    """
    support = support if support is not None else scm.support
    desc = scm.descendants(list(intervention))
    for node, val in intervention.items():
        if node in support:
            lo, hi = support[node]
            if not lo <= val <= hi:
                raise CounterfactualRefusal(f"NO_COMMON_SUPPORT: {node}={val} outside historical support [{lo}, {hi}]")
    if mode == "OPERATIONAL":
        present = [d for d in desc if episode.get(d) is not None]
        if present:
            raise CounterfactualRefusal(f"FUTURE_OUTCOME_IN_OPERATIONAL_CALL: episode carries realised {present}")
        values = {n: episode.get(n) for n in scm.order}
        values.update(intervention)
        pred = {}
        for node in scm.order:
            if node in intervention or node not in scm.mechanisms:
                continue
            values[node] = scm.f(node, values)
            sd = scm.residual_sd.get(node)
            pred[node] = {"mean": values[node], "sd": sd}
        return {"label": "OPERATIONAL_PREDICTIVE_DISTRIBUTION", "mode": mode, "prediction": pred,
                "abduction": None}
    if mode != "RETROSPECTIVE":
        raise ValueError(f"unknown mode {mode}")
    factual = {}
    for node in scm.order:
        v = episode.get(node)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            raise CounterfactualRefusal(f"ABDUCTION_NEEDS_OBSERVED_OUTCOME: {node} missing in episode "
                                        f"{episode.get('episode_id')}")
        factual[node] = float(v)
    u = {node: factual[node] - scm.f(node, factual) for node in scm.order if node in scm.mechanisms}
    cf = dict(factual)
    cf.update({k: float(v) for k, v in intervention.items()})
    model_based = {}
    for node in scm.order:
        if node in intervention or node not in scm.mechanisms or node not in desc:
            continue
        base = scm.f(node, cf)
        model_based[node] = base
        cf[node] = base + u[node]
    # factual reconstruction: the null action must return the factual episode exactly
    rec = dict(factual)
    for node in scm.order:
        if node in scm.mechanisms:
            rec[node] = scm.f(node, rec) + u[node]
    rec_err = max((abs(rec[n] - factual[n]) for n in scm.mechanisms), default=0.0)
    return {
        "label": CF_LABEL,
        "mode": mode,
        "episode_id": episode.get("episode_id"),
        "abduction": {f"U_{k}": v for k, v in u.items()},
        "action": dict(intervention),
        "factual": factual,
        "prediction": cf,
        "model_based": model_based,
        "delta": {n: factual[n] - cf[n] for n in model_based},
        "propagation": {"order": list(scm.order), "descendants": [n for n in scm.order if n in desc],
                        "kept_fixed": [n for n in scm.order if n not in desc and n not in intervention]},
        "reconstruction_max_abs_error": float(rec_err),
        "note": "estimate under the declared SCM; the counterfactual outcome was never observed",
    }


def rung3_population(episodes, *, treatment, outcome, adjustment_cols, a0, rung2_state, mediators=(),
                     placebo_outcome=None, analog_k=5, analog_band_sd=0.25, min_analogs=20, time_key=None,
                     quantiles=(0.01, 0.99), seed=1729):
    """Same-episode counterfactuals for every episode of a population, with the rung-3 checks.

    The SCM: W exogenous; M_j = g_j(A, W) + U_Mj; Y = f(A, W, M) + U_Y (linear additive noise,
    fitted on the TRAIN episodes passed). The action sets A := a0 (must lie in A's historical
    support); W and all abducted U are kept; mediators and Y are propagated.
    """
    cols, n = _columns(episodes)
    order_idx = _order(cols, time_key, n)
    cols = {k: v[order_idx] for k, v in cols.items()}
    names = [*adjustment_cols, treatment, *mediators, outcome]
    mk = _complete(cols, names)
    reasons = []
    block = {"state": NOT_IDENTIFIED, "label": "NONE", "reasons": reasons, "sensitivity": {}}
    if rung2_state != IDENTIFIED:
        reasons.append("RUNG2_NOT_IDENTIFIED")
    if mk.sum() < 2 * min_analogs:
        reasons.append("TOO_FEW_EPISODES")
        return block, []
    sub = {k: np.asarray(v)[mk] for k, v in cols.items()}
    a = _num(sub, treatment)
    lo, hi = np.quantile(a, quantiles)
    if not lo <= a0 <= hi:
        reasons.append("NO_COMMON_SUPPORT")
        return block, []
    safe = {c: f"v{i}" for i, c in enumerate(names)}
    data = {safe[c]: _num(sub, c) for c in names}
    w_s = [safe[c] for c in adjustment_cols]
    m_s = [safe[c] for c in mediators]
    a_s, y_s = safe[treatment], safe[outcome]

    def build(parents_y, med=True):
        par = {m: [a_s, *w_s] for m in (m_s if med else [])}
        par[y_s] = parents_y
        order = [*w_s, a_s, *(m_s if med else []), y_s]
        return fit_linear_scm(data, order=order, parents=par)

    scm = build([a_s, *w_s, *m_s])
    scm.support = {a_s: (float(lo), float(hi))}
    rows, recon = [], 0.0
    for i in range(int(mk.sum())):
        ep = {k: float(data[k][i]) for k in scm.order}
        ep["episode_id"] = str(sub.get("episode_id", np.arange(len(a)))[i])
        cfr = counterfactual_same_episode(scm, ep, intervention={a_s: float(a0)})
        recon = max(recon, cfr["reconstruction_max_abs_error"])
        rows.append({"episode_id": ep["episode_id"], "A": ep[a_s], "y_factual": ep[y_s],
                     "y_counterfactual": cfr["prediction"][y_s], "delta": cfr["delta"][y_s],
                     "model_based": cfr["model_based"][y_s], "u_y": cfr["abduction"][f"U_{y_s}"],
                     "mediators_cf": {c: cfr["prediction"][safe[c]] for c in mediators}})
    delta = np.array([r["delta"] for r in rows])
    ycf = np.array([r["y_counterfactual"] for r in rows])
    # historical analog pairs: episodes that actually had A near a0 with the nearest pre-event context
    rng = np.random.default_rng(seed)
    w = st.as_matrix([data[c] for c in w_s], len(a))
    wz = (w - w.mean(0)) / np.where(w.std(0) > 0, w.std(0), 1.0) if w.shape[1] else w
    band = analog_band_sd * float(np.std(a))
    pool = np.where(np.abs(a - a0) <= band)[0]
    analog_gap = []
    if len(pool) >= max(analog_k, min_analogs):
        far = np.where(np.abs(a - a0) > band)[0]
        for i in far:
            d = np.sum((wz[pool] - wz[i]) ** 2, axis=1) if w.shape[1] else rng.random(len(pool))
            nn = pool[np.argsort(d)[:analog_k]]
            analog_gap.append(ycf[i] - float(np.mean(data[y_s][nn])))
    checks = {"reconstruction_max_abs_error": recon}
    if analog_gap:
        g = np.array(analog_gap)
        se = float(np.std(g) / math.sqrt(len(g))) if len(g) > 1 else float("inf")
        z = float(np.mean(g) / se) if se > 0 else 0.0
        checks.update(analog_pairs=len(g), analog_mean_gap=float(np.mean(g)), analog_gap_z=z,
                      analog_state="CONSISTENT" if abs(z) <= 2.58 else "INCONSISTENT")
    else:
        checks.update(analog_pairs=0, analog_state="NO_ANALOGS")
    # placebos: null action returns the factual episode; A cannot move a pre-event outcome
    null = counterfactual_same_episode(scm, {**{k: float(data[k][0]) for k in scm.order}, "episode_id": "null"},
                                       intervention={a_s: float(data[a_s][0])})
    checks["null_action_max_abs_delta"] = float(max(abs(v) for v in null["delta"].values()) if null["delta"] else 0.0)
    plac_state = "PASS" if checks["null_action_max_abs_delta"] < 1e-9 else "FAIL"
    if placebo_outcome and placebo_outcome in sub:
        pv = _num(sub, placebo_outcome)
        okp = np.isfinite(pv)
        if okp.sum() >= 2 * min_analogs:
            x = st.add_const(np.column_stack([a[okp], w[okp]]) if w.shape[1] else a[okp][:, None])
            f = st.ols(x, pv[okp])
            pd_ = float(f["beta"][1]) * (a0 - a[okp])
            z = float(f["beta"][1] / f["se"][1]) if f["se"][1] > 0 else 0.0
            checks.update(placebo_outcome=placebo_outcome, placebo_mean_delta=float(np.mean(pd_)), placebo_z=z)
            if abs(z) > 2.58:
                plac_state = "FAIL"
    checks["placebo_state"] = plac_state
    # model sensitivity: alternative compatible SCMs, compared episode by episode
    def deltas(model, extra=None):
        out_d = []
        for i in range(len(a)):
            fact = {k: float(data[k][i]) for k in model.order}
            act = {**fact, a_s: float(a0), **({} if extra is None else extra(float(a0)))}
            out_d.append(model.f(y_s, fact) - model.f(y_s, act))
        return np.array(out_d)

    alts = {"declared_with_mediators": delta}
    try:
        alts["no_mediator_total_effect"] = deltas(build([a_s, *w_s], med=False))
        sq = f"{a_s}_sq"
        data[sq] = data[a_s] ** 2
        quad = fit_linear_scm(data, order=[*w_s, a_s, sq, y_s], parents={y_s: [a_s, sq, *w_s]})
        alts["quadratic_dose"] = deltas(quad, extra=lambda v: {sq: v * v})
    except (np.linalg.LinAlgError, ValueError) as trouble:  # pragma: no cover - reported, not hidden
        alts["alternative_error"] = str(trouble)
    arrs = {k: v for k, v in alts.items() if isinstance(v, np.ndarray)}
    big = np.abs(delta) > 0.1 * float(np.std(delta)) if np.std(delta) > 0 else np.zeros(len(delta), bool)
    agree = {k: float(np.mean(np.sign(v[big]) == np.sign(delta[big]))) if big.any() else 1.0 for k, v in arrs.items()}
    worst = min(agree.values())
    u = np.array([r["u_y"] for r in rows])
    het = st.corr(np.abs(u), np.abs(a - a.mean()))
    sens = {"alternatives_mean_abs_delta": json.dumps({k: float(np.mean(np.abs(v))) for k, v in arrs.items()}),
            "alternatives_sign_agreement": json.dumps(agree), "alternatives_worst_sign_agreement": worst,
            "abs_residual_vs_abs_dose_corr": het, **{k: v for k, v in checks.items() if not isinstance(v, str)},
            "analog_state": checks["analog_state"], "placebo_state": plac_state}
    signs = {0} if worst >= 0.9 else {0, 1}
    if recon > 1e-8:
        reasons.append("FACTUAL_RECONSTRUCTION_FAILED")
    if plac_state != "PASS":
        reasons.append("PLACEBO_FAILED")
    if checks["analog_state"] == "INCONSISTENT":
        reasons.append("ANALOG_PAIRS_INCONSISTENT")
    if checks["analog_state"] == "NO_ANALOGS":
        reasons.append("NO_HISTORICAL_ANALOGS")
    if len(signs) > 1:
        reasons.append("MODEL_DEPENDENT_COUNTERFACTUAL")
    inv = {v: k for k, v in safe.items()}
    far_mask = np.abs(a - a0) > band
    if far_mask.any():
        sel = far_mask
        sens["counterfactual_population"] = "episodes with |A - a0| > band (treated relative to the action)"
    else:
        sel = np.ones(len(a), dtype=bool)
        sens["counterfactual_population"] = "all episodes"
    sens["counterfactual_population_n"] = int(sel.sum())
    block["sensitivity"] = sens
    if not reasons:
        block.update(
            state=CF_STATE, label=CF_LABEL,
            scm={"order": [inv[n] for n in scm.order], "equations": {inv[k]: _rename(v, inv) for k, v in scm.equations.items()},
                 "noise": "ADDITIVE", "invertible": True, "fit_digest": scm.fit_digest(),
                 "library": "causal_inference_provider.ps3c linear additive-noise SCM (numpy)",
                 "alternatives_considered": sorted(alts)},
            abduction={"U_" + outcome: {"mean": float(np.mean(u)), "sd": float(np.std(u)), "n": float(len(u))}},
            action={treatment: float(a0)},
            prediction={"factual": float(np.mean(np.array([r["y_factual"] for r in rows])[sel])),
                        "counterfactual": float(np.mean(ycf[sel])), "delta": float(np.mean(delta[sel])),
                        "model_based": float(np.mean(np.array([r["model_based"] for r in rows])[sel])),
                        "propagated_nodes": [*mediators, outcome], "barrier_reread": None,
                        "uncertainty": {"delta_p05": float(np.percentile(delta[sel], 5)),
                                        "delta_p95": float(np.percentile(delta[sel], 95)),
                                        "residual_sd": float(np.std(u)), "n": int(sel.sum())}})
    block["reasons"] = sorted(set(reasons))
    return block, rows


def _rename(eq, inv):
    for k in sorted(inv, key=len, reverse=True):
        eq = eq.replace(k, inv[k])
    return eq


# ------------------------------------------------------------------------------------------------- dossier


_RUNG1_KEYS = {"state", "estimators", "effective_n", "conditioning_set", "multiplicity", "evidence"}
_RUNG2_KEYS = {"state", "reasons", "estimand", "contrast", "dag", "adjustment", "excluded_from_adjustment",
               "assumptions_declared", "assumptions_evidence", "assumptions_unverified", "population",
               "estimator", "support", "placebo", "sensitivity", "estimate"}
_RUNG3_KEYS = {"state", "label", "reasons", "scm", "abduction", "action", "prediction", "sensitivity"}


def _clean_sens(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v) if math.isfinite(float(v)) else None
        elif v is None or isinstance(v, str):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) if isinstance(x, (int, float)) else x for x in v]
        else:
            out[k] = json.dumps(v, default=str)
    return out


def dossier(*, dossier_id, producer_revision, subject, data_manifest, treatment, rung1, rung2=None, rung3=None,
            emission=None, limitations, produced_at=None, module="causal_inference_provider.ps3c"):
    """Assemble one ``causal_dossier.v1`` document; rung states stay separate; NOT_* never become zero."""
    r1 = {k: v for k, v in rung1.items() if k in _RUNG1_KEYS}
    r2 = {k: v for k, v in (rung2 or {"state": NOT_EVALUATED, "reasons": [], "estimand": "not evaluated"}).items()
          if k in _RUNG2_KEYS}
    if "sensitivity" in r2:
        r2["sensitivity"] = {k: v for k, v in _clean_sens(r2["sensitivity"]).items() if not isinstance(v, list)}
    r3 = {k: v for k, v in (rung3 or {"state": NOT_EVALUATED, "label": "NONE"}).items() if k in _RUNG3_KEYS}
    if "sensitivity" in r3:
        r3["sensitivity"] = _clean_sens(r3["sensitivity"])
    asset = data_manifest.get("asset_appearance", {}).get("state")
    if asset == "NOT_EXECUTABLE_NO_CONTRACTED_PRICE":
        r1 = {"state": NOT_EVALUATED}
        if r2["state"] == IDENTIFIED:
            r2 = {"state": NOT_IDENTIFIED, "reasons": ["NOT_EXECUTABLE_NO_CONTRACTED_PRICE"],
                  "estimand": r2["estimand"], "estimate": None}
        if r3["state"] == CF_STATE:
            r3 = {"state": NOT_IDENTIFIED, "label": "NONE", "reasons": ["NOT_EXECUTABLE_NO_CONTRACTED_PRICE"]}
    if r2["state"] != IDENTIFIED and r3["state"] == CF_STATE:
        raise ValueError("rung 3 cannot be a counterfactual under a declared SCM when rung 2 is not identified")
    if r2["state"] != IDENTIFIED and r2.get("estimate") is not None:
        raise ValueError("a rung-2 state other than identified must not carry an estimate")
    if r3["state"] == CF_STATE:
        level = "COUNTERFACTUAL_SENSITIVITY"
    elif r2["state"] == IDENTIFIED:
        level = "IDENTIFIED_EFFECT"
    elif r1["state"] == "ASSOCIATION_REPORTED":
        level = "ASSOCIATION"
    else:
        level = "NONE"
    if asset == "NOT_EXECUTABLE_NO_CONTRACTED_PRICE":
        level = "NONE"
    doc = {
        "schema": "causal_dossier.v1",
        "dossier_id": dossier_id,
        "produced_at": produced_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "producer": {"repository": "causal-inference", "revision": producer_revision, "module": module},
        "subject": subject,
        "data_manifest": data_manifest,
        "treatment": treatment,
        "rung1": r1,
        "rung2": r2,
        "rung3": r3,
        "emission": emission or {"operational_use": "RETROSPECTIVE_ONLY", "emittable_from": {}},
        "selection": {"causal_evidence_level": level, "cf_eligible": r3["state"] == CF_STATE,
                      "reason_code": "NOT_IDENTIFIED_IS_NOT_REJECTION" if r2["state"] != IDENTIFIED else "EVIDENCE_RECORDED"},
        "limitations": list(limitations) or ["none stated"],
    }
    return json.loads(json.dumps(doc, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def validate_dossier(doc):
    """Validate against the vendored schema; returns the list of error messages (empty when valid)."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - environment without jsonschema
        return ["JSONSCHEMA_NOT_INSTALLED"]
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in validator.iter_errors(doc)]
