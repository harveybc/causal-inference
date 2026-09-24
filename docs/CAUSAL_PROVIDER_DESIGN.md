# Minimal causal provider: test design and traceability

## Discovery and requirements (S0-S3)

Actor: an offline analyst using an M5PHET chatbot, with an explicit numeric
dataset, outcome, binary treatment, pre-treatment adjustment set, estimand and
identifying assumptions. Text alone is never causal evidence. No financial
conclusion, broker, GPU, real holdout, or training sweep is in scope.

The original checkout has exploratory changes. All implementation belongs to the
isolated `feat/m5phet-causal-provider` worktree based on `1bd7669`.

| Requirement | Observable acceptance / tests |
| --- | --- |
| R1: real numeric engine | `test_known_effect_and_actual_uncertainty`, `test_matches_real_library`: confounded IID fixture, ATE 2, biased unadjusted difference, proper library ATE CI, CPU <30 s |
| R2: no implicit identification | `test_no_implicit_assumptions`, missing assumptions, overlap refusals; no numeric payload |
| R3: explicit lifecycle | entry-point discovery, load/fit/infer states, forbid fit during infer, reload invalidation |
| R4: defensive input handling | bad data, role collisions, nonfinite values, unsupported treatment, resource limits |
| R5: reproducibility and behavior | negative control, negative effect, replay, units scaling, wider CI, mutation isolation |
| R6: usable independent installation | clean venv, wheel install, pip check, real CLI/example outside source tree |
| R7: honest integration boundary | handoff doc, no M5PHET edits or claim of web/chat acceptance |

## Architecture and component design (S4-S8)

Existing `causal_regime_analysis.py` uses DoWhy and EconML CausalForestDML,
but auto-selects controls and allows unidentified estimates. Reuse the EconML
engine family, not those application-specific assumptions. Choose LinearDML
with X=None and declared constant treatment effect; hence its single effect
equals the population ATE. Do not label a propensity-weighted heterogeneous
effect as an ATE. Nuisances: standardized linear regression for E[Y|W],
standardized logistic regression for P[T=1|W]; explicitly require correct
specification. Three shuffled stratified folds, fixed seed 1729; HC1 covariance
through EconML StatsModelsInference. No bespoke estimator or averaged pointwise
confidence bounds. No post-treatment variables or automatic variable selection.

All eight assumptions in the test config must be literally true. This is
conditional identification supplied by the caller, NOT empirical certification.
The declaration of sufficient adjustment includes conditional exchangeability
and exclusion of colliders. Pre-treatment status alone is insufficient.
IID sampling is mandatory; temporal/clustered data are unsupported.

Provider is a separate `provider/` distribution, avoiding the legacy root
rl-optimizer package, generic `app` namespace and unrelated GPU/RL dependencies.
No dependency on M5PHET. Entry point resolves to a no-argument class. Config and
data are separate; loaded config contains no fitted state. `fit(data)` is an
explicit bounded offline job. `infer()` only reads a detached result from that
job; no new dataset/estimand, implicit training, or external state references.
State is in-memory and returned by defensive copy; load or failed fit clears
old results. No pickle deserialization. Consumers serialize results, not models.

Preflight rejects nonnumeric/missing/nonfinite selected columns, missing roles,
bad treatment levels, ambiguous labels, rank-deficient adjustments, too few
samples per arm, and resource excess. Cross-fitted propensity diagnostics must
lie in [0.05, 0.95]; never clip, trim, or silently change the target population.
This conservative screen cannot prove positivity or absence of confounding.
Bounds: 100 to 10,000 rows, at most 20 adjustments, one CPU numerical thread.

Unit tests exercise validation and invalidation; integration tests discover the
installed entry point and compare to independent direct EconML calls; system
tests exercise the actual CLI and wheel; alpha tests use only synthetic IID
data. Structural mocks only forbid hidden fitting, never provide numeric
acceptance. Public-data confirmation and domain revalidation are deferred.

## Main integration coordination

Read-only inspection of M5PHET docs/INTERFACES.md on 2026-09-24 found the
general provider contract SPECIFIED, NOT_IMPLEMENTED, explicitly forbidding fit
inside infer. This provider follows that boundary and documents its local v1
contract rather than inventing compatibility with an unimplemented validator.
The main integrator must dispatch an explicit fit job before infer; do not route
an infer-only chat request into training. There is no agent messaging channel
available in this session; the handoff is an artifact and conversation report,
not a claim of agreement. No prompt interpreter is included.

## Primary engine references

- https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html
- https://www.pywhy.org/EconML/spec/estimation/dml.html

Consulted 2026-09-24. Dependency version is pinned to EconML 0.16.0.

## Chat handoff update (2026-09-24)

Main now has a runtime in its separate m5phet-chat worktree, discovered after
the integration update. It calls load(state_ref), infer(request, state).
Add a separate entry-point adapter to the local core. Advertise infer only;
fit stays an explicit CLI operation before service startup. Store a JSON fitted
study result (no executable pickle) under a content-addressed state ref, with
config, row identity, engine versions and availability clock. Inference is a
report of that fitted analysis, not a new-population prediction. Chat data is a
small exact population reference; new raw datasets require the explicit fit CLI.

`chat_request(prompt, data, config)` returns draft2 infer requests with
output_schema.targets=[effect]; parameters must contain the entire identifying
config. Only the documented report prompt is accepted; it cannot override
parameters. `chat_examples()` lists only explicitly prepared development
artifacts, never real user datasets or studies. Missing artifacts return no
examples and are never built on discovery/request. File hashes are local
integrity checks, not signatures or governance approval. The state directory is
trusted local operator storage.

R8 acceptance: test_chat.py covers installed discovery, actual prefit CLI,
immutable inference, absent demos, no automatic fitting, clock/config/population
binding, tamper refusal and prompt limits. These tests precede adapter code.

R9 (main environment update): the base adapter must serve without importing any
scientific fit dependency, and must enforce an administrator state allowlist
before reading JSON. Main's newer NumPy/sklearn versions are incompatible with
the chosen EconML fitting stack. Keep dependencies in optional [fit], run the
fit CLI in its own venv, and serve the completed result in main unchanged. Tests
for dependency import refusal and allowlist/symlink handling were captured red
before this change. No new analyzer operation is needed for fitted-study reports;
new analyses remain an explicit CLI job, not an infer side effect.
