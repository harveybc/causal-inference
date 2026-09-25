# Causal provider handoff

## Install and prepare before starting the service

From the root of this causal-inference worktree, using Python 3.12:

```bash
python3.12 -m venv .venv-causal
.venv-causal/bin/python -m pip install -c provider/constraints-cpu-py312.txt './provider[fit,test]'
.venv-causal/bin/python -m pip check
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
.venv-causal/bin/python -m pytest provider/tests -q
export CAUSAL_INFERENCE_STATE_DIR="$PWD/.causal-studies"
.venv-causal/bin/python -m causal_inference_provider prepare-demo --state-dir "$CAUSAL_INFERENCE_STATE_DIR"
.venv-causal/bin/python -m causal_inference_provider prepare-demo --with-modifier --state-dir "$CAUSAL_INFERENCE_STATE_DIR"
```

`--with-modifier` fits a SECOND, separate study (`study_id: demo-modifier-v1`,
estimand `CATE`) that declares one binary effect modifier, so conditional-effect
questions about its declared subgroups can be answered. It reads, changes and
replaces nothing already retained; without it the first study is the only one, and
a conditional question about it is refused with its reason.

For an existing web environment, use that environment's Python to
`pip install ./provider` instead. The base inference adapter is standard-library
only, with NO runtime dependencies; it neither imports nor installs NumPy,
Pandas, SciPy, sklearn, EconML, TensorFlow, Torch, OpenRL, brokers or a text model.
This allows main's newer scientific packages to remain unchanged. The optional
`[fit]` extra installs EconML and its scientific dependencies (including
SHAP/LightGBM); use a separate fit venv because EconML 0.16 requires older sklearn
than main currently uses. Only the explicit fit CLI needs that extra. Constraints
reproduce the tested Python 3.12 fit environment. No Laya backend or GPU is used.

Do NOT install the repository root or its requirements.txt for this provider:
the legacy root distribution is unrelated rl-optimizer scaffolding. The provider
is an isolated subproject, with unique module names and no generic app package.

Start the main-owned M5PHET web service in the SAME environment with the SAME
CAUSAL_INFERENCE_STATE_DIR, after preparation. This means the same serving
environment that installed the base adapter; the fitting environment can be
separate. Registry snapshots known states
at registration, so prepare before starting it. No service is started by this
provider. `chat_examples()` then exposes the prepared SYNTHETIC/DEVELOPMENT
study. With no preparation it returns an empty list, never trains on discovery.

The state directory is configured by the administrator, never through chat.
At provider startup its regular content-addressed JSON files become a fixed
allowlist. To narrow it, set CAUSAL_INFERENCE_STATE_REFS to comma-separated
`causal-ate:<digest>` references before startup. The API also accepts an explicit
state_refs list, including [] to allow nothing. load() checks membership before
opening a file, rejects path strings and symlinks, bounds file size and verifies
its content hash. There is NO pickle/joblib deserialization. Requests cannot
change the directory or allowlist.

## Exact main-facing contract

- Entry-point group: `m5phet.providers`
- Entry-point name / provider.name: `causal_inference`
- Factory: `causal_inference_provider.chat:M5PHETCausalProvider`
- Runtime operations: `infer` only; families: `["causal_inference"]`
- Output kinds: `["causal_effect"]`; output schema: `{"targets": ["effect"]}`
- Uncertainty: `econml_statsmodels_HC1_normal`
- State ref: `causal-ate:<artifact_sha256>`; load(state_ref) returns a mapping
  containing digest, state_ref, task_id, available_at, config, population, result,
  and -- for a study fitted with an identifier -- manifest (study_id, provenance,
  estimator, effect_modifiers, n, seed and, for synthetic data, the generating
  equations and the effects they imply). task_id is `causal-<estimand>.v1:<config
  sha256>`, so a request for one estimand never binds a study fitted for the other.
- `infer(request, state)` returns outputs.effect with status, why, payload,
  uncertainty; plus the exact fitted population. No fit or model mutation.
- `chat_request(prompt, data, config)` returns a full m5phet.task.draft2 infer
  request with stable task identity, population and identifying parameters.
- `chat_examples()` returns `{title, prompt, data, config}` objects. Only demo
  artifacts are exposed, never the operator's real studies.
- `chat_slots()` declares one slot, `study`, whose allowed values are the retained
  studies: a study's own `study_id` when it has one, else its estimand and roles.
  A study fitted with a modifier answers to phrasings that name the modifier or the
  conditional reading ("modifier study", "estudio con modificador"), never to the
  bare role phrases the constant-effect study on the same roles answers to.
- `question_types()` declares `ate` and `cate`. `cate` takes the subgroup under
  either spelling, `subgroup` or `condition`, and neither is required at the
  envelope layer so that the named study itself states which subgroups it carries.
- `answer_questions(state, questions, data, as_of)` answers from ONE named study.
  The state names it by `state_ref`, by `study` (a declared slot value), or by
  `causal_graph`; a graph or a name no retained study matches is refused with the
  names and graphs that are retained. `cate` on a study without a modifier is
  refused `NOT_ESTIMABLE`; a subgroup naming a variable the study did not declare
  as a modifier is refused `NOT_ESTIMABLE` naming it and the modifiers it declares.
  No answer carries a number the artifact does not carry, and no p-value is derived.

The only accepted prompt, when no slot value is resolved, is
`Report the configured ATE and its uncertainty.` (case-insensitive); with a resolved
`study` slot value the question may be any sentence, and no word of it enters the
request. It is a report command, not natural-language identification.
Changing causal meaning requires changing structured parameters and explicitly
fitting another study. The provider never invents an adjustment set or assumption.

Example data is the small fitted-population reference
`{"data_sha256": "<actual selected-data digest>", "n_rows": 2400}`. Actual
values come from chat_examples(), not placeholders. Config contains
`input: "json"`, provider/family/output_kind, state, as_of, and parameters equal
to the explicit identifying config below. Request state and population must
exactly match that fitted dataset identity; raw data uploads are not fit jobs.

Successful response shape:

```json
{
  "outputs": {
    "effect": {
      "status": "OK",
      "why": null,
      "uncertainty": "econml_statsmodels_HC1_normal",
      "payload": {
        "estimand": "ATE",
        "estimate": 2.0293820021534623,
        "interval": [1.931308938896731, 2.1274550654101936],
        "unit": "synthetic outcome units",
        "assumptions": ["consistency", "constant_effect", "iid_sampling", "no_interference", "nuisance_models_correct", "positivity", "pre_treatment_adjustment", "sufficient_adjustment"],
        "diagnostics": {"identification": "conditional_on_user_assumptions"}
      }
    }
  }
}
```

This excerpt abbreviates diagnostics and omits the population mapping. The real
payload also includes confidence level, contrast, counts, cross-fitted propensity
range, versions, data/config hashes and analysis ID. A refusal has payload=null;
it never reports a dummy effect. Main owns the output validator and outer status
aggregation. Preserve NOT_IDENTIFIED per output rather than translating it to a
successful generic answer. Hashes establish local identity, not governance or
authentication. The state directory must be trusted operator storage.

## Explicit new-data analysis

Create your JSON config only after justifying its assumptions. The demo config
is valid for its synthetic process only, NOT a default declaration for real data:

```json
{
  "estimand": "ATE",
  "treatment": "treatment",
  "outcome": "outcome",
  "adjustments": ["baseline"],
  "unit": "synthetic outcome units",
  "alpha": 0.05,
  "assumptions": {
    "sufficient_adjustment": true,
    "pre_treatment_adjustment": true,
    "consistency": true,
    "no_interference": true,
    "positivity": true,
    "iid_sampling": true,
    "constant_effect": true,
    "nuisance_models_correct": true
  }
}
```

`sufficient_adjustment` asserts conditional exchangeability given exactly the
named adjustment columns and a valid backdoor set, excluding collider bias.
`pre_treatment_adjustment` separately asserts temporal eligibility. Consistency
requires well-defined 0/1 interventions; no_interference rules out spillovers.
Positivity is assumed as well as screened. IID excludes temporal and clustered
dependence. Constant effect is essential: this adapter's single DML coefficient
is an ATE only under that restriction. Nuisance specification asserts a linear
conditional outcome mean and logistic-linear propensity in the selected numeric
adjustments. An empty adjustment list is explicit and still requires the same
identifying declarations. Unknown/missing assumptions are never filled in.

```bash
.venv-causal/bin/python -m causal_inference_provider fit \
  --data study.csv --config study.json --state-dir "$CAUSAL_INFERENCE_STATE_DIR"
```

Input CSV must have unique column labels and numeric selected columns. CSV
numbers are parsed explicitly; the Python records API does not coerce strings.
No imputation, dropped rows, automatic discretization, column selection, clipping
or trimming. Additional unselected columns do not influence the estimate.
Exit 0 means an artifact was written; exit 2 means refusal/error with a structured
JSON result and no new artifact. New studies are not added to chat demo examples.

For an in-process offline job, use CausalInferenceProvider (not the registered
inference adapter): load(config), fit(DataFrame or list of numeric records),
infer(). The core's load config is distinct from the adapter's load state_ref.
Failed fit/load clears old output. Local result envelopes use
`causal-inference.provider.v1`; the adapter wraps them for M5PHET.

## Evidence and genuine limits

[Recorded acceptance](CAUSAL_PROVIDER_ACCEPTANCE.json): 79 focused tests pass,
clean fit/serving environments verified, and actual main-environment
Engine.execute returns OK without altering its installed scientific packages.

Synthetic process (ATE study): Z~Bernoulli(0.5), T|Z~Bernoulli(0.2+0.6Z),
Y=2T+4Z+Normal(0,1), n=2400, seed 17. Synthetic process (CATE study, study_id
`demo-modifier-v1`): baseline~Bernoulli(0.5), confounder~Normal(0,1),
treatment~Bernoulli(sigmoid(-0.6+1.2*baseline+0.35*confounder)),
outcome=1.0*treatment+2.0*treatment*baseline+3.0*baseline+1.5*confounder+Normal(0,1),
n=2400, seed 4373 -- so the effect is 1 where baseline==0, 3 where baseline==1 and
2 on average. Those equations and their implied effects are written into the
study's own manifest when it is fitted, so the artifact carries the truth its
estimate is checked against. Both are calibration evidence only.
EconML LinearDML provides the estimate and HC1 normal interval directly; tests
compare them against a separate real library invocation. No placeholder estimator,
mocked numeric acceptance, financial conclusion, GPU, broker or real holdout.

Only binary 0/1 treatment, a numeric outcome and IID sampling are supported, with
the effect either constant (estimand `ATE`) or varying with exactly ONE declared
binary effect modifier (estimand `CATE`, effect linear in that modifier, which a
single binary modifier satisfies exactly). A `CATE` study computes the effect at
each declared level WHILE FITTING and carries both, with their intervals and
subgroup counts, in its artifact; inference reads them and never refits, subsets
or interpolates. Subgroups are read only as `<modifier> == 0` / `<modifier> == 1`:
an inequality, another level, a cut of a continuous variable or a conjunction is a
different estimand and needs another explicit fit. Each subgroup needs at least 20
treated and 20 control observations, and no subgroup is pooled to reach it. No
ATT, more than one modifier, continuous treatment, instruments, time-series or
clustered uncertainty, causal discovery, sensitivity/refutation suite, public
data confirmation or domain revalidation. Overlap screening is conservative
and can reject identifiable studies. Passing it never proves identification.
100-10000 rows, <=20 adjustments, >=20 observations per arm. Numeric kernels use
one thread; there is no hard wall-clock scheduler. Only the small fixture is
certified under 30 seconds. Instances are not concurrent fit-job containers.

The stored artifact is a completed analysis result, not an executable fitted
model for predicting other populations. New data, roles, units, alpha or
assumptions require another explicit fit. Artifact available_at is a local
completion clock, not evidence of real source-data availability or governance.

Main integration was probed read-only again after its new validator appeared:
actual Engine.execute() returned OK and the real effect payload with matching
task/state/population bindings. Removing sufficient_adjustment returned
outputs.effect.status=NOT_IDENTIFIED and payload=null; main's outer result status
was INVALID_INPUT. Main owns that status aggregation and browser acceptance.
The inspected main worktree was based on 3f4a833 with active changes, not a sealed
release. No M5PHET files or dirty original causal-inference files were edited.
