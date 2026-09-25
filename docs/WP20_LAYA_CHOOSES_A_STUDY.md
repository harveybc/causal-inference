# WP20 — Laya chooses a causal study, and what happened when it did

`choose-study` profiles a CSV, asks Laya one declared choice at a time through `m5phet.decide`, and composes the
`m5phet.causal_study_spec.v1` document those decisions describe. This page is the record of the first real run of
that path against the real checkpoint on the private worker's GPU, 2026-09-25.

**Read the outcome before the numbers.** The run did not produce a fitted study. Laya gave **all four columns the
role `confounder`** — the column literally named `treatment` included — so the composition named no treatment and no
outcome, and `study_spec.validate_spec` refused it `NO_TREATMENT`. Nothing was repaired and nothing was fitted. That
is the result this package exists to obtain: the plumbing works end to end, and the zero-shot checkpoint does not
assign causal roles from a column profile.

## The command

```
python -m causal_inference_provider choose-study \
  --dataset <the synthetic modifier CSV> \
  --problem "Estimate the effect of treatment on outcome; baseline may modify the effect; confounder is a common cause" \
  --out study_spec.json --study-id laya-chosen-modifier-v1 --provenance DEVELOPMENT
```

Exit 2. `--out` received nothing: there is no spec to write when the composition is refused.

The data are WP10's synthetic modifier set, regenerated from its own generator and seed
(`causal_inference_provider.example.modifier_example_data`, `numpy.default_rng(4373)`, 2400 rows) and written to a
CSV. Its generating equations put the effect at **1.0 where `baseline == 0`, 3.0 where `baseline == 1`, 2.0 on
average**.

## What Laya was shown

The profile, and only the profile. Per column: dtype, missing fraction, distinct values, binary, and for a numeric
column mean, sd, minimum and maximum at six declared decimals. No row is rendered into a state text.

| column | dtype | missing | distinct | binary | mean | sd | min | max |
|---|---|---|---|---|---|---|---|---|
| `baseline` | int64 | 0.0 | 2 | yes (0/1) | 0.505417 | 0.500075 | 0.0 | 1.0 |
| `confounder` | float64 | 0.0 | 2400 | no | 0.006057 | 1.046969 | -3.988474 | 3.393396 |
| `treatment` | int64 | 0.0 | 2 | yes (0/1) | 0.490417 | 0.500012 | 0.0 | 1.0 |
| `outcome` | float64 | 0.0 | 2400 | no | 2.638351 | 3.228397 | -5.292165 | 12.196217 |

Each role question also carried the problem sentence, the dataset's column names and row count, and the roles already
given to the columns asked before it.

## The eight decisions

Checkpoint `laya-checkpoint:bd12df887789924672d1c848319caf9866a172183e6959beab369d9e20aaaa89` for all eight;
`backend: laya`; `execution_authorized: false`; stamped 2026-09-25T06:41:05Z … 06:42:52Z. Probabilities are the
model's own, copied verbatim, **uncalibrated**: they are this checkpoint's head outputs for these exact wordings, not
accuracies and not evidence that a choice is right.

| question | chosen | uncalibrated probabilities | record digest |
|---|---|---|---|
| `baseline` | `confounder` | treatment 0.1036 · outcome 0.1079 · **confounder 0.4085** · modifier 0.2196 · exclude 0.1605 | `ab6fe2b1…` |
| `confounder` | `confounder` | treatment 0.0907 · outcome 0.0993 · **confounder 0.3591** · modifier 0.1179 · exclude 0.3330 | `853a2ff9…` |
| `treatment` | `confounder` | treatment 0.2090 · outcome 0.0917 · **confounder 0.3549** · modifier 0.1286 · exclude 0.2159 | `7e9a0aa4…` |
| `outcome` | `confounder` | treatment 0.1273 · outcome 0.1607 · **confounder 0.2999** · modifier 0.1239 · exclude 0.2882 | `0f751c1c…` |
| `estimator` | `LinearDML` | **LinearDML 0.3126** · DRLearner 0.2236 · SparseLinearDML 0.2361 · DML 0.2278 | `2417d26a…` |
| `model_y` | `ridge` | lasso 0.3416 · **ridge 0.3623** · gradient_boosting 0.1256 · random_forest 0.1705 | `4368685d…` |
| `model_t` | `ridge` | lasso 0.3278 · **ridge 0.3614** · gradient_boosting 0.1268 · random_forest 0.1841 | `d7aa6216…` |
| `confidence_level` | `0.90` | 0.95 0.3747 · **0.90 0.6253** | `971362ca…` |

Two things in that table are worth naming precisely.

**The role choices are near-uniform.** The winning option carries 0.30–0.41 of the mass over five options, and the
column named `treatment` gave the `treatment` role 0.2090 — its second-highest, but not its choice. This is the same
picture WP17 recorded for the `transform` decision (0.3532 / 0.3249 / 0.3219 over three options): the checkpoint
barely separates these options. A near-uniform distribution that still has to name a label will name one, and here
it named the same one four times.

**The estimator options were filtered correctly.** Because the roles Laya had already given declared no modifier, the
estimand those roles imply is `ATE`, and `CausalForestDML` — which the space declares reports `CATE` only — was not
on the list Laya was shown. The filter reads the space's own `estimands` declaration, so it cannot drift from what
the fit can actually build.

## The refusal

```
NO_TREATMENT: no column is the treatment; a causal study is about an intervention, and this spec names none.
```

All eight decision records were written (content-addressed, under the studies directory's `decisions/`) and are
returned with the refusal. No study was fitted, nothing was retained under `laya-chosen-modifier-v1`, and the three
studies already on this host were not touched.

## hand_spec vs laya_spec, same rows

Both rows below are the same 2400-row CSV (`data_sha256 3ed79e3d…`). The third row is a diagnostic, not a study Laya
chose: it takes the hand role assignment and transplants **only** the estimator, nuisance models and confidence level
Laya decided, to isolate which part of the choice failed. Its estimator decision was made about an `ATE` study, so it
is being reused outside the question it answered; it is reported for that reason and no other.

| | roles | estimator | model_y / model_t | level | ATE (known 2.0) | `baseline == 0` (known 1.0) | `baseline == 1` (known 3.0) |
|---|---|---|---|---|---|---|---|
| `hand_spec` (WP20's example) | hand | LinearDML | lasso / lasso | 0.95 | 2.0308 [1.9439, 2.1177] ✓ | 0.9303 [0.8011, 1.0595] ✓ | 3.1077 [2.9912, 3.2243] ✓ |
| `laya_spec` | **all four `confounder`** | LinearDML | ridge / ridge | 0.90 | **NOT FITTED — `NO_TREATMENT`** | — | — |
| `laya_partial` (diagnostic) | hand | LinearDML | ridge / ridge | 0.90 | 2.0309 [1.9579, 2.1038] ✓ | 0.9312 [0.8227, 1.0397] ✓ | 3.1070 [3.0092, 3.2048] ✗ |

`✓` / `✗` say only whether the interval covers the number the generating equations put there. On this one synthetic
draw the diagnostic row's `baseline == 1` interval misses 3.0 by 0.009 at the 90% level Laya chose, where the same
study at 95% covers it; a 90% interval is built to miss one time in ten, so this is an observation about one draw and
not a measurement of anything.

**What the comparison does and does not establish.** It establishes that the fit path recovers known synthetic
effects from a hand-written spec, and that Laya's non-role choices (LinearDML, ridge, 0.90) change those recoveries
very little on this data. It establishes **nothing** about real data, about any market, or about whether a chooser is
useful: the recovery is synthetic by construction, and the decisions are uncalibrated. `causal_accuracy` remains
refused.

## Where this leaves WP20

Done: the study space is offered as declared options, every choice is a recorded decision bound to a state digest,
an estimator that cannot serve the estimand is never offered, a composition that does not validate is refused by the
validator's own name with the decisions kept, and a spec the chooser produces is fitted by `prepare-study` without an
edit (proved in `tests/test_choose_study.py`).

Not done, and not attempted: a study Laya actually chose. On this checkpoint and this state text the role assignment
is degenerate. Whether that is the state text, the wording of the question, the zero-shot checkpoint or the task
itself is open, and the honest next step is a measurement — the same four columns asked many times, and a closure
table — not a reworded prompt until the answer looks right.
