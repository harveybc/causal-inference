# PS3-C: the three causal rungs over the episodes this programme actually holds

Lane C of the orders `SATOSHI_PROGRESSIVE_SELECTION_AND_MODULAR_CONTINUATION_2026_09_30` (predictor master
`ac125db9`), subplan `FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md` section 5, stage PS3-C of
section 10.1. Written 2026-10-01 from read-only inspection of the committed trees named below.

**Status: SPECIFICATION. NO_NEW_MEASUREMENT.** Nothing in this document was fitted, re-fitted or re-run for it.
Every number is read from a committed artifact and its path is given; every method is a design with its acceptance
tests written red (`provider/tests/test_fs10_fs12_ps3c.py`). The identification status of every study this machine
retains today is `NOT_IDENTIFIED`, for the concrete reasons in section 1.3, and this specification does not change it.

What this document does NOT do: it does not install a parallel causal engine in feature-eng, predictor, Calendar or
M5PHET (the orders forbid it; the engine lives here), does not require a VAE, does not require a physical intervention
on the market, and does not wait for PS3-R or for complete A/B lanes.

## 0. Where the existing pieces are, so nothing is rebuilt

| piece | where it lives today | tip read | what it is |
|---|---|---|---|
| treatment-effect provider (ATE, CATE by declared modifier, continuous treatment, assumptions declared by the caller, overlap screens, `NOT_IDENTIFIED` refusals) | `causal-inference/provider/src/causal_inference_provider/provider.py` | origin/master `bb11d64` | EconML `LinearDML` 0.16.0, fit is an explicit offline CLI job, inference never fits |
| event-study registration and the three questions it answers (`impulse_response`, `sensitivity`, `counterfactual_path`) | `causal-inference/provider/src/causal_inference_provider/event_study.py` | `bb11d64` | registers a `m5phet.event_projections.v1` document by digest; carries the identification verdict verbatim; refuses `ate`/`cate` as `NOT_ESTIMABLE` |
| point-in-time calendar store (arrivals, `release_surprise` vs `available_surprise`, refusals by name) | `feature-eng/app/economic_calendar.py` | branch `satoshi/rp150-calendar-acceptance-20260926` `ca22c00` | the audited release boundary the event rows are built on |
| event rows (step 1), association (rung 1), local projections with HAC, held-out skill, superposition and placebo (rung 2), model-based counterfactual (step 4), expectations (`MODEL_BASED_EXPECTATION`) | `feature-eng/feature_eng_m5phet/{events,association,local_projections,counterfactual,expectations}.py` | `ca22c00` | CPU, deterministic, seed 1729; estimator Jorda (2005) local projection |
| the measured data manifest | `feature-eng/docs/EVENT_STUDY_DATA_STATUS.md`, `docs/CALENDAR_DATASET_INVENTORY.md`, `docs/evidence/calendar_dataset_inventory.json` | `ca22c00` | every count in section 1 below comes from there |
| the as-of admission of calendar-study resources | `data-gov` worktree `calendar-study-admission`, `docs/CALENDAR_STUDY_ADMISSION_METHOD_STATE.json` | S12_ALPHA_ACCEPTANCE | CAL-ADM-01..04: exact pins, refuse unknown availability, require measured consensus/observed overlap |

Not re-read for this document and therefore UNVERIFIED here: whether the feature-eng branch `ca22c00` is merged
anywhere (its tip is not on feature-eng `origin/master d3d003d`), and whether the registered study
`eurusd-events-assumed-clock-v1` still exists in the operator's `CAUSAL_INFERENCE_STATE_DIR`.

### 0.1 Where the PS3-C study code lives, and what of this repository is NOT used

This repository is a fork of `rl-optimizer` whose root packaging still identifies the inherited package
(`setup.py` declares `name='rl-optimizer'`, the root `requirements.txt` pulls GPU/RL dependencies, `app/` is the
inherited CLI, `tests/` is the inherited suite, and the root research scripts are one-off artifacts). **None of that
is used by PS3-C**, and the README's instruction not to install the root package blindly stands.

The PS3-C study code is a module of the isolated provider distribution:

| item | value |
|---|---|
| distribution | `causal-inference-m5phet` 0.1.0, `provider/pyproject.toml`, packages under `provider/src` |
| package | `causal_inference_provider` (base adapter is standard-library only; `[fit]` extra pins EconML 0.16.0 in its own venv) |
| entry point | group `m5phet.providers`, name `causal_inference`, factory `causal_inference_provider.chat:M5PHETCausalProvider` |
| new module | `causal_inference_provider.ps3c` (`rung2_effect`, `AdditiveSCM`, `counterfactual_same_episode`, `dossier`), specified by the red tests in section 6; fitting stays an explicit offline CLI job (`python -m causal_inference_provider ...`), inference reads retained JSON, no pickle |
| retained state | `CAUSAL_INFERENCE_STATE_DIR` under the operator's allowlist, as the treatment-effect and event studies already do |
| consumers | Calendar (feature-eng `app/economic_calendar.py`, data-gov calendar-study admission) and M5PHET consume **only** the `causal_dossier.v1` contract (section 7); they import nothing from this package and implement no causal arithmetic of their own |
| not used | root `setup.py`, root `requirements.txt`, `app/`, `tests/`, `causal_regime_analysis*.py`, `cluster_regime_analysis.py`, `cross_asset_*.py`, `momentum_analysis.py`, `nfp_event_response_poc.py`, `run_rpcmci_only.py`, `results*/` |

### 0.2 The RL/SAC arm (FS13, FS14) is not one of the six lanes — owner unassigned

FS13 (internal selection and purge by real supports; RL reward separate from MAE) and FS14 (heuristic strategy and
SAC compared with representation, costs and periods declared) name an RL comparison that no lane of the
2026-09-30 orders owns. This lane does not take it. **Owner: UNASSIGNED.** What the RL side would consume, when it
is assigned: the `representation_candidate_card.v1` (its `evaluation.probes[]` with `target: J_policy`, its
`identity` and `cost`) and the `SelectionRelease` of subplan 10.3 — not the causal dossier, which carries no policy
objective. The dossier's `subject.target` enum includes `J_policy` only so a future RL-side study can be filed
under the same contract; no such study is specified here.

## 1. Data and unit of analysis (subplan 5.1)

### 1.1 The sources, measured by `feature_eng_m5phet.calendar_inventory` on 2026-09-26

| resource | rows | consensus | actual | publication instant | span |
|---|---:|---|---|---|---|
| `feature-eng/tests/data/economic_calendar_2011_2021.csv` (no header, no `provenance.json`) | 121,658; 54,275 carry actual AND consensus | yes | yes (114,045) | **no** — naive wall clock; measured as fixed UTC−05:00 2012-05→2018-01, America/New_York from 2018-03; 13,201 releases `CLOCK_PERIOD_UNDETERMINED` | 2011-01-01 → 2021-04-26 |
| `financial-data/economic_calendar/release_actuals/fxmacrodata/announcements.parquet` | 18,147; 18,015 with an instant | **no** (its own README) | yes (17,961) | yes, `announcement_datetime_utc`, tz-aware — *declared by the dataset, not verified by anyone here* | 2024-12-12 → 2026-05-01 |
| `financial-data/economic_calendar/scheduled_events/fxmacrodata/release_calendar.parquet` | 896 | no | no | yes | 2026-02-05 → 2027-07-14 (forward) |
| FRED `release_actuals/*` (9 series) and `fred_release_date_proxy` | 432 per series; 1,200 | **`COLUMN_PRESENT_BUT_EMPTY` in every row** | yes | date proxy only | 1990 → 2025 |
| `financial-data/market_data/forex/g10/eurusd/{5m,15m,1h,4h}.parquet` (HistData, provenance digests present) | 5m: 1,552,028 bars | — | — | bar clock | 2005-01-03 → 2025-12-31 |

**The join of consensus to an observed instant is 0 of 54,275** (`calendar_join`, 2026-09-25): the spans are
disjoint (14,766 rows, 27.2 %, fail only on the date), not the matching. No source on this machine carries
consensus, actual and observed publication instant for the same release. That is a purchasing decision recorded in
`financial-data/economic_calendar/release_surprises/stage13_consensus_gap.md`, not a bug, and it is not resolved here.

The inventory of 2026-09-26 found none of the five calendar resources registered; **later the same day the data-gov
registry registered all five** (`data-gov/docs/08_RESOURCE_REGISTRY.md`, "The five economic-calendar registrations
(2026-09-26)"), with `execution_authorized: False` in every row, a named `absences` list per resource, and the
measured verdict that no consensus source overlaps an observed-clock source (a 1,326-day gap). A registration grants
nothing: a resource with no availability contract stays closed. Section 1.4 binds the rungs to the resources that do
hold a TRAIN contract.

### 1.4 Binding to sealed TRAIN contracts (lane B / M03, received 2026-10-01)

Source of truth: feature-eng branch `satoshi/b-selection-ps0-ps2-20261001` @ `4255f74`,
`docs/feature_metrics/laneB/FINANCIAL_TRAIN_CONTRACTS.v1.json`, sha256
`bfaf2cf63814a0be5756f648f3f5c949b97d96b1f0d0c29f674bc0d36a371406` — **re-hashed by this lane from the fetched
blob: matches**. It lists 198 C127 census appearances over 53 entities (4h/1h/15m/5m), each with `dataset_id`,
`appearance_id`, `physical_sha256`, `contract_sha256`, chronological 0.6/0.2/0.2 partitions, period and variables;
plus the ETH 4h model-ready view and the local `d4`. Everything else in the census (6,555 appearance-columns) has no
TRAIN split and is not profiled or selected from.

| episode source | TRAIN contract | binding decision |
|---|---|---|
| `economic_calendar__release_actuals__fxmacrodata__announcements` — 4 census appearances (e.g. `app_00d9ca67…`, `app_097b81b8…`, `app_127800fd…`, `app_de5da682…`), period 2025-01-31 → 2026-04-30 | **YES** (C127) | the treatment source for every rung: actuals and declared instants, **no consensus** → expectation is `MODEL_BASED_EXPECTATION` → rung 2/3 `NOT_IDENTIFIED` by the schema rule; rung 1 admissible |
| `economic_calendar__scheduled_events__fxmacrodata__release_calendar` — 4 appearances, 2026-02-05 → 2027-07-14 | YES | the forward schedule only: defines episode existence (`C` in the DAG), never a surprise |
| `feature-eng/tests/data/economic_calendar_2011_2021.csv` (lake `none`, consensus + actuals, assumed clock) | **NO** | **NOT_ADMISSIBLE_NO_CONTRACT** — the retained `eurusd-events-assumed-clock-v1` study rests on it and stays DEVELOPMENT evidence |
| FRED `release_actuals/*`, `fred_release_date_proxy` | NO (not among the 198 as calendar entities; FRED series appear as covariate entities only) | NOT_ADMISSIBLE_NO_CONTRACT as episode sources; admissible as `W_pre` covariates where their appearance is contracted |
| EURUSD bars `financial-data/market_data/forex/g10/eurusd/*.parquet` | **NO** — no EURUSD appearance is among the 198 | the uncontracted files are **NOT_ADMISSIBLE_NO_CONTRACT**; the study's outcome variable is **`NOT_EXECUTABLE_NO_CONTRACTED_PRICE`** until a EURUSD price appearance is sealed through the governed route. **Ruling (Satoshi, 2026-10-01, owner open question 13): no substitution of the asset** — `usdcad`, `gbpjpy`, `nzdusd` do not stand in; lane B prepares the sealing request |
| ETH 4h model-ready view (TRAIN `[0, 13699)` = 2017-09-28 → 2023-12-31, has CLOSE) | YES (immutable predictor contract) | a possible outcome series for a crypto-asset event study; not the FX question of section 5.1; not bound here |

Caveat carried verbatim from lane B: `availability_time` is UNAVAILABLE for every census variable — no per-family
availability contract instance exists. Point-in-time episode use therefore needs that contract first; until then the
`published_at` of every contracted calendar appearance is **a declared assumption**, and `data_manifest.publication_clock`
must not be written as `OBSERVED_*` from a census appearance alone. The raw `announcements.parquet` (sha `d8dd8c13…`)
and the C127 appearances (different `physical_sha256`, resampled to a frequency) are different objects; a dossier names
the appearance it read.

Consequence for the three rungs today: the treatment side is contracted (fxmacrodata appearances), the outcome side
is not. Rung 1 is `NOT_EVALUATED` until the EURUSD price appearance is sealed; rung 2 and rung 3 are
`NOT_IDENTIFIED` for every cell by construction (`EXPECTATION_IS_MODEL_BASED`, no observed availability clock) and
would stay so even after sealing. No contracted source moves that.

### 1.5 The single binding slot: `data_manifest.asset_appearance`

Every rung reads the outcome price series through one slot of the dossier contract and nowhere else, so that the
moment a sealed EURUSD appearance exists the study binds to it **without redesign**:

| state | fields (validated by `causal_dossier.v1.schema.json`) | effect on the rungs |
|---|---|---|
| `CONTRACTED` | `appearance_id` (`app_` + 24 hex), `dataset_id`, `entity`, `resource_sha256` (the appearance's `physical_sha256`), `contract_id`, `contract_sha256`, `train_rows` `[start, end)`, `frequency`, `period`, `contracts_document_sha256` (the FINANCIAL_TRAIN_CONTRACTS document the ids were read from) | rung 1 may run inside `train_rows` only; rungs 2–3 decided by their own checks |
| `NOT_EXECUTABLE_NO_CONTRACTED_PRICE` | `entity`, `reason`, `open_question` | rung 1 forced `NOT_EVALUATED`; rungs 2–3 in {`NOT_EVALUATED`, `NOT_IDENTIFIED`}; `selection.causal_evidence_level` forced `NONE` |

Rules the schema enforces and the predictor tests exercise for both states: a branch name, a short digest, a
one-element `train_rows` or an unknown frequency is refused; a missing binding field is refused; any state other than
the two is refused; supplying the contracted slot and nothing else lets a rung-1 `ASSOCIATION_REPORTED` validate.
The retained development studies on the uncontracted bars keep their verdicts as DEVELOPMENT evidence and are
filed with the slot in the `NOT_EXECUTABLE_NO_CONTRACTED_PRICE` state.

### 1.2 The episode (one row = one identified release, never one filled hour)

Required fields, their provenance today, and the refusal when absent (names reused from `app/economic_calendar.py`
and `events.py`; new ones are marked NEW):

| field | subplan 5.1 requirement | available today | refusal when absent |
|---|---|---|---|
| `event_key`, `event_type` (economy \| release), `country` | yes | archive: free text; fxmacrodata: `currency`, `indicator`; synonym table in `calendar_join` | `RELEASE_NAME_NOT_IN_THE_ANNOUNCEMENT_ARCHIVE` |
| `published_at` (source clock), `received_at` (our clock), `vintage` | three separate clocks | archive: none (assumed); fxmacrodata: declared `announcement_datetime_utc`; `received_at`: file-grain `acquired_at` only | `MISSING_PUBLICATION_CLOCK`; `ASSUMED_SCHEDULED_PUBLICATION` is a declared assumption, never a measurement |
| `consensus_prior` with its own publication instant | yes | archive: value, no clock (`consensus_clock: ASSUMED_BEFORE_RELEASE`); fxmacrodata: none; WP28 substitutes `MODEL_BASED_EXPECTATION`, which is never called a consensus | `NO_EXPECTATION` / `NO_CONSENSUS` |
| `actual_initial`, `revisions[]` (separate arrivals) | first release vs later revisions | archive: `actual`, `previous` only; no revision arrivals exist in any source | NEW `REVISIONS_UNAVAILABLE` (state, not a value) |
| `unit`, `period` | required for a difference to be a surprise | archive: `data_format` is a magnitude marker (`%`,`B`,`K`,`M`,`T`), 28,380 rows none; fxmacrodata: no unit column | `UNIT_REQUIRED` |
| `W_pre`: pre-event realized vol, hour/day-of-week (UTC), regime state, neighbour surprises at NEGATIVE offsets inside `W` | covariates observable before `published_at` | built by `events.py` / `local_projections.py` from the 5m bars | `INSUFFICIENT_HISTORY`, `NON_POSITIVE_RESIDUAL_SCALE` |
| `neighbours[]`: other releases inside the window with signed offsets | vector of neighbouring events | built; positive-offset neighbours are listed but EXCLUDED from controls (collider) | — |
| outcome paths with bar clock/resolution | known clock and resolution | 5m bars; `BARS_MISSING_AT_HORIZON` when a bar is missing on the path | `BARS_MISSING_AT_HORIZON` |

Treatment `A` is the standardized surprise `(actual_initial − consensus_prior) / sigma_type`, `sigma_type` from
surprises published strictly before the release (`events.py`). `A = 0` means "published at consensus", never "no
release". A policy-rate *decision* is a second treatment variable, not a dose of the same one. If only the
surprise-at-receipt is available, the estimand is declared as the receipt-clock object and labelled
`available_surprise`; it is not credited with the initial market impact.

Overlapping outcome windows are not independent episodes: inference is clustered by episode and by calendar block
(HAC with the lag rule of `local_projections._hac_lags`; block bootstrap for anything the HAC does not cover),
and the first complete bar after the instant is where an hourly outcome starts, which omits the initial impact
by construction — the 5m path is the only resolution that sees it, and it exists.

### 1.3 Identification status of every retained study today (FS12, stated here so no reader infers otherwise)

| study / artifact | clock | expectation | verdict | concrete reasons (verbatim codes) |
|---|---|---|---|---|
| `eurusd-events-assumed-clock-v1` (archive 2011–2021, registered as `causal-event-study:` manifest) | `ASSUMED_SCHEDULED_PUBLICATION`, anchor corrected to the measured UTC−05:00 / America/New_York periods | consensus from the archive (no clock) | **NOT_IDENTIFIED** | `ASSUMED_PUBLICATION_CLOCK`; placebo fails on **40 of 40** (event type, horizon, outcome); held-out skill 23 of 40 under the corrected anchor (15 of 40 under the earlier UTC misreading) |
| WP28 observed-clock run (fxmacrodata 2024-12 → 2025-12, **not registered**) | `OBSERVED_ACTUAL_PUBLICATION` (declared by the dataset) | `MODEL_BASED_EXPECTATION` (`SEASONAL_NAIVE(m=1)` 9,762 rows; `AR(p≤4,BIC)` 1,373) | **NOT_IDENTIFIED** | `EXPECTATION_IS_MODEL_BASED`; `PLACEBO_FAILED` on 652 of 660; placebo passes 8 of 660; superposition `ADDITIVE_HOLDS` 10 of 10; beta intervals excluding zero 114 of 650; 2,198 of 9,815 releases (22.4 %) have stimulus exactly 0.0; NFP, CPI, PPI, retail sales, GDP and policy rates produced **zero** rows; `USD | initial_jobless_claims` stamped on Saturdays → 0 rows |
| `counterfactual_path` answers from either study | inherits | inherits | `MODEL_BASED_COUNTERFACTUAL`, flagged `NOT_IDENTIFIED_BY_CONSTRUCTION` where the clock was assumed | a fitted regression evaluated twice; see section 4.2 for why this is not rung 3 |
| provider demo `causal-ate:` study | n/a (synthetic IID) | n/a | `conditional_on_user_assumptions` | SYNTHETIC/DEVELOPMENT; not a market study |

A consensus feed over 2025-01 → today would remove `EXPECTATION_IS_MODEL_BASED` and restore the monthly releases
(a consensus row needs no vintage history); it would not give the fxmacrodata instants a provenance. The placebo
would then be the only check between a study and `PLACEBO_CONSISTENT` — which the code already says "is NOT a claim
of identification".

## 2. Targets (subplan section 3), mapped onto the episode

| target | definition on the episode | horizon grid | existing field | paired naive on identical rows |
|---|---:|---|---|---|
| `Y_s(e,h)` | cumulative log return of EURUSD from the first complete bar at/after `published_at` to `+h` | h = 1..6 **hours** (60, 120, 180, 240, 300, 360 min); the event-study grid 5/15/30/60/240 min covers only h = 1 h and 4 h — the grid is extended, not re-interpreted | `log_return` at `horizon_minutes` | persistence (0 return) and the rung-1 naive-by-sign mean (`association.naive_response_by_sign`) fitted on the same training episodes |
| `Y_l(e,h)` | same, h = 24, 48, ..., 144 hours | 6 horizons; 144 h windows overlap across episodes — clustered inference mandatory | not built today (`events.py` max horizon used: 240 min) | persistence; sign-mean |
| `Y_b(e)` | first barrier hit after entry at the first complete bar: TP at `close + tp_multiplier·ATR`, SL at `close − sl_multiplier·ATR` (heuristic-strategy `plugin_long_short_predictions.py` params `tp_multiplier`, `sl_multiplier`), **close-only signals, never trailed**, timeout = the plugin's five-day cap; versioned intrabar ambiguity rule (on a 5m bar both barriers inside the range → `AMBIGUOUS_INTRABAR`, censored, counted) | one per episode per rule version | not built today | base rate of each barrier from training episodes; log-loss/Brier against it |
| `J_policy` | out of scope for PS3-C (RL evaluation by episodes, agent-multi/gym-fx) | — | — | — |

Price conversion from returns uses the price known at the first complete bar and is verified against the strategy
plugin, whose interface is not changed. Volatility used to normalize is the pre-event realized vol of `W_pre` only.
Y_b uses fixed rules or out-of-fold forecasts, never an ideal prediction.

## 3. Rung 1 — association and predictive relevance (subplan 5.2)

**Question.** With the information available at `t`, what changes in the distribution of `Y_s`, `Y_l`, `Y_b`?
**Estimand.** `P(Y_h | X_history, calendar, W_pre)` on internal chronological folds of TRAIN.

| element | specification | existing / new |
|---|---|---|
| unit | episode (section 1.2); hourly rows for candidate-feature probes outside events | existing rows; hourly probes are PS3-R/PS5 territory |
| estimators | Pearson/Spearman with `n`; naive response by sign/tercile/pre-vol tercile with cut points; with-vs-without-candidate probes on paired budget and population; conditional dependence: GCMI as screen, confirmed by a nonlinear estimator on the flagged cases; PCMCI+ (tigramite `5.2.1.25`, `5a87687`) as a **structure proposer under its assumptions**, never as proof | `association.py` existing; GCMI/PCMCI+ NEW, not implemented |
| conditioning set | only variables observed at `t` (`W_pre`, calendar); FS01 forbids anything else | existing refusals |
| multiplicity | per family: estimator, effective n, p and q, permutation count B with the floor `(b+1)/(B+1) ≥ 1/201` stated; direction stability reported separately from MI (MI has no sign) | NEW fields in the dossier |
| regime | per year/regime; a regime-specific signal is not discarded for failing elsewhere | NEW |
| outputs | predictive evidence by head and regime, cost, extraction priority; `rung1.state ∈ {ASSOCIATION_REPORTED, TOO_FEW_EVENTS, ZERO_VARIANCE, NOT_EVALUATED}` | dossier contract |

Nothing on rung 1 carries a causal word. `NOT_EVALUATED` is not zero.

## 4. Rung 2 — effect of historically observed interventions (subplan 5.3)

**Question.** How would the response change if the published surprise were `a` instead of `a0`, for a comparable
population of episodes of one event type?
**Estimand.** `E[Y_h | do(A=a)] − E[Y_h | do(A=a0)]` within the observed dose support of that event type, or
`CATE(w)` by declared pre-event modifier. Continuous dose: a dose-response estimator or pre-declared dose intervals,
never a fake binary (`provider.py` WP22 continuous treatment, residual-variance screen).

### 4.1 Temporal DAG (declared, contestable; each edge is an assumption, listed so it can be attacked)

```
C (calendar: type, scheduled instant)        ──► existence of the episode (selection, not a cause of A)
W_pre (pre-event vol, hour/dow, regime,
       neighbour surprises at NEGATIVE offsets,
       consensus_prior as a level)           ──► A (surprise)      [assumed ABSENT under rational expectations;
                                                                     tested, not assumed: the residual-variance screen]
W_pre                                        ──► Y_h
A                                            ──► M_1 (impact 0–5 min), M_2 (realized vol 0–h),
                                                 M_3 (later releases' surprises at POSITIVE offsets)   ──► Y_h
A                                            ──► Y_h   (direct)
U_cal (unobserved: feed timing error, revisions, unmeasured regime)  ──► A and Y_h   [the confounding that is NOT
                                                                                      removable by W_pre; sensitivity]
```

Adjustment set for the **total** effect: `W_pre` only. Explicitly NOT adjusted: `M_1..M_3` (mediators/descendants),
any positive-offset neighbour (collider through the same news flow), the same-window outcome at a shorter horizon
(descendant), post-release volatility. Lagged treatments (same type, previous releases) enter `W_pre` as history.

Sequential episodes whose later release depends on an earlier one (e.g. a rate decision after a CPI print) are
either treated with a time-varying-treatment estimator over a sequential DAG or excluded to separated episodes;
the dossier says which (`sequential_treatment_policy`).

### 4.2 Support, overlap and balance (checked, never clipped)

| check | rule | state when it fails |
|---|---|---|
| dose support | the contrast `(a, a0)` must lie inside the empirical support of `A` for that event type, by declared quantiles, with `n ≥ 20` episodes per side of the contrast (provider's `MIN` arms rule) | `NO_COMMON_SUPPORT` → `NOT_IDENTIFIED` |
| overlap, binary/interval dose | cross-fitted propensity in `[0.05, 0.95]` for every episode, no trimming (provider rule) | `OVERLAP_SCREEN_FAILED` → `NOT_IDENTIFIED` |
| overlap, continuous dose | residual variance of `A` after cross-fitted adjustment on `W_pre` above the declared floor (provider WP22 screen) | `TREATMENT_PREDICTED_BY_CONTROLS` → `NOT_IDENTIFIED` |
| balance | standardized mean differences of `W_pre` across the contrast after weighting ≤ declared bound; effective sample size reported | `IMBALANCE` → sensitivity only |
| event density | episodes whose window contains an untreated instant for the placebo; on a dense calendar the placebo may be `NOT_RUN` | `PLACEBO_NOT_RUN` → `NOT_IDENTIFIED` |

### 4.3 Estimators (in order of preference, each named in the dossier with its library and pin)

1. g-computation: `m_h(a, w) = E[Y_h | A=a, W=w]` fitted on TRAIN folds; effect = mean over the declared population
   of `m_h(a, W_e) − m_h(a0, W_e)`. DoWhy `v0.14` (`178ecc9c`) identifies the estimand from the declared DAG; the
   outcome model is the local projection already fitted (`local_projections.py`) when its design equals `W_pre`,
   which it does today (const, `surprise`, `pre_event_realized_vol`, hour/dow, negative-offset neighbour sum).
2. EconML `LinearDML` (0.16.0 pinned by the provider; 0.17.0 `f0fc2e7` exists upstream, not adopted here) through
   `causal_inference_provider.provider` for a declared constant effect or CATE by one modifier; continuous treatment
   per WP22. Its eight assumptions must be declared true by the caller and the dossier copies them verbatim.
3. Matching/weighting as alternatives, reported beside (1)–(2), never as proof of no confounding.
4. Instrument / discontinuity / natural experiment only if one concretely exists and its assumptions are declared;
   a residualized surprise is not an instrument.

### 4.4 Placebo and refutation battery (can refute, cannot prove)

| test | what it does | pass rule |
|---|---|---|
| pseudo-event placebo (existing) | 200 seeded pseudo-instants with no release inside the exclusion span, dressed with the type's empirical surprise distribution | `beta` interval contains 0 AND does not overlap the real one, on every tested cell |
| pre-event placebo outcome (NEW) | regress `Y` over `[t−h, t)` on `A` | interval contains 0 |
| negative-control outcome (NEW) | an outcome `A` cannot affect (a series closed at `t`, or a prior-day return) | interval contains 0 |
| random-confounder / subset refuters (DoWhy) | add random common cause; bootstrap subsets | estimate stable within declared tolerance |
| sensitivity to unmeasured confounding | Rosenbaum-style bounds or E-value on the identified cells | reported; no pass/fail |
| specification and fold stability | HAC lag rule, holdout fraction, window `W`, pre-vol span, by year | reported |

**Rung 2 state.** `IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS` only when: a publication clock is observed (not
assumed), the expectation is a published consensus (not model-based), the adjustment set is declared and satisfies
the back-door criterion on the declared DAG, support/overlap/balance pass, and the placebo battery passes on the
cell. Otherwise `NOT_IDENTIFIED` with reasons, and the rung-1 evidence stays. Today every cell is `NOT_IDENTIFIED`.

## 5. Rung 3 — counterfactual of the same episode (subplan 5.4)

**Question.** In this observed episode `e`, what would `Y_h` have been had the surprise been `a0` (e.g. zero), holding
that episode's own circumstances?

### 5.1 The structural model (a declared assumption, not a property any network certifies)

```
W_pre,e           exogenous (observed)
A_e      = a_e    observed treatment
M_e      = g(A_e, W_pre,e) + U_M,e            mediators, in temporal order (M_1 then M_2 then M_3)
Y_h,e    = f_h(A_e, W_pre,e, M_e) + U_h,e     additive noise, invariant mechanism
```

`f_h`, `g` fitted on TRAIN episodes of that event type and context (rung-2 outcome models can be reused when their
design equals this one). Non-additive or non-invertible mechanisms: infer the posterior of `U` and report it; a point
counterfactual from a non-invertible model is refused `NON_INVERTIBLE_MECHANISM_POSTERIOR_REQUIRED`.

### 5.2 The three steps, and what is preserved at each

| step | computation | preserved | refused |
|---|---|---|---|
| **abduction** | `u_M,e = m_e − g(a_e, w_e)`; `u_h,e = y_e,h − f_h(a_e, w_e, m_e)` | the episode's own perturbations | missing observed `y_e,h` → `ABDUCTION_NEEDS_OBSERVED_OUTCOME` |
| **action** | replace the mechanism of `A` by `A := a0`; keep `w_e`, `u_M,e`, `u_h,e` | `W_pre`, all inferred `u` | `a0` outside the type's support → `NO_COMMON_SUPPORT` |
| **prediction** | `m_cf = g(a0, w_e) + u_M,e` (descendants recomputed in order); `y_cf,h = f_h(a0, w_e, m_cf) + u_h,e` | — | mediators held fixed by hand → `MEDIATOR_NOT_PROPAGATED` |

Output per episode and horizon: `y_factual`, `y_cf`, `delta = y_factual − y_cf`, the inferred `u` with their
uncertainty, the mechanism identity (fit digest, TRAIN fold, design), and, where a 5m trajectory exists, the
counterfactual path through the strategy's barrier rule so `Y_b` can be re-read (one terminal return does not
determine TP/SL).

**Why the existing `counterfactual_path` is not rung 3.** `feature_eng_m5phet.counterfactual` evaluates the fitted
projection twice and reports `predicted_observed − predicted_counterfactual`, printing `observed_outcome` beside
them. It never computes `u_e` and never adds it back: its counterfactual is `f_h(a0, w_e)`, the population mechanism
at `a0`, not `f_h(a0, w_e) + u_e`, the same episode at `a0`. For the additive single-equation case the *difference*
coincides, but the counterfactual *level*, the barrier re-reading and any non-linear or mediated mechanism do not.
Its label `MODEL_BASED_COUNTERFACTUAL` is accurate and stays; it is the rung-2 model's statement, retained as such.

### 5.3 Library path

DoWhy-GCM (`v0.14`, `178ecc9c`): `InvertibleStructuralCausalModel` with `AdditiveNoiseModel` per node,
`gcm.counterfactual_samples(model, {"A": lambda a: a0}, observed_data=episode)` performs abduction–action–prediction
over the declared graph in topological order. The fit is an explicit offline job (provider rule), the fitted
mechanisms are retained as JSON-serializable parameters and a digest, and inference reads them; no pickle.

### 5.4 No live future — the rule, stated as a contract

The retrospective study may read `y_e,h` for abduction after the episode happened. **An operational feature emitted
at or before `published_at + h` may not contain `u_e,h`, `y_cf,e,h`, `delta_e,h` or any function of them.** The
operational variant uses only `W_pre`, the calendar, and the predictive distribution of `U` fitted on prior
episodes. A dossier records per quantity the earliest instant it may be emitted (`emittable_from`); a consumer
joining a rung-3 quantity to a row stamped before that instant is a leak, and FS01/FS11 tests must catch it.

**Rung 3 state.** `COUNTERFACTUAL_UNDER_DECLARED_SCM` only when rung 2 is identified for the cell AND the structural
block (DAG, equations, noise assumptions, invertibility, support, alternatives considered) is declared. If different
compatible models give different answers, report the sensitivity or bounds, not one number. Today: no cell qualifies;
state `NOT_IDENTIFIED` inherited from rung 2.

## 5.5 Intervening on the network is not intervening on the market (subplan 5.5)

Replacing, permuting or ablating latents, branches or inputs of the predictor or the policy and measuring the change
in its outputs is a **controlled experiment on the model**. It measures functional dependence and utility of a
representation to a head. It is recorded in the representation card (`representation_candidate_card.v1`,
`model_intervention_evidence`), never in the causal dossier, and it never changes a rung state. Permuting a branch
can create out-of-distribution combinations; calendar matching between branches does not guarantee coherence with
the others. It is contrasted with withdrawal-and-refit under paired budgets, by branch and by group. The three rungs
above are about the market under the declared DAG; this paragraph is about the network.

## 6. Behavioural tests FS10, FS11, FS12 (written red; `provider/tests/test_fs10_fs12_ps3c.py`)

The missing mechanism is one module, `causal_inference_provider.ps3c`, with four names the tests pin:

| name | contract |
|---|---|
| `rung2_effect(episodes, *, treatment, outcome, adjustment, contrast, dag, support)` | returns a rung-2 block; with `adjustment=None` returns `NOT_IDENTIFIED` reason `ADJUSTMENT_SET_NOT_DECLARED` and `estimate=None`; with a declared set that is a valid back-door set on `dag`, support passing, returns `IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS` with an estimate close to the planted effect; with no common support returns `NOT_IDENTIFIED` reason `NO_COMMON_SUPPORT` |
| `AdditiveSCM(order, mechanisms)` | a declared SCM over named nodes with additive noise and explicit topological order |
| `counterfactual_same_episode(scm, episode, *, intervention, mode)` | `mode="RETROSPECTIVE"`: abduction, action, prediction with `u` preserved and descendants propagated; `mode="OPERATIONAL"`: refuses `FUTURE_OUTCOME_IN_OPERATIONAL_CALL` when the episode carries any outcome realized after its emission instant |
| `dossier(...)` | assembles the `causal_dossier.v1` document with the three rung states separate |

| test | what it proves today | why it is red | what turns it green |
|---|---|---|---|
| `test_fs10_filtered_event_contrast_is_not_do` | a mean difference over filtered episodes recovers the confounded number, not the planted effect — computed in the test from the planted world, so the trap is real | `ps3c` does not exist (`MECHANISM_MISSING`) | `rung2_effect` refusing the undeclared adjustment and recovering the planted effect with the declared one |
| `test_fs10_existing_event_study_refuses_do_questions` | **GREEN today**: `event_study.answer(manifest, "ate", ...)` is refused `NOT_ESTIMABLE`; an event study carries no contrast | — | — (already in place; kept so a regression is visible) |
| `test_fs11_counterfactual_preserves_inferred_perturbation` | the planted world's `y_cf = f(a0,w) + u_e` differs from `f(a0,w)` by exactly `u_e`, and only the first is the same-episode counterfactual | `ps3c` missing | `counterfactual_same_episode` returning `y_cf` with `u_e` preserved |
| `test_fs11_descendants_are_propagated_not_frozen` | a mediator `m = g(a,w)+u_m` must be recomputed under `a0` with `u_m` kept | `ps3c` missing | propagation in topological order |
| `test_fs11_operational_mode_refuses_live_future` | an operational call that receives a realized outcome must refuse by name | `ps3c` missing | the `FUTURE_OUTCOME_IN_OPERATIONAL_CALL` refusal |
| `test_fs12_provider_keeps_not_identified_without_declared_assumptions` | **GREEN today**: `CausalInferenceProvider.load` with undeclared assumptions returns `status NOT_IDENTIFIED`, `payload None` | — | — |
| `test_fs12_no_common_support_keeps_not_identified` | a contrast outside the type's dose support must stay `NOT_IDENTIFIED` with `NO_COMMON_SUPPORT`, and rung-1 evidence must still be present in the block | `ps3c` missing | support check in `rung2_effect` |
| `test_fs12_confounded_world_without_adjustment_keeps_not_identified` | with a planted unmeasured confounder and the adjustment set missing it, the state stays `NOT_IDENTIFIED` (`BACKDOOR_NOT_SATISFIED`) even though an estimate could be computed | `ps3c` missing | back-door check against the declared DAG |

Run: `cd provider && CUDA_VISIBLE_DEVICES="" PYTHONPATH=src ~/.local/bin/crispdm-run -q -m 1G -- python -m pytest
tests/test_fs10_fs12_ps3c.py -q`. Expected today: 2 passed, 6 failed, every failure message beginning
`MECHANISM_MISSING`.

## 7. Consumer contract

Calendar (feature-eng `app/economic_calendar.py`, data-gov calendar admission) and M5PHET consume
`docs/contracts/causal_dossier.v1.schema.json` on predictor branch `satoshi/c-contracts-20261001`, never this
repository's internals. One document per (episode or episode population, event type, head, horizon), three rung
states separate, `NOT_EVALUATED` and `NOT_IDENTIFIED` distinct from zero, `emittable_from` per quantity.

## 8. Not done, and whose decision it is

- No fit, no re-fit, no new measurement; the tests are red by design (6 of 8).
- `causal_inference_provider.ps3c` is not implemented; the tests are its specification.
- Y_l (24..144 h) and Y_b episode outcomes are not built in `events.py`; the grid extension is specified only.
- GCMI, PCMCI+, pre-event placebo, negative-control outcome, E-value: specified, not implemented.
- A consensus feed with an observed publication instant (purchase) is the only thing that can move any cell out of
  `NOT_IDENTIFIED`; owner's decision, recorded in `stage13_consensus_gap.md`.
- A sealed EURUSD price appearance (owner open question 13, lane B preparing the sealing request) is the only thing
  that moves rung 1 out of `NOT_EVALUATED`; it binds through `data_manifest.asset_appearance` (section 1.5).
- Registration of the five calendar resources under data-gov contracts: operator act.
- The fxmacrodata instants are declared by the dataset; nothing here can verify them (65 of 66 types show a fixed
  posting rule; `initial_jobless_claims` stamped on Saturdays).

## References (methods; none claims financial SOTA)

- J. Pearl, "Causal inference in statistics: An overview," Statistics Surveys 3, 2009.
- Ò. Jordà, "Estimation and Inference of Impulse Responses by Local Projections," AER 95(1), 2005.
- T. Andersen, T. Bollerslev, F. Diebold, C. Vega, "Micro Effects of Macro Announcements," AER 93(1), 2003.
- R. Gürkaynak, B. Sack, E. Swanson, "Do Actions Speak Louder Than Words?," IJCB 1(1), 2005.
- PyWhy DoWhy `v0.14` (`178ecc9c690a02f2801c1f70da2695f5744186cc`), GCM counterfactuals:
  https://www.pywhy.org/dowhy/ (user guide, "Computing Counterfactuals"); EconML `v0.16.0`
  (`8e80e89a0fef058b219e2754bb4ed551e5df52c7`), as pinned by the provider.
- J. Runge, "Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series
  datasets," UAI 2020; tigramite `5.2.1.25` (`5a8768754e6103755b006e9357e21c1a58534927`).
