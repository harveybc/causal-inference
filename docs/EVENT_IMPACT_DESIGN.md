# Event impact: the causal effect of scheduled macro releases on EUR/USD volatility and trend

Lane CAUSAL, 2026-10-19, branch `satoshi/event-impact-20261019`. Estimator:
`provider/src/causal_inference_provider/event_impact.py`; acceptance tests EI01–EI14:
`provider/tests/test_event_impact.py`; descriptive run: `provider/tools/event_impact_fixture_run.py`, output under
`docs/evidence/event_impact_20261019/`.

**Status of every number in this lane: DESCRIPTIVE / FIXTURE / CLOCK_UNVERIFIED / FIXTURE_RESEARCH_ONLY.** The only
real data read is the two feature-eng test fixtures (an Investing.com calendar export 2011–2021, research-only, and
EUR/USD hourly bars 2005–2020). Rows at or after 2025-01-01 are refused by every loader. Real 2020–2024 calendar
data is being acquired separately; the design and its synthetic validation are what this document delivers.

## 1. The medical framing (the estimand language)

| clinical term | here |
|---|---|
| **patient state X** | the market state at the release instant, read from bars strictly before the release bar: realized volatility at two scales (24 h, 120 h) and their ratio, trend strength at two scales (`abs(sum r) / sum abs r`), the sign of the short trend, and an **unsupervised regime label** (tercile cells of `vol_ratio` × `trend_strength_short`, cuts fitted on the training period only — causal filtering, no look-ahead). A second, **declared arm** is the feature set the I6-DMO campaign froze for the RL / price-regression line: 19 features under feature manifest sha256 `2ff34fa21d5841da1befdd4d898db911128d959a5e460d04cc45b54955416470` (predictor, `docs/audits/evidence/I6N_STLFSI4_EXCLUSION_20261012/REGENERATION.json` and `docs/i6n/direct_multi_output_static/best_registry/MG2M_REFERENCE.json`, MG2_SELECTED). That set was selected for **price regression on 2024 development data, not for effect estimation**; only its four OHLC-computable members (`px.rv5`, `px.log_co`, `px.close_loc`, `rg.atr_ratio`) exist on the fixture, so on the fixture the arm is `PARTIAL_OHLC_SUBSET_ONLY`. |
| **drug T** | the event family (e.g. `United States \| Nonfarm Payrolls`) |
| **dose** | the standardized surprise `z = (actual − consensus) / scale`, **signed** for trend and taken in absolute value, or ignored, for volatility (the volatility effect is of the release itself; the dose modulates it) |
| **effect Y(τ)** | at lag τ after the dose: (a) the excess absolute log return (realized-volatility unit) and (b) the signed log return per unit of z. **Two separate decay curves**, each with its own fitted half-life and bootstrap interval, and a paired-bootstrap test of whether they differ (`compare_half_lives`). The same half-life is never assumed. |
| **estimand** | `CATE(X, dose, τ)`: the heterogeneous effect by patient state, by dose and by lag. The X-arm is kept only if it improves **out-of-sample** effect prediction on the decision period (chronological, purged); if it adds noise the X-free estimator is kept and the fact is reported (`cate_out_of_sample`). |
| **drug interactions** | concurrent releases within the window: `concurrency_test` regresses the post-window realized volatility on one indicator per family present plus a two-or-more indicator; the latter's interval gives `ADDITIVE_HOLDS` or `INTERACTIONS_PRESENT`. |

## 2. Pearl's ladder, from data only

1. **Association.** `event_mean[τ]`: the mean absolute return at lag τ across releases. It co-occurs with elevated
   volatility; that alone says nothing about cause, because the release hour is itself a high-volatility hour.
2. **Intervention.** The release versus **matched control windows**: for each release bar, the `n_controls = 5` bars
   of the **same hour of day, same day of week**, within one year, with **no covered release inside
   `[c − window, c + window]`**, nearest in **log pre-release realized volatility** (24 bars). The contrast
   `excess[τ] = event_mean[τ] − control_mean[τ]` is the interventional quantity; EI05 proves it removes a planted
   hour-of-day profile that biases the unmatched baseline.
3. **Counterfactual.** Only through matched observed cases. **No synthetic generator exists in the estimator.** The
   one in the tests plants a known world so the estimator can be shown to recover it; it informs no estimate.

What matching does NOT remove: unscheduled news inside the window, and the publication clock itself. Both are
stated in the output's `identification` block; the verdict is `DESCRIPTIVE_MATCHED_CONTRAST_NOT_CONFIRMATORY`.

## 3. Estimators

### 3.1 Impulse response and half-life (volatility)

For lags τ ∈ [−6, +window] the paths of `abs_ret` over events and over each event's controls are averaged; the
excess, its 95 % bootstrap interval (resampling events with their matched controls, 400 draws), the ratio
`event/control`, the peak, the cumulative excess and the **pre-release placebo** (mean excess over τ < 0) are
reported. The half-life is fitted on τ ≥ 0 as `excess[τ] = A · 2^(−τ/H)` by least squares over a log grid of H with
A closed-form per H; its interval is the bootstrap distribution of H. H at the grid floor (0.25 h) is reported
`LOW_WITHIN_RELEASE_BAR` (hourly bars cannot resolve it), at the cap `HIGH_NOT_RESOLVED_BY_WINDOW`.

### 3.2 Trend: surprise-conditional drift and its own decay

Per the announcement literature (Andersen, Bollerslev, Diebold and Vega 2003) the signed response exists only
conditional on the surprise. `surprise_drift` regresses the forward return `log(close[p+h] / close[p−1])` on z at
horizons 1/2/4/8 bars with HC1 errors; the slope is the trend effect per unit z, the intercept is the unconditional
drift. The family verdict is Bonferroni-corrected across horizons: `SURPRISE_CONDITIONAL_DRIFT` (slope significant,
intercept not), `UNCONDITIONAL_DRIFT_PRESENT`, `NO_DRIFT_DETECTED`. `trend_impulse_response` fits the per-lag slope
β[τ] of `log_ret[p+τ]` on z and its decay `β[τ]·sign(β[0]) = B · 2^(−τ/H_trend)`.

### 3.3 Surprise standardization (scale fitted on prior releases only)

`z_i = surprise_i / sd(surprise_j : j < i)`, with at least 8 prior releases; the first 8 releases of a family get
no z. EI08 proves that changing a later release leaves every earlier z unchanged. Consensus is read from the
calendar's `forecast` column as published before the release; the `actual_text` qualifier is NOT used (it disagrees
with the sign of `actual − forecast` in roughly 10 % of rows), and `previous` is never used for z. No winsorizing is
applied; extreme z (e.g. the 2020 pandemic NFP prints) are visible in the scatter and are a known lever on the slope.

### 3.4 Heterogeneity

`heterogeneous_effects(y, z, strata, covariates)` returns the slope per stratum under one interface. With EconML
installed and at least 60 rows it is a **cross-fitted `LinearDML`** (5 folds, random-forest nuisances for the
outcome and for the continuous treatment z, the stratum one-hot as effect modifier X, backward-looking state
`vol_long`, `trend_strength_short`, `trend_sign_short` as W); below 60 rows, or without EconML, a stratified HC1
OLS. `estimator` names which one was used per family (`econml_linear_dml_crossfit` / `stratified_ols_hc1`) and
`estimator_reason` why. `cate_out_of_sample` is the keep/drop rule for the X-arm (section 1).

Environment (owner-authorized, project venv only, CPU wheels, never the trading-stack env): `.venv-causal` built
from `provider/constraints-cpu-py312.txt` with `./provider[fit,test,event]` — econml 0.16.0, scikit-learn 1.6.1,
numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, matplotlib 3.9.2 (the `[event]` extra, pinned in the constraints file).

### 3.5 Aggregation kernel (the feature at any t)

```
vol_feature(t)   = Σ_{released e: t_e ≤ t}            V_f(e) · 2^(−(t − t_e)/H_f(e))
trend_feature(t) = Σ_{released e: t_e ≤ t}            B_f(e) · z_e · 2^(−(t − t_e)/H_trend,f(e))
anticipation(t)  = Σ_{scheduled e: t < t_e ≤ t + Hz}  V_f(e) · 2^(−(t_e − t)/H_a)
```

Realized terms read only events released at or before t; the anticipation term reads the **schedule only** (never
z). `cumulative_impact` refuses `FUTURE_SHIFTED_EVENT` for a released event without an observed surprise, and EI09
proves a future event's z never enters any term.

### 3.6 Event selection (the analogue of feature selection)

`select_events`: trade direction from the training period only (`sign(slope_train)`); on the **decision period
only**, per family, the per-release return `direction · sign(z) · forward_return(4 bars)`, its mean, standard
deviation, effect per unit risk, t and p; the decision-period matched volatility impulse; **Benjamini–Hochberg** at
α = 0.1 across families; a minimum of 20 events on each side; selected = BH pass with positive mean; the set is
**frozen** in `SELECTED_EVENTS_MANIFEST.json` with the input digests and the run's sha256, before any confirmation
period is read.

## 4. Leakage rules (enforced by typed refusal)

| rule | code | where |
|---|---|---|
| no STL / loess / lowess / centred / two-sided / seasonal-trend-residual series as input or target: such a series at bar t reads bars after t. STL is allowed only as a retrospective diagnostic on closed data, never fitted on or predicted. | `FORBIDDEN_TARGET_TRANSFORM` | `refuse_forbidden_target`, checked by `impulse_response` on its target |
| availability ≤ t: an event's actual is never used before its release; a response that precedes its recorded release (significant pre-release placebo ≥ half the post-release peak) is refused — the clock or the schedule is shifted | `FUTURE_SHIFTED_EVENT` | `assert_released`, `impulse_response`, `cumulative_impact` |
| the schedule is known in advance; anticipation reads it, never the outcome | by construction | `cumulative_impact` |
| consensus frozen pre-release; scale fitted on strictly prior releases | `CONSENSUS_NOT_FROZEN` (by construction, EI08) | `standardize_surprises` |
| TEST rows (≥ 2025-01-01) are never read | `TEST_ROWS_FORBIDDEN` | every loader and `bars_from_prices` |
| X-arm judged out of sample, purged (one week), chronological | — | `cate_out_of_sample` |

Known limitation of the placebo refusal: a family published 90 minutes after another covered family (ISM
Non-Manufacturing at 10:00 ET after the 08:30 ET block) shows pre-release excess that is contamination, not a
shifted clock; it is refused all the same and must be re-estimated with a pre-window that excludes the earlier
release, which this version does not do.

## 5. Clocks (detected from the data, not assumed)

* **Calendar fixture.** Not a legal time zone. From the Nonfarm Payrolls rows (published 08:30 US Eastern every
  month): 8:30 in November–March and 7:30 in April–October through 2018-01 → a fixed UTC−5 clock without daylight
  saving; afterwards an exporter pseudo-DST (UTC−4 over roughly February/March → September/November of 2018–2021).
  The loader builds a **per-row offset table from anchor releases** of known publication instant (eleven US 08:30
  ET families), assigns every row the nearest anchor's offset, and marks a row between two disagreeing anchors
  `SWITCH_AMBIGUOUS` (1,054 of 117,116 rows read). The months with two offsets in the run are exactly 2018-02,
  2018-09, 2019-02, 2019-11, 2020-02, 2020-09. Mapping check on NFP 2011-01-07: local 08:30 → 13:30 UTC,
  actual 103, forecast 150 (K). `previous` embeds `<Revised From x>` text; only its leading number is parsed.
* **Hourly fixture.** The NFP release bar's absolute return is 2.15× its same-bar-one-week-away controls at
  offset 0 and below 1.35× at every other offset in [−3, +5], so the bars are on a **UTC clock labelled by open
  time** (Sunday bars start at 21:00/22:00 UTC, consistent with a 17:00 New York open). CLOCK_UNVERIFIED: inferred,
  not read from a broker contract. Each family also carries a `clock_check` (best offset in [−3, +3]);
  `CLOCK_MISMATCH` is reported, not corrected (Retail Sales +2, GDP −3, ZEW +1 in the run).

## 6. The fixture run (DESCRIPTIVE, 2011-01 → 2020-04, decision period from 2018-01-01)

`docs/evidence/event_impact_20261019/event_impact_fixture_run.json`, `HALF_LIFE_TABLE.md`,
`SELECTED_EVENTS_MANIFEST.json`, figures `impulso_volatilidad_por_familia`, `impulso_tendencia_por_familia`,
`sorpresa_vs_deriva_por_familia`, `tabla_vidas_medias` (PNG + PDF, Spanish labels). Only aggregates and digests are
written; no calendar row is copied (research-only licence).

Half-lives (hours, matched excess of |log return|, window 24 h):

| family | n | ratio at release bar | vol t½ [95 %] | trend t½ [95 %] | differ? | drift verdict (Bonferroni) | slope 4 h (bp/z), p |
|---|---|---|---|---|---|---|---|
| US Nonfarm Payrolls | 90 | 3.31 | 0.81 [0.47, 1.20] | 0.33 [0.25, 0.58] | DIFFER | NO_DRIFT_DETECTED | −9.7, 0.06 |
| US CPI | 145 | 1.84 | 0.57 [0.25, 1.89] | 0.25 (floor) | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | −2.6, 0.32 |
| US Core CPI | 145 | 1.84 | 0.57 [0.25, 1.89] | 0.25 (floor) | NOT_DISTINGUISHABLE | SURPRISE_CONDITIONAL_DRIFT | −6.4, 0.01 |
| US Retail Sales | 92 | 2.23 | 0.35 [0.25, 0.95] | 0.25 (floor) | NOT_DISTINGUISHABLE | NO_DRIFT_DETECTED | −5.9, 0.07 |
| US Core Retail Sales | 80 | 2.22 | 0.28 | 0.25 (floor) | NOT_DISTINGUISHABLE | SURPRISE_CONDITIONAL_DRIFT | −8.9, <0.01 |
| US ISM Manufacturing | 87 | 1.62 | 0.41 | 0.25 (floor) | — | NO_DRIFT_DETECTED | −5.5, 0.04 |
| US GDP | 64 | 1.88 | 1.44 | 0.25 (floor) | — | NO_DRIFT_DETECTED | −3.5, 0.20 |
| US Unemployment Rate | 90 | 3.31 | 0.81 | 0.38 | — | NO_DRIFT_DETECTED | −4.7, 0.36 |
| US ADP | 68 | 2.00 | 0.28 | 0.25 (floor) | — | NO_DRIFT_DETECTED | −3.9, 0.21 |
| US Fed rate decision | 23 | 4.38 | 0.60 | — (no surprise variance) | — | NOT_FITTED | — |
| Euro Zone CPI | 229 | 0.86 | 19.1 (not a spike) | 0.25 (floor) | — | NO_DRIFT_DETECTED | −0.9, 0.62 |
| German ZEW | 111 | 1.10 | 3.32 | 0.25 (floor) | — | NO_DRIFT_DETECTED | +1.8, 0.48 |
| German Ifo | 83 | 1.52 | 0.25 (floor) | 12 (cap) | — | NO_DRIFT_DETECTED | +0.9, 0.74 |
| German Manufacturing PMI | 154 | 1.79 | 0.25 (floor) | 0.45 | — | UNCONDITIONAL_DRIFT_PRESENT | +8.8, <0.01 |
| ECB rate decision | — | — | — | — | — | REFUSED TOO_FEW_EVENTS (18 matched) | — |
| US ISM Non-Manufacturing | — | — | — | — | — | REFUSED FUTURE_SHIFTED_EVENT (pre-release excess; section 4) | — |

The exact intervals for every row are in the JSON and in `HALF_LIFE_TABLE.md`; NFP and Unemployment Rate share
the same release instant and therefore the same volatility row.

**Surprise–drift verdict.** Across the USD families the slope is negative at every horizon (a USD-positive
surprise lowers EUR/USD) and the intercept is never significant: the drift exists only through the surprise,
as the literature says. With the Bonferroni family rule only Core CPI and Core Retail Sales reach
`SURPRISE_CONDITIONAL_DRIFT`; NFP is −9.7 bp/z at 4 h with p = 0.06 (p = 0.04–0.05 at 1, 2 and 8 h), carried by
the 2020 prints, so it is `NO_DRIFT_DETECTED` under the pre-registered rule. German Manufacturing PMI is the one
`UNCONDITIONAL_DRIFT_PRESENT` and its positive slope is the EUR-side sign. Trend decays are faster than volatility
decays everywhere they are resolved (NFP: DIFFER, trend − vol = −0.48 h, 95 % [−0.90, −0.10]).

**Patient-state arms (CATE).** The cross-fitted LinearDML ran for 13 of 14 fitted families (Fed decision: 23
rows, stratified fallback). NFP by pre-release volatility tercile: low −25.5 bp/z [−41.6, −9.4], mid −13.5
[−27.2, +0.2], high −3.3 [−8.6, +2.1] — the surprise moves EUR/USD most when the market was calm before, which is
the heterogeneity the medical framing asks for. Out of sample on the decision period, the unsupervised regime arm
was `X_ARM_DROPPED_ADDS_NOISE` for 13 of 13 families; the OHLC subset of the RL feature set was kept for 5 and
dropped for 8. The X-free estimator therefore remains the default; the per-family verdicts are in `state_arms`.

**Concurrency.** Over 1,391 release windows of the fitted families, 853 with two or more covered releases inside
24 h, the two-or-more coefficient on post-window realized volatility is +2.5e-4 with 95 % [−1.05e-3, +1.55e-3]:
`ADDITIVE_HOLDS` at this resolution.

**Event selection.** On the 2018-01 → 2020-04 decision period, NFP ranks first (effect per unit risk 0.48,
p = 0.011), then ADP (0.36, p = 0.056) and German PMI (0.30, p = 0.023); **none passes Benjamini–Hochberg at
α = 0.1 across 13 families**, so the frozen selected set is **empty**. The manifest records the ranking and digests
so the strategy lane can consume it; it is not a licence to trade any family.

**Comparison with the earlier NFP POC (vol ratio 1.07, drift n.s.).** The POC scanned broker offsets by comparing
first-Friday bars to the previous Thursday at the same hour and chose "UTC+2, bar 15". The fixture is on a UTC
clock and NFP is published at 13:30 UTC in winter and 12:30 UTC in summer, so bar 15 is two to three hours after
the release, where the excess has mostly dissipated (half-life 0.81 h). Replaying the POC's own measure here gives
1.12; at the release bar against matched same-hour same-weekday controls the ratio is 3.31 (excess 95 %
[14, 26] bp). The POC also used first Fridays instead of the calendar (NFP moves around holidays) and one-bar
open-to-close moves instead of |log return|. Its drift verdict does not change: the unconditional drift is ~0 here
too; the signed effect exists only through the surprise, which the POC could not observe.

## 7. Acceptance tests (all in `provider/tests/test_event_impact.py`, synthetic world planted in the test only)

| id | what is proved |
|---|---|
| EI01 | the matched impulse response recovers a planted additive impulse (interval covers it, within 25 %, peak at lag 0) |
| EI02 | the fitted volatility half-life recovers the planted 4 h (interval covers it, fit R² > 0.8) |
| EI03 | the surprise-conditional slope recovers the planted drift per z and the intercept covers zero; verdict `SURPRISE_CONDITIONAL_DRIFT` |
| EI04 | with no planted drift the verdict is `NO_DRIFT_DETECTED` and every slope interval covers zero |
| EI05 | matched controls remove a planted 5× hour-of-day profile (excess at lag 0 straddles zero) while the unmatched baseline is biased by more than half the control level; controls share hour and weekday |
| EI06 | a schedule shifted 6 h after the planted response is refused `FUTURE_SHIFTED_EVENT`; `assert_released` refuses a release after `as_of` |
| EI07 | `stl_trend`, `loess_resid`, `close_centered_ma`, `seasonal_component`, `lowess_24`, `two_sided_filter` are refused `FORBIDDEN_TARGET_TRANSFORM` both directly and as an impulse-response target |
| EI08 | the standardization scale uses strictly prior releases: the first 8 z are NaN, a later change leaves earlier z unchanged, the scale equals the prior sample sd |
| EI09 | the aggregation kernel equals its closed form; a future event's z never enters; a released event without an actual is refused |
| EI10 | rows at or after 2025-01-01 are refused; heterogeneity recovers two planted stratum slopes under one interface |
| EI11 | the trend decay is estimated separately (planted 2 h vs 4 h), recovered, and told apart; pre-release betas straddle zero |
| EI12 | the X-arm is kept when heterogeneity is real and dropped when it is noise, judged out of sample with a purge |
| EI13 | the concurrency test runs on overlapping families and returns a verdict |
| EI14 | selection ranks on the decision period only, refuses a family with too few events, controls FDR, freezes |

Run: `PYTHONPATH=provider/src CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 120s -n ci-event -- python3 -m pytest
provider/tests/test_event_impact.py -q` → 20 passed.

## 8. What this lane does not claim

No confirmatory measurement; no identification beyond the matched contrast; no broker clock; no 2020–2024 real
calendar yet; no `CausalForestDML` (the families hold 23–229 releases, too few for a forest; LinearDML with
cross-fitting is the EconML estimator used); no trading eligibility for any family.

Test counts: `provider/tests/test_event_impact.py` 20 passed under both interpreters (anaconda 3.12 without
EconML → fallback path; `.venv-causal` with EconML → DML path). Full `provider/tests` under `.venv-causal`:
327 passed, 7 failed, 1 error, 15 skipped — the failures are in `test_fs_causal*.py`, `test_ps3c_batch.py` and
`test_ps3c_ladder.py`, which import optional dependencies the fit venv does not carry; none touches this lane's
files and the same tests fail on the branch base.
