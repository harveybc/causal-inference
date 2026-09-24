# causal-inference

## Minimal M5PHET causal provider

The CPU-only ATE provider lives in the isolated `provider/` subproject; see
[installation, explicit demo preparation and contract](docs/M5PHET_CAUSAL_PROVIDER.md).
It is inference-only over studies that were fitted on purpose beforehand, it requires
caller-declared identifying assumptions, and it never fits during chat inference.

It reports no accuracy, and refuses to: a causal estimate has no held-out truth,
because the counterfactual outcome of a row is never observed. It reports the
estimate, its interval, the assumptions it rests on and its diagnostics.

The application instructions below are the historical tool's, not the provider's
installation path.

**Status: experimental research repository — unverified.** This repository
hosts exploratory causal-inference research for regime-based trading
strategies. Nothing here is a released tool: the scripts are one-off
research artifacts, the inherited application code is unmaintained, and no
claim in this README has been verified by running the code for this
document beyond inspecting the committed tree.

## What this repository actually is

Two distinct layers share this repository:

1. **An inherited fork of
   [rl-optimizer](https://github.com/harveybc/rl-optimizer).** The
   repository began as a copy of that project and most of the tree still
   comes from it: the `app/` package (CLI, config handling, plugin loader,
   data handler), the plugin modules under `app/plugins/`, the legacy
   `tests/` suite and the helper `.bat`/`.sh` scripts.

2. **Causal research scripts** added on top for regime-analysis
   experiments (see below).

### Unrepaired inherited package identity — disclosure

The package metadata still identifies the **inherited `rl-optimizer`
package**: [`setup.py`](setup.py) declares `name='rl-optimizer'`, a
`rl_optimizer=app.main:main` console script, `rl_optimizer.*` entry-point
groups and the upstream `rl-optimizer` project URL. This has **not** been
repaired: installing this repository would install a package named
`rl-optimizer`, not `causal-inference`. Several declared entry points also
reference module paths that do not exist in the committed tree (for
example `app.plugins.optimizer_plugin_openrl` — the file actually lives at
`app/plugins/transformation/optimizer_plugin_openrl.py`, and the NEAT
optimizer plugins are absent entirely). Code ownership and packaging
identity will be separated in a future repair; until then treat the
packaging metadata as stale.

## Causal research content committed on `master`

- [`causal_regime_analysis.py`](causal_regime_analysis.py) — the main
  research script. It applies three techniques to 12 regime features
  derived from EURUSD hourly OHLC data (resampled to 4h):
  1. NOTEARS causal discovery (DAG structure among features),
  2. Invariant Causal Prediction (stability of feature/forward-return
     relationships across regimes),
  3. DoWhy refutation tests (placebo and random-confounder checks).

  It expects its input CSV from a sibling `feature-eng` checkout (path
  overridable via the `OHLC_FILE` environment variable). The input data is
  **not** committed here.
- [`results/causal_analysis_results.json`](results/causal_analysis_results.json)
  — the recorded output of one historical run of that script (ICP scores,
  p-values and per-regime coefficients per feature). It documents a
  specific past run and is not regenerated automatically; it has not been
  independently reproduced for this README.

Additional causal research scripts (a v2 regime analysis, cluster-based
regime analysis and cross-asset comparison/audit work) exist only as
**uncommitted work-in-progress in the owner's local working tree**. They
are not part of the repository history and are deliberately not described
here; they will be documented if and when they are committed.

## Inherited application code (unverified)

The `app/` package provides a plugin-based CLI inherited from
rl-optimizer: [`app/main.py`](app/main.py), configuration merging
([`app/config_handler.py`](app/config_handler.py),
[`app/config_merger.py`](app/config_merger.py)), a plugin loader
([`app/plugin_loader.py`](app/plugin_loader.py)) and plugin modules under
[`app/plugins/`](app/plugins/) (inference, preprocessing and
transformation groups, including an OpenRL PPO agent and
prediction/custom environment plugins). The legacy
[`tests/`](tests/) suite also predates the causal work. **None of this has
been exercised or verified in the causal-inference context** — no claim is
made that the CLI runs, that the plugins load, or that the tests pass.

## Dependencies

[`requirements.txt`](requirements.txt) mixes inherited RL dependencies
(`tensorflow-gpu`, `openrl`, `neat-python`, `stable-baselines3`, `gym`)
with general scientific packages. The causal scripts additionally import
libraries (e.g. for NOTEARS/DoWhy) that are not all pinned there. No
tested installation procedure is claimed.

## Relationship to sibling repositories

- [rl-optimizer](https://github.com/harveybc/rl-optimizer) — the upstream
  project this repository was forked from; its packaging identity is still
  present here (see disclosure above).
- [feature-eng](https://github.com/harveybc/feature-eng) — source of the
  regime features and the OHLC test data consumed by
  `causal_regime_analysis.py`.

## License

[`LICENSE.txt`](LICENSE.txt) (inherited from the upstream project).

---

*This README describes only what is committed on `master` as of
2026-08-10. Experimental, unverified research — not a supported tool, not
financial advice.*
