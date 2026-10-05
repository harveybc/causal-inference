# Phase-1 worker traceability

| Requirement | Structural evidence | Behavioral evidence |
|---|---|---|
| Config-driven EURUSD/ETH targets | `TargetDefinition`; `feature_selection_unit.v1` schema | `test_eth_and_eurusd_target_definitions_are_config_driven` |
| Existing scientific method is reused | `FB.run_feature`; `FB.finalize` calls | focused legacy suite (`test_fs_causal*`) |
| One TRAIN column receives the established PS1 families | frequency-aware profiler; explicit gaps/outliers/PACF/entropy | `test_ps1_has_established_families_frequency_aware_lags_and_no_silent_gap_fill` |
| Raw cells cannot masquerade as final decisions | unit envelope has raw `causal_evidence` and empty `selection_decisions` | warehouse integration assertion |
| Global BH/FDR needs the complete inventory | `finalize_inventory` verifies every terminal first | `test_inventory_finalizer_requires_every_terminal` |
| Missing inputs are explicit | `_unavailable_cells` | missing target/context and real finalizer tests |
| Retry is byte-preserving | content-addressed identity and terminal verification | replay test |
| Interrupted publication recovers | staging directory plus terminal-last publication | stale-staging test |
| Changed data cannot replay stale evidence | dataset and target byte digests in identity | changed-input test |
| Warehouse handoff is typed and offline | exact six-family contract from `data-warehouse@50bddf3` | imported pinned warehouse validator; no network API in module |
| Remote-safe worker transport | one stdin request; host-local sealed deployment resolver; one stdout result | `test_stdio_worker_resolves_host_local_deployment_and_emits_one_json` |
| Cross-host global closure | bounded raw-cell payload retained inside each terminal; no path fields | multiprocess test deletes both remote output trees before finalization |
| Global FDR has every p-value | exact 14-cell EURUSD / 6-cell ETH denominator and p-value validation | multi-result subprocess finalizer test |
| Profile refresh cannot rerun causal work | explicit `PROFILE_ONLY` unit/deployment mode and separate publisher | forbidden `FB.run_feature` test |
| Profile transport is bounded and path-free | envelope plus authenticated request, no causal payload, 2 MB limit | remote directories deleted before subprocess merge |
| Profile and causal closure cannot be confused | distinct PLAN mode, result mode, output schema and CLI | causal finalizer rejection test |
| Full EURUSD profile denominator is mandatory | PLAN hash plus exact terminal filename set | explicit incomplete 366-column denial test |
| Profile merge detects mutation | terminal, request, envelope and row digest verification | envelope mutation with resealed outer terminal is rejected |
| Final OLAP envelope uses disjoint authenticated sources | four profile families plus two adopted causal families | 366-feature composition fixture |
| Adopted causal denominator is exact | 14 target cells, rungs 1/2/3 and one decision for every feature | missing feature and duplicate causal row denials |
| Bundle adoption is path-safe | self-hashed report and confined relative envelope paths | real `adoption/warehouse_envelopes` layout in subprocess test |
| Adopted bytes are bound to the predictor bundle | self-hashed `BUNDLE_MANIFEST.json`, size and SHA-256 for report and envelopes | mutation denial before composition |
| Composition never recomputes causality | envelope-only coordinator command; no call to `fs_causal*` | source inspection plus full provider regression |
| Context cannot duplicate structural columns | ordered deduplication excludes timestamp and feature column | repeated history/pre-return/calendar regression with 1-D feature |

The full provider suite requires the optional `fit` environment. In the current
interpreter, the focused phase-1 and legacy causal suites are the governing
verification because they exercise the changed scientific path without EconML.
