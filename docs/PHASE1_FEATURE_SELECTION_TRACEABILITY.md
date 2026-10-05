# Phase-1 worker traceability

| Requirement | Structural evidence | Behavioral evidence |
|---|---|---|
| Config-driven EURUSD/ETH targets | `TargetDefinition`; `feature_selection_unit.v1` schema | `test_eth_and_eurusd_target_definitions_are_config_driven` |
| Existing scientific method is reused | `FB.run_feature`; `FB.finalize` calls | focused legacy suite (`test_fs_causal*`) |
| One TRAIN column receives a PS1 profile | `profile_series`; supplied-profile path | `test_run_column_is_idempotent_and_profiles_missing_dates` |
| Raw cells cannot masquerade as final states | envelope state `RAW_AWAITING_GLOBAL_BH_FDR` | `test_inventory_finalizer_requires_every_terminal` |
| Global BH/FDR needs the complete inventory | `finalize_inventory` verifies every terminal first | `test_inventory_finalizer_requires_every_terminal` |
| Missing inputs are explicit | `_unavailable_cells` | missing target/context and real finalizer tests |
| Retry is byte-preserving | content-addressed identity and terminal verification | replay test |
| Interrupted publication recovers | staging directory plus terminal-last publication | stale-staging test |
| Changed data cannot replay stale evidence | dataset and target byte digests in identity | changed-input test |
| Warehouse handoff is typed and offline | `feature_selection_envelope.v1` contract | envelope assertion; no network API in module |

The full provider suite requires the optional `fit` environment. In the current
interpreter, the focused phase-1 and legacy causal suites are the governing
verification because they exercise the changed scientific path without EconML.
