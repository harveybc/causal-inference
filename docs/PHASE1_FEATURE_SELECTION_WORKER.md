# Phase-1 per-column worker

`feature-selection-column` is the CPU-only boundary between an inventory unit
and phase-1 feature-selection evidence. It does not select a feature by itself.

## One column

```bash
feature-selection-column run-column --config unit.json
```

The config follows
`provider/src/causal_inference_provider/contracts/feature_selection_unit.schema.json`.
Both `EURUSD` and `ETH` use the target definitions in the config. Target names,
horizons, context columns, mediator, volatility-regime column and placebo column
are not inferred from the asset name.

The command reads `TRAIN` only, computes or accepts a PS1 profile, calls the
existing three-rung implementation, and atomically publishes:

- `units/<unit-key>/ps1_profile.json`
- `units/<unit-key>/raw_causal_cells.jsonl`
- `units/<unit-key>/feature_record.json`
- `units/<unit-key>/terminal.json`
- `outbox/<unit-key>.json` (`feature_selection_envelope.v1`)

An identical replay returns `REPLAY` without changing bytes. A missing dataset,
target, context or fold yields an `UNAVAILABLE` unit terminal with explicit
`UNAVAILABLE` or `NOT_APPLICABLE` evidence cells; it is never converted to a
score.

## Orchestrated stdio transport

Remote workers use the path-free `stdio-json-v1` boundary from
`predictor@a83f7495`:

```bash
feature-selection-column --stdio --deployment-manifest deployment.json
```

The process reads exactly one `phase1.column_request.v1` object and prints
exactly one `phase1.column_result.v1` object. Standard output contains no logs.
Its result state is only `COMPLETED`, `UNAVAILABLE`, or `FAILED`;
`NOT_APPLICABLE` remains a metric or cell state. The request's inventory row
selects a resource key and feature column, while all host-local paths, target
definitions, folds, clocks and output roots come from the sealed, versioned
`phase1.column_worker_deployment.v1` manifest. Consequently an SSH worker does
not dereference a coordinator-local temporary path.

## Complete inventory

```bash
feature-selection-column finalize-inventory --manifest inventory.json
```

The manifest has schema `feature_selection_inventory.v1`, an `inventory_id`, a
warehouse run identity, an `output_dir`, and a `units` array containing
`unit_id` and `terminal_path`. A non-completed terminal is rejected unless that
unit explicitly carries the `UNAVAILABLE` disposition. Finalization refuses if
any unit or artifact is missing or altered. Only after
the complete denominator is verified does it call `fs_causal_batch.finalize`,
which applies global BH/FDR per declared target and rung. The final terminal is
`inventory_terminal.json`; `causal_evidence.jsonl` contains the final states and
`feature_selection_envelope.json` carries the final `selection_decisions` rows.

Each envelope has exactly the warehouse-owned top-level fields `schema_version`,
`run`, `rows`, and `envelope_sha256`; `rows` always contains all six row
families and every row has its own digest. This module writes outbox documents
but never contacts a live service. Loading the envelope into the warehouse
belongs to its owner process.
