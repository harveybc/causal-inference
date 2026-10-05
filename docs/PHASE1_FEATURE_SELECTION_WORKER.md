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
target, context or fold is retained as `NOT_AVAILABLE` or `NOT_APPLICABLE`; it is
never converted to a score.

## Complete inventory

```bash
feature-selection-column finalize-inventory --manifest inventory.json
```

The manifest has schema `feature_selection_inventory.v1`, an `inventory_id`, an
`output_dir`, and a `units` array containing `unit_id` and `terminal_path`.
Finalization refuses if any unit or artifact is missing or altered. Only after
the complete denominator is verified does it call `fs_causal_batch.finalize`,
which applies global BH/FDR per declared target and rung. The final terminal is
`inventory_terminal.json`; `causal_evidence.jsonl` contains the final states.

This module writes outbox documents but never contacts a live service. Loading
`feature_selection_envelope.v1` into the warehouse belongs to its owner process.
