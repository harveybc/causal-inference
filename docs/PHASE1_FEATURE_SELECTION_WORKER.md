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

Every completed or measured-unavailable result carries a bounded
`phase1.column_finalization_payload.v1`. It contains the authenticated request,
the feature record and the complete raw causal denominator: 14 target cells for
EURUSD or 6 for ETH. It contains no source series or training arrays. The
payload is self-hashed, capped at 2,000,000 canonical JSON bytes and retained
inside the orchestrator terminal, so finalization never depends on a remote
worker directory.

The coordinator closes retained orchestrator terminals with the same placeholder
interface used by `predictor@a83f7495`:

```bash
feature-selection-column finalize-terminals \
  --plan PLAN.json --terminals terminals/ --output finalizer-result.json
```

This operation authenticates the plan, every terminal, embedded worker result,
request, inventory row, unit envelope and finalization payload. It then applies
global BH/FDR from the transported raw p-values, without reading datasets or
recomputing a causal cell, and atomically writes `phase1.finalizer_result.v1`.
Its embedded warehouse envelope contains the final causal evidence and
selection-decision rows.

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

The transport schemas are retained beside the unit schema as
`phase1_column_finalization_payload.schema.json` and
`phase1_finalizer_result.schema.json`; profile merge output uses
`phase1_profile_merge_result.schema.json`.

## Profile-only refresh

`PROFILE_ONLY` is a separate campaign mode for recomputing the current PS1 and
pair-relation tables without rerunning a causal cell. The mode must be declared
both in the host-local population manifest and in `PLAN.json`; it is deliberately
absent from `phase1.column_request.v1`, so request identity and host-local data
resolution are unchanged.

Each stdout result remains `phase1.column_result.v1`, carries
`mode: PROFILE_ONLY`, the authenticated request and a warehouse-valid envelope.
Only `sampling_quality`, `variable_profiles`, `information_metrics` and
`pair_relations` contain rows. `causal_evidence` and `selection_decisions` are
present because the warehouse owns a fixed six-family contract, but are empty.
There is no `finalization_payload`. The complete stdout object is limited to
2,000,000 canonical JSON bytes.

The coordinator merges retained results with:

```bash
feature-selection-column merge-profile-terminals \
  --plan PLAN.json \
  --terminals retained-terminals/ \
  --output profile-merge-result.json
```

The command authenticates the complete PLAN denominator (366 items for the
current EURUSD inventory), every terminal, request, inventory row, campaign and
warehouse row. It never reads a worker path. Rows are sorted canonically and
the output is a deterministic `phase1.profile_merge_result.v1` containing one
warehouse envelope. The causal finalizer refuses a `PROFILE_ONLY` plan, so this
metric refresh cannot authorize feature selection or masquerade as BH/FDR
closure. A transported `UNAVAILABLE` result is explicit but cannot enter a
`PROFILE_METRICS_COMPLETE` merge: every item in the declared denominator must
be `COMPLETED`.

## Final EURUSD OLAP envelope

After the 366/366 profile merge, the coordinator can combine those current
metrics with the already adopted EURUSD causal evidence from the predictor
deployment bundle:

```bash
feature-selection-column combine-adopted-eurusd \
  --profile-result profile-merge-result.json \
  --adoption-dir <predictor-bundle-root> \
  --output feature-selection-envelope.json
```

`--adoption-dir` may name either the bundle root containing
`adoption/ADOPTION.json` or that `adoption` directory itself. Paths named by
the report are resolved beneath the bundle root and cannot escape it.

The command authenticates the profile result, all six warehouse families and
row digests in every adopted envelope, `BUNDLE_MANIFEST.json`, the adoption
report, both inventory namespaces, 366-feature equality and the complete EURUSD
denominator of 14 target cells, three causal rungs and one decision per cell.
Duplicate or missing cells are rejected. It takes the four noncausal families
only from the profile merge and the two causal families only from adopted
evidence; no causal method is called.

The fresh profile orchestrator inventory and the retained semantic adoption
inventory are allowed to differ. All 366 adopted envelopes must nevertheless
share exactly one nonempty 64-character inventory identity, and every adopted
row must carry the population digest derived from that adopted identity. The
output binds both source inventories in its input identity and normalizes all
six row families to a new combined EURUSD population digest derived from both.
The output itself, rather than a parallel wrapper, is one deterministic
`feature_selection_envelope.v1` with a new run, campaign, code, input and
inventory identity suitable for warehouse ingestion.
