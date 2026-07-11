# ADR 0004 — Trainable displacement-field native cutover

- Status: Accepted
- Change class: [arch]
- Date: 2026-07-11
- Related: backlog MIG-526-01; ADR 0001; ADR 0002

## Context

The trainable `DisplacementField` is RITK's remaining dense-field optimizer
boundary. Its Burn implementation owns four coupled responsibilities:

1. `D` parameter tensors and reverse-mode gradient registration;
2. named persistence of components plus origin, spacing, and direction;
3. physical-to-index mapping followed by differentiable interpolation; and
4. integration with registration optimizers through Burn `Module` /
   `AutodiffModule` visitation.

Commit `e75d8748` restored the Burn module plumbing but did not update the
migration allowlist. The surface is load-bearing: `DisplacementFieldTransform`
uses its record and autodiff implementations, and the registration optimizer
traits require `AutodiffModule`. Deleting only `module.rs`, hiding it from the
audit, or wrapping it with a native facade would remove capability or create a
compatibility shim.

The current Coeus contracts are insufficient for a native same-change cutover:

- `coeus_nn::Module::parameters()` returns unnamed `Var`s even though
  `Parameter` owns a name;
- `coeus_tensor::StateDict` is an eager, unbounded bespoke reader and does not
  archive domain metadata;
- Coeus exposes differentiable trilinear interpolation, but the field contract
  also ships 2-D interpolation and boundary semantics; and
- the active RITK optimizers are coupled to Burn module visitation rather than
  the already-proven Coeus optimizer path.

## Decision

Migrate the trainable field as one provider-first vertical scope. No parallel
field type, branded trait, forwarding wrapper, or dual record format is
introduced.

### Provider increments

1. **Named parameter collection in Coeus.** Extend the canonical module seam
   so nested modules expose stable parameter names with their `Var`s. Preserve
   the current zero-cost static module structure; do not add dynamic dispatch.
2. **Bounded archived state.** Replace the eager `StateDict` reader with a
   bounded, validated rkyv archive contract that stores tensor shape/data and
   displacement metadata. Compatibility decoding, if required for existing
   external checkpoints, lives at an explicit import boundary and is deleted
   after conversion; RITK does not own a second serializer.
   Coeus commit `f52c095` delivers the bounded archive contract; the consumer
   cutover assigns stable names to the field geometry tensors.
3. **Dimension-complete interpolation.** Add one Coeus interpolation operation
   family parameterized by dimension and boundary policy, with 2-D and 3-D
   forward/backward implementations sharing the same contract. CPU laws and
   backend differential tests cover value and gradient semantics.
4. **Optimizer consumption.** Route trainable-field registration through the
   existing Coeus `Var` optimizer path. Optimizers consume the named parameter
   collection directly; they do not emulate Burn visitors.

### Atomic RITK consumer set

The removal increment updates these consumers together:

- `ritk-transform/displacement_field/{core,grid,module,resample,transform}`;
- `ritk-transform` public re-exports and transform tests;
- `ritk-registration` displacement-registration tests and the optimizer trait,
  Adam, momentum, gradient-descent, regular-step, adaptive-stochastic,
  multiresolution, engine, and global-MI bounds that currently require
  `AutodiffModule`;
- composition code whose generic bounds require Burn `Module`; and
- PM/audit artifacts and the Burn allowlist.

The static displacement field remains a separate non-trainable value type. It
may share interpolation and geometry operations, but it must not acquire
parameter or optimizer responsibilities.

## Rejected alternatives

- **Allowlist-only closure.** Restores CI but does not migrate capability. The
  admitted entry remains owned by MIG-526-01 and is deleted by the cutover.
- **RITK-owned substrate trait.** Recreates the old Burn module API over Coeus
  and becomes a permanent shim.
- **Delete the trainable field.** Removes a registration capability instead of
  replacing it.
- **3-D-only native field.** Narrows the shipped 2-D/3-D contract to make the
  migration easier and is therefore not acceptable.

## Verification contract

- Exact constant and affine displacement values in 2-D and 3-D.
- Finite-difference gradient checks for every component parameter.
- CPU/backend differential interpolation tests with bounds derived from
  interpolation depth and scalar epsilon.
- State archive round-trip, malformed/truncated input, allocation bounds, and
  metadata equality.
- Registration optimization decreases the same analytical objective from the
  same initialization.
- `cargo nextest run` for Coeus providers, `ritk-transform`, and
  `ritk-registration`; warning-denied Clippy, Rustdoc, doctests, and a clean
  Burn audit with `displacement_field/module.rs` deleted from the allowlist.

## Revisit trigger

Revisit only if Coeus gains an equivalent dimension-complete field primitive
or an external persisted-checkpoint requirement proves that the current Burn
record format must remain readable beyond the migration window.
