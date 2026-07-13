# ADR 0005 — Static displacement and SSMMorph native boundary

- Status: Accepted
- Change class: [major]
- Date: 2026-07-11
- Related: backlog MIG-528-01; ADR 0004

## Context

SSMMorph already computes its displacement with Coeus, but the registration
boundary copied fixed and moving Burn images to host vectors, rebuilt Coeus
inputs, then copied the Coeus output back into three Burn tensors. The returned
`StaticDisplacementField` duplicated the trainable field's physical geometry,
coordinate mapping, replicated-border interpolation, and resampling logic.
Only the SSMMorph boundary and its integration test consume this static type.

## Decision

Migrate the connected boundary atomically:

1. native Coeus images enter SSMMorph by zero-copy tensor reshape;
2. the Coeus displacement output is sliced into Coeus component views without
   host extraction;
3. static and trainable fields share one geometry/validation/grid module;
4. static sampling uses the canonical `linear_interpolation::<D>` family and
   the `Replicate` ZST; and
5. the Burn static-field implementation and converted allowlist entries are
   deleted without compatibility wrappers.

The static field remains a distinct non-trainable type. It stores raw Coeus
tensors and does not implement `Module` or expose optimizer parameters.

## Rejected alternatives

- Retain the Burn image API and hide copies in conversion helpers: preserves
  the legacy boundary and violates zero-copy ownership.
- Reuse the trainable field with gradients disabled: gives static state a
  parameter-oriented representation and mixes responsibilities.
- Keep parallel Burn and Coeus static types: duplicates the operation family
  and creates a permanent compatibility path.

## Verification

- exact constant-field transform and resampling laws in 2-D;
- native SSMMorph integration with exact component shape and identity-output
  values from the zero-initialized projection;
- full transform and registration nextest suites;
- warning-denied Clippy, Rustdoc, doctests, and a clean migration audit.
