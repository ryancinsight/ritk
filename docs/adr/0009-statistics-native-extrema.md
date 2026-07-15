# ADR 0009 — Native statistics extrema boundary

- Status: Accepted
- Change class: [major]
- Date: 2026-07-15
- Related: ADR 0002, MIG-654-03

## Context

`minimum_position` and `maximum_position` accepted the legacy generic
`Image<B, D>` boundary only to extract a host `Vec<f32>` before running their
substrate-independent argmin/argmax core. Repository search finds no in-tree
caller of that signature. The native image substrate already supplies a
contiguous borrowed `data_slice` with a typed host-access failure.

Keeping the unused legacy boundary would retain a Burn-typed public contract
after the operation itself has no provider-specific computation.

## Decision

Replace both public functions under their existing names with
`native::Image<f32, B, D>` inputs. They return
`anyhow::Result<Option<[usize; D]>>`: `None` represents an empty image and
`Err` represents a violated host-access contract. The existing row-major,
lowest-flat-index tie policy and single O(n) pass remain in the shared private
core. No legacy overload, alias, or forwarding adapter remains.

## Migration

Callers construct or retain a native image and propagate extraction failure:

```rust,ignore
let coordinate = ritk_statistics::minimum_position(&image)?;
```

The former `Option<[usize; D]>` result becomes
`Result<Option<[usize; D]>>`. A caller must handle storage failure separately
from an empty image.

## Consequences

The statistics package releases `0.2.0`. Its direct `burn-ndarray` test
dependency remains until every remaining legacy statistics operation is
removed; this ADR does not claim crate-wide Burn deletion.

## Verification

The focused native suite asserts 1-D and 3-D extrema, first-index ties,
single-voxel behavior, negative values, and a row-major flat-index round trip.
Compile, warnings-denied Clippy, doctests, and package documentation verify the
new public boundary. This is compile-time and empirical evidence, not a formal
proof.
