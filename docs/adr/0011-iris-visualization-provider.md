# ADR 0011 — Iris visualization-provider boundary

- Status: Accepted
- Change class: [arch] [major]
- Date: 2026-07-21
- Related: VIS-665-01, [Iris ADR 0001](https://github.com/ryancinsight/iris/blob/e2edd47615454111b4b0df2e68dc6076161ba457/docs/adr/0001-domain-neutral-visualization-contract.md)

## Context

RITK contained two independent named-color implementations. `ritk-snap`
defined eight display maps and `ritk-vtk` defined five presets with a second
set of interpolation functions. Both compute the same domain-neutral function
from a normalized scalar to RGBA. The duplication gave the two packages
different public enums and allowed their color laws to drift.

The public Iris repository now owns validated normalized coordinates, named
zero-sized color strategies, runtime enum dispatch, and const-generic lookup
tables. File formats, medical windowing, GPU resource mechanics, UI state, and
clinical interpretation remain outside Iris.

## Decision

RITK consumes Iris at immutable revision
`e2edd47615454111b4b0df2e68dc6076161ba457`.

- `ritk-snap` re-exports `iris::color::NamedColorMap` and delegates every CPU
  and GPU lookup-table sample to Iris.
- `ritk-vtk` constructs its fixed 256-entry table from
  `LookupTable<NamedColorMap, 256>`.
- RITK deletes both local interpolation implementations and migrates every
  in-tree caller in the same change.
- Window/level remains RITK-owned. Its 8-bit result crosses the boundary
  through `Normalized::from_u8`, including the existing deterministic
  non-finite input policy.

The replacement is intentionally breaking. `Colormap` and `ColormapPreset`
are removed; callers use `NamedColorMap`. No forwarding aliases or adapters
remain.

## Rejected alternatives

- A RITK wrapper enum would preserve a duplicate public vocabulary and require
  conversion match arms whenever Iris adds a map.
- Retaining either interpolation implementation would leave two authorities
  for the same mathematical function.
- Moving medical windowing or VTK serialization into Iris would cross the
  domain-neutral provider boundary.
- Dynamic trait dispatch would add a vtable to a closed runtime selection;
  `NamedColorMap` provides exhaustiveness-checked enum dispatch instead.

## Consequences

Color sampling has one public owner and one implementation. RITK gains all ten
Iris maps through the same stable enum, while fixed lookup tables use inline
array storage with no heap allocation or strategy field. New maps are
implemented and verified in Iris before consumer adoption.

External source migration is `Colormap` or `ColormapPreset` to
`iris::color::NamedColorMap`; variant names shared by the old contracts remain
unchanged.

## Verification

Iris exhaustively verifies the 8-bit normalization grid and samples every map
over the normalized domain. RITK Snap pins display boundary vectors and
non-finite windowing behavior. RITK VTK compares every one of its 256 table
nodes for every named map bit-for-bit with direct Iris sampling. Package
format, warning-denied Clippy, Nextest, doctest, and Rustdoc gates cover the
consumer revision. The focused suite passes 943 of 943 tests.
`cargo-semver-checks` classifies only the intended removals described above:
one VTK enum and the Snap enum/module.
