# ADR 0018 - One coordinate-transform API on `Image`

- Status: Accepted
- Date: 2026-08-13
- Board item: [FIX-690-01](../../backlog.md#fix-690-01-minor---consolidate-the-image-coordinate-transform-api)

## Context

`Image` carried two public single-point coordinate-transform APIs, in adjacent
`impl` blocks with identical bounds:

| Pair | Honours `CoordinateMap` | Singular direction | External callers |
| --- | --- | --- | --- |
| `continuous_index_to_physical_point` / `physical_point_to_continuous_index` | yes | typed error | 0 |
| `transform_continuous_index_to_physical_point` / `transform_physical_point_to_continuous_index` | no | `expect` panic | 18 |

The map-aware pair was correct and unused. The `transform_`-prefixed pair
applied `point = origin + D S index` unconditionally and had every caller —
`ritk-filter`'s displacement resampler among them.

For a Cartesian raster the two agree exactly, which is why the divergence went
unnoticed: `Image::new` sets `CoordinateMap::Cartesian` and only an explicit
`with_coordinate_map` changes it, so no test covered a non-Cartesian image
through the single-point path. On a `CurvilinearArray` or `PhasedArray3D`
image the index pair is a beam and a sample, not a raster coordinate, and the
Cartesian formula returns a point in no physical space at all. On the fixture
in `types.rs` the far sample of the centre beam maps to (32 m, 63 m) instead
of (66 mm, 9 mm) — an error of roughly 30 m, silent and unsignalled.

The batch forms (`world_to_index_native`, `index_to_world_native`) had always
dispatched on the map, so the two granularities of the same operation
disagreed with each other.

`transform_physical_point_to_continuous_index` additionally reached a
`.expect("direction matrix must be invertible")` on a singular direction
matrix, which is input-dependent: direction cosines are read from file
headers, and a degenerate one is a malformed input rather than a programmer
error.

## Decision

Keep one pair, `continuous_index_to_physical_point` and
`physical_point_to_continuous_index`, and delete the `transform_`-prefixed
pair. All 18 call sites move in the same change.

The surviving names are the ones the batch-form documentation already
cross-references, they carry no redundant `transform_` prefix on a type whose
method this plainly is, and they match the ITK vocabulary the rest of the
crate follows.

`physical_point_to_continuous_index` returns `anyhow::Result<Point<D>>`, so a
singular direction matrix and a point outside a beam-space acquisition are
both reported rather than panicking or returning a fabricated index.

No deprecated re-export or forwarding wrapper is introduced: a shim would
preserve exactly the wrong-answer path this removes.

## Consequences

`ritk-image` 0.3.0 -> 0.4.0. Consumers rename the call and handle the
`Result`:

```rust
// before
let index = image.transform_physical_point_to_continuous_index(&point);
let point = image.transform_continuous_index_to_physical_point(&index);

// after
let index = image.physical_point_to_continuous_index(&point)?;
let point = image.continuous_index_to_physical_point(&index);
```

A caller that knows its image is Cartesian with an invertible direction can
`expect` on the `Result`; the in-repo test call sites do exactly that, which
keeps the assumption written down at the site that makes it.

Beam-space images now transform correctly through the single-point path, so
`ritk-filter`'s displacement resampler is correct on curvilinear and
phased-array input for the first time.

## Verification

`single_point_transform_honours_the_curvilinear_map` asserts the single-point
transform agrees with the batch form on the curvilinear fixture. The batch
form is an independent oracle rather than a restatement: it reaches the
geometry through a different code path, and it was already correct while the
single-point path was not, so this pins the consolidated method to behaviour
that predates it. The test fails against the deleted implementation with a
~30 m discrepancy.
