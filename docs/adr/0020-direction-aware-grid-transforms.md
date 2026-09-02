# ADR 0020: Direction-aware index/world transforms in grid-sweeping filters

- Status: Accepted
- Date: 2026-08-18
- Board item: ATLAS-RITK-TRANSFORM-DIRECTION-081

> **Revision (2026-08-19):** RITK-PARITY-171 showed that the Python NumPy
> boundary still constructed `[Z,Y,X]` images with the native identity
> direction. The boundary now stores the axial permutation and maps the
> Python `(disp_z, disp_y, disp_x)` API to the provider's physical
> `(x,y,z)` inputs for the TPS and iterative inverters. The original
> SimpleITK `<1e-4` oracles are green after a fresh wheel build.

## Context

ADR 0018 consolidated `Image`'s single-point coordinate transforms onto
`continuous_index_to_physical_point` / `physical_point_to_continuous_index`,
both of which apply the direction cosines and dispatch on the `CoordinateMap`.
Filters that sweep a whole grid never adopted that seam. Three of them compose
their own index/world mapping from `origin` and `spacing` alone:

| Site | Effect of the missing direction |
| --- | --- |
| `surface/marching_cubes.rs` `phys` | Mesh vertices placed on the index axes rather than the acquisition axes. |
| `iterative_inverse_displacement.rs` `phys` / `idx` (in both `apply` and `apply_native`) | The line search runs in the index frame while the displacement vectors it adds are world vectors — two different frames. |
| `inverse_displacement.rs` `world` (in both `apply` and `apply_native`) | Thin-plate-spline landmarks fitted in the index frame. |

The direction matrix is the identity only for an axis-aligned acquisition.
Clinical CT and MR are routinely oblique, so these produce plausible, silently
displaced geometry rather than an error — the failure mode ADR 0018 identified
for the single-point path, in the filters that path did not reach.

Two further facts emerged while measuring the defect and shape this decision.

First, the affected filters do not assume the *identity* direction. They assume
the *axial permutation* — index axis 0 (depth) along world z, axis 1 along y,
axis 2 along x — because they pair `spacing[2]` with the world x component and
`spacing[0]` with z. In this crate's convention the identity direction instead
sends the depth axis to world x. So the pre-existing behaviour was correct for a
conventional axial volume and wrong for everything else, including the identity
direction their own test fixtures used. `inverse_displacement`'s module doc did
record "axis-aligned (identity direction) is assumed", but the assumption it
named was not the assumption the code made, and no caller guaranteed either.

Second, the seam cannot simply be called per point.
`physical_point_to_continuous_index` recomputes a Gauss-Jordan inversion of the
direction matrix on every call. That is correct and cheap for a handful of
points and ruinous inside a per-voxel coordinate-descent loop that evaluates it
several times per voxel per iteration.

## Decision

Introduce one internal `CartesianGridGeometry` in `ritk-filter` that holds the
image's origin, spacing, direction and the *hoisted* inverse direction, and
answers both directions of the transform. All four displacement-filter sites
route through it; the four hand-rolled mappings are deleted.

It implements the same Cartesian formulas as the ADR 0018 seam —
`point = origin + D S index` and `index = S^-1 D^-1 (point - origin)` — and is
restricted to `CoordinateMap::Cartesian`, returning an error otherwise, so it
cannot answer a beam-space acquisition with a Cartesian result. Its constructor
also rejects a singular direction, which is a malformed file header rather than
a programmer error, matching ADR 0018's reasoning. This makes
`InverseDisplacementField::apply` and
`IterativeInverseDisplacementField::apply` fallible.

The inverse uses `Direction::try_inverse`, not the transpose. Orthonormality is
never validated anywhere in `ritk-spatial`, non-orthonormal directions are
representable, and every other transform path in the workspace uses the general
inverse; a transpose would be a silent scale error on exactly those inputs.

`MarchingCubesFilter` gains a `direction` field rather than routing through the
same type, because it takes a raw `&[f32]` and a shape and has no `Image` to
read geometry from. It keeps its existing `(ix, iy, iz)` component order for
`spacing`, and its direction *columns* are indexed to match, so the identity
default reproduces the previous output bit for bit. A caller holding an `Image`
must reverse both, which `ritk-snap`'s surface export now does in one named
function.

`FodVolume::world_to_voxel` in `ritk-diffusion` is left direction-free. It is
not a dropped matrix: `FodVolume` carries no direction, has no production
constructor, and nothing upstream of it holds an `Image`. It defines the frame
its own queries are answered in, and its points in, peak directions out, and
accumulated streamlines all live in that frame. The contract is now stated in
its Rustdoc together with what it requires of a future volume-level CSD
pipeline built from an oblique series.

## Consequences

`ritk-filter` takes a breaking change: two `apply` methods return
`anyhow::Result`, and `MarchingCubesFilter` gains a public field. In-repo
callers move in the same change. The Python bindings call the fallible
`apply_native` forms, but their NumPy construction boundary must also preserve
the `[Z,Y,X]` to physical `(X,Y,Z)` mapping; RITK-PARITY-171 closes that
consumer-side contract gap.

Numerical output changes for any volume whose direction is not the axial
permutation, which is the point. It is unchanged for volumes that are.

Two adjacent defects were found and fixed as part of wiring the direction
through, because the direction fix is meaningless without them:

- `ritk-snap`'s surface export passed `LoadedVolume`'s tensor-order spacing
  into the filter's `(ix, iy, iz)`-order parameter, transposing x and z on
  anisotropic volumes.
- `inverse_displacement` indexed `origin` by tensor axis while `origin` is in
  LPS component order. This was numerically inert — the thin-plate spline is
  translation-equivariant and both the landmarks and the evaluation points
  carried the same wrong offset, so it cancelled exactly — but it is gone with
  the rewritten geometry.

## Verification

Each fixed site has a regression test on a deliberately oblique direction (the
exact 3-4-5 rotation of the (x, y) plane, orthonormal with determinant 1, so
expected coordinates are exact rationals).

`CartesianGridGeometry`'s tests assert hand-computed physical coordinates in
both directions, a fractional round trip that would expose a transpose-for-
inverse substitution as a scale error, and a singular-direction rejection.
`marching_cubes` asserts three hand-computed oblique vertex positions.

The two displacement filters are verified by rigid-motion equivariance instead
of hand-computed coordinates, because their outputs are the result of an
iterative search and a linear solve: inverting on a grid with direction `R·A`
carrying components `R·u` must give exactly `R·v`. Both runs execute an
identical iteration sequence, so the identity is exact rather than
convergence-limited, and it is an oracle independent of the filters' own
arithmetic. A direction-blind implementation returns the same `v` for both
grids and fails.

Every one of these tests was confirmed to fail against the pre-fix
direction-free composition, and the axis-aligned tests were confirmed to pass
unchanged under both, which is what makes the oblique fixtures load-bearing:
an axis-aligned fixture cannot distinguish the two implementations at all.

## Revision history

- 2026-08-19: Revised for RITK-PARITY-171 to close the Python NumPy direction
  and displacement-component mapping gap; the SimpleITK parity oracles remain
  unchanged.
