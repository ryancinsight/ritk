# ADR 0021 - Two host-data accessors on `Image`, one grid-transform family

- Status: Accepted
- Date: 2026-08-18
- Board item: ATLAS-RITK-VIEWS-047 (Atlas meta-repo board)
- Supersedes the "Deviation from the item's acceptance oracle" section of ADR
  [0019](0019-borrowed-region-views.md)

## Context

ADR 0019 deferred two clauses of its item and recorded the reason: `Image`'s
seven host-data accessors had "681 call sites across roughly 250 files"
(`data_slice` alone 562), and coordinate transforms were "17 distinct
implementations across 22 sites".

Re-measuring both before starting changed the shape of the first one
materially. The 681 figure counts every call to any of the seven accessors, but
`data_slice` is one of the accessors that survives a collapse to two — its 565
sites are not a migration, they are the evidence for keeping it. The accessors
that must go carry **60 call sites**, not 681. The deferral was correct about
the direction and wrong about the size, and the wrong size is what made it look
undoable.

Current counts (this revision):

| Accessor | Sites | Distinct behaviour |
| --- | ---: | --- |
| `data_slice() -> Result<&[T]>` | 565 | borrow, or fail on a strided layout |
| `data_cow_on(&B) -> Cow<[T]>` | 59 | borrow, or materialise a compact copy |
| `data_cow()` | 12 | `data_cow_on` on `B::default()` |
| `data_vec()` | 18 | `data_cow_on(..).into_owned()` |
| `data_vec_on(&B)` | 4 | `data_cow_on(..).into_owned()` |
| `try_data_vec()` | 13 | the same, in a `Result` that is always `Ok` |
| `try_data_vec_on(&B)` | 15 | the same, in a `Result` that is always `Ok` |

Only the first two rows describe a behaviour; the other five are a convenience
axis (`B::default()`) and an ownership axis (`.into_owned()`) crossed with it.
The `try_` pair is worse than redundant: its own Rustdoc stated that
"extraction succeeds for every valid image", so 28 call sites carried a `?` or a
`.context(..)?` on a branch that cannot be taken — fallibility as decoration.

The transform count re-measured at 17 distinct implementations across 22 sites
in the four crates ADR 0019 named, confirming that figure, but the audit's scope
was too narrow: **ten further sites** exist in `ritk-snap` (4), `ritk-io` (3),
`ritk-cli` (1), `ritk-connectome` (1) and `ritk-vtk` (1), for a workspace total
of **32 sites / 25 distinct implementations / 10 direction-free sites**. Two of
the 22 that ADR 0019 counted are no longer implementations: `inverse_displacement`
and `iterative_inverse_displacement` became consumers of `CartesianGridGeometry`
in ADR 0020.

## Decision

### Two accessors

`Image` keeps `data_slice` and `data_cow_on` and nothing else. The choice
follows the caller evidence directly: 565 sites want a contiguous borrow and
already handle the strided failure, 59 want the layout-independent form, and the
50 that want ownership want it as a consequence of one of those two, which
`.into_owned()` states at the site that pays for it.

The default-backend convenience goes with them. `Image::from_flat`/`from_flat_on`
keeps its pair because construction has no other route; extraction does, and the
budget is two.

All 60 sites are migrated in this change, across every crate that had one. The
28 dead `?`/`.context(..)?` branches are deleted rather than preserved, which is
the visible part of the diff outside the accessor module.

### One grid-transform family

`CartesianGridGeometry` — added by ADR 0020 as a `pub(crate)` 3-D type inside
`ritk-filter` — moves to `ritk-spatial` as a public type generic over rank. That
is the deepest common ancestor of every crate holding one of the 32 sites, so
for the first time the family has a home all of them can reach; a `pub(crate)`
type in a leaf crate is not a family, whatever its contents.

`Image::grid_geometry()` in `ritk-image::transform::grid` is the entry point:
one call from any image, dispatching the beam-space rejection and the singular
direction to typed errors. It is the third granularity alongside ADR 0018's
per-point pair and ADR 0020's per-tensor batch, and it is the one a per-voxel
sweep needs.

The error type is `NonCartesianGrid` rather than `anyhow::Error`, because the
two failures are distinguishable and a library caller may want to distinguish
them. The old message named displacement-field inversion, its first caller,
which was a leak from the type's origin.

Four sites are converted to consumers in this change:

- `InverseDisplacementField::{apply, apply_native}` and
  `IterativeInverseDisplacementField::{apply, apply_native}` had byte-identical
  cores — 179 and 143 lines respectively, verified line-for-line — wrapped in a
  legacy-vs-native I/O shell. Each pair collapses to one private
  `invert_components` taking host buffers and a `&CartesianGridGeometry<3>`;
  the entry points keep only their extraction, their degenerate-case return, and
  their rebuild. `inverse_displacement.rs` goes 738 → 583 lines,
  `iterative_inverse_displacement.rs` 393 → 263.
- `DicomOrient::{apply, apply_native}` carried a verbatim-duplicated corner-origin
  loop; both now call `geometry.point(corner)`, which is the same formula.

## Deferred, with counts

Thirty of the 32 sites remain (the two `DicomOrient` sites became consumers),
across 24 distinct implementations. They divide into two kinds and only one of
them is duplication:

- **20 sites are direction-aware and correct**, each doing the affine inline
  where routing to `grid_geometry()` is a mechanical substitution: `ritk-filter`
  `cpr_helpers`, `cpr`, `marching_cubes`, `resample`, `fractal_dimension`,
  `transform::{roi, pad, shrink}`; `ritk-transform`
  `displacement_field::geometry` (2); `ritk-registration` `label_transfer`,
  `classical::native::transform`; `ritk-snap` `cursor_info`,
  `rtstruct_overlay`, `rt_dose_analytics` (a verbatim duplicate of the
  previous), `rtdose_overlay`; `ritk-io` `dicom::writer::{metadata, series}`
  and `dicom::seg::converters` (slice-axis only); `ritk-cli` `tract`.
- **10 sites are direction-free and wrong for an oblique volume.**
  `ritk-filter::transform::expand` `apply`/`apply_native` are the sharp case:
  every other member of the ROI/pad/shrink/orient/expand geometry-update family
  honours the direction, so an oblique volume's origin shifts along index axes
  instead of world axes. `sources.rs` (4 sites, one formula) is defensible —
  synthetic phantoms define their own frame — but undocumented, unlike
  `ritk-diffusion`'s `FodVolume::world_to_voxel`, which states its
  direction-free contract; `ritk-diffusion`'s `NoddiVolume::direction_at` does
  not. `ritk-connectome::label_at` and `ritk-vtk`'s threshold point emission
  are the remaining two.

The direction-free set is a correctness change with a numeric consequence for
oblique inputs, and `expand`'s fix cannot be bit-identical to its current
arithmetic even under the identity direction (`o - 0.5s + 0.5s'` versus
`o + D(S · δ)` associate differently). It therefore belongs in its own change
with its own oblique fixtures, exactly as ADR 0020 handled the previous batch —
not folded into an accessor migration.

## Consequences

**Breaking.** Five public methods are removed from `Image`:
`data_cow`, `data_vec`, `data_vec_on`, `try_data_vec`, `try_data_vec_on`. An
external caller migrates by appending `.into_owned()` and, for the
default-backend forms, passing `&B::default()`. No re-export, `#[deprecated]`
shim or forwarding wrapper is introduced; the substitution is local at every
site.

`CartesianGridGeometry` becomes public API of `ritk-spatial` and generic over
rank, along with the `NonCartesianGrid` error. `ritk-filter::grid_geometry` is
deleted rather than re-exported.

`ritk-image` gains `Image::grid_geometry()` and a `transform::grid` module.

## Verification

The accessor migration is behaviour-preserving by construction: every removed
method had a one-expression body over a surviving one, and each site was
rewritten to that expression. The evidence is the existing suite — no test was
changed to accommodate it — plus `cargo semver-checks`, which is asked to report
the five removals as the declared breaks they are.

`CartesianGridGeometry`'s test module moves to `ritk-spatial` and gains three
tests for what the move and the generalisation newly claim: a 2-D instantiation
producing the hand-computed point and its round trip (the rank generalisation is
one formula, not a second code path), `axis_direction` returning the direction
*column* rather than the row (the two differ in sign under the oblique fixture,
so a transpose is visible), and a curvilinear map rejected rather than mapped
affinely (the constructor's whole reason for taking a map). The existing six —
oblique point, oblique index, non-lattice round trip, displacement without
translation, identity-direction bit-identity, singular-direction rejection —
carry over unchanged.

The two displacement inverters and `DicomOrient` are covered by their existing
value tests, which is the evidence that the extraction preserved semantics: the
cores were verified line-identical by `diff` before extraction, so a behavioural
change could only enter through the seams, and the seams are the parts the tests
already pin.
