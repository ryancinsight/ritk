# ADR 0019 - Borrowed region views on `Image`

- Status: Accepted
- Date: 2026-08-17
- Board item: ATLAS-RITK-VIEWS-047 (Atlas meta-repo board)

## Context

`Image` exposed seven host-data accessors and no way to read part of a volume.
`data_slice` borrows the whole buffer or fails on a strided layout; `data_cow`
borrows when contiguous and materialises the whole volume when not; the
`data_vec` family always owns. Every one of them is all-or-nothing, so a filter
that needed a neighbourhood took the entire flat buffer and did its own index
arithmetic against it, and a filter handed a strided image paid a whole-volume
copy to read anything at all.

The underlying tensor does not close this gap. `coeus_tensor::Tensor::slice`
produces a strided view cheaply — its `Shape`/`Strides` are `SmallVec<[usize;
4]>`, inline at every rank this crate uses — but a strided `Tensor` cannot be
*read*: `as_slice` and `iter` both assert contiguity, so the only route from a
tensor view to its values is `to_contiguous`/`to_vec`, which copies the region.
Views are free to make and impossible to use.

A sub-region is also not merely a sub-array here. Its origin moves by
`direction · (spacing ⊙ start)`; a view that forgets the shift reports the
parent's physical position for every voxel it contains.

Two premises recorded on the board did not survive checking and did not drive
this decision. ritk is not GAT-free — `ritk-snap`'s `series_tree` declares
`type Str<'b>` / `type Path<'b>` — and the linear interpolation kernel's
three-allocations-per-voxel defect was already fixed in `7635f1aa`, which
replaced the per-axis `vec!`s with `[_; MAX_RANK]` stack scratch. What remains
true is that nothing on the image data plane could express a borrowed region.

## Decision

Add `VoxelRegion<'a, T, D>` in a new `ritk-image::region` module: a borrowed,
possibly strided rectangular view holding shape, strides, offset and its own
corrected physical metadata, all in fixed-size arrays, so constructing one and
narrowing it further allocates nothing at any rank. `Image::region()` is the
single entry point and accepts strided images, which `data_slice` refuses and
`data_cow` can only serve by materialising.

Region operations are `subregion`, `clipped_window` (the shrinking-boundary
convention ITK's box filters use), `get`, `iter`, `subregions` (fixed-extent
tiling) and `rows`.

`rows` is a lending walker behind a `RowWalker` trait carrying `type Item<'a>`.
Its rows have two different owners: when the innermost axis is unit-stride —
every axis-aligned sub-region of an ordinary volume — a row is a direct borrow
of the source buffer; when it is not, the row is gathered into a scratch buffer
the walker owns and reuses, so a strided traversal costs one allocation rather
than one per row. An item that borrows the source in one case and the walker in
the other cannot be an `Iterator`, whose `Item` has no lifetime to tie to
`&mut self`. That is the whole justification for the generic associated type.

Everything whose items borrow only the source — `iter`, `subregions` — stays a
plain `Iterator`. A GAT would buy them nothing and would cost them the adaptor
ecosystem, so the seam is deliberately not applied uniformly.

The region type lives in `ritk-image` rather than upstream in coeus because the
metadata correction is this crate's domain, and because it needs no coeus change:
it is built entirely on `Tensor::layout()`, `Layout::{shape, strides, offset}`
and `storage().as_slice()`, all already public.

`BoxSigmaImageFilter` is migrated as the first consumer. Its two entry points
carried byte-identical bodies, each preceded by a whole-volume `Vec` copy
(`extract_vec_infallible` and `native::extract_image_vec`); both now share one
`sigma_over` kernel reading through a region. `apply` gains a
`B::DeviceBuffer<f32>: CpuAddressableStorage<f32>` bound, which is the breaking
part of this change.

## Deviation from the item's acceptance oracle

The oracle asked for `≤2` accessors on `Image` and one coordinate-transform
family. Neither is done here, and neither should have been attempted in one
change:

- The accessors have **681 call sites across roughly 250 files** (`data_slice`
  alone 562). Collapsing them is a mechanical but genuinely large migration, and
  a partial pass would leave exactly the compatibility shims the standards
  forbid.
- Coordinate transforms are **not three families but 17 distinct
  implementations across 22 sites**. ADR 0018 already consolidated the
  single-point pair on `Image`; the remaining sites sit in `ritk-filter`,
  `ritk-transform`, `ritk-registration` and `ritk-diffusion`, and **eight of
  them are direction-free**, silently wrong for oblique geometry. That is a
  correctness item, not a duplication item, and it deserves its own ADR.

Both are filed rather than half-done. This change delivers the seam, one
migrated filter, and the measurement.

## Consequences

`ritk-image` gains a module and five exported names; `ritk-filter`'s
`BoxSigmaImageFilter::apply` gains a trait bound on `B::DeviceBuffer<f32>`. Both
in-repo call patterns already satisfy it — tests use `SequentialBackend`,
`ritk-python` uses `MoiraiBackend` — so no call site changes, but it is a
breaking signature for any external caller with a backend whose device buffer is
not CPU-addressable. Such a caller could not have used the filter regardless,
since the old body extracted to host memory unconditionally.

No shim, re-export or forwarding wrapper is introduced.

`Image::region()` accepting strided input is the capability that was previously
absent: a permuted or sliced image is now readable in place instead of only
through `data_cow`'s whole-volume materialisation.

## Verification

Thirteen tests in `ritk-image::region` cover row-major traversal order, stride
inheritance, bounds rejection, clipped-window shrinking at edges and corners,
empty regions, tiling with a dropped ragged tail, `ExactSizeIterator`/fused
contracts, and both `rows` paths. Two are value oracles rather than restatements
of the implementation: `subregion_shifts_origin_by_direction_times_spacing`
computes the expected origin by hand under a 90° rotation and anisotropic
spacing, and `subregion_origin_agrees_with_the_canonical_forward_transform`
cross-checks the same number against `continuous_index_to_physical_point`, an
independent code path.

`box_sigma`'s three existing SimpleITK-pinned value tests pass unchanged, which
is the evidence that the rewrite preserved semantics.

The allocation claim is measured, not asserted.
`crates/ritk-filter/tests/box_sigma_allocation.rs` installs a counting global
allocator and reports, for a 64³ `f32` volume (262 144 voxels, 1 048 576 B):

| Operation | Allocations | Bytes |
| --- | ---: | ---: |
| one whole-volume host copy (`try_data_vec`) | 2 | 2 097 152 |
| bare parallel collect, output only | 2 | 1 048 608 |
| `BoxSigmaImageFilter::apply_native` | 3 | 1 048 648 |

The filter allocates one block and 40 bytes above the scheduler's own floor for
producing the output — 72 bytes above the output volume itself. The previous
implementation's `extract_image_vec` would add a further 1 048 576 B. Per-voxel
or per-window allocation would show as a count near 262 144 rather than 3.

Limits: the counter is process-wide and records bytes *requested* at the
allocator, not peak resident set, so it bounds copying from above rather than
measuring footprint. The parallel runtime allocates its worker buffers on first
substantial use (841 allocations / 7.4 MB), which the test pays in a full-size
warm-up before measuring; without that warm-up the figure describes pool
construction rather than either kernel. `dhat` is not a dependency anywhere in
this workspace and was not added for a total an exact counter already provides;
the tradeoff is that this reports totals rather than per-call-stack attribution.
