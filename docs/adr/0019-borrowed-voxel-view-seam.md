# ADR 0019 - Borrowed voxel views on `Image`

- Status: Proposed
- Date: 2026-08-13
- Board item: [ARCH-696-01](../../backlog.md#arch-696-01-archmajor---borrowed-voxel-view-seam)

## Context

`Image` has no indexed access to its own voxels. Every route to pixel data is
a *flat logical row-major sequence*, and there are seven of them:

| Method | Shape | Backend | Cost on a strided or offset tensor |
| --- | --- | --- | --- |
| `data_slice` | borrow or fail | — | fails |
| `data_cow_on` | borrow or materialize | explicit | whole-buffer copy |
| `data_vec_on` | own | explicit | whole-buffer copy |
| `try_data_vec_on` | own, fallible | explicit | whole-buffer copy |
| `data_cow` | borrow or materialize | `B::default()` | whole-buffer copy |
| `data_vec` | own | `B::default()` | whole-buffer copy |
| `try_data_vec` | own | `B::default()` | whole-buffer copy |

That is a 3×2 cartesian fan-out (three answers to "how do I pay for the copy"
× two answers to "which backend") plus one strict-borrow outlier. The backend
is a *parameter*: `data_cow()` is `data_cow_on(&B::default())` and nothing
else. It is not an API dimension and should not have produced three methods.

The deeper problem is that all seven answer the same question — *give me a
flat host slice* — and that question forces a copy the caller usually does not
need. A filter, an interpolator or a neighbourhood operator wants **indexed
access to voxels**, which a shape, a stride and an offset express directly and
for free. Only serialization genuinely needs linearization.

The consequences are visible today:

- `ritk-interpolation`'s linear kernel calls `data.to_contiguous()` on the
  volume and again on the index tensor before reading either, then recomputes
  row-major strides into a fresh `Vec` per call. Sampling 1000 points from a
  64³ volume copies 262 144 voxels to read 8000 of them.
- `ritk-filter`'s `sample_moving_at_world` calls `moving.data_cow()` — a whole
  moving volume — to sample a point list.
- There is no borrowed tile, no neighbourhood iterator and no `chunks_exact`
  seam, so no filter can be written against a window without materializing.

`Tensor::to_contiguous()` is also sharper than it looks. Its free path
requires `is_contiguous() && offset == 0`, but `Layout::is_contiguous()`
inspects strides only. A row slice of a contiguous tensor therefore *reports
contiguous* and still costs a full copy. `is_contiguous()` is not a valid
predicate for "this is free", and nothing in RITK currently distinguishes the
two.

RITK has no lending iterator and no `Item<'_>` GAT anywhere, so the shape of a
lending seam is an open question rather than an established local pattern.
(RITK is not GAT-free — `ritk-snap`'s `series_tree` declares `type Str<'b>` and
`type Path<'b>` — but no GAT is an iterator item.)

The scale of the accessor surface, by call site:

| accessor | production | consuming crates |
| --- | --- | --- |
| `data_slice` | 76 | ritk-segmentation 36, ritk-filter 17, ritk-cli 10, ritk-statistics 8, others 5 |
| `data_cow_on` | 27 | 13 crates, almost all format writers |
| `try_data_vec_on` | 14 | ritk-filter only |
| `try_data_vec` | 12 | ritk-io 7, ritk-registration 3, ritk-tensor-ops 2 |
| `data_cow` | 9 | ritk-filter only |
| `data_vec` | 7 | ritk-registration 4, ritk-cli 2, ritk-filter 1 |
| `data_vec_on` | 3 | ritk-python only |

152 production call sites, plus roughly 480 in tests — of which about 434 are
`data_slice`. Any single change that collapses this surface is unreviewable,
which is why this ADR separates the seam from its adoption.

## Decision

### The view type is `leto::ArrayView<'a, T, N>`, not a RITK type

Coeus stores a layout; `coeus_leto::to_leto_view(&CoeusLayout, &'a [T]) ->
Result<ArrayView<'a, T, N>>` already converts one into a borrowed leto view,
validating the layout's footprint against the slice so an out-of-bounds layout
is rejected rather than producing an unsound view. leto owns the borrowed-array
vocabulary — `slice`, `get`, `axis_iter`, `lanes`, `reshape`, `permute` — and
the iterator family over it: `Tiles`, `Windows`, `ExactChunks`, `AxisChunks`,
`Lanes`, all yielding `ArrayView`. `ArrayView` is `Copy`.

RITK therefore adds an *adapter*, not a vocabulary:

```rust
pub fn tensor_view<T, B, const N: usize>(&Tensor<T, B>) -> Result<ArrayView<'_, T, N>>
impl Image<T, B, D> { pub fn view(&self) -> Result<ArrayView<'_, T, D>> }
```

Defining `RitkImageView` would fork leto's vocabulary and forfeit its whole
iterator family for no capability. The adapter reads `tensor.storage()`
directly rather than `Tensor::as_slice`, because `as_slice` pre-applies the
offset and asserts contiguity — both of which the layout already encodes.

### No GAT. A plain `Iterator` suffices, and RITK writes no iterator at all

The lending question resolves against RITK needing a GAT, on the same grounds
leto used when it *removed* one:

> The GAT is warranted only when an item genuinely borrows from the iterator —
> a reused scratch buffer, for example. An iterator whose items borrow the same
> data the iterator borrows is a plain `Iterator` and must be declared as one:
> the narrower item lifetime would forfeit `IntoIterator`, `zip`, `enumerate`,
> `rev`, `ExactSizeIterator` and every parallel bridge for no capability.
>
> — `leto/crates/leto/src/application/iter/lending.rs:17`

RITK's case is that case, and the check is mechanical: an image tile or
neighbourhood iterator would hold `&'a [T]` (the host storage, kept alive by
the tensor's `Arc`) plus a `Copy` layout, and build each item from that slice.
The item borrows `'a`, never the iterator. So it is a plain `Iterator`, and
`leto::Tiles` — which holds exactly `&'a [T]` plus a `Layout<N>` — already *is*
that iterator. RITK does not need to write one.

This is not an assumption transplanted from leto's conclusion; it is the same
predicate evaluated against RITK's operands, which happen to be leto's
operands once `tensor_view` has run.

One future case would invert the answer: an iterator that yields *padded or
clamped* boundary neighbourhoods must materialize each neighbourhood into a
reused scratch buffer, and its item then borrows the iterator. That is a
genuine lending iterator, and it must implement `leto::LendingIterator` — which
leto deliberately retained for exactly this case — rather than motivate a RITK
GAT.

### The seven accessors collapse to two operations

Two questions survive, because they have different answers:

| Operation | Method | Contract |
| --- | --- | --- |
| indexed access | `view()` | borrowed, layout-preserving, never copies, never linearizes |
| linearization | `host_cow_on(backend)` | logical row-major, borrows when the layout permits |

`data_slice` is subsumed: `view()` accepts every layout it accepts and more,
without the failure mode. `data_vec*`/`try_data_vec*` are
`host_cow_on(..).into_owned()` written out three times; callers that need
ownership write `.into_owned()`, which is where the copy belongs — visible at
the call site that wants it. The `B::default()` convenience overloads are
deleted rather than duplicated, since a caller holding no backend can pass
`&B::default()` in the same character count.

Seven methods become two, and the second is already Coeus's own
`Tensor::host_cow_on` — so `Image` re-exposes it rather than wrapping it.

### The batch coordinate-transform families are a separate, live defect

ADR 0018 collapsed the two *single-point* transform pairs. The equivalent
divergence survives one granularity up and is not yet recorded:

| Batch method | Honours `CoordinateMap` | Singular direction |
| --- | --- | --- |
| `physical_points_to_continuous_indices` | **no** | typed error |
| `world_to_index_native_on` / `index_to_world_native_on` | yes | panics |

This is precisely the ADR-0018 defect — a map-blind implementation beside a
map-aware one, agreeing on Cartesian rasters and diverging on beam-space
acquisitions. ADR 0018 fixed it at point granularity only. It is named here
because the audit that produced this ADR found it, but it is **not** in this
ADR's scope: it is a coordinate-transform decision, not a data-access one, and
it belongs in an ADR-0018 successor. Filed as a board item.

The surviving point-vs-batch split is *not* duplication and is retained: the
two granularities have different cost models, which is the boundary
ARCH-695-01 established for this crate.

## Rejected alternatives

**A RITK-owned `ImageView<'a, T, D>`.** Forks leto's borrowed-array vocabulary
and its five iterator types, and would have to re-derive `slice`, `permute`,
`axis_iter` and `lanes`. Rejected: the substrate owns this concern, and a
consumer-owned duplicate is the additive defect the standards forbid.

**A RITK GAT lending iterator.** Rejected on the mechanical test above: the
items borrow the host slice, not the iterator, so the GAT would narrow the item
lifetime to `'this` and forfeit `IntoIterator`, `zip`, `rev`,
`ExactSizeIterator` and the parallel bridges — the exact compositions tiled
filters need — for nothing.

**Keeping `data_slice` and fixing only its error path.** Rejected: the failure
is not in the error path but in the flat contract, which cannot express a
stride. Making it fail more nicely preserves the copy.

**`Image::view()` returning `Option`.** Rejected: the failure carries
information (non-host-addressable backend, rank mismatch, layout overrun) that
a caller needs in a diagnostic; `Option` discards it.

**Deleting `native.rs` as a duplicate of `coeus_ops::linear_interpolation`.**
Considered because `native.rs`'s module doc claims the Image-backed sister
"delegates to the same math", which is false — `trilinear.rs` calls
`coeus_ops`, `native.rs` implements its own. But they are not
interchangeable: `native.rs` is generic over `T: FloatElement` and
`coeus_ops::linear_interpolation` is pinned to `f32`, so the collapse would
*lose* the generic path. Deferred to a board item, upstream-first.

## Migration path

The seam is additive, so consumers migrate one at a time and nothing breaks in
between. Ordered by value:

1. **`ritk-interpolation` linear kernel** — this increment. The hottest path,
   and the only one that is simultaneously the `f32`-pinned case, the
   rank-decoded-per-sample case and the double-`to_contiguous` case. Its three
   siblings (`nearest`, `sinc`, `bspline`) carry the identical
   `to_contiguous().as_slice()` pair and follow it directly.
2. **The registration metrics**, which are the largest waste in the repo:
   `metric/{mse,ncc,lncc,ngf}/native.rs` each call `data_vec()` — a full owned
   copy of the fixed volume — on *every* metric evaluation inside the optimizer
   loop, only to `zip` and reduce it. Case (a): an iterator, not even a view.
3. `ritk-filter`'s `sample_moving_at_world` and the resample family, which
   `data_cow()` a whole moving volume per sample batch.
4. The offset-indexing consumers, which are what the view is actually for:
   `ritk-filter`'s `projection/ops.rs` (`vals[z*ny*nx + y*nx + x]` on all three
   axes), `transform/{roi,shrink,pad}.rs`, and `ritk-segmentation`'s four
   level-set filters, whose per-iteration 6/26-neighbour stencils are
   `ArrayView::windows`.
5. The remaining `data_slice` callers, which gain strided-input support.
6. Only once every consumer of a given accessor has moved does that accessor
   get deleted — in one change, with no re-export, per the compatibility-soup
   rule. `Image` is at 0.4.0; the deletion is the 0.5.0 break.

Two consolidations fall out of the traversal and should be taken with it, not
after: `map_flat_image`/`map_flat_pair` is copy-pasted between
`ritk-filter/src/native_support.rs` and `ritk-segmentation/src/native_support.rs`
with the same signature and different cost (one owns via `try_data_vec_on`, one
borrows via `data_slice`), and it fans out to most of both crates' call sites —
so it is the single highest-leverage migration point. And `ritk-python`'s
`with_image_slice`/`with_image_pair_slices` closure seam (~50 call sites) is a
hand-rolled borrowed-access pattern that the view supersedes.

No shim, no `#[deprecated]` re-export, no forwarding wrapper at any step: the
old accessors keep working untouched until their last caller leaves.

## Verification

The claim is that the seam removes copies, and value equality cannot show it —
a materializing implementation satisfies value equality too. Provenance can: a
borrowed view's `data()` pointer lies inside the source allocation, and a copy's
does not. Every zero-copy assertion is therefore a pointer-identity assertion,
including a differential case that puts `to_contiguous()` beside the view on
the same offset tensor and shows the one allocating where the other does not.

Behaviour preservation for the migrated kernel is carried by its existing
suite, unchanged: the kernel's contract is fixed by tests that predate this
change, so they are an independent oracle for it. The kernel additionally gains
a case it could not previously express — sampling a strided volume without a
prior materialization — asserted against the same values read from the
contiguous equivalent.
