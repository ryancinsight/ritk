# ADR 0003 — CLI / Python consumer cutover to the native substrate

- Status: Accepted
- Change class: [arch] (umbrella; each command/crate slice is [minor] or [major])
- Date: 2026-07-03
- Related: ADR 0002 (core Image Burn→Coeus migration), `docs/coeus_migration.md`,
  backlog MIG-489…495 (naming, un-gating, all-format native I/O parity)

## Context

ADR 0002 established the strategy: build native (Atlas-substrate) capability
bottom-up, then remove Burn top-down (consumers first, core `Image` last), with
the `burn-migration-audit` token counts as the done-metric.

Since ADR 0002 the **standard image-I/O layer reached native parity**
(MIG-493/494/495): all standard file formats wired through the CLI format matrix
(nifti, nrrd, analyze, mgh, metaimage, png, jpeg, tiff) have Atlas-native
readers, and all except png have Atlas-native writers behind the unified
`ImageReader<I>` / `ImageWriter<I>` contract, each verified byte-identical /
value-identical to its Burn counterpart. A later DICOM integration added
Atlas-native scalar DICOM series readers on both DICOM facades, verified against
the legacy Burn loaders. The `coeus` cargo feature was removed (MIG-492): the
native path is the mainline, always compiled and tested.

Yet the audit token counts have **not** dropped. The reason is the consumer
chain, traced concretely:

```
ritk-cli / ritk-python  (read_image → Image<Backend,3>, Backend = NdArray<f32>)
    ↓ consume Burn-typed
ritk-io  (re-exports Burn read_*/write_* from every format crate)
    ↓
format crates  (Burn read_*/write_* still LIVE because ritk-io/cli/python call them)
```

`crates/ritk-cli/src/commands/mod.rs` defines `type Backend = NdArray<f32>`
and `read_image() -> Image<Backend,3>` / `write_image(path, &Image<Backend,3>,
fmt)`; every command (`convert`, `filter`, `normalize`, `register`, `resample`,
`segment`, `stats`, `viewer`) flows an `Image<Backend,3>` (Burn) from
`read_image` through processing to `write_image`. `ritk-python`'s
`io::read_image` mirrors this, wrapping the Burn image into `PyImage`.

**A format crate's Burn reader/writer can be deleted only once no consumer
calls it.** The consumers are cli and python. So the cutover of these two
entry points is the gate that finally lets Burn code be deleted and the audit
counts fall.

### The blocking subtlety: the image type flows through the whole pipeline

`read_image` returns `Image<Backend,3>` and that value is consumed by
Burn-typed processing (`ritk-filter`, `ritk-registration`, `ritk-segmentation`,
`ritk-transform`, `ritk-statistics`). Naively switching `read_image` to return
`native::Image<f32,B,3>` breaks **every** command at once, because the
processing crates accept Burn images. So the cutover is not a mechanical
one-function change — it is gated on each processing crate accepting native
images. That is why this ADR is required before implementation.

## Decision

Cut over **incrementally, per command**, not in one breaking change.

1. **Parallel native I/O helpers in cli/python.** Add `read_image_native`
   (format-dispatch over the native `ImageReader` adapters, on `SequentialBackend`)
   and `write_image_native` / `write_image_native_inferred` alongside the
   existing Burn `read_image`/`write_image`. Commands migrate one at a time by
   switching which helper they call; un-migrated commands keep the Burn helpers.
   The two paths coexist exactly as the `native` format modules coexist with the
   Burn ones during ADR 0002 — a transitional duplication that folds away when
   the last command is cut over and the Burn helpers are deleted.

2. **Order commands by processing depth** (shallowest first — each step turns
   more Burn readers/writers dead):
   - **Phase A — pure I/O (no processing):** `convert`. Reads then writes; only
     touches `shape()`/`spacing()` for logging, both present on the native
     `Image`. **Format coverage is partial** and must be handled explicitly:
     the native path exists for `{nifti, nrrd, analyze, mgh, metaimage, tiff,
     jpeg}` (read+write), `png` (read only — no writer for any substrate), and
     DICOM directories (read only through the native DICOM series reader),
     while `vtk` has **no** native reader or writer. cli's
     `read_image` also does not currently wire `minc` at all (a pre-existing
     gap, out of scope here). Therefore native `convert` dispatches
     **native-first with a Burn fallback for `vtk` and DICOM outputs**: the
     native helpers handle the native volume formats end-to-end plus DICOM
     reads, and VTK or DICOM-output cases route to the existing Burn helpers
     until those formats gain native writer coverage. This keeps `convert`
     fully functional while removing Burn from the common (volume-format) path.
   - **Phase B — read + reduce (no image output):** `stats`. Needs native
     statistics — already available (`ritk_statistics::native::compute_statistics`).
   - **Phase C — read + elementwise op + write:** `normalize`, `resample`.
     Need native filter/interpolation ops (`ritk-filter` / `ritk-interpolation`
     have `native` modules; audit coverage vs. the Burn op set per command).
   - **Phase D — deep processing:** `filter`, `register`, `segment`. Gated on
     native migration of `ritk-filter` (879 audit tokens), `ritk-registration`
     (1374), `ritk-segmentation` (284). Each is its own [major] with its own
     ADR; the native registration seams already exist (ADR 0001).
   - **`viewer`:** GUI shell. CLI viewer DICOM decode now uses the
     metadata-rich native DICOM loader, then explicitly bridges into the
     current Burn-typed `ritk-snap` viewer core until that rendering/core path
     migrates.

3. **Delete Burn per format crate only when dead.** After a command migrates,
   re-run `burn-migration-audit`; when a format crate's Burn reader/writer has
   no remaining caller (cli, python, ritk-io re-export, or test), delete it and
   drop `burn` from that crate's manifest. This is the step that moves the SSOT
   number.

## Alternatives considered

- **Convert native→Burn as the whole migration strategy** (native reader →
  immediately build a Burn `Image` forever): rejected. It would let
  format-crate Burn readers die but not the Burn `Image` type or the processing
  surface, and it would institutionalize a copy on every load. A temporary,
  named `native_image_to_burn` bridge is acceptable only at still-Burn-typed
  command/viewer boundaries and is deleted as those processing surfaces migrate.
- **Big-bang cutover of `read_image`/`write_image`**: rejected. Breaks all 8
  commands simultaneously and cannot be landed as small green vertical slices;
  violates the WIP limit and DORA small-batch discipline.
- **A `dyn`-dispatched image abstraction over both substrates**: rejected.
  A trait object on the hot image path is a throughput defect (integrity
  STRONG-DEFAULT); the substrate is a compile-time-known type, so generics /
  the existing `ImageReader<I>`/`ImageWriter<I>` contract are the zero-cost seam.

## Consequences

- Transitional duplication (`read_image` + `read_image_native`) in cli/python
  during the cutover — bounded, and deleted when the last command migrates.
- Each command cutover is an independently verifiable slice: differential test
  that the native command output matches the Burn command output (byte-identical
  for the writers, value-identical for the readers, per MIG-494/495).
- The audit counts stay flat until Phase D crates migrate; Phases A–C are
  correctness-preserving plumbing that make the format-crate Burn I/O dead but
  leave the processing Burn surface. This is expected and must be stated in
  status reports so the flat counts are not read as "no progress".

## Verification plan

- Per command: a differential test reading a fixture, running the command via
  the native path and the Burn path, asserting identical output file bytes
  (writers) / identical decoded voxels + metadata (readers).
- `burn-migration-audit` re-run after each format-crate Burn deletion; the delta
  is the evidence a slice moved the SSOT.
- Full `cargo nextest run` on cli/python after each command cutover; clippy
  `-D warnings`; doc sync of the command's help text and any migration notes.

## Completed CLI increments

**Phase A — native `convert`.** Added `read_image_native` / `write_image_native`
to `ritk-cli::commands` dispatching over the native `ImageReader`/`ImageWriter`
adapters (`ritk_io::format::<fmt>::native::*`) on `SequentialBackend`, routed
`convert` through them for the native volume formats with a Burn fallback for
VTK and native-missing write targets, and added a differential test (`convert`
via native vs. Burn produces byte-identical output). Change class [minor]. This
validates the per-command pattern end-to-end and is the template every later
command follows.

**CLI native loading slice.** Routed the shared `read_image` helper through
native readers for every native-readable CLI format, including DICOM, and kept
only VTK on the legacy Burn reader. Because the remaining processing commands
still accept `Image<Backend, 3>`, this slice uses an explicit
`native_image_to_burn` bridge at the command boundary; value-semantic tests
assert that the bridge preserves shape, origin, spacing, direction, and voxel
values. The headless CLI viewer now loads scalar DICOM studies through
`load_native_dicom_series_with_metadata` and uses the same explicit bridge into
the current Burn-typed `ritk-snap` core. Evidence tier: value-semantic
differential tests (`ritk-cli dicom` 5/5 and `ritk-cli native` 6/6).

A DRY note for the implementation: cli and python each re-implement the
format→reader match today. The native helpers should ideally live once in
`ritk-io` (`read_image_native`/`write_image_native`) and be shared by both
consumers, rather than duplicated — fold that consolidation in when python's
Phase A lands, so the two consumers share one native dispatch SSOT.

## Python Phase A update (2026-07-05)

`ritk-io` now owns the shared `read_image_native`/`write_image_native` consumer
helpers over the native reader/writer adapters. `ritk-python` uses those helpers
directly for `ritk.io.read_image`/`ritk.io.write_image`, removing the Python
I/O module's local Burn `NdArray` reader/writer dispatch. The public Python
`Image` processing surface still wraps the legacy Burn image type until the
processing crates migrate; the only conversion remains at the PyImage boundary.
