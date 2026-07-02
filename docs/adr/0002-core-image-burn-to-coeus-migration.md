# ADR 0002 — Core `Image` / tensor-substrate Burn→Coeus migration strategy

- Status: Accepted
- Change class: [arch]
- Date: 2026-06-30
- Related: `docs/coeus_migration.md`, ADR 0001 (registration traits), backlog
  MIG-471…482 (parallel Coeus registration capability)

## Context

`cargo run -p xtask -- burn-migration-audit` (this sprint) shows every RITK
crate still depends on Burn (manifest=true) and the source-token surface is
concentrated in `ritk-registration` (1374), `ritk-filter` (879),
`ritk-interpolation` (594), `ritk-transform` (407), `ritk-model` (333),
`ritk-segmentation` (284), `ritk-io` (243). The other stated migration targets
are already clean in RITK: **no `rayon`, `tokio`, `nalgebra`, `ndarray`, or
`rustfft` direct dependencies remain** (audit + manifest grep). Burn is the
entire remaining substrate surface.

Twelve sprints (MIG-471…482) built a complete, individually-verified,
**parallel** Coeus-autograd registration capability (loss ×2, sampling,
transforms ×2, seams, generic metric, optimizer, driver) behind the `coeus`
feature. Likewise `ritk_image::coeus::Image` and per-format `read_*_coeus`
paths exist alongside the Burn ones. **None of this has reduced the Burn
surface**: the audit token counts are essentially unchanged because the Burn
code paths are all still present and load-bearing.

### Why leaf crates cannot drop Burn in isolation (the bottleneck)

Traced this sprint: `ritk-vtk::read_vtk<B: Backend> -> Image<B,3>` feeds
`ritk-io`'s `ImageReader<B,3>`/`ImageWriter<B,3>` trait hub, consumed by
`ritk-cli` and `ritk-python` on a concrete Burn backend. Every format crate
funnels through the same two chokepoints:

1. `ritk_core`'s Burn `Image<B: Backend, const D>` (the vocabulary type), and
2. `ritk-io`'s `ImageReader`/`ImageWriter` traits (the I/O contract).

A leaf crate's Burn dependency cannot be removed while its public API returns/
accepts the Burn `Image`, and that API cannot change until its consumers
(`ritk-io` → `ritk-cli`/`ritk-python`) accept the Coeus `Image`. So although
Coeus *capability* is added bottom-up (leaf readers, primitives), Burn
*removal* must proceed top-down from the consumers, converging on the core
`Image` type last.

## Decision

**Strategy B — parallel Coeus paths, incremental crate-by-crate cutover,
top-down Burn removal.** Rejected alternatives:

- **A. One tensor-substrate trait** abstracting Burn `Backend` and Coeus
  `ComputeBackend` behind a RITK-owned contract so a single generic code path
  serves both. Rejected: Burn and Coeus tensor APIs are structurally different
  (op sets, autodiff models, storage); a unifying trait wide enough to carry
  reshape/slice/gather/matmul/autodiff for both is a larger, longer-lived
  abstraction than the migration it serves, and it would outlive its purpose
  (once Burn is gone the abstraction is dead weight). It also violates
  seam-first cost discipline (a trait built for two implementors where one is
  scheduled for deletion).
- **C. Wholesale swap** of Burn `Image` for Coeus `Image` across all crates in
  one change. Rejected: unbounded blast radius, un-reviewable, cannot keep the
  tree releasable (git_discipline), and cannot be validated incrementally.

Strategy B keeps Burn as the production backend and grows/cuts over the Coeus
path per crate, matching `docs/coeus_migration.md`'s sequence.

### Cutover ordering (the actionable part)

Capability is added bottom-up (mostly done); **removal** is gated top-down.
The migration proceeds by making a consumer accept the Coeus `Image`, then
cutting its providers over, then deleting their Burn paths:

1. **Prerequisite — Coeus `Image` parity.** `ritk_image::coeus::Image` must
   expose the accessor/host-extraction surface the format writers, CLI, and
   Python boundary need (shape, spacing/origin/direction metadata, contiguous
   host slice, flat construction). Gap-audit each consumer's usage before
   cutover; extend the Coeus `Image` (and Coeus itself) to fill gaps, tested.
2. **`ritk-io` I/O contract.** Introduce Coeus-typed `ImageReader`/`ImageWriter`
   surfaces (or generalize the existing ones over the image type) and route the
   per-format `read_*_coeus`/`write_*_coeus` paths through them. Both Burn and
   Coeus dispatch coexist behind the `coeus` feature until consumers move.
3. **Consumers (`ritk-cli`, `ritk-python`).** Switch their concrete backend
   type from the Burn `NdArray` to a Coeus CPU backend; they become the first
   crates whose Burn path is *removed*.
4. **Format leaf crates.** Once no consumer needs the Burn reader/writer,
   delete each leaf's Burn-generic `read_*`/`write_*` and its `burn`/
   `burn-ndarray` dependency — measurable audit progress (manifest=false).
5. **Compute crates** (`ritk-filter`, `ritk-interpolation`, `ritk-statistics`,
   `ritk-transform`, `ritk-registration`, `ritk-model`, `ritk-segmentation`):
   migrate each on the same add-Coeus-path → cut-consumers → remove-Burn cycle.
   `ritk-registration`'s differentiable path already has its Coeus
   implementation (MIG-471…482); its removal step is gated on the metric/engine
   callers moving.
6. **`ritk_core` `Image`.** Remove the Burn `Image` last, once nothing
   constructs or consumes it.

### Removal criterion (per crate)

A crate's Burn dependency is removed only when: no crate-local or downstream
caller references its Burn-generic API; its Coeus path has value-semantic
differential parity coverage vs. the (about-to-be-deleted) Burn path; and the
audit token count for the crate reaches zero. The Burn path is deleted in the
same change that removes the last caller (no compatibility shim — integrity).

## Consequences

- The parallel Coeus capability built so far (registration + Coeus `Image` +
  Coeus readers) is confirmed as the correct **capability-add** phase; the
  next distinct phase is **cutover/removal**, which is top-down and starts at
  `ritk-io` + its consumers, not at the leaves.
- Each cutover step is a normal [minor]/[major] increment (a crate's public
  image type changing is [major] for that crate); no single [arch] mega-change.
- Progress is now measurable by the audit token counts trending to zero,
  crate by crate — a concrete definition of "done" the migration previously
  lacked.

## Verification

- The audit tool (`burn-migration-audit`) is the migration SSOT; each cutover
  increment must lower a crate's `source_tokens` and, when complete, flip its
  `manifest` to false, with differential parity tests proving the Coeus path
  matches the removed Burn path before deletion.
