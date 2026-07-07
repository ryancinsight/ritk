# Moirai parallel-surface SSOT formalization (kwavers-Atlas-migration-push)

This artifact declares the **Atlas-typed `moirai-parallel` macro-thread
/ task-rail** as the Single Source of Truth (SSOT) for all parallel
fan-out across the ritk workspace, replacing the legacy `rayon` +
`tokio` + `ndarray::Zip::indexed().par_*` fan-out surfaces. This
sub-batch formalizes the SSOT alignment under the
kwavers-Atlas-migration-push ceremony naming. ritk HEAD at discovery time =
`7aaae9eb fix(ritk-nifti): Decode Int16 payloads` was the
pre-ceremony tip; the SSOT formalization chore itself landed at
`2fa640ee chore(ritk): Moirai parallel-surface SSOT formalization
and Atlas-provider audit`, with the accuracy-slips docs fixup
layered at `docs:` follow-up — both anchored under the
`kwavers-Atlas-migration-push` ceremony anchor.

## Surface inventory

### Legacy fan-out (NOT present in production)

- **0** call-sites of `rayon::` / `tokio::` / `par_iter` /
  `ndarray::Zip::indexed().par_*` across the 31 ritk-* production
  crates' source.
- The tokens appear only as string literals in
  `repos/ritk/xtask/src/migration_audit.rs`'s `LEGACY_SOURCE_TOKENS`
  const list (banned patterns), where the audit *detects* their
  presence and fails the build — exactly the SSOT enforcement
  mechanism we want.

### Atlas-typed moirai coverage (in flight)

- **2** `use moirai;` import sites in 2 files within `ritk-filter`:
  - `crates/ritk-filter/src/edge/separable_gradient/mod.rs:30`
  - `crates/ritk-filter/src/morphology/binary_erode.rs:38`
  - Wiring routed through `moirai`'s `ParallelSlice{,Mut}` prelude.
- Discovery provenance: ripgrep scan of `D:/atlas/repos/ritk/crates/`
  for the regex `^use moirai` returned exactly the two lines above.
- Additional rollout across the rest of the ritk-* production
  crates is fanned out on a per-crate Atlas sub-batch basis
  (lexically independent of this SSOT formalization).

## Out-of-scope partition (NOT addressed here)

- The vast in-flight `burn -> coeus` tensor migration (≈ 259 dirty
  files across `ritk-*` production crates at the time of this
  sub-batch) is **orthogonal** to the parallel-surface SSOT work:
  it swaps the *tensor backend* (Burn → Coeus), not the *parallel
  runtime* (Rayon/Tokio → Moirai). It must remain scoped under its
  own future sub-batch to avoid scope collision with this ceremony.

## Validation

- **0** production-source references to `rayon::` / `tokio::` /
  `par_iter` / `ndarray::Zip::indexed().par_*` confirmed via
  ripgrep across `D:/atlas/repos/ritk/crates/`.
- Informational only — no code paths change in this chore commit.
- Future regressions gated by
  `cd ritk && cargo run -p xtask -- migration-audit`.

## Ceremony linkage

This doc lands under the same `kwavers-Atlas-migration-push` SSOT
formalization ceremony as the cfdrs counterpart
(`D:/atlas/repos/cfdrs/docs/atlas-migration/moirai-ssot.md`) and
the canonical kwavers sub-batch anchor (`HEAD = 702e4f125 chore
(deps): drop unused ndarray/rayon feature`). It marks ritk's
moirai-phase as functionally complete for parallelism-surface
SSOT enforcement, leaving the `burn -> coeus` migration as a
separate, dedicated future sub-batch.
