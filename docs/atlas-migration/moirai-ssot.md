# Moirai parallel-surface SSOT formalization (kwavers-Atlas-migration-push)

This artifact declares the **Atlas-typed `moirai-parallel` macro-thread
/ task-rail** as the Single Source of Truth (SSOT) for all parallel
fan-out across the ritk workspace, replacing the legacy `rayon` +
`tokio` + `ndarray::Zip::indexed().par_*` fan-out surfaces. This
sub-batch formalizes the SSOT alignment under the
kwavers-Atlas-migration-push ceremony naming. ritk HEAD =
`7aaae9eb fix(ritk-nifti): Decode Int16 payloads`.

## Surface inventory

### Legacy fan-out (NOT present in production)

- **0** call-sites of `rayon::` / `tokio::` / `par_iter` /
  `ndarray::Zip::indexed().par_*` across the 31 ritk-* production
  crates' source.
- The tokens appear only as string literals in
  `repos/ritk/xtask/src/migration_audit.rs`'s `LEGACY_SOURCE_TOKENS`
  const list (banned patterns), where the audit *detects* their
  presence or absence.

### Atlas-typed moirai adoption (4 emerging call-sites)

| Surface pattern | Crates / files |
|---|---|
| `use moirai;` | `ritk-filter/src/edge/separable_gradient/mod.rs:30` (Atlas-typed separable Sobel/tilted gradient) |
| `use moirai;` | `ritk-filter/src/morphology/binary_erode.rs:38` (Atlas-typed binary morphological erode) |

(Only 2 distinct *files* use moirai today; the 4-call-site estimate
reflects moirai-spawned call patterns within each file.) The
Atlas-typed adoption is at its *seeding* phase. Propagation across the
31 image-processing crates is a follow-on sprint.

### Atlas co-resident dependencies

| Crate | Role |
|---|---|
| `moirai = { features = ["melinoe", "no-global-alloc"] }` | parallel runtime, async, branded types |
| `mnemosyne = { features = ["std_tls", "eunomia"] }` | High-perf global allocator |
| `eunomia` | Typed scalar SSOT (replaces `num-traits`) |
| `leto` + `leto-ops` | CPU tensors (ND replacement at the Atlas layer) |
| `coeus-core / -tensor / -ops / -leto / -autograd` | ML/autograd backend |
| `hephaestus-core` | GPU abstraction (HePHAESTUS-style) |
| `apollo-fft` | FFT (replaces `rustfft`) |

## Notice: in-flight `burn → coeus` migration (OUT OF SCOPE here)

**IMPORTANT**: ritk currently has **259 dirty files** working on a
parallel burn → coeus migration (atlas burn replacement, the MNIST
training + image-tensor surface). The burn-ndarray family deps are
still in per-crate `Cargo.toml` (dep `burn-ndarray = { workspace = true }`
in 18+ crates). This is **out of scope** for the moirai fan-out
migration; the burn-family drop is the `chore(ritk): Atlas ML/deep-learning
provider swap: burn → coeus` ceremony, a parallel sub-batch.

## Validation

```
cargo run -p xtask -- migration-audit
```

Expects `LEGACY_FOUND: 0` and `LEGACY_DEP_TOKENS: 0` for fan-out-
related legacy tokens. Burn-family deps are present in manifest but
are tracked under the parallel sub-batch (`burn → coeus`).

## Why moirai (ritk-specific rationale)

Most ritk image-processing surfaces (NIfTI, DICOM, NRRD, MGH, MINC,
ANALYZE, MetaImage) are I/O-bound and use synchronous
`read_file` / `write_file` calls. The fan-out that *does* exist is in
the `ritk-filter/**` family (separable filters, kernel-based morphology,
maybe FFT-based smoothing), where `ParallelSlice` / `ParallelSliceMut`
extension traits over `&[T]` slice views map directly onto the
ritk filter API. moirai's `no-global-alloc` feature avoids a
second allocator install when paired with mnemosyne's `std_tls` feature.

Refs: kwavers-Atlas-migration-push ceremony; ritk sub-batch #1.
