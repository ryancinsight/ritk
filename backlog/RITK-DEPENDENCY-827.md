<a id="RITK-DEPENDENCY-827"></a>

## RITK-DEPENDENCY-827 — Canonical dependency lock resolves; Coeus 0.12 caller migration blocks CI [patch][major] — blocked

- status: blocked
- priority: correctness
- owner: RITK dependency integration
- change class: [patch] canonical lock repair; [major] subsequent public error-contract migration
- target: PR #827; existing RITK package versions; no release tag
- last update: 2026-10-13

### Contract and delivered dependency repair

Preserve all seven commits from `7e45ee8289734cfaf32d0683bbfc7f70bebc22c2`
and their final Cargo.toml contents. Current main `5622a84b` is an ancestor;
no rebase, requirement reversal or compatibility facade is needed.

Resolve the canonical lock outside Atlas path patches. Every selected version
must satisfy its incoming manifest requirements, every first-party package must
retain a Git source, and locked verification must leave Cargo.lock unchanged.

Metis #499 corrects Consus compression/raster from ^0.1 to ^0.2 and admits
Moirai WebView 0.2.0 after verifying its consumed run/resource APIs. Its patch
release is 0.3.2. Compression and raster source trees are identical between
Metis's old Consus lock `6b7544f5` and current Consus `cf30dc29`.

RITK locks the published Metis PR revision rather than inserting a path patch,
source alias or consumer-local implementation. The lock selects Consus 0.2.0,
Moirai 0.8.2, Coeus 0.12.0, Leto 0.45.2, Eunomia 0.9.1 and Apollo FFT 0.28.0.
Full unqualified updates require the upstream Metis correction to reach main;
the precise Metis update and locked consumer resolution already succeed.

### Verification on macOS with Rust 1.97.0

- Locked metadata with all features resolves 880 packages and 44 workspace members.
- Shared lockfile guard passes; provider identity excess is zero and no unused
  patch tables remain.
- Workspace dependency alignment and formatting pass.
- Python script tests pass 37/37 with CI's Metis scripts path and output root.
- Clippy, nextest compilation and rustdoc fail on the Coeus caller contracts below.
- Keep-going all-feature/all-target checking reports 321 distinct primary
  location/message diagnostics in 32 files: ritk-image 55, ritk-model 266.
  Production compilation reports 20 image errors and 246 model errors; test
  compilation adds failures. Later dependent crates remain unverified.
- Python wheel/parity execution remains blocked by the Rust compilation failures.
  Local maturin/pytest are absent; no wheel or binding-runtime pass is claimed.

### Remaining Coeus contract migration [major]

Coeus 0.12's tensor allocation/construction, host extraction, contiguous
materialization, arithmetic, Var construction and module initialization now
return typed Results. Existing callers still use infallible values:

- ritk-image/src/access.rs:59 returns host_cow_on directly as Cow.
- ritk-image/src/types.rs:83 passes fallible from_slice_on to Image::new.
- ritk-image/src/transform/batch.rs:58-61 treats tensor construction,
  subtraction and matrix multiplication as infallible.
- ritk-model/src/affine/network.rs:96 accesses Conv.weight on a Result;
  autograd and model stages retain further infallible assumptions.

The next increment needs an ADR for public error contracts, propagation of
backend/initialization errors through every affected caller, and existing
value-semantic tests. Panicking adapters, old-name shims, ignored errors and
dependency downgrades do not close this contract gap. Rerun the complete gates
after the real caller migration; current compilation does not establish that
these are the only remaining failures.

### Separate Consus finding

Current Consus standalone updates reject its Moirai ^0.7.0 requirements against
0.8.2. That optional HDF5 execution edge is absent from RITK's all-feature
resolved graph, so it does not block this lock repair. Consus #170 tracks that
provider change independently. Current Moirai manifests declare no Metis edge.
