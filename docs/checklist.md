# Sprint Checklist — v0.50.80

Sprint: 318 — Parzen Correctness, Cleanup, Doc Fixes
Status: COMPLETE

---

## Sprint 318 Items

- [x] **CORRECT-318-01**: Masked-cache fingerprint `Option<f32>` → `Option<u64>` (SipHash-1-3)
- [x] **FIX-318-02**: Remove duplicate `ParzenConfig` from private import in `direct/mod.rs:106`
- [x] **FIX-318-03**: `MAX_PARZEN_BINS` cfg-gated, `support_bins()` removed, `MIN_HALF_WIDTH` re-export removed
- [x] **FIX-318-04**: Fix bench build break — add HistogramPool `None` args + remove unused `SparseWFixedEntry` import
- [x] **FIX-318-05**: Remove dead `validate_num_bins` from `ritk-python/src/metrics/mod.rs`
- [x] **CLEAN-318-01**: Add `#![allow(dead_code)]` to `tests/common/mod.rs` (shared test utilities)
- [x] **DOC-318-01**: Fix 30 doc warnings across ritk-registration (preprocessing, validation, demons, diffeomorphic, bspline_ffd, parzen)
- [x] **CLIPPY-318-01**: Fix 2 clippy warnings in `direct/types.rs` (int_plus_one, doc_lazy_continuation)
- [x] `cargo check --workspace --all-targets`: 0 errors, 0 warnings
- [x] `cargo clippy -p ritk-registration --lib`: 0 warnings
- [x] `cargo doc --no-deps -p ritk-registration`: 0 warnings
- [x] `cargo test -p ritk-registration --features direct-parzen --lib`: 385 passed, 0 failed, 1 ignored
- [x] CHANGELOG.md, backlog.md, checklist.md, gap_audit.md updated

---

## Phase Exit Criteria

- [x] All selected gaps implemented and value-semantically verified
- [x] Diagnostics clean for `ritk-registration` (check, clippy, doc, test)
- [x] Artifacts synchronized

---

## Next Sprint Target: v0.50.81

Candidates (pick one):
1. **ARCH-319-01** — Parallelize tensor-path dispatch (non-cached Parzen path) `[minor]`
2. **ARCH-319-02** — Remove deprecated `compute_joint_histogram_from_cache_direct` `[patch]`
3. **ARCH-319-03** — Fix 12 clippy warnings in `ritk-core` `[minor]`
