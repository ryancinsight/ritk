# Backlog

Sprint target version: **0.50.81** (next increment)

Triage order: correctness gaps → architecture drift → missing tests → documentation drift → cleanup.

---

## Active / Next Sprint

| ID | Priority | Class | Title | Status |
|----|----------|-------|-------|--------|
| ARCH-319-01 | High | [minor] | Tensor-path dispatch parallelization (non-cached path) | OPEN |
| ARCH-319-02 | Medium | [patch] | Remove deprecated `compute_joint_histogram_from_cache_direct` re-export | OPEN |
| ARCH-319-03 | Medium | [minor] | ritk-core clippy: fix 12 warnings (too_many_arguments, needless_range_loop, doc) | OPEN |
| ARCH-319-04 | Low | [patch] | ritk-model clippy: fix manual_pattern_char_comparison | OPEN |

---

## Completed

| ID | Sprint | Class | Title |
|----|--------|-------|-------|
| CORRECT-318-01 | 318 | [patch] | Masked-cache fingerprint hardened from probabilistic (f32 sum) to deterministic (u64 SipHash) |
| FIX-318-02 | 318 | [patch] | Sprint 317 build break — duplicate ParzenConfig import in direct/mod.rs |
| FIX-318-03 | 318 | [patch] | Dead-code cleanup — MAX_PARZEN_BINS cfg-gated, support_bins() removed, MIN_HALF_WIDTH re-export removed |
| FIX-318-04 | 318 | [patch] | Bench build break — HistogramPool missing args + SparseWFixedEntry unused import in parzen_direct.rs |
| FIX-318-05 | 318 | [patch] | Dead-code: validate_num_bins removed from ritk-python metrics |
| CLEAN-318-01 | 318 | [patch] | Dead-code: `#![allow(dead_code)]` on tests/common/mod.rs (shared test utilities) |
| DOC-318-01 | 318 | [patch] | 30 doc warnings → 0 in ritk-registration (unresolved links, unclosed HTML tags, code blocks) |
| CLIPPY-318-01 | 318 | [patch] | 2 clippy warnings → 0 in ritk-registration (int_plus_one, doc_lazy_continuation) |
| ASYNC-GPU-274 | 274 | [minor] | Async double-buffered GPU readback (MIP + VR) — `Maintain::Wait` removed |
| DIMSE-SCU-273 | 273 | [minor] | DIMSE SCU (C-ECHO/C-FIND/C-STORE/C-MOVE) via dicom-ul |
| GPU-PERF-272 | 272 | [minor] | GPU pipeline perf: in-shader WL+LUT, packed u32 RGBA, GpuFrameCache, zero-copy upload |

---

## Residual Risk

- `ritk-core` has 12 pre-existing clippy warnings (too_many_arguments, needless_range_loop, doc) unrelated to Parzen/IO work.
- `ritk-python`, `ritk-vtk` have pre-existing diagnostics errors unrelated to GPU/IO work.
- `render_mip` / `render_vr` return `None` on the first call per volume/param change (1-frame latency);
  callers that previously assumed `Some` on every call must handle `None` by falling back to CPU.
- GPU async readback is single-buffered (only one `staging_buf` per pass); a second in-flight frame
  cannot be submitted until the first `map_async` completes. Double staging buffers are a follow-up.
