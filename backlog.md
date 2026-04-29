## Sprint 71 — Completed

**Status**: Completed
**Phase**: Closure
**Goal**: Expose `zscore_normalize` mask parity in the Python stub surface; add Python-level smoke coverage for masked z-score; verify the current API surface remains aligned with the compiled binding; preserve backward-compatible default behavior.

### Gaps closed
| ID | Gap | Root cause | Resolution | Tag |
|---|---|---|---|---|
| GAP-R71-01 | `zscore_normalize` Python stub lacks optional `mask` parity | Compiled binding accepts `mask=None`, but the stub file still exposed `def zscore_normalize(image: Image) -> Image` | Updated `crates/ritk-python/python/ritk/_ritk/statistics.pyi` to include `mask: Image | None = None` | [patch] |
| GAP-R71-02 | `zscore_normalize(mask=...)` smoke coverage absent in Python test suite | Existing regression test covered shape mismatch, but no positive Python-level smoke case asserted masked dispatch and output shape/value semantics | Added `test_zscore_normalize_masked_matches_foreground_shape` to `crates/ritk-python/tests/test_statistics_bindings.py` | [patch] |
| GAP-R71-03 | Python API contract drift between stub and runtime for `zscore_normalize` | The runtime binding has optional mask support; the stub and smoke suite must reflect the compiled callable signature | Audited `test_smoke.py` and `test_statistics_bindings.py`; no additional change required | [patch] |
| GAP-R71-04 | Sprint artifact drift after prior closure | Sprint 70 artifacts were complete, but Sprint 71 tracking needed a fresh entry before implementation proceeded | Updated `backlog.md`, `checklist.md`, and `gap_audit.md` after verification | [patch] |

### Architecture decisions
- `zscore_normalize` stub/runtime parity is now explicit in `statistics.pyi`.
- Masked z-score smoke coverage uses matching-shape foreground voxels and asserts computed value semantics, not existence-only behavior.
- The existing mismatch test remains valid and unchanged.

### Verification
| Check | Result |
|---|---|
| `cargo check --workspace --tests` | 0 errors, 0 warnings |
| `cargo test -p ritk-python --lib` | 10/10 passed |
| Python regression test target for Sprint 71 | passed for `test_zscore_normalize_masked_matches_foreground_shape` and `test_zscore_normalize_mask_shape_mismatch_raises` |

### Updated artifacts
- `backlog.md`: Sprint 71 marked completed; gaps recorded as closed by stub update, smoke test addition, or audit.
- `checklist.md`: Sprint 71 checklist items marked complete.
- `gap_audit.md`: Sprint 71 closure notes updated with the stub/runtime parity evidence and masked z-score tests.

### Residual risk
- None identified from the selected Sprint 71 gaps.

---

## Sprint 70 — Completed

**Status**: Completed
**Phase**: Closure
**Goal**: Audit `white_stripe_normalize` Python binding parameter surface; add negative tests for `zscore_normalize` with mismatched mask shape; audit `run_lddmm` convergence and learning-rate parameter wiring; add `minmax_normalize_range` Python-level integration test to the pytest test suite.

### Gaps closed
| ID | Gap | Root cause | Resolution | Tag |
|---|---|---|---|---|
| GAP-R70-01 | `white_stripe_normalize` Python binding parameter surface audit | `white_stripe_normalize` already exposes `mask`, `contrast`, and `width` and validates contrast with `PyValueError` | Audited `crates/ritk-python/src/statistics.rs`; no code change required | [patch] |
| GAP-R70-02 | `zscore_normalize(mask=...)` missing negative test for shape-mismatched mask | Requested Python `mask=` path is present; binding now validates `mask.shape == image.shape` and raises `PyValueError` on mismatch | Added shape-validation guard in `crates/ritk-python/src/statistics.rs`; added `test_zscore_normalize_mask_shape_mismatch_raises` to `crates/ritk-python/tests/test_statistics_bindings.py` | [patch] |
| GAP-R70-03 | `run_lddmm` `learning_rate` parameter parity audit | `LDDMMConfig` has a `learning_rate` field; verify `RegisterArgs.learning_rate` is wired in `run_lddmm` | Audited `crates/ritk-cli/src/commands/register.rs`; wiring already present, no code change required | [patch] |
| GAP-R70-04 | `minmax_normalize_range` guard absent from pytest test suite | GAP-R69-01 added Rust unit tests for `validate_range` but no Python-level test exercises the `PyValueError` path | Added `test_minmax_normalize_range_inverted_bounds_raises` to `crates/ritk-python/tests/test_statistics_bindings.py` | [patch] |

### Architecture decisions
- No public API changes were required.
- GAP-R70-01 and GAP-R70-03 were closed by source audit only.
- GAP-R70-02 was closed by adding a deterministic precondition check at the Python boundary and a value-semantic regression test.
- GAP-R70-04 was closed by adding a value-semantic Python regression test to the existing `ritk-python` suite.

### Verification
| Check | Result |
|---|---|
| `cargo check --workspace --tests` | 0 errors, 0 warnings |
| `cargo test -p ritk-python --lib` | 10/10 passed |
| Python regression test target for Sprint 70 | passed for `test_minmax_normalize_range_inverted_bounds_raises` and `test_zscore_normalize_mask_shape_mismatch_raises` |

### Updated artifacts
- `backlog.md`: Sprint 70 marked completed; gaps recorded as closed by audit or test addition.
- `checklist.md`: Sprint 70 checklist items marked complete.
- `gap_audit.md`: Sprint 70 closure notes updated with the audited source evidence and added tests.

### Residual risk
- None identified from the selected Sprint 70 gaps.

### Next action
- Sprint 71 planning: source-audit-only closure or new patch-class item from backlog.

- `autonomous`
