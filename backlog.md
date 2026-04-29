## Sprint 75 — Completed

**Status**: Completed
**Phase**: Closure
**Goal**: Close the SyN translation recovery gap (open since Sprint 74). Root cause: incorrect CC gradient force formula in all three diffeomorphic SyN variants (`mod.rs`, `multires_syn.rs`, `bspline_syn.rs`) plus absence of step-size normalization. Fix verified via new Rust unit test `syn_recovers_translation_ncc_improves` and new Python parity test `test_syn_register_ncc_improves_on_shifted_gaussian_blob`.

### Gaps closed
| ID | Gap | Root cause | Resolution | Tag |
|---|---|---|---|---|
| GAP-R75-01 | SyN CC gradient force formula inverted — translation not recovered | All three `cc_forces` functions used `force_scale = -2*cc_num/(var_i*var_j)`. Since `cc_num = CC*sqrt(var_i*var_j)`, this equals `-2*CC/sqrt(var_i*var_j)`, which for CC > 0 pushes the velocity field in the wrong direction (gradient descent on CC instead of ascent) | Replaced with Avants 2008 eq. 10: `force_scale = (J_W-&#956;_J)/sqrt(var_i*var_j) &#8722; CC*(I_W-&#956;_I)/var_i` in all three `cc_forces` functions (`diffeomorphic/mod.rs`, `diffeomorphic/multires_syn.rs`, `diffeomorphic/bspline_syn.rs`) | [patch] |
| GAP-R75-02 | No step-size normalization — force magnitude depended on image intensity scale | Velocity field update `v += u` accumulated raw CC gradient forces; Gaussian smoothing after each step dissipated small forces before they could accumulate | Added `gradient_step: f64 = 0.25` to `SyNConfig` and `MultiResSyNConfig`; forces normalised per iteration so max|u| = gradient_step (inf-norm) before accumulation. `BSplineSyNConfig` also receives the field (consistent API) | [minor] |
| GAP-R75-03 | `gradient_step` missing from Python `syn_register` / `multires_syn_register` / `bspline_syn_register` bindings | Bindings were not updated to expose the new config field | Added `gradient_step: float = 0.25` to all three Python function signatures, PyO3 pyi stubs, and doc-strings; `build_atlas` inner `MultiResSyNConfig` literal fixed | [minor] |
| GAP-R75-04 | No Python parity test for SyN NCC improvement | `test_syn_register_ncc_improves_on_shifted_gaussian_blob` missing from `test_simpleitk_parity.py` Section 5 | Added test: Gaussian blob sigma=4 in 24&#179; volume, 4-voxel x-shift; `syn_register` 50 iter, gradient_step=0.25, sigma_smooth=1.5; asserts NCC_after > NCC_before AND NCC_after &#8805; 0.80; passes on rebuilt wheel | [minor] |

### Architecture decisions
- Force formula is gradient **ascent** on CC (minimise 1&#8722;CC). Avants 2008 eq. 10 first term `(J_W&#8722;&#956;_J)/sqrt(&#963;_I&#178;·&#963;_J&#178;)` is the primary force; the second term `&#8722;CC·(I_W&#8722;&#956;_I)/&#963;_I&#178;` provides second-order curvature correction. Both terms are implemented.
- Gaussian blob images (not linear-ramp images) are the canonical synthetic test for SyN translation recovery. Local CC of a linear ramp is shift-invariant (near 1.0 for any x-offset), so the gradient is near zero and cannot drive convergence.
- `gradient_step = 0.25` matches the ANTs default `gradientStep`. This is the canonical default; the parameter is exposed at the Python and CLI layers so users can tune for large-deformation cases.
- `BSplineSyNConfig::gradient_step` is added for API consistency but is currently unused in the BSplineSyn register loop (BSplineSyn accumulates to a CP lattice whose implicit scale provides magnitude control via `accumulate_to_cp`). If BSplineSyn is found to need normalization in a future sprint, the field is already present.

### Verification
| Check | Result |
|---|---------|
| `cargo test -p ritk-registration diffeomorphic` | 56/56 pass including `syn_recovers_translation_ncc_improves` |
| `cargo test -p ritk-registration atlas` | 28/28 pass |
| `cargo check --workspace --tests` | 0 errors, 0 warnings |
| `py -m pytest test_simpleitk_parity.py test_vtk_parity.py test_ct_mri_registration_parity.py -v` | 54 passed, 4 skipped (Elastix) in 24.41 s |
| `ritk` wheel rebuilt with `--auditwheel repair` (MSVC toolchain) | Installed successfully; `import ritk` OK |

### Updated artifacts
- `checklist.md`: Sprint 75 checklist items marked complete.
- `gap_audit.md`: GAP-R75-01..04 closed; SyN translation recovery risk removed; updated risk posture.

### Residual risk
- GAP-R08 (Elastix parity) — Medium: 4 Elastix tests still skipped (Elastix absent). ASGD optimizer and parameter-map interface remain absent. Not affected by this sprint.

---

## Sprint 74 — Completed

**Status**: Completed
**Phase**: Closure
**Goal**: Fix Python wheel DLL load failure on Windows; document the `ritk-python` build workflow; extend VTK parity tests with 8 CT/MRI-relevant operations; extend SimpleITK parity tests with 5 registration quality tests; add real-DICOM CT/MRI cross-modal parity test file.

### Gaps closed
| ID | Gap | Root cause | Resolution | Tag |
|---|---|---|---|---|
| GAP-R74-01 | Python wheel DLL load failure on Windows (MinGW runtime vs MSVC Python ABI) | Default toolchain `nightly-x86_64-pc-windows-gnu` links `_ritk.dll` against `libgcc_s_seh-1.dll`, `libstdc++-6.dll`, `libwinpthread-1.dll`; CPython 3.13 (MSVC ABI) cannot locate these DLLs via the default search path | Built wheel with `rustup run nightly-x86_64-pc-windows-msvc py -m maturin build --release --auditwheel repair`; maturin copies MinGW DLLs into `ritk.libs/` inside the wheel and patches the DLL search path at import time | [patch] |
| GAP-R74-02 | No build/test documentation for `ritk-python` | Wheel build process was undocumented; `--auditwheel repair` requirement was unknown | Created `crates/ritk-python/README.md` with build requirements, correct build command, test execution instructions, module API table, architecture description, and DICOM I/O dispatch documentation | [patch] |
| GAP-R74-03 | VTK parity tests lacked CT/MRI-relevant operations | `test_vtk_parity.py` covered only basic filters; no resampling, CT HU statistics, cross-modal NCC, anisotropic diffusion, cast, or spacing tests existed | Added 8 tests to `test_vtk_parity.py` (now 18 total): threshold, reslice identity, CT bimodal statistics, cross-modal NCC premise, histogram mass conservation, anisotropic diffusion spike reduction, integer&#8594;float cast, gradient magnitude with 0.5 mm spacing | [minor] |
| GAP-R74-04 | SimpleITK parity tests lacked registration quality tests | `test_simpleitk_parity.py` had no tests comparing RITK registration output quality against analytical or reference expectations | Added Section 5 (5 tests): BSpline FFD NCC improvement on Gaussian blob (NCC &#8805; 0.80), Symmetric Demons NCC improvement (NCC &#8805; 0.90), histogram matching vs SimpleITK (Pearson r &#8805; 0.99), histogram matching median shift, Thirion Demons NCC improvement | [minor] |
| GAP-R74-05 | No Python-level CT/MRI DICOM parity tests using real MRI-DIR data | No Python test exercised the full RITK I/O + statistics pipeline on real DICOM data and compared results against SimpleITK | Created `crates/ritk-python/tests/test_ct_mri_registration_parity.py` with 4 real-DICOM tests (skipif data absent): CT statistics vs SimpleITK, MRI statistics vs SimpleITK, cross-modal NCC < 0.5, histogram matching reduces distribution gap | [minor] |

### Architecture decisions
- `--auditwheel repair` is the canonical Windows build path; it bundles all MinGW runtime DLLs into the wheel, making `ritk` self-contained on MSVC Python installations.
- BSpline FFD NCC tests require smooth (Gaussian-blurred) input images. Binary sphere images produce near-zero interior gradients that cause the optimiser to declare convergence after the first iteration (rel_change < 1e-6).
- SyN translation recovery is not testable with the current synthetic volumes; velocity fields do not accumulate for pure translations under sigma_smooth=1.0–3.0. Symmetric Demons is used as the high-quality diffeomorphic parity reference.
- CT/MRI DICOM parity tests use `@pytest.mark.skipif(not _DATA_PRESENT, ...)` consistent with the `#[ignore]` pattern in Rust integration tests.
- VTK `DiffusionThreshold` means "diffuse faces with gradient **below** threshold" (same polarity as Perona-Malik conductance); set threshold > spike gradient to diffuse the spike.

### Verification
| Check | Result |
|---|---|
| `py -m pytest test_vtk_parity.py test_simpleitk_parity.py test_ct_mri_registration_parity.py -v` | 53 passed, 4 skipped (Elastix) in 18.79 s |
| `cargo check --workspace --tests` | 0 errors, 0 warnings |
| `ritk` wheel import (CPython 3.13, MSVC ABI, `--auditwheel repair`) | Confirmed working |
| CT/MRI DICOM parity (4 real-data tests) | 4/4 pass with MRI-DIR data present |

### Updated artifacts
- `checklist.md`: Sprint 74 checklist items marked complete.
- `gap_audit.md`: Sprint 74 closure notes added; SyN translation recovery gap recorded as open Medium risk.

### Residual risk
- SyN translation recovery — Medium: `syn_register` does not converge on synthetic translation test cases. The `warped_fixed` output equals the original fixed image identically, suggesting velocity fields do not accumulate. Requires investigation in `diffeomorphic/mod.rs` velocity field update loop.
- GAP-R08 (Elastix parity) — Medium: 4 Elastix tests exist and are skipped (Elastix absent in current environment). ASGD optimizer and parameter-map interface remain absent.

---

## Sprint 73 — Completed

**Status**: Completed
**Phase**: Closure
**Goal**: Download a proper CT/MRI DICOM combo for registration testing; add VTK filter parity tests against SimpleITK; add CT/MRI DICOM registration integration tests; fix all remaining ritk-snap compiler warnings.

### Gaps closed
| ID | Gap | Root cause | Resolution | Tag |
|---|---|---|---|---|
| GAP-R73-01 | 3 `ritk-snap` compiler warnings (unused doc comment, unused mut, dead code `step_slice`) | Warnings introduced in Sprint 72 implementation; `step_slice` was defined but never called | Changed `///` &#8594; `//` on nested closure doc comment in `loader.rs:302`; removed `mut` from `let mut try_add` in `loader.rs:304`; replaced 4 direct `step_slice_for_axis(self.axis, ±1)` call sites in `app.rs` with `self.step_slice(±1)` | [patch] |
| GAP-R73-02 | Paired CT test data absent — only porcine phantom MRI existed without matching CT | Sprint 72 downloaded MRI but not the CT from the same phantom | Downloaded 409-slice MRI-DIR CT (512×512, 0.390625 mm pixel spacing, 0.625 mm slice thickness, CC BY 4.0, PatientID=MRI-DIR-zzmeatphantom) from TCIA to `test_data/3_head_ct_mridir/DICOM/`; updated `test_data/README.md` | [patch] |
| GAP-R73-03 | No VTK filter parity tests | `test_simpleitk_parity.py` covered SimpleITK but no VTK comparison existed | Created `crates/ritk-python/tests/test_vtk_parity.py` with 10 VTK 9.6.1 &#8596; SimpleITK 2.5.4 parity tests: Gaussian (constant invariant + NRMSE < 0.15), gradient magnitude (analytical + Pearson r > 0.95), Laplacian (&#8711;²=0), median spike suppression, binary erosion (A&#8854;B&#8838;A), binary dilation (A&#8838;A&#8853;B), scalar range; 10/10 pass | [minor] |
| GAP-R73-04 | No CT/MRI DICOM registration integration tests | No Rust test exercised the BSpline FFD pipeline on real DICOM data | Created `crates/ritk-registration/tests/ct_mri_dicom_registration_test.rs` with 4 `#[ignore]` tests: CT DICOM metadata invariants, MRI DICOM metadata invariants, BSpline FFD NCC improvement on stride-16 32³ CT sub-volume (2-voxel x-shift, NCC_after > NCC_before &#8743; &#8805; 0.80), cross-modal intensity statistics differ | [minor] |

### Architecture decisions
- MRI-DIR porcine phantom CT (same anatomy as existing T2 MRI, gold fiducial ground truth) is the canonical CT&#8596;MRI test pair; no synthetic or mismatched data.
- VTK parity tests use `pytest.importorskip` for graceful skip when VTK/SimpleITK are absent; consistent with Elastix `@skipif` pattern.
- `step_slice` closes the dead-code gap without new logic: it is the existing `step_slice_for_axis(self.axis, delta)` wrapper; call sites consolidate to it.
- CT/MRI integration tests are `#[ignore]` (require 79.9 MB downloaded data); run explicitly with `-- --ignored`.
- VTK gradient/Laplacian filters require `SetDimensionality(3)`; default=2 silently skips the z-axis — documented in `test_vtk_parity.py` at module scope.

### Verification
| Check | Result |
|---|---|
| `cargo check -p ritk-snap --tests` | 0 errors, 0 warnings |
| `cargo check --test ct_mri_dicom_registration_test -p ritk-registration` | 0 errors, 0 warnings |
| `pytest crates/ritk-python/tests/test_vtk_parity.py -v` | 10/10 pass in 1.23 s |
| CT download: 409 DCM files, modality=CT, PatientID=MRI-DIR-zzmeatphantom | Verified |

### Updated artifacts
- `backlog.md`: Sprint 73 marked completed; all 4 gaps recorded as closed.
- `checklist.md`: Sprint 73 checklist items marked complete.
- `gap_audit.md`: Sprint 73 closure notes added; GAP-R07 confirmed closed (Sprint 66); GAP-R08 risk posture updated.

### Residual risk
- GAP-R08 (Elastix parameter-map interface, ASGD optimizer, Transformix path) remains Medium — parity tests exist but are skipped because Elastix is absent in the current Python environment.
- CT/MRI integration tests require manual download trigger (`cargo test -- --ignored`); not part of the standard CI pass.

---

## Sprint 72 — Completed

**Status**: Completed
**Phase**: Closure
**Goal**: Implement ritk-snap as a complete DICOM viewer binary with eframe/egui GUI shell, multi-planar MPR layout, DICOM series browser, 7 colormaps, 18 clinical W/L presets, measurement tools (Length, Angle, ROI, HU-point), NIfTI loading, DICOM overlay, and PNG slice export; add cranial MRI DICOM test data.

### Gaps closed
| ID | Gap | Root cause | Resolution | Tag |
|---|---|---|---|---|
| GAP-R72-01 | ritk-snap had no GUI application shell | No eframe/egui binary or SnapApp struct existed | Implemented `SnapApp` with `eframe::App` in `app.rs`; `main.rs` launches via `run_app`; 19 source files added across `render/`, `tools/`, `dicom/`, and `ui/` submodules | [minor] |
| GAP-R72-02 | No DICOM series browser in ritk-snap | No sidebar or tree widget existed | Implemented `SidebarPanel` with Patient&#8594;Study&#8594;Series tree via `scan_dicom_directory` in `ui/sidebar.rs` and `dicom/series_tree.rs` | [minor] |
| GAP-R72-03 | No MPR (multi-planar reconstruction) in viewer | No multi-viewport layout existed | Implemented 2×2 `MprLayout` with axial/coronal/sagittal viewports in `ui/layout.rs` and `ui/viewport.rs` | [minor] |
| GAP-R72-04 | No W/L presets in viewer | No window/level preset registry existed | Implemented `WindowPreset` with 14 CT + 4 MR clinical presets in `ui/window_presets.rs`; exposed via View menu | [minor] |
| GAP-R72-05 | No measurement tools in viewer | No interaction tool infrastructure existed | Implemented Length (mm), Angle (°), Rect ROI, Ellipse ROI, HU-point in `tools/kind.rs`, `tools/interaction.rs`, and `ui/measurements.rs` | [minor] |
| GAP-R72-06 | No NIfTI loading in viewer | GUI had no file-open path | Implemented `load_nifti_volume` dispatch via `ritk-io` in the GUI file-open handler | [minor] |
| GAP-R72-07 | No DICOM overlay in viewer | No viewport annotation layer existed | Implemented 4-corner DICOM text overlay + patient orientation labels in `ui/overlay.rs` | [minor] |
| GAP-R72-08 | No slice export in viewer | No export path existed in the GUI | Implemented PNG export via `rfd` file dialog in `ui/toolbar.rs` | [minor] |
| GAP-R72-09 | Missing cranial MRI DICOM test data | CT-to-MRI registration tests lacked real MRI input | Downloaded MRI-DIR T2 head phantom DICOM (94 slices, CC BY 4.0, TCIA) to `test_data/2_head_mri_t2/DICOM/`; documented in `test_data/README.md` | [patch] |
| GAP-R72-10 | No colormaps in viewer | No LUT rendering infrastructure existed | Implemented 7 colormaps with piecewise-linear LUT in `render/colormap.rs` and `render/slice_render.rs`; 42+ colormap tests added | [minor] |

### Architecture decisions
- `SnapApp` implements `eframe::App`; all domain logic is separated from the GUI shell per the `GuiBackend` trait boundary.
- `render/`, `tools/`, `dicom/`, and `ui/` submodules each own a single bounded responsibility; cross-module access is unidirectional.
- `WindowPreset` encodes W/L variation as data (not cloned functions); presets are selected by name at runtime.
- Colormap LUTs are piecewise-linear over `[0.0, 1.0]` and parameterized by control-point tables, not hardcoded per-colormap functions.
- NIfTI and DICOM load paths share the `LoadedVolume` type; no duplication of volume representation.

### Verification
| Check | Result |
|---|---|
| `cargo check --workspace --tests` | 0 errors, 0 warnings |
| Total test count | 102 tests pass (up from 42) |
| Commit | a3b08bd pushed to origin/main |

### Updated artifacts
- `backlog.md`: Sprint 72 marked completed; all 10 gaps recorded as closed.
- `checklist.md`: Sprint 72 checklist items marked complete.
- `gap_audit.md`: Sprint 72 closure notes added.

### Residual risk
- None identified from the selected Sprint 72 gaps.

---

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
