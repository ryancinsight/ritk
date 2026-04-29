## Sprint 72 — Completed

- [x] GAP-R72-01: implemented `SnapApp` eframe/egui binary with full multi-viewport viewer in `crates/ritk-snap/src/app.rs` and `main.rs`; 19 new source files added across `render/`, `tools/`, `dicom/`, and `ui/` submodules
- [x] GAP-R72-02: implemented `SidebarPanel` with Patient→Study→Series tree via `scan_dicom_directory` in `crates/ritk-snap/src/ui/sidebar.rs` and `dicom/series_tree.rs`
- [x] GAP-R72-03: implemented 2×2 MPR layout with axial/coronal/sagittal viewports in `crates/ritk-snap/src/ui/layout.rs` and `ui/viewport.rs`
- [x] GAP-R72-04: implemented `WindowPreset` with 14 CT + 4 MR clinical presets in `crates/ritk-snap/src/ui/window_presets.rs`; exposed via View → Window menu
- [x] GAP-R72-05: implemented Length (mm), Angle (°), Rect/Ellipse ROI, and HU-point tools in `crates/ritk-snap/src/tools/kind.rs`, `tools/interaction.rs`, and `ui/measurements.rs` with mm-accurate computation from DICOM pixel-spacing metadata
- [x] GAP-R72-06: implemented `load_nifti_volume` dispatch via `ritk-io` in the GUI file-open handler in `crates/ritk-snap/src/app.rs`; `LoadedVolume` carries the volume with affine metadata
- [x] GAP-R72-07: implemented 4-corner DICOM text overlay + patient orientation labels (L/R, A/P, S/I) in `crates/ritk-snap/src/ui/overlay.rs`
- [x] GAP-R72-08: implemented PNG slice export via `rfd` file dialog in `crates/ritk-snap/src/ui/toolbar.rs`
- [x] GAP-R72-09: downloaded MRI-DIR T2 head phantom DICOM (94 slices, CC BY 4.0, TCIA) to `test_data/2_head_mri_t2/DICOM/`; provenance, license, and intended use documented in `test_data/README.md`
- [x] GAP-R72-10: implemented 7 colormaps with piecewise-linear LUT in `crates/ritk-snap/src/render/colormap.rs` and `render/slice_render.rs`; 42+ colormap and render tests added
- [x] Verify: `cargo check --workspace --tests` 0 errors after each change
- [x] Verify: 102 tests pass workspace-wide (up from 42 pre-Sprint-72 baseline)
- [x] Update backlog.md and gap_audit.md on Sprint 72 closure

---

## Sprint 71 — Completed

- [x] GAP-R71-01: updated `crates/ritk-python/python/ritk/_ritk/statistics.pyi` so `zscore_normalize` exposes optional `mask: Image | None = None`
- [x] GAP-R71-02: added a positive Python-level smoke test for `zscore_normalize(image, mask=...)` with matching shapes and asserted computed output values
- [x] GAP-R71-03: verified `test_smoke.py` and `test_statistics_bindings.py` match the compiled `zscore_normalize` callable signature
- [x] GAP-R71-04: synced `backlog.md`, `checklist.md`, and `gap_audit.md` after verification
- [x] Verify: `cargo check --workspace --tests` 0 errors after each change
- [x] Verify: `cargo test -p ritk-python --lib` all pass
- [x] Update backlog.md and gap_audit.md on Sprint 71 closure

---

## Sprint 70 — Completed

- [x] GAP-R70-01: audited `white_stripe_normalize` Python binding; `mask`, `contrast`, and `width` are exposed and validated in `crates/ritk-python/src/statistics.rs`; no code change required
- [x] GAP-R70-02: added negative test for `zscore_normalize(mask=...)` with shape-mismatched mask in `crates/ritk-python/tests/test_statistics_bindings.py`; asserts value-semantic error boundary
- [x] GAP-R70-03: audited `run_lddmm` `learning_rate` wiring; `RegisterArgs.learning_rate` is already plumbed into `LddmmConfig`; no code change required
- [x] GAP-R70-04: added `test_minmax_normalize_range_inverted_bounds_raises` to `crates/ritk-python/tests/test_statistics_bindings.py`; asserts Python-level `PyValueError` path
- [x] Verify: `cargo check --workspace --tests` 0 errors after each change
- [x] Verify: `cargo test -p ritk-python --lib` all pass
- [x] Update backlog.md and gap_audit.md on Sprint 70 closure

---

## Sprint 69 — minmax_normalize_range validation, run_multires_syn convergence wiring, zscore masked CLI integration tests, ritk-python CI audit — Completed

- [x] GAP-R69-01: `validate_range(target_min, target_max) -> Result<(), String>` helper added to `ritk-python/src/statistics.rs`; called in `minmax_normalize_range` before delegate; error mapped to `PyValueError`; 4 unit tests added
- [x] GAP-R69-02: `convergence_threshold: 1e-6` replaced with `convergence_threshold: args.convergence_threshold` in `run_multires_syn` `MultiResSyNConfig` literal
- [x] GAP-R69-03: `pub mask: Option<PathBuf>` added to `NormalizeArgs`; `zscore` arm dispatches `normalize_masked` when mask is `Some`; `default_args` test helper updated; 2 CLI integration tests added
- [x] GAP-R69-04: audited `python_ci.yml`; `cargo test -p ritk-python --lib -- --test-threads=4` step already present; gap closed without code change
- [x] Verify: `cargo check --workspace --tests` — 0 errors
- [x] Verify: `cargo test -p ritk-core --lib` — 777/777 passed (unchanged)
- [x] Verify: `cargo test -p ritk-python --lib` — 10/10 passed (+4 from Sprint 68 baseline of 6)
- [x] Verify: `cargo test -p ritk-cli` — 197/197 passed (+2 from Sprint 68 baseline of 195)
- [x] Updated backlog.md and gap_audit.md on Sprint 69 closure

---

## Sprint 68 — zscore masked variant, bspline-syn convergence wiring, marker-watershed integration tests, percentile validation tests — Completed

- [x] GAP-R68-01: `ZScoreNormalizer::normalize_masked` added to `ritk-core/src/statistics/normalization/zscore.rs`; `zscore_normalize` Python binding extended with `#[pyo3(signature=(image, mask=None))]`; dispatches `normalize_masked` when mask is provided; falls back to `normalize` otherwise; 3 core tests added (masked stats, empty-mask fallback, metadata preservation)
- [x] GAP-R68-02: `convergence_threshold: 1e-6` hard-code removed from `run_bspline_syn`; replaced with `convergence_threshold: args.convergence_threshold`; `RegisterArgs.convergence_threshold` docstring updated to name both BSpline FFD and BSpline SyN
- [x] GAP-R68-03: `test_segment_marker_watershed_creates_output_with_correct_shape` and `test_segment_marker_watershed_output_contains_both_basin_labels` added to `segment.rs` tests; both helpers (`make_uniform_gradient_image`, `make_two_seed_marker_image`) co-located in `mod tests`; tests assert shape=[3,3,3] and label presence value-semantically
- [x] GAP-R68-04: `validate_percentiles(p: &[f64]) -> Result<(), String>` extracted as private helper in `ritk-python/src/statistics.rs`; inline validation in `nyul_udupa_normalize` refactored to call helper; 6 `#[cfg(test)]` tests added (empty, single-element, equal-pair, descending-pair, minimal valid, standard Nyul 13-element set)
- [x] Verify: `cargo check --workspace --tests` — 0 errors, 0 warnings
- [x] Verify: `cargo test -p ritk-core --lib` — 777 passed, 0 failed (was 774; +3)
- [x] Verify: `cargo test -p ritk-python --lib` — 6 passed, 0 failed (new)
- [x] Verify: `cargo test -p ritk-cli` — 195 passed, 0 failed (was 193; +2)
- [x] backlog.md updated (Sprint 68 closed, Sprint 69 planned)
- [x] gap_audit.md updated (Sprint 68 gap closures recorded)

---

## Sprint 67 — Python normalize parity, seeded watershed binding, adversarial region-growing tests, BSpline FFD CLI audit — Completed

- [x] GAP-R67-01: `histogram_match` extended with `#[pyo3(signature=(source,reference,num_bins=256))]`; guard `num_bins < 2 → PyValueError`; `nyul_udupa_normalize` extended with `percentiles: Option<Vec<f64>>`; pre-GIL validation; dispatches `NyulUdupaNormalizer::with_percentiles` or `::new()`
- [x] GAP-R67-02: `MarkerControlledWatershed` added to `use` imports in `segmentation.rs`; `marker_watershed_segment` function added before `register`; registered in submodule under `// Watershed`
- [x] GAP-R67-03: 5 adversarial tests added to `confidence_connected.rs` (multi-seed isolation, large-k gradient expansion, corner seed, zero-max-iterations, inclusive boundary values); 4 adversarial tests added to `neighborhood_connected.rs` (multi-seed isolation, boundary-radius clamping, large uniform image, noisy-shell boundary rejection)
- [x] GAP-R67-04: `convergence_threshold: f64` field added to `RegisterArgs` (default `0.00001`); `..Default::default()` removed from `run_bspline_ffd`; all 6 `BSplineFFDConfig` fields explicitly set; 22 test struct literals updated
- [x] Verify: `cargo check -p ritk-python` — 0 errors, 0 warnings
- [x] Verify: `cargo check -p ritk-cli` — 0 errors, 0 warnings
- [x] Verify: `cargo test -p ritk-core --lib` — 774 passed, 0 failed (was 765; +9)
- [x] Verify: `cargo test -p ritk-cli` — 193 passed, 0 failed (no change)
- [x] backlog.md updated (Sprint 67 closed, Sprint 68 planned)

---

## Sprint 66 — statistics re-exports, K-Means parity, CLI normalize command — Completed

- [x] GAP-R66-01: `statistics/mod.rs` `pub use normalization::` expanded to include `NyulUdupaNormalizer`, `WhiteStripeNormalizer`, `WhiteStripeConfig`, `MriContrast`, `WhiteStripeResult`
- [x] GAP-R66-02: `crates/ritk-cli/src/commands/normalize.rs` created with methods: `histogram-match`, `nyul`, `zscore`, `minmax`, `white-stripe`; `pub mod normalize` added to `commands/mod.rs`; `Normalize` variant + dispatch arm added to `main.rs`
- [x] GAP-R66-03: BSpline FFD gap confirmed already closed (prior sprint); backlog note corrected
- [x] GAP-R66-04: K-Means CLI parity — `kmeans_max_iterations`, `kmeans_tolerance`, `kmeans_seed` optional args added to `SegmentArgs`; `run_kmeans` applies them; `Default` initialises all to `None`
- [x] GAP-R66-04: K-Means Python parity — `kmeans_segment` signature extended with `max_iterations=None`, `tolerance=None`, `seed=None` via pyo3 signature attribute
- [x] Verify: `cargo check --workspace --tests` — 0 errors, 0 warnings
- [x] Verify: `cargo test -p ritk-core --lib` — 765 passed, 0 failed (no change from Sprint 65)
- [x] Verify: `cargo test -p ritk-io --lib` — 454 passed, 0 failed (no change)
- [x] Verify: `cargo test -p ritk-cli` — 193 passed, 0 failed (was 181; +12)
- [x] backlog.md updated (Sprint 66 closed, Sprint 67 planned)

---

## Sprint 65 — BinaryThreshold, MarkerControlledWatershed, Multi-Otsu Adversarial Tests, CLI/Python Integration — Completed

- [x] GAP-R65-01: `BinaryThreshold` struct + `binary_threshold` fn + `apply_binary_threshold_to_slice` added to `threshold/binary.rs`; re-exported in `threshold/mod.rs` and `segmentation/mod.rs`
- [x] GAP-R65-02: `MarkerControlledWatershed` added to `watershed/marker_controlled.rs`; FIFO priority-queue flooding with `grad_bits: u32` + `seq: u64` ordering; re-exported in `watershed/mod.rs` and `segmentation/mod.rs`; two QueueEntry bugs fixed (neg-bits ordering + FIFO tie-breaking)
- [x] GAP-R65-03: 10 adversarial multi-Otsu tests added to `multi_otsu.rs`: K=4 thresholds/labels, K=5 thresholds/labels, σ²_B = P₁·P₂·(μ₁−μ₂)² invariant, monotone-input monotone-output, K > distinct values, single-voxel degenerate
- [x] GAP-R65-04: CLI `binary` method (`run_binary`) + `marker-watershed` method (`run_marker_watershed`) added to `segment.rs`; `markers: Option<String>` added to `SegmentArgs`
- [x] GAP-R65-05: Python `binary_threshold_segment` binding added to `ritk-python/src/segmentation.rs`; registered in `lib.rs`
- [x] Bug fix: `QueueEntry` in `marker_controlled.rs` — replaced `neg_grad_bits: u64` (broken IEEE 754 ordering) with `grad_bits: u32` (reversed comparison) + `seq: u64` (FIFO tie-breaking)
- [x] Verify: `cargo check --workspace --tests` — 0 errors, 0 warnings
- [x] Verify: `cargo test -p ritk-core --lib` — 765 passed, 0 failed (was 724; +41)
- [x] Verify: `cargo test -p ritk-io --lib` — 454 passed, 0 failed (no change)
- [x] Verify: `cargo test -p ritk-cli` — 181 passed, 0 failed (was 177; +4)
- [x] backlog.md and gap_audit.md updated

---

## Sprint 64 — RT Dose/Plan Writers, VTI Binary CellData Coverage, crate-level re-exports — Completed

- [x] GAP-R64-01: add `write_rt_dose` to `rt_dose.rs` + value-semantic round-trip test
- [x] GAP-R64-02: add `write_rt_plan` to `rt_plan.rs` + value-semantic round-trip test
- [x] GAP-R64-03: add `RtDoseGrid`, `RtPlanInfo`, `RtBeamInfo`, `RtFractionGroup`, `write_dicom_seg`, `read_rt_dose`, `write_rt_dose`, `read_rt_plan`, `write_rt_plan` and VTI binary-appended functions to `ritk-io/src/lib.rs` pub-use surface
- [x] GAP-R64-04: extend VTI binary-appended tests to cover CellData + two-array offset correctness
- [x] Verify: `cargo check --workspace --tests` 0 errors after each change
- [x] Verify: `cargo test -p ritk-io --lib` all pass after each change
- [x] Update backlog.md and gap_audit.md on Sprint 64 closure

---

## Sprint 63 — CT Bed Separation Filter, Viewer Selection, and Modality-Aware Geometry Audit — Completed

- [x] Close GAP-R63-01: add a core CT bed separation filter for masking table/bed voxels while preserving patient foreground
  - `BedSeparationFilter` + `BedSeparationConfig` in `ritk-core/filter/intensity/bed_separation.rs`
  - threshold_foreground → keep_largest_component → binary_closing → binary_opening → apply_mask pipeline
- [x] Close GAP-R63-02: expose `bed-separation` as a selectable filter in `ritk-snap` and CLI filter dispatch
  - `FilterKind` enum (`BedSeparation(BedSeparationConfig)`, `Gaussian { sigma }`, `Median { radius }`) in `ritk-snap/src/lib.rs`
  - `apply_filter` method on `ViewerCore<B, 3>` — concrete dispatch, no dyn Trait, ownership-preserving via take/restore
  - `run_bed_separation` in `ritk-cli/src/commands/filter.rs` (was already present; wired to `--filter bed-separation`)
- [x] Close GAP-R63-03: audit CT, MRI, and ultrasound geometry/orientation handling so visualization uses a unified display contract
  - `ModalityDisplay::for_modality` in `ritk-snap/src/lib.rs`: CT→(center=-400,width=1500), MR→(600,1200), US→(128,256)
  - Geometry summary uses loaded image geometry as authoritative source (unchanged invariant, confirmed by test)
- [x] Close GAP-R63-04 (residual): DICOMDIR multi-series selection by SeriesInstanceUID
  - `per_file_series_uids: Vec<Option<String>>` parallel vec built in `scan_dicom_directory` scan loop
  - Series-UID grouping block after plurality-dim filter: selects unique plurality UID; warns on multi-series DICOMDIR
  - `first_series_instance_uid` overridden with selected UID
- [x] Close GAP-R63-05 (residual): DICOM-SEG writer
  - `write_dicom_seg` in `seg.rs`: BINARY MSB-first packing (inverse of `unpack_pixel_data`); FRACTIONAL byte-per-voxel
  - SegmentSequence SQ with per-segment label/description/algorithm items
  - FileMetaTableBuilder with SEG_SOP_CLASS_UID + Explicit VR LE transfer syntax
- [x] Close GAP-R63-06 (residual): VTI binary-appended format
  - `write_vti_binary_appended_bytes` + `write_vti_binary_appended_to_file` in `image_xml/writer.rs`
  - `read_vti_binary_appended_bytes` + `read_vti_binary_appended` in `image_xml/reader.rs`
  - uint32-LE length prefix + f32-LE data per array; lexicographic array sort for deterministic offsets
- [x] Close GAP-R63-07 (residual): RT Dose reader
  - `read_rt_dose` + `RtDoseGrid` in new `ritk-io/src/format/dicom/rt_dose.rs`
  - DoseGridScaling × u32-LE PixelData → `dose_gy: Vec<f64>`; GridFrameOffsetVector; IPP/IOP/PixelSpacing
- [x] Close GAP-R63-08 (residual): RT Plan reader
  - `read_rt_plan` + `RtPlanInfo` + `RtBeamInfo` + `RtFractionGroup` in new `ritk-io/src/format/dicom/rt_plan.rs`
  - BeamSequence (3-level SQ) + FractionGroupSequence with ReferencedBeamSequence
- [x] Add `ritk-core` tests for bed separation: `test_threshold_foreground`, `test_keep_largest_component_selects_body`, `test_mask_preserves_foreground_and_removes_background`, `test_apply_uses_outside_value`, `test_binary_morphology_round_trip_identity_radius_zero`
- [x] Add `ritk-snap` selection tests:
  - `test_filter_kind_bed_separation_dispatch_replaces_study_image` — shape preserved; outside_value applied to sub-threshold voxels
  - `test_filter_kind_no_study_returns_status_message` — message contains "no study"
- [x] Add modality-aware viewer tests:
  - `test_modality_display_ct_window_parameters` — CT:(-400,1500); MR:(600,1200); US:(128,256); None:(128,256)
- [x] Add `test_scan_directory_selects_most_populated_series_when_same_dimensions` (reader.rs) — Series A (3 slices) wins over B (1 slice); SeriesUID confirmed
- [x] Add VTI binary-appended tests:
  - `test_write_vti_binary_appended_header_contains_appended_format`
  - `test_write_vti_binary_appended_roundtrip` — 8-voxel 2×2×2; all values within 1e-6
  - `test_write_vti_binary_appended_offset_correctness` — offset[1]=12 analytically
- [x] Add DICOM-SEG writer tests:
  - `test_write_dicom_seg_binary_roundtrip` — 4×4×2 BINARY; pack/unpack symmetry verified
  - `test_write_dicom_seg_fractional_roundtrip` — 2×2×1 FRACTIONAL; {0,128,200,255} boundary values
  - `test_write_dicom_seg_rejects_mismatched_frame_count` — pixel_data.len()≠n_frames → Err
- [x] Add RT Dose tests: missing-file, wrong-SOP-class, synthetic-grid (1000×0.001=1.0 Gy)
- [x] Add RT Plan tests: missing-file, wrong-SOP-class, synthetic-plan (2 beams, 30 fractions)
- [x] Fix `HeadlessViewerBackend::Error = std::io::Error` in `ritk-cli/src/commands/viewer.rs` (satisfies StdError bound)
- [x] Revert `load_dicom_series` to `Result<Image<B,3>>` — backward-compatible; metadata variant is `load_dicom_series_with_metadata`
- [x] Verify: `cargo check --workspace --tests`: 0 errors
- [x] Verify: `cargo test -p ritk-io --lib -- --test-threads=4`: 445 passed, 0 failed
- [x] Verify: `cargo test -p ritk-snap --lib -- --test-threads=4`: 7 passed, 0 failed
- [x] Verify: `cargo test -p ritk-cli -- --test-threads=4`: 177 passed, 0 failed
- [ ] Sprint 64: RT Dose writer (GAP-R64-01)
- [ ] Sprint 64: RT Plan writer (GAP-R64-02)
- [ ] Sprint 64: crate-level re-exports (GAP-R64-03)
- [ ] Sprint 64: VTI binary-appended CellData coverage (GAP-R64-04)

---

## Sprint 61 — DICOM Direction Matrix Fix + Cross-Slice IOP/PixelSpacing Validation — Completed

- [x] GAP-C61-01 (GAP-R60-03): Fix `load_from_series` direction matrix — `from_row_slice` → `from_column_slice` (`reader.rs` L1254)
  - `metadata.direction = [rx,ry,rz, cx,cy,cz, nx,ny,nz]` is column-major (groups of 3 = columns)
  - `from_column_slice` assigns column 0=[rx,ry,rz], col 1=[cx,cy,cz], col 2=[nx,ny,nz] — ITK convention
  - `from_row_slice` on same data produces transpose — wrong for any non-identity IOP
  - Now consistent with `multiframe.rs` which has used `from_column_slice` since Sprint 46

- [x] GAP-C61-02 (GAP-R60-01): Add cross-slice IOP consistency check in `scan_dicom_directory`
  - Threshold: `IOP_CONSISTENCY_THRESHOLD = 1e-4` (>100× DS roundtrip encoding error)
  - Policy: warn-and-continue via `tracing::warn!` with slice_index and max_iop_deviation
  - Canonical IOP: first post-sort (lowest-position) slice

- [x] GAP-C61-03 (GAP-R60-02): Add cross-slice PixelSpacing consistency check in `scan_dicom_directory`
  - Threshold: `PIXEL_SPACING_CONSISTENCY_THRESHOLD = 1e-4` mm
  - Policy: warn-and-continue via `tracing::warn!` with slice_index and max_spacing_deviation
  - Canonical PixelSpacing: first post-sort slice

- [x] Add `test_load_from_series_oblique_direction_uses_column_slice_convention` — coronal IOP [1,0,0,0,0,-1]; asserts dir[(2,1)]=-1.0, dir[(1,2)]=+1.0 within 1e-5
- [x] Add `test_scan_directory_warns_on_inconsistent_iop` — axial+coronal slices in same dir; scan returns Ok; direction[0..6] ≈ first-slice IOP
- [x] Add `test_scan_directory_warns_on_inconsistent_pixel_spacing` — 0.8+1.0mm spacing slices; scan returns Ok; spacing[0..2] ≈ first-slice spacing
- [x] Verify: `cargo test -p ritk-io` — 428 passed, 0 failed (+3 from Sprint 60 baseline of 425)
- [ ] Sprint 62: DICOM-SEG writer (GAP-R60-04)
- [ ] Sprint 62: VTI binary-appended format (GAP-R60-05)
- [ ] Sprint 62: RT Dose / RT Plan readers (GAP-R60-06)
- [ ] Sprint 62: Gantry tilt handling and affine consistency audit (GAP-R62-01..03)

---

## Sprint 59 — DICOM-SEG Reader, DICOM-RT Structure Set Reader, VTK XML ImageData (.vti) Reader/Writer — Completed

- [x] Close GAP-R58-01: DICOM-SEG (Segmentation Storage) reader — `seg.rs`
- [x] Add `DicomSegmentInfo` and `DicomSegmentation` domain types
- [x] Implement `read_dicom_seg`: SOP class guard, header parsing, Segment Sequence (0062,0002), Per-Frame FG (5200,9230), Shared FG (5200,9229), BINARY bit-unpack (MSB-first, frame_bytes=⌈rows×cols/8⌉), FRACTIONAL byte-per-pixel
- [x] Add 6 SEG tests: missing file, wrong SOP, binary 4×4 single-frame (all-ones unpack), two-frame two-segment, pixel spacing round-trip, per-frame image position
- [x] Close GAP-R58-02: DICOM-RT Structure Set reader → VTK PolyData — `rt_struct.rs`
- [x] Add `RtContour`, `RtRoiInfo`, `RtStructureSet` domain types
- [x] Implement `read_rt_struct`: SOP class guard, StructureSetROI (3006,0020), RTROIObservations (3006,0080), ROIContour (3006,0039) with nested ContourSequence (3006,0040); HashMap → sorted Vec
- [x] Implement `rt_roi_to_polydata`: CLOSED_PLANAR→polygons, OPEN_PLANAR→lines, POINT→vertices, running offset indexing, f64→f32 coordinate cast
- [x] Add 8 RT struct tests: missing file, wrong SOP, single ROI CLOSED_PLANAR (point values), two ROIs sorted by number, interpreted type, polydata CLOSED_PLANAR, polydata OPEN_PLANAR, polydata mixed contours
- [x] Close GAP-R58-03: VTK XML ImageData (.vti) reader/writer — `format/vtk/image_xml/`
- [x] Add `VtkImageData` domain type to `vtk_data_object.rs`: `whole_extent [i64;6]`, `origin/spacing [f64;3]`, `point_data`/`cell_data` HashMaps; `n_points()`, `n_cells()`, `validate()`
- [x] Add `VtkDataObject::ImageData(VtkImageData)` variant
- [x] Add 4 VtkImageData domain tests: n_points/n_cells, validate ok, validate wrong scalar len, ImageData variant
- [x] Create `format/vtk/image_xml/writer.rs` (ASCII-inline VTI writer, 10 tests)
- [x] Create `format/vtk/image_xml/reader.rs` (ASCII-inline VTI reader, 10 tests)
- [x] Update `format/vtk/mod.rs` to expose `image_xml`
- [x] Update `format/dicom/mod.rs`: add `pub mod seg; pub mod rt_struct;` + re-exports
- [x] Fix `seg.rs` compile errors: replace `.to_int::<u16>()` with `.to_str().ok().and_then(parse)` pattern; fix `debug!` macro syntax
- [x] Verify: `cargo check -p ritk-io --tests` zero errors, zero warnings; `cargo test -p ritk-io --lib` 415 passed, 0 failed (+35 from Sprint 58 baseline of 380)
- [ ] Sprint 60: DICOM-SEG writer (write segmentation masks as DICOM-SEG)
- [ ] Sprint 60: RT Dose / RT Plan readers (dose grid and beam geometry)
- [ ] Sprint 60: VTK Rectilinear Grid XML (.vtr) reader/writer
- [ ] Sprint 60: VTI binary-appended format for large volumes

---

## Sprint 58 — VtkCellType + VTU Reader/Writer, DICOM Enhanced Multiframe, JPEG 2000 Lossless Round-Trip, Build Fix — Completed

- [x] Close GAP-R57-01: JPEG 2000 lossless round-trip test via openjpeg-sys FFI encoder
- [x] Add `write_jpeg2000_lossless_dicom_file` helper (unsafe openjpeg-sys, lossless J2K, 16-bit)
- [x] Add `test_decode_compressed_frame_jpeg2000_lossless_round_trip` (max_error == 0.0, ISO 15444-1)
- [x] Add `VtkCellType` enum (34 VTK standard cell type codes) to `vtk_data_object.rs`
- [x] Change `VtkUnstructuredGrid.cell_types` from `Vec<u8>` to `Vec<VtkCellType>`
- [x] Update `unstruct_grid.rs` reader/writer to use `VtkCellType`
- [x] Create `format/vtk/unstructured_xml/writer.rs` (VTU ASCII-inline writer, 10 tests)
- [x] Create `format/vtk/unstructured_xml/reader.rs` (VTU ASCII-inline reader, 16 tests)
- [x] Update `format/vtk/mod.rs` to expose `unstructured_xml`
- [x] Add `PerFrameInfo` struct to `multiframe.rs` (per-frame functional group data)
- [x] Add `per_frame: Vec<PerFrameInfo>` field to `MultiFrameInfo`
- [x] Implement `extract_functional_groups` (parses (5200,9229) and (5200,9230))
- [x] Update `load_dicom_multiframe` to apply per-frame rescale when available
- [x] Update `read_multiframe_info` to populate per_frame
- [x] Add 5 per-frame functional groups tests
- [x] Update `build.rs` to detect and emit libstdc++ search path
- [x] Update `.cargo/config.toml`: add `-lstdc++` link arg for x86_64-pc-windows-gnu
- [ ] Sprint 59: DICOM-SEG (Segmentation Object) reader for MITK parity
- [ ] Sprint 59: DICOM-RT structure set reader (VTK mesh output)
- [ ] Sprint 59: File format parity audit for remaining ITK formats

---

## Sprint 57 — JPEG-LS + JPEG 2000 Codec Integration — Completed

- [x] Enable `charls` feature on `dicom-transfer-syntax-registry` (workspace Cargo.toml)
- [x] Add `charls = { version = "0.4", features = ["static"] }` to workspace.dependencies
- [x] Add `charls = { workspace = true }` to ritk-io `[dependencies]` for static feature unification
- [x] Enable `openjpeg-sys` feature on `dicom-transfer-syntax-registry`
- [x] Add `openjpeg-sys = "1.0"` to workspace.dependencies
- [x] Add `openjpeg-sys = { workspace = true }` to ritk-io `[dev-dependencies]`
- [x] Add `[env]` section to `.cargo/config.toml` with clang/clang-cl per-target entries (`force = false`)
- [x] Add LLVM/Clang installation steps to CI test job (Linux, macOS, Windows)
- [x] Add `JpegLsLossless`, `JpegLsLossy`, `Jpeg2000Lossless`, `Jpeg2000Lossy` to `is_codec_supported()`
- [x] Update `is_codec_supported()` doc comment — remove "Not yet supported" section; add charls/OpenJPEG rows to table
- [x] Update codec.rs doc table — add JPEG-LS and JPEG 2000 rows
- [x] Rename `test_is_codec_supported_jpeg_ls_false` → `test_is_codec_supported_jpeg_ls_true`
- [x] Rename `test_is_codec_supported_jpeg2000_false` → `test_is_codec_supported_jpeg2000_true`
- [x] Add `test_decode_compressed_frame_jpegls_lossless_round_trip` (max_error = 0.0)
- [x] Add `test_decode_compressed_frame_jpegls_near_lossless_round_trip` (max_error ≤ 2.0)
- [x] Sprint 58: JPEG 2000 round-trip test via openjpeg-sys FFI encoding helper

---

## Sprint 56 -- Completed

- [x] DICOM-RLE-NATIVE-R56: `packbits_decode` + `decode_rle_lossless_frame` + RLE bypass in `decode_compressed_frame`
  - `dicom-transfer-syntax-registry v0.8.2` RLE decoder off-by-one: `start = spp − byte_offset = 1` (not 0) for 8-bit grayscale
  - Silently forces `dst[0] = 0` and loses `dst[N−1]` for any file where `pixel[0] ≠ 0`
  - Post-hoc correction impossible without permanent data loss (last pixel is unrecoverable)
  - `packbits_decode(input, expected_len)` implements the strict left inverse of `packbits_encode` per PS3.5 Annex G.3.1
  - `decode_rle_lossless_frame` parses the 64-byte DICOM RLE header, decodes each byte-plane segment via `packbits_decode`,
    reassembles into LE pixel bytes per PS3.5 §G.5: `raw[p×S×B + s×B + j] = segment[s×B + (B−1−j)][p]`
  - Fragment bytes accessed via `Value::PixelSequence(seq).fragments()[frame_idx].to_vec()`
    (dicom-rs stores pixel fragments as `Vec<u8>`, not `PrimitiveValue`)
  - `decode_compressed_frame` detects `RleLossless` via `obj.meta().transfer_syntax()` and dispatches to native decoder
  - Correct for `bits_allocated ∈ {8, 16}` and any `samples_per_pixel`; upstream codec still used for all other TSes

- [x] DICOM-RLE-UNRESTRICTED-RT-R56: `test_decode_compressed_frame_rle_lossless_unrestricted_round_trip`
  - New test: `pixel[0] = 42` (non-zero); encodes all N=16 pixels with `build_rle_fragment_8bit(&original)`
  - Would FAIL with upstream decoder (which forces `dst[0] = 0`); MUST pass with native decoder
  - Asserts `decoded[0] == 42.0` explicitly (not just `max_error`)
  - Verifies `max_error == 0.0` over all 16 pixels
  - Exercises `pixel[0] = 42, [50,50,50]` (literal + repeat), `[75,80,85,90]` (literal), `[100×4]` (repeat), `[120,130,140,150]` (literal)

- [x] Updated `test_decode_compressed_frame_rle_lossless_round_trip`
  - Changed `build_rle_fragment_8bit(&original[1..])` → `build_rle_fragment_8bit(&original)` (full 16 pixels)
  - Removed upstream-bug offset-compensation proof from docstring
  - Updated docstring to describe native decoder correctness
  - Test still exercises both PackBits run types (repeat + literal) in the same 4×4 frame

- [x] Sprint 56 verification: **337 passed, 0 failed**, zero warnings, zero errors

---

## Sprint 55 -- Completed

- [x] DICOM-CODEC-DOC-R55: Update `codec.rs` module docstring to list all 8 supported codecs
  - Sprint 53 docstring table listed only 3 codecs (JPEG Baseline, JPEG Lossless FOP, RLE)
  - Updated to 8-row table with `Feature` column covering all pure-Rust codecs
  - Added JPEG Extended (`.51`), JPEG Lossless NH (`.57`), JPEG XL variants (`.110`/`.111`/`.112`)
  - Replaced "Extension points" section with "Not yet supported" section (correct UIDs + C/C++ feature names)
  - Added JPEG Extended tolerance contract and RLE Lossless exact-fidelity contract to module docstring

- [x] DICOM-CODEC-EXT-RT-R55: `test_decode_compressed_frame_jpeg_extended_round_trip` in `codec.rs`
  - JPEG Extended (1.2.840.10008.1.2.4.51) was `is_codec_supported()=true` but had no round-trip test
  - SOF0 JPEG frame encapsulated under TS `.51`; `jpeg-decoder` handles both SOF0 and SOF1
  - Analytical tolerance: ≤ 16 (Q75 DC≤4 + AC(1,0)≤3 + AC(0,1)≤3 + AC(1,1)≤3 + margin = 16)
  - Test asserts: pixel count == 16, values ∈ [0,255], `max_error ≤ 16.0`

- [x] DICOM-CODEC-RLE-RT-R55: RLE Lossless round-trip test + DICOM PackBits encoder in `codec.rs`
  - RLE Lossless (1.2.840.10008.1.2.5) was `is_codec_supported()=true` but had no round-trip test
  - Implemented `packbits_encode` per DICOM PS3.5 Annex G.3.1 (repeat and literal runs, even-length pad)
  - Implemented `build_rle_fragment_8bit` per DICOM PS3.5 Annex G.4.1 (64-byte RLE header + segment)
  - Identified upstream `dicom-transfer-syntax-registry v0.8.2` RLE off-by-one: `start=1` for 8-bit
    grayscale forces `dst[0]=0`; `dst[i]=decoded_segment[i-1]` for i ∈ [1, N-1]
  - Offset-compensation proof: set `original[0]=0`, encode `original[1..]`; all 16 values match exactly
  - Test exercises both repeat runs ([50,50,50]) and literal runs ([75,80,85,90]) in same frame
  - Test asserts: pixel count == 16, values ∈ [0,255], `max_error == 0.0`

- [x] CI-MATRIX-R55: Extend CI test matrix to Windows and macOS
  - `test` job in `.github/workflows/ci.yml` converted from single `ubuntu-latest` to matrix
  - `strategy.matrix.os: [ubuntu-latest, windows-latest, macos-latest]`
  - `runs-on`, job `name`, cache `key`, and `restore-keys` all parameterized on `matrix.os`
  - All other jobs (`fmt`, `clippy`, `dependency-alignment`, `python-wheel`) remain Ubuntu-only
  - `python-wheel: needs: test` preserved; waits for all three matrix variants to succeed

- [x] Sprint 55 verification: **336 passed, 0 failed**, zero warnings, zero errors

---

## Sprint 54 -- Completed

- [x] DICOM-CODEC-EXT-TS1-R54: Add `JpegExtended` (1.2.840.10008.1.2.4.51) to `TransferSyntaxKind`
  - JPEG Extended (Process 2 & 4), lossy 12-bit; covered by existing `jpeg` feature (zero new deps)
  - `is_compressed()=true`, `is_lossless()=false`, `is_codec_supported()=true`
  - Tests: `test_from_uid_jpeg_extended`, `test_is_compressed_jpeg_extended_true`,
    `test_is_lossless_jpeg_extended_false`, `test_is_codec_supported_jpeg_extended_true`

- [x] DICOM-CODEC-EXT-TS2-R54: Add `JpegLosslessNonHierarchical` (1.2.840.10008.1.2.4.57) to `TransferSyntaxKind`
  - JPEG Lossless, Non-Hierarchical (Process 14); covered by existing `jpeg` feature
  - `is_compressed()=true`, `is_lossless()=true`, `is_codec_supported()=true`
  - Tests: `test_from_uid_jpeg_lossless_non_hierarchical`, `test_is_compressed_jpeg_lossless_nh_true`,
    `test_is_lossless_jpeg_lossless_nh_true`, `test_is_codec_supported_jpeg_lossless_nh_true`

- [x] DICOM-CODEC-JXL-DEP-R54: Enable `jpegxl` feature of `dicom-transfer-syntax-registry`
  - Added `dicom-transfer-syntax-registry = { version = "0.8", features = ["native", "jpegxl"] }` to workspace
  - Pure-Rust: `jxl-oxide` (decoder) + `zune-jpegxl` + `zune-core` (encoder); no native library dependency
  - Added `dicom-transfer-syntax-registry = { workspace = true }` to `ritk-io` dependencies
  - Added `zune-jpegxl` and `zune-core` as dev-dependencies for test data generation

- [x] DICOM-CODEC-JXL-TS1-R54: Add `JpegXlLossless` (1.2.840.10008.1.2.4.110) to `TransferSyntaxKind`
  - `is_compressed()=true`, `is_lossless()=true`, `is_codec_supported()=true`
  - ISO 18181-1 modular path; lossless invariant: `max|decoded[i] − original[i]| = 0`
  - Tests: `test_from_uid_jpeg_xl_lossless`, `test_is_compressed_jpeg_xl_lossless_true`,
    `test_is_lossless_jpeg_xl_lossless_true`, `test_is_codec_supported_jpeg_xl_lossless_true`

- [x] DICOM-CODEC-JXL-TS2-R54: Add `JpegXlJpegRecompression` (1.2.840.10008.1.2.4.111) to `TransferSyntaxKind`
  - `is_compressed()=true`, `is_lossless()=false`, `is_codec_supported()=true`
  - Decoder-only support via `JpegXlAdapter`
  - Tests: `test_from_uid_jpeg_xl_recompression`, `test_is_codec_supported_jpeg_xl_recompression_true`,
    `test_is_lossless_jpeg_xl_false`

- [x] DICOM-CODEC-JXL-TS3-R54: Add `JpegXl` (1.2.840.10008.1.2.4.112) to `TransferSyntaxKind`
  - `is_compressed()=true`, `is_lossless()=false` (not guaranteed by TS), `is_codec_supported()=true`
  - Tests: `test_from_uid_jpeg_xl`, `test_is_codec_supported_jpeg_xl_true`

- [x] DICOM-CODEC-JXL-RT-R54: JXL Lossless round-trip test (`codec.rs`)
  - 4×4 8-bit frame encoded via `zune-jpegxl` → wrapped in DICOM Part 10 (TS 1.2.840.10008.1.2.4.110)
  - Decoded via `decode_compressed_frame` → `max_error == 0.0` (JXL modular path: lossless by spec)
  - Test: `test_decode_compressed_frame_jxl_lossless_round_trip`

- [x] DICOM-TS-SEM-R54: Correct `is_compressed()` semantics for `DeflatedExplicitVrLittleEndian`
  - Removed from `is_compressed()`: per DICOM PS3.5 Table A-1, `is_compressed()` = pixel-data encapsulation
  - `DeflatedExplicitVrLittleEndian` compresses the dataset byte-stream, not pixel fragments
  - All formal invariants preserved: `is_natively_supported() ⟹ !is_compressed() ∧ !is_big_endian()`
  - Test: `test_is_compressed_deflated_false`

- [x] Formal invariant tests updated to cover all 16 known variants (was 11)
  - `test_codec_supported_implies_compressed` — exhaustive over 16 variants
  - `test_natively_supported_and_codec_supported_are_disjoint` — exhaustive over 16 variants
  - `test_natively_supported_implies_not_compressed_and_not_big_endian` — exhaustive over 16 variants
  - `test_uid_roundtrip_all_known` — exhaustive over 16 variants

- [x] Sprint 54 verification: **334 passed, 0 failed**, 0.09 s, zero warnings, zero errors

---

## Sprint 53 -- Completed

- [x] DICOM-CODEC-DEP-R53: Add `dicom-pixeldata = "0.8"` with `native` feature as direct ritk-io dependency
  - `dicom-pixeldata 0.8.2` was already a transitive dep via `dicom = "0.8.0"`; promoted to direct dep
  - `jpeg-decoder`, `jpeg-encoder`, and `dicom-rle` codecs already compiled in via `native` default feature
  - No new downloads or lock-file changes required

- [x] DICOM-CODEC-MODULE-R53: New `codec.rs` module with `pub(super) fn decode_compressed_frame`
  - Single dispatch entry point for all codec-supported compressed transfer syntaxes
  - Calls `PixelDecoder::decode_pixel_data_frame`, extracts raw bytes via `.data()`
  - Applies existing `decode_pixel_bytes` linear modality LUT (DICOM PS3.3 C.7.6.3.1.4)
  - Zero new unsafe code; `pub(super)` visibility keeps codec path internal to the dicom module

- [x] DICOM-CODEC-TS-PRED-R53: Add `is_codec_supported()` predicate to `TransferSyntaxKind`
  - Returns `true` for `JpegBaseline`, `JpegLosslessFirstOrderPrediction`, `RleLossless`
  - Returns `false` for JPEG-LS (requires `charls` feature) and JPEG 2000 (requires `openjp2` feature)
  - Invariants: `is_codec_supported() ⟹ is_compressed()`; `is_natively_supported() ⟹ !is_codec_supported()`
  - Tests: `test_codec_supported_implies_compressed`, `test_natively_supported_and_codec_supported_are_disjoint`

- [x] DICOM-CODEC-GUARD-R53: Relax compressed-TS guard in `load_from_series` and `load_dicom_multiframe`
  - Guard changed from `is_compressed()` to `is_compressed() && !is_codec_supported()`
  - JPEG Baseline, JPEG Lossless FOP, RLE Lossless now pass through to the decode path
  - JPEG-LS and JPEG 2000 are still correctly rejected (no codec registered)
  - Existing guard tests updated: TS changed from JPEG Baseline to JPEG-LS Lossless (1.2.840.10008.1.2.4.80)

- [x] DICOM-CODEC-READER-R53: Codec dispatch in `read_slice_pixels`
  - Detects TS from `slice.transfer_syntax_uid`; dispatches to `codec::decode_compressed_frame` when `is_codec_supported()`
  - Native (uncompressed) path unchanged; conditional branch is the only structural change
  - E2E test: `test_load_series_jpeg_baseline_codec_round_trip` — Secondary Capture DICOM with JPEG Baseline TS
    loads successfully and per-pixel error ≤ 16 (JPEG Q75 analytically-derived bound)

- [x] DICOM-CODEC-MF-R53: Codec dispatch in `load_dicom_multiframe`
  - When `ts.is_codec_supported()`, decodes each frame individually via `codec::decode_compressed_frame(&obj, frame_idx, ...)`
  - Uncompressed path unchanged; codec path builds `all_floats` by extending frame-by-frame
  - E2E test: `test_load_multiframe_jpeg_baseline_codec_round_trip` — 2-frame JPEG Baseline multiframe DICOM
    loads successfully with correct shape [2,4,4] and per-frame error ≤ 16

- [x] DICOM-CODEC-RT1-R53: `test_decode_compressed_frame_jpeg_baseline_round_trip` (codec.rs)
  - 4×4 8-bit JPEG Baseline: decoded pixel count == 16, all values in [0, 255], max error ≤ 16
  - Tolerance derivation: DC quantization step ≤ 4 + 3 primary AC terms ≤ 3 each + higher-order margin

- [x] DICOM-CODEC-RT2-R53: `test_decode_compressed_frame_rescale_contract` (codec.rs)
  - Uniform 4×4 patch: `scaled[i] == base[i] × 2.0 + 10.0` within 0.01 floating-point epsilon
  - Verifies modality LUT linearity independent of JPEG spatial quantization effects

---

## Sprint 52 -- Completed

- [x] DICOM-SERIES-UID-MONO-R52: Fix `generate_series_uid` monotonicity in writer.rs
  - `AtomicU64` static counter added (`COUNTER`)
  - Format `2.25.<ns>.<seq>` guarantees distinct UIDs within a process
  - Symmetric with Sprint 51 fix for `generate_multiframe_uid`
  - Test: `test_series_uid_distinct_on_rapid_successive_calls`

- [x] DICOM-TS-BE-R52: Remove ExplicitVrBigEndian from `is_natively_supported()`
  - `decode_pixel_bytes` uses `u16::from_le_bytes` / `i16::from_le_bytes` exclusively
  - Applying LE decode to BE bytes produces `bswap(x)` — silently incorrect intensities
  - ExplicitVrBigEndian retired per DICOM PS 3.5 (withdrawn 2004)

- [x] DICOM-TS-DEFLATE-R52: Remove DeflatedExplicitVrLittleEndian from `is_natively_supported()`
  - Both readers reject Deflated via `is_compressed()`
  - Prior classification violated invariant `is_natively_supported() => !is_compressed()`
  - Test: `test_is_natively_supported_deflated_false`

- [x] DICOM-TS-BIGENDIAN-PRED-R52: Add `is_big_endian()` predicate to `TransferSyntaxKind`
  - Returns `true` only for `ExplicitVrBigEndian`
  - Enables precise rejection guard distinct from the compressed-TS check
  - Tests: `test_big_endian_is_big_endian_true`, `test_explicit_vr_le_is_not_big_endian`

- [x] DICOM-TS-READER-GUARD-R52: Add BigEndian guard to series reader (`load_from_series`)
  - Guard added alongside existing `is_compressed()` check
  - Returns `Err` with message containing `"big-endian"` before any pixel decode
  - Test: `test_load_series_big_endian_ts_errors`

- [x] DICOM-TS-MF-GUARD-R52: Add BigEndian guard to multiframe reader (`load_dicom_multiframe`)
  - Guard added alongside existing `is_compressed()` check
  - Returns `Err` with message containing `"big-endian"` before any pixel decode
  - Test: `test_multiframe_rejects_big_endian_ts`

- [x] DICOM-TS-INVARIANT-R52: Formal invariant `is_natively_supported() ⟹ !is_compressed() ∧ !is_big_endian()`
  - Property test exhaustively verifies over all 11 known `TransferSyntaxKind` variants
  - Test: `test_natively_supported_implies_not_compressed_and_not_big_endian`
  - Positive coverage: `test_implicit_vr_le_is_natively_supported`, `test_explicit_vr_le_is_natively_supported`

- [x] HYGIENE-SCRATCH-R52: Delete 37 scratch/temp files from repository root
  - Removed: `TransformParameters.0.txt`, all `_*.py` scripts, `_*.txt` scratch files,
    `dg_test.tmp`, `fix_docs.py`, `gen_morph.py`, `gen_sprint27.py`, `result.0.nii`,
    `sizes.csv`, `sprint27_write.py`, `test2.py`, `test_out.rs`, `test_out.txt`,
    `test_output.txt`, `test_sprint.rs`, all `write_*.py` scripts, and other ad-hoc artifacts

- [x] HYGIENE-GITIGNORE-R52: Append `*.tmp`, `*.nii`, `sizes.csv` to `.gitignore`
  - Prior patterns (`_*.tmp`, `result.*.nii`) were narrower; broadened to prevent future commits

---

## Sprint 51 -- Completed

- [x] DICOM-MF-UID-R51: Add StudyInstanceUID (0020,000D) and SeriesInstanceUID (0020,000E) to multiframe writer
  - Both Type 1 mandatory under SC Multi-Frame IOD (PS3.3 A.8.5.2)
  - Generated via `generate_multiframe_uid()` — two independent calls per write operation
  - UIDs guaranteed distinct within a process (atomic counter + nanosecond clock)

- [x] DICOM-MF-TYPE2-R51: Add six Type 2 mandatory tags to multiframe writer
  - (0010,0010) PatientName — empty default
  - (0010,0020) PatientID — empty default
  - (0008,0020) StudyDate — empty default
  - (0008,0090) ReferringPhysicianName — empty default
  - (0020,0010) StudyID — empty default
  - (0020,0011) SeriesNumber — empty default

- [x] DICOM-MF-PR-R51: Honor PixelRepresentation in `load_dicom_multiframe`
  - `MultiFrameInfo.pixel_representation: u16` field added (default 0 = unsigned)
  - `extract_multiframe_header` extracts (0028,0103)
  - `load_dicom_multiframe` uses `super::reader::decode_pixel_bytes` (now `pub(super)`)
  - Signed i16 pixels correctly decoded via two's-complement arithmetic

- [x] DICOM-MF-UID-MONO-R51: Fix `generate_multiframe_uid` monotonicity
  - `AtomicU64` static counter added
  - Format `2.25.<ns>.<seq>` guarantees distinct UIDs even on low-resolution Windows clocks

- [x] DICOM-MF-UID-TEST-R51: `test_multiframe_has_study_and_series_uids`
  - Writes via `write_dicom_multiframe`, opens file
  - Asserts StudyInstanceUID and SeriesInstanceUID present, non-empty, and distinct

- [x] DICOM-MF-TYPE2-TEST-R51: `test_multiframe_has_type2_patient_study_series_tags`
  - Asserts all six Type 2 tags present in emitted file (presence check; value may be empty)

- [x] DICOM-MF-SIGNED-TEST-R51: `test_load_multiframe_signed_i16_roundtrip`
  - Manually constructs DICOM file with PixelRepresentation=1 (signed)
  - Input: [-1000, 0, 1000, 2000] as i16 LE bytes, identity rescale
  - Asserts decoded f32 within 0.5 of analytical ground truth

---

## Sprint 50 -- Completed

- [x] DICOM-D1-PIXELDECODE-R50: Centralized `decode_pixel_bytes` helper in reader.rs
  - Handles 8-bit unsigned, 16-bit unsigned, 16-bit signed (PixelRepresentation=1)
  - Applied per DICOM PS3.3 C.7.6.3.1.4 linear modality LUT: F(x) = x × slope + intercept
  - Verified by 5 unit tests with analytically derived expected values

- [x] DICOM-D2-FILEDETECT-R50: Canonicalize `is_likely_dicom_file`
  - Accepts: `.dcm`, `.dicom`, `.ima` (case-insensitive)
  - Rejects: `.hdr`, `.img` (Analyze 7.5), `.raw` (unstructured binary)
  - Extensionless: probe DICM magic at byte offset 128 (DICOM PS3.10 §7.1)

- [x] DICOM-D3-WRITERDRY-R50: DRY fix in writer.rs
  - `write_dicom_series` references `DICOM_SOP_CLASS_SECONDARY_CAPTURE` constant
  - SamplesPerPixel (0028,0002) added to `writer_exclusion_tags`

- [x] DICOM-D4-WINDOWMETA-R50: Window/pixel metadata in `DicomSliceMetadata`
  - New fields: `pixel_representation`, `bits_allocated`, `window_center`, `window_width`
  - `DicomSliceMetadata::default()` implemented
  - `known_handled_tags` updated for (0028,0002), (0028,0103), (0028,1050), (0028,1051)
  - Signed i16 read verified by `test_read_slice_pixels_signed_i16_roundtrip`

---

## Sprint 49 -- Completed

- [x] DICOM-TYPE2-META-R49: Add Type 2 fallback tags to `write_dicom_series_with_metadata(None)`
  - Five Type 2 mandatory tags absent when `metadata=None`: (0008,0090) ReferringPhysicianName,
    (0010,0010) PatientName, (0010,0020) PatientID, (0008,0020) StudyDate, (0020,0011) SeriesNumber
  - Inserted unconditional defaults before the `if let Some(m) = metadata` block
  - The conditional block overrides via `obj.put()` when metadata provides non-None values
  - All five tags now present in every emitted slice regardless of metadata argument
  - Verified by `test_metadata_writer_none_metadata_type2_tags`

- [x] DICOM-E2E-ROUNDTRIP-BASIC-R49: `test_write_series_load_series_intensity_roundtrip` in `reader.rs`
  - Writes 4×4×4 image (intensities 0..63) via `write_dicom_series`
  - Loads via `load_dicom_series`
  - Asserts per-voxel `|decoded − original| ≤ 65535 × 0.5e-6 + 0.5e-6 + slope/2`
  - Tolerance analytically derived: DS `{:.6}` format → at most 0.5e-6 rounding per coefficient;
    accumulated over max u16 (65535) → ≈0.033; quantization adds slope/2 ≈ 1.14e-4

- [x] DICOM-E2E-ROUNDTRIP-META-R49: `test_write_metadata_series_load_series_intensity_roundtrip` in `reader.rs`
  - Writes 3×4×4 image with origin [5,10,-20], spacing [0.5,0.5,1.5] via `write_dicom_series_with_metadata`
  - Loads via `load_dicom_series`
  - Asserts per-voxel intensity error within same DS-precision analytical bound
  - Asserts origin round-trips within 1e-4 mm on all three axes
  - Asserts spacing round-trips within 1e-4 mm on all three axes

- [x] DICOM-TYPE2-META-TEST-R49: `test_metadata_writer_none_metadata_type2_tags` in `writer.rs`
  - Calls `write_dicom_series_with_metadata(None)`, opens first slice
  - Asserts all five Type 2 tags present: (0010,0010), (0010,0020), (0008,0090), (0008,0020), (0020,0011)

- [x] GAP-AUDIT-IO-SYNC-R49: Update `gap_audit.md` sections 6.1, 6.2, 6.4, 6.6, 6.8
  - Section 6.1 MetaImage: Critical → Closed (Sprint 2); "Planned location" replaced with implementation bullets
  - Section 6.2 NRRD: High → Closed (Sprint 2); Teem prose replaced with implementation bullets
  - Section 6.4 VTK Image: Medium → Closed (Sprint 8); "Planned location" replaced with implementation bullets
  - Section 6.6 Analyze: Low → Closed (Sprint 2); legacy prose replaced with implementation bullets
  - Section 6.8 JPEG 2D: Low → Closed (Sprint 8); lossy-artifact prose replaced with implementation bullets
  - Section 8.5 priority matrix was already correct (all Closed); no change needed there

---

## Sprint 48 -- Completed

- [x] DICOM-TS-GUARD-MF-R48: Compressed TS detection in `load_dicom_multiframe`
  - After `open_file`, reads `obj.meta().transfer_syntax()` and calls `TransferSyntaxKind::from_uid(ts_uid).is_compressed()`
  - Returns `Err` with TS UID and path before any pixel decode; prevents silent garbage-intensity output
  - Added `use super::transfer_syntax::TransferSyntaxKind` to `multiframe.rs` imports
  - Verified by `test_load_multiframe_compressed_ts_errors`

- [x] DICOM-TS-GUARD-SERIES-R48: Compressed TS detection in `load_from_series`
  - Pre-decode loop over `slices.iter()` checks each `DicomSliceMetadata.transfer_syntax_uid`
  - Uses `TransferSyntaxKind::from_uid(ts).is_compressed()`; bails with TS UID and slice path on first compressed hit
  - Added `use super::transfer_syntax::TransferSyntaxKind` to `reader.rs` imports
  - Verified by `test_load_series_compressed_ts_errors`

- [x] DICOM-INFO-RESCALE-R48: Add `rescale_slope: f64` and `rescale_intercept: f64` to `MultiFrameInfo`
  - Two new public fields: `rescale_slope` (default 1.0) and `rescale_intercept` (default 0.0)
  - Populated from (0028,1053) and (0028,1052) inside `extract_multiframe_header`
  - Exposes the linear transform without requiring a second file open
  - Verified by `test_multiframe_info_rescale_slope_intercept_populated`

- [x] DICOM-MF-LOAD-DRY-R48: Extract `extract_multiframe_header` — eliminate header parse duplication
  - Private `fn extract_multiframe_header(path: &Path, obj: &InMemDicomObject) -> MultiFrameInfo`
  - Encapsulates all header element reads: n_frames, rows, cols, bits_allocated, pixel_spacing,
    frame_thickness, modality, sop_class_uid, image_position, image_orientation, rescale_slope, rescale_intercept
  - Both `read_multiframe_info` and `load_dicom_multiframe` delegate to it; each opens the file once
  - Zero header-field duplication remains

- [x] DICOM-CLAMP-SERIES-R48: Fix missing `.clamp(0.0, 65535.0)` in both series writer pixel encoding paths
  - `write_dicom_series` and `write_dicom_series_with_metadata` were both missing the clamp before `as u16`
  - Added `.round().clamp(0.0, 65535.0) as u16` to both per-slice pixel encoding closures
  - `write_multiframe_impl` already had the correct form; all three writers now consistent
  - Verified by `test_series_pixel_clamp_u16_range`

- [x] DICOM-CONV-TYPE-R48: Add `ConversionType` (0008,0064) = "WSD" to all three writers
  - SC Equipment Module (PS3.3 C.8.6.1) mandates ConversionType as Type 1
  - "WSD" (Workstation) added after Modality in `write_dicom_series`, `write_dicom_series_with_metadata`, and `write_multiframe_impl`
  - Added `writer_tag_key(0x0008, 0x0064)` to `writer_exclusion_tags()` to prevent preservation duplication
  - Verified by `test_series_writer_has_conversion_type_wsd` and `test_multiframe_has_conversion_type_wsd`

- [x] DICOM-TYPE2-BASIC-R48: Add Type 2 mandatory tags to `write_dicom_series`
  - Five Type 2 tags absent from the basic series writer added with empty/default values per DICOM PS3.3
  - (0008,0090) ReferringPhysicianName="", (0010,0010) PatientName="", (0010,0020) PatientID="",
    (0008,0020) StudyDate="", (0020,0011) SeriesNumber="0"
  - Tags (0008,0090) and (0020,0011) added to `writer_exclusion_tags()`
  - Verified by `test_basic_series_writer_has_type2_patient_tags`

- [x] DICOM-TS-GUARD-MF-TEST-R48: `test_load_multiframe_compressed_ts_errors` in `multiframe.rs`
  - Writes DICOM with JPEG Baseline TS (1.2.840.10008.1.2.4.50) via `FileMetaTableBuilder`
  - Asserts `load_dicom_multiframe` returns `Err`; error message contains TS UID or "compress"

- [x] DICOM-INFO-RESCALE-TEST-R48: `test_multiframe_info_rescale_slope_intercept_populated` in `multiframe.rs`
  - 1×5×5 image range [0.0, 24.0]; analytical slope = 24.0/65535.0, intercept = 0.0
  - Asserts |info.rescale_slope − expected| < 5×10⁻⁷ and |info.rescale_intercept − 0.0| < 5×10⁻⁷
  - Tolerance derived from DS `{:.6}` format precision

- [x] DICOM-CONV-TYPE-MF-TEST-R48: `test_multiframe_has_conversion_type_wsd` in `multiframe.rs`
  - Writes via `write_dicom_multiframe`, opens with `open_file`, reads (0008,0064), asserts trimmed == "WSD"

- [x] DICOM-CLAMP-TEST-R48: `test_series_pixel_clamp_u16_range` in `writer.rs`
  - 16 analytically-spaced f32 values 0→65535 (step = 65535/15); all encoded u16 ≤ 65535

- [x] DICOM-CONV-TYPE-SERIES-TEST-R48: `test_series_writer_has_conversion_type_wsd` in `writer.rs`
  - Writes via `write_dicom_series`, opens first slice, reads (0008,0064), asserts trimmed == "WSD"

- [x] DICOM-TYPE2-TEST-R48: `test_basic_series_writer_has_type2_patient_tags` in `writer.rs`
  - Writes via `write_dicom_series`, opens first slice
  - Asserts presence of (0010,0010), (0010,0020), (0008,0090), (0008,0020), (0020,0011)

- [x] DICOM-TS-GUARD-SERIES-TEST-R48: `test_load_series_compressed_ts_errors` in `reader.rs`
  - Writes single CT slice with JPEG Baseline TS in file meta
  - Verifies `scan_dicom_directory` captures compressed TS in slice metadata
  - Verifies `load_dicom_series` returns `Err` with TS UID in error message

---

## Sprint 47 -- Completed

- [x] DICOM-SPP-MF-R47: Add `SamplesPerPixel` (0028,0002) = 1 to `write_multiframe_impl`
  - Type 1 mandatory tag in DICOM Image Pixel Module (PS3.3 C.7.6.3.1.1)
  - Was absent from multi-frame writer; emitted `DataElement::new(Tag(0x0028,0x0002), VR::US, PrimitiveValue::from(1_u16))` before Rows element
  - Verified by `test_written_multiframe_has_samples_per_pixel_one`

- [x] DICOM-SPP-SERIES-R47: Add `SamplesPerPixel` (0028,0002) = 1 to both series writers
  - Same mandatory tag absent from `write_dicom_series` and `write_dicom_series_with_metadata`
  - Fixed in both per-slice emission loops before `Rows` element
  - Verified by `test_series_writer_has_samples_per_pixel_one`

- [x] DICOM-INST-NUM-MF-R47: Add `InstanceNumber` (0020,0013) to multi-frame writer
  - Type 2 required tag for SC Image Module; emitted from `config.instance_number` (default 1)
  - Verified by `test_writer_config_instance_number_propagated` (asserts value == 42 with explicit config)

- [x] DICOM-DRY-DS-R47: Extract `parse_ds_backslash<const N: usize>` private helper
  - Six duplicated DS backslash-parse closures across `read_multiframe_info` and `load_dicom_multiframe`
  - Replaced with `fn parse_ds_backslash<const N: usize>(s: &str) -> Option<[f64; N]>`
  - Const generic parameter encodes field-width variation; zero logic duplication remains
  - All callers: pixel_spacing (N=2), image_position (N=3), image_orientation (N=6) in both functions

- [x] DICOM-CONFIG-R47: Add `MultiFrameWriterConfig` builder struct and `write_dicom_multiframe_with_config`
  - `MultiFrameWriterConfig { sop_class_uid: String, spatial: Option<MultiFrameSpatialMetadata>, instance_number: u32 }`
  - `Default` impl: sop_class = `MF_GRAYSCALE_WORD_SC_UID`, instance_number = 1, spatial = None
  - `write_dicom_multiframe_with_config` accepts explicit `&MultiFrameWriterConfig`
  - `write_dicom_multiframe` and `write_dicom_multiframe_with_options` delegate via config construction; public signatures unchanged
  - `write_multiframe_impl` refactored to take `config: &MultiFrameWriterConfig`

- [x] DICOM-REEXPORT-R47: Fix `mod.rs` re-export gap for multi-frame types
  - `MultiFrameSpatialMetadata`, `write_dicom_multiframe_with_options`, `MultiFrameWriterConfig`, `write_dicom_multiframe_with_config` added to `pub use multiframe::{...}`
  - Module doc Public API section expanded with Series I/O, Multi-Frame I/O, and Object Model subsections

- [x] DICOM-SPP-MF-TEST-R47: Add `test_written_multiframe_has_samples_per_pixel_one` in `multiframe.rs`
  - Writes via `write_dicom_multiframe`, reads `(0028,0002)` via `open_file`, asserts parsed u16 == 1

- [x] DICOM-INST-NUM-TEST-R47: Add `test_writer_config_instance_number_propagated` in `multiframe.rs`
  - Writes via `write_dicom_multiframe_with_config` with `instance_number=42`
  - Reads `(0020,0013)` via `open_file`, asserts parsed u32 == 42

- [x] DICOM-NEG-RT-TEST-R47: Add `test_round_trip_negative_intensity_image` in `multiframe.rs`
  - 24-sample image spanning [-1024, 500]; analytical slope = 1524.0/65535.0 ≈ 0.02325
  - Asserts |recovered − original| ≤ slope + 1.0 for all samples

- [x] DICOM-FLAT-RT-TEST-R47: Add `test_round_trip_flat_image_exact` in `multiframe.rs`
  - Constant image (42.75_f32, exactly representable in f32 and DS "{:.6}" format)
  - Verifies slope=1.0 / all-zeros u16 branch; asserts |recovered − 42.75| ≤ f32::EPSILON

- [x] DICOM-SPP-SERIES-TEST-R47: Add `test_series_writer_has_samples_per_pixel_one` in `writer.rs`
  - Writes via `write_dicom_series`, opens first slice, asserts `(0028,0002)` == 1

---

## Sprint 46 -- Completed

- [x] DICOM-MF-SOP-FIX-R46: Fix `write_dicom_multiframe` SOP class to Multi-Frame Grayscale Word SC
  - Was emitting `1.2.840.10008.5.1.4.1.1.7` (Single-frame Secondary Capture)
  - Corrected to `1.2.840.10008.5.1.4.1.1.7.3` (Multi-Frame Grayscale Word Secondary Capture)
  - Extracted `MF_GRAYSCALE_WORD_SC_UID` const — single authoritative reference used by DataElement and FileMetaTableBuilder
  - Updated existing SOP class assertions in `test_multiframe_info_and_roundtrip_writer_read_consistency` and `test_read_multiframe_info_reports_scalar_defaults_for_single_frame`

- [x] DICOM-MF-SPATIAL-R46: Add `MultiFrameSpatialMetadata` and `write_dicom_multiframe_with_options`
  - `MultiFrameSpatialMetadata { origin: [f64;3], pixel_spacing: [f64;2], slice_thickness: f64, image_orientation: [f64;6], modality: String }`
  - `write_dicom_multiframe_with_options(path, image, Option<&MultiFrameSpatialMetadata>)` — emits IPP/IOP/PixelSpacing/SliceThickness/Modality when Some
  - Shared private `write_multiframe_impl`; `write_dicom_multiframe` public signature unchanged

- [x] DICOM-MF-INFO-SPATIAL-R46: Extend `MultiFrameInfo` and `read_multiframe_info` with IPP/IOP fields
  - Added `image_position: Option<[f64; 3]>` and `image_orientation: Option<[f64; 6]>` to `MultiFrameInfo`
  - `read_multiframe_info` now parses (0020,0032) ImagePositionPatient and (0020,0037) ImageOrientationPatient

- [x] DICOM-MF-LOAD-SPATIAL-R46: `load_dicom_multiframe` derives origin and direction from IPP/IOP
  - Origin set from (0020,0032) when present; defaults to [0,0,0]
  - Direction derived via `SMatrix::from_column_slice` with cols [row_cosines, col_cosines, normal]; normal = row × col
  - Previously hardcoded `Point::new([0,0,0])` and `Direction::identity()`

- [x] DICOM-MF-SOP-TEST-R46: Add `test_multiframe_sop_class_is_mf_grayscale_word` in `multiframe.rs`
  - Writes via `write_dicom_multiframe`, reads info, asserts `sop_class_uid == Some("1.2.840.10008.5.1.4.1.1.7.3")`

- [x] DICOM-MF-SPATIAL-TEST-R46: Add `test_write_multiframe_with_spatial_metadata_round_trip` in `multiframe.rs`
  - Writes with origin=[10,20,-50], pixel_spacing=[0.8,0.8], slice_thickness=2.5, IOP=identity row/col, modality="CT"
  - Asserts `read_multiframe_info` IPP ±1e-4, IOP ±1e-4, modality exact
  - Asserts `load_dicom_multiframe` loaded origin ±1e-4
  - Asserts pixel reconstruction error ≤ slope + 1.0

- [x] DICOM-LOAD-DIR-FIX-R46: Fix `load_from_series` to use `metadata.direction` instead of `Direction::identity()`
  - `load_from_series` was ignoring `metadata.direction: [f64; 9]` and constructing Image with identity direction
  - Fixed to `Direction::from_row_slice(&metadata.direction)`

- [x] DICOM-READER-BVR-FIX-R46: Fix binary-VR routing in `scan_dicom_directory` top-level preservation loop
  - Same dicom-rs 0.8 `to_str()` bug from Sprint 45 `parse_sequence_item` fix also present in top-level loop
  - VR::OB/OW/OD/OF/OL/UN elements were stored as `DicomValue::Text` instead of routed to `preservation.preserved`
  - Fixed by adding the same `is_binary_vr` gate before the `to_str()` branch

- [x] DICOM-SCAN-PRIV-RT-R46: Add `test_scan_preserves_private_text_and_bytes_through_write_read_cycle` in `reader.rs`
  - Writes 1-slice series with private LO text tag (0009,0010)="PRIV_ROUND_TRIP_VALUE" and OB bytes tag (0019,1001)=[0xAB,0xCD,0xEF,0x01]
  - Reads back via `scan_dicom_directory`
  - Asserts text in `preservation.object` with exact value; bytes in `preservation.preserved` with exact payload
  - Closes the "private-tag round-trip on the general series reader/writer path" gap

---

## Sprint 45 -- Completed

- [x] DICOM-TS-BUG-R45: Fix transfer_syntax_uid read in `scan_dicom_directory`
  - Was reading Tag(0x0008,0x0070) = Manufacturer into `transfer_syntax_uid`
  - Fixed both per-slice and `first_transfer_syntax_uid` reads to use `obj.meta().transfer_syntax()`
  - Transfer syntax now correctly populated from DICOM file meta table

- [x] DICOM-SEQ-OB-FIX-R45: Fix binary VR preservation in `parse_sequence_item`
  - `to_str()` on VR::OB in dicom-rs 0.8 returns a decimal-formatted string rather than an error
  - Added explicit `is_binary_vr` gate: VR::OB | VR::OW | VR::OD | VR::OF | VR::OL | VR::UN go directly to the bytes branch
  - Non-binary VRs still use `to_str()` with `to_bytes()` fallback
  - Restores Sprint 43 invariant; `test_scan_private_sequence_is_preserved_in_object_model` passes

- [x] DICOM-SPATIAL-RT-R45: Add `test_scan_metadata_round_trip_spatial_fields` in `reader.rs`
  - Writes 3-slice CT series via `write_dicom_series_with_metadata` with origin=[10,20,-50], spacing=[0.8,0.8,2.5], identity direction, modality="CT"
  - Reads back via `scan_dicom_directory` and asserts all DicomReadMetadata fields to ±1e-4
  - Asserts per-slice IOP, pixel_spacing, and IPP z-position for all 3 slices

- [x] DICOM-RESCALE-RT-R45: Add `test_scan_metadata_round_trip_rescale_params` in `reader.rs`
  - Writes 2-slice CT image spanning [-1024, 1024] float intensities
  - Verifies slope > 0 and intercept is finite for all slices
  - Verifies first-voxel quantization error is bounded by slope/2

- [x] DICOM-TS-RT-R45: Add `test_scan_metadata_round_trip_transfer_syntax` in `reader.rs`
  - Writes series with Explicit VR LE transfer syntax
  - Asserts `transfer_syntax_uid == Some("1.2.840.10008.1.2.1")` for every slice
  - Directly validates the bug fix above

- [x] AUDIT-R02B-R45: Close GAP-R02b in gap_audit.md
  - Marked GAP-R02b as Closed (Sprint 45 audit)
  - `InverseConsistentDiffeomorphicDemonsRegistration` and `MultiResDemonsRegistration` confirmed implemented
  - Both exposed in Python as `inverse_consistent_demons_register` and `multires_demons_register`
  - Both included in smoke test required list

## Sprint 43 -- Completed

- [x] DICOM-OBJECT-MODEL-R43: add nested-sequence and preserved-bytes round-trip coverage for `ritk_io::format::dicom::writer_object`
  - Extended `writer_object` tests to exercise `DicomObjectModel` sequence nodes, nested `DicomSequenceItem` content, and raw preserved byte nodes
  - Added direct file round-trip assertions against `dicom::object::open_file` for SQ / OB emission
  - Verifies `model_to_in_mem` emits sequence items, private tags, and byte payloads with value-semantic checks

## Sprint 42 -- Completed

- [x] SMOKE-FILTER-DT-R42: Add `"distance_transform"` to filter smoke required list in test_smoke.py
  - test_python_api_parity.py would fail: distance_transform registered in filter.rs but not in smoke required
  - Added after "resample_image" in test_filter_public_functions_exist required list
  - Closes the parity gap introduced when distance_transform was added in Sprint 41

- [x] SMOKE-SEG-LS-R42: Add `"label_shape_statistics"` to segmentation smoke required list in test_smoke.py
  - Added after "skeletonization" in test_segmentation_public_functions_exist required list
  - Closes the parity gap introduced when label_shape_statistics was added in Sprint 41

- [x] SMOKE-STAT-CLIS-R42: Add `"compute_label_intensity_statistics"` to statistics smoke required list in test_smoke.py
  - Added after "nyul_udupa_normalize" in test_statistics_public_functions_exist required list
  - Closes the parity gap introduced when compute_label_intensity_statistics was added in Sprint 41

- [x] CI-STATS-TEST-R42: Add test_statistics_bindings.py to CI pytest invocation in python_ci.yml
  - Inserted between test_segmentation_bindings.py and test_smoke.py in the pytest run step
  - 4 value-semantic statistics tests now execute on every CI run

- [x] AUDIT-ATLAS-R42: Sync gap_audit.md section 7.2 and 7.3
  - Joint label fusion: ✗ → ✓ ritk.registration.joint_label_fusion_py (Sprint 8)
  - Atlas building: ✗ → ✓ ritk.registration.build_atlas (Sprint 8)
  - Transform I/O: ✗ → ✓ ritk.io.read_transform / write_transform (Sprint 8)
  - Function counts updated: 53+ (Sprint 7) → 91+ (Sprint 41)
  - Stale "remaining work" items replaced with accurate current state

## Sprint 41 -- Completed

- [x] LABEL-STATS-PY-R41: Python binding for compute_label_intensity_statistics
  - `compute_label_intensity_statistics(label_image, intensity_image) -> list[dict]` in statistics.rs
  - Zero-copy via nested `with_tensor_slice` (no clone().into_data())
  - Returns list of dicts: label (int), count (int), min (float), max (float), mean (float), std (float)
  - Background (label 0) excluded; results sorted ascending by label
  - Registered in statistics submodule register()
  - Stub added to statistics.pyi
  - 4 value-semantic tests in test_statistics_bindings.py (NEW FILE):
    - single label 2 voxels: mean=4.0, std=1.0
    - background excluded: empty list
    - two labels sorted ascending
    - four voxels: mean=2.5, std=sqrt(1.25) analytically derived
  - cargo build -p ritk-python clean; 719/719 ritk-core tests pass

- [x] DIST-TRANSFORM-R41: Python binding for distance_transform
  - `distance_transform(image, foreground_threshold=0.5, squared=False)` in filter.rs
  - Maps to DistanceTransform::transform / DistanceTransform::squared from ritk_core::segmentation
  - `use ritk_core::segmentation::DistanceTransform` import added
  - Registered in filter submodule register()
  - Stub added to filter.pyi
  - 3 value-semantic tests appended to test_segmentation_bindings.py:
    - all-foreground → all zeros
    - single-voxel foreground: adjacent voxel = 1.0 (unit spacing), background all > 0
    - squared output = euclidean^2 (element-wise allclose)

- [x] LABEL-SHAPE-R41: Python binding for label_shape_statistics
  - `label_shape_statistics(mask, connectivity=6) -> list[dict]` in segmentation.rs
  - Delegates to ConnectedComponentsFilter::with_connectivity(connectivity).apply(mask)
  - Returns list of dicts: label, voxel_count, centroid ([z,y,x] f64), bounding_box_min/max ([z,y,x] i64)
  - Background excluded; connectivity validated (ValueError for non-6/26)
  - Registered in segmentation submodule register()
  - Stub added to segmentation.pyi
  - 3 value-semantic tests appended to test_segmentation_bindings.py:
    - single voxel: centroid=[2.0,1.0,3.0], bounding_box_min=max=[2,1,3]
    - two components sorted by label
    - invalid connectivity=18 → ValueError

- [ ] PYTHON-CI-VALIDATION: hosted GitHub Actions matrix run (deferred)

## Sprint 40 -- Completed

- [x] ZEROCOPY-ARCH-R40-LIYKOT: from_slice variants for Li/Yen/Kapur/Triangle/MultiOtsu
  - compute_li_threshold_from_slice(slice, num_bins, max_iterations) -> f32
  - compute_yen_threshold_from_slice(slice, num_bins) -> f32
  - compute_kapur_threshold_from_slice(slice, num_bins) -> f32
  - compute_triangle_threshold_from_slice(slice, num_bins) -> f32
  - compute_multi_otsu_thresholds_from_slice(slice, num_classes, num_bins) -> Vec<f32>
  - Each compute_X_threshold_impl delegates to from_slice variant (no duplication)
  - threshold/mod.rs exports all 5 new functions
  - Python bindings for all 5 threshold functions migrated to with_tensor_slice + inline apply
  - FUTURE SPRINT comments removed (fulfilled)
  - 5 parity tests added: bit-identical assert_eq! vs filter-struct compute
  - cargo build -p ritk-core clean; cargo build -p ritk-python clean
  - 83/83 threshold tests pass

- [x] LABEL-STATS-R40: per-label intensity statistics (ITK LabelStatisticsImageFilter equivalent)
  - New file: crates/ritk-core/src/statistics/label_statistics.rs
  - LabelIntensityStatistics { label: u32, count: usize, min: f32, max: f32, mean: f32, std: f32 }
  - compute_label_intensity_statistics<B>(label_image, intensity_image) -> Vec<LabelIntensityStatistics>
  - compute_label_intensity_statistics_from_slices(label_slice, intensity_slice) -> Vec<LabelIntensityStatistics>
  - Single O(N) rayon fold/reduce pass; f64 accumulation for numerical stability
  - statistics/mod.rs updated to export LabelIntensityStatistics and both compute functions
  - 9 tests: single voxel, known stats (n=4: mean=2.5, std=sqrt(1.25)), two labels, background excluded, uniform (std=0), image API parity, length-mismatch panic, shape-mismatch panic, sorted output
  - 740/740 ritk-core tests pass (719 unit + 21 integration); 0 failed

- [ ] PYTHON-CI-VALIDATION: hosted GitHub Actions matrix run (deferred)

## Sprint 39 -- Completed

- [x] PERF-STATS-R39: replace par_sort_unstable_by with 3x select_nth_unstable_by in compute_from_values
  - Phase 2 changed from O(N log N) to O(N) amortized introselect/pdqselect
  - Processing order: i75 -> i50 -> i25 (highest-to-lowest) preserves partition invariant
  - Slice bounds proof: i25 = n/4 <= i50 = n/2 <= i75 = 3n/4 < n (integer floor division)
  - 1 parity test added: test_select_percentiles_match_sort_parity (bit-identical assert_eq! vs full sort, n=1000 LCG)
  - cargo check clean; 705/705 ritk-core tests pass (was 704; +1 parity test)

- [x] PERF-DG-R39: discrete_gaussian conv1d_replicate vectorization + parallel Z-axis + cache-friendly Y-axis
  - conv1d_replicate: fill+SAXPY-per-kernel-position with analytic boundary split (i_start/i_end); interior loop has no branch; LLVM-vectorizable
  - dim=1 Y-axis: reordered from (ix,iy,kj) to (kj,iy,ix) SAXPY-over-rows; contiguous src_row/dst_row slices; eliminates per-Z-slab buf[ny] allocation
  - dim=0 Z-axis: par_chunks_mut(nyx).enumerate() + SAXPY over contiguous src_z slices; parallelizes nz independent output slices; eliminates serial for yx in 0..nyx loop
  - All changes produce bit-identical output (SAXPY terms added in same kj=0..ksz-1 order per output element)
  - 726/726 ritk-core tests pass (705 unit + 21 integration)

- [ ] ZEROCOPY-ARCH-R38-LIYKOT: from_slice variants for Li/Yen/Kapur/Triangle/MultiOtsu (deferred)
- [ ] PYTHON-CI-VALIDATION: hosted GitHub Actions matrix run (deferred)

## Sprint 38 -- Completed

- [x] ZEROCOPY-ARCH-R38-CORE: add zero-copy APIs to ritk-core
  - Image::into_tensor(self) -> Tensor<B,D> and into_parts(self)
  - compute_from_values -> pub in image_statistics.rs
  - compute_otsu_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 in otsu.rs
  - estimate_noise_mad_from_slice and estimate_noise_mad_masked_from_slices in noise_estimation.rs
  - GradientMagnitudeFilter::apply_from_slice in gradient_magnitude.rs
  - 2 parity tests added: test_apply_from_slice_matches_apply, test_compute_otsu_from_slice_matches_filter (bit-identical assertions)
  - cargo check clean; 702/702 ritk-core tests pass

- [x] ZEROCOPY-ARCH-R38-PY: with_tensor_slice + Python binding hot-path migration
  - with_tensor_slice<R,F>(tensor: &Tensor<Backend,3>, f: F) -> R in image.rs
    path: clone tensor (arc refcount+1, O(1)) -> into_primitive() -> TensorPrimitive::Float(NdArrayTensor::F32(arc_array)) -> as_slice_memory_order() -> Option<&[f32]>
  - Updated: image_to_vec, to_numpy, compute_statistics, masked_statistics, estimate_noise, otsu_threshold, gradient_magnitude
  - 56/56 Python SimpleITK parity tests pass
  - Benchmark results (64^3, release): otsu 18.74ms->0.83ms (22.6x); gradient_mag 6.55ms->0.49ms (13.4x); to_numpy ~0.32ms (SITK parity); stats 6.94ms->1.19ms (5.9x)

- [ ] ZEROCOPY-ARCH-R38-LIYKOT: from_slice variants for Li/Yen/Kapur/Triangle/MultiOtsu thresholds (deferred, marked with FUTURE SPRINT comments in segmentation.rs)

- [ ] PERF-DG-R39: discrete_gaussian further optimization (still 3.63x slower than SITK; Z-axis parallelization or tiling strategy)

- [ ] PYTHON-CI-VALIDATION: hosted GitHub Actions matrix run (still deferred)

## Sprint 37 -- Completed

- [x] ZEROCOPY-R37: replace all redundant as_slice().to_vec() patterns with into_vec()
  - affected files: ritk-python/src/image.rs (image_to_vec + to_numpy), gradient_magnitude.rs (extract_vec), 13 other files (binary_threshold, rescale, sigmoid, threshold, windowing, hit_or_miss, label_morphology, top_hat, discrete_gaussian test helper, parity.rs, frangi x5)
  - into_vec::<f32>() consumes TensorData and transmutes Vec<u8> to Vec<f32> via bytemuck; zero-copy when alignment matches (NdArray backend guarantees this)
  - GradientMagnitude: 7.1ms -> 6.55ms measured improvement
  - cargo check clean; 702/702 ritk-core tests pass

- [x] PERF-DG-R37: DiscreteGaussian separable convolution on flat Vec<f32>
  - convolve_separable<const D: usize>: D==3 -> convolve3d_dim, other -> convolve_nd_dim_serial
  - convolve3d_dim: dim-2 (X) rayon par_chunks_mut; dim-1 (Y) rayon Z-slab parallel; dim-0 (Z) serial (safe, no unsafe needed)
  - replaces Burn conv1d (permute + reshape + TensorCat padding + conv1d + reshape + inverse permute) with 3 direct array passes
  - DiscreteGaussian: 13.9ms -> 9.01ms (1.54x speedup)
  - 12/12 DiscreteGaussian unit tests pass; 702/702 ritk-core tests pass
  - 30/30 SimpleITK parity tests pass (including 4 Elastix)

- [ ] ZEROCOPY-ARCH-R38: architectural zero-copy (deferred) -- store raw ndarray in PyImage bypassing Burn tensor abstraction

- [ ] PYTHON-CI-VALIDATION: hosted GitHub Actions matrix run (deferred Sprint 30-37)

## Sprint 34 -- Completed

- [x] PY-STUB-PARSER-FIX-R34: fix correctness bug in parse_top_level_stub_reexports
  - root cause: ASSIGN_PATTERN regex matches `name = value` lines only; __init__.pyi uses `from ... import X as X` (ImportFrom AST nodes) — regex returns empty set causing missing_stub_exports assertion to always fail
  - fix: replaced with ast.parse + ast.ImportFrom walk, consistent with python_api_drift_report.py and parse_top_level_reexports; ASSIGN_PATTERN constant removed; import re retained (WRAP_PATTERN still uses it)
  - verified: python -m py_compile exits 0; parse_top_level_stub_reexports now returns {Image, filter, image, io, registration, segmentation, statistics}; missing_stub_exports = []
- [x] PY-CI-DRIFT-REPORT-R34: wire drift report into CI as always-run diagnostic step
  - new step in python_ci.yml: if: always(), continue-on-error: true, shell: bash
  - positioned after parity/smoke pytest gate and before Rust unit tests
  - prints full per-module and top-level drift summary in CI logs on every run including on test failure
- [x] PY-CI-NUMPY-SEGBINDINGS-R34: add numpy and segmentation bindings tests to CI
  - pip install step: maturin[patchelf] pytest numpy
  - pytest invocation now includes test_segmentation_bindings.py (alphabetical: parity → segmentation_bindings → smoke)
  - segmentation_bindings tests: connected_components (6-conn two-blob count, 26-conn diagonal merge, invalid-connectivity ValueError), chan_vese/geodesic/shape_detection/threshold/laplacian level-set shape + finite + nonzero-variance assertions
- [x] XTASK-PARITY-REPORT-R34: add cargo xtask python-parity-report subcommand
  - PythonParityReport variant added to Commands enum in xtask/src/main.rs
  - python_parity_report handler uses std::process::Command to invoke drift report helper
  - --python flag selects interpreter (default: python)
  - exits non-zero and emits anyhow error on drift; exits 0 with info log on clean
  - cargo check -p xtask: Finished dev profile, zero errors, zero warnings
- [ ] PYTHON-CI-VALIDATION: deferred — hosted GitHub Actions matrix run required

## Sprint 33 -- Completed

- [x] PY-API-PARITY-GUARD: add automated parity test for PyO3 register() exports vs stub files and smoke-test required lists
  - crates/ritk-python/tests/test_python_api_parity.py derives exported names from wrap_pyfunction! registrations in filter.rs, registration.rs, segmentation.rs, statistics.rs, and io.rs
  - asserts every registered function is present in the corresponding stub file and in the required list of the matching smoke test
  - closes the manual-drift class that caused Sprint 31 stub and smoke mismatches
- [x] PY-SMOKE-SURFACE-EXPANSION-R32: extend smoke required lists to full registered Python API surface
  - io: 4 callable exports covered
  - filter: 33 callable exports covered
  - segmentation: 24 callable exports covered
  - registration: 13 callable exports covered
  - statistics: 14 callable exports covered
- [x] PYTHON-CI-HARDENING-R33: run parity and smoke tests against the built wheel artifact
  - replace editable-install workflow path with wheel build plus explicit wheel installation
  - execute crates/ritk-python/tests/test_python_api_parity.py and crates/ritk-python/tests/test_smoke.py in CI
  - uninstall any preexisting ritk package before wheel installation to preserve artifact-under-test correctness
- [x] PY-IO-PARITY-R33: extend parity guard and smoke coverage to the io submodule
  - crates/ritk-python/tests/test_python_api_parity.py now validates io.rs registrations against io.pyi and test_io_public_functions_exist
  - crates/ritk-python/tests/test_smoke.py now asserts read_image, write_image, read_transform, and write_transform are callable
- [x] PY-TOPLEVEL-CONTRACT-R33: guard top-level Python package exports, __all__, and __version__ contract
  - crates/ritk-python/tests/test_python_api_parity.py now validates ritk/__init__.py and ritk/__init__.pyi re-exports, __all__ membership/order, and non-empty __version__
  - crates/ritk-python/tests/test_smoke.py now asserts top-level ritk exports exist and ritk.__all__ matches the stable public contract
- [x] PY-DRIFT-REPORT-R33: add human-readable Python API drift report helper
  - crates/ritk-python/tests/python_api_drift_report.py now prints per-module and top-level drift summaries for Rust registrations, .pyi stubs, smoke-test required lists, __all__, and __version__
  - local execution currently reports clean parity across filter, io, registration, segmentation, statistics, and the top-level ritk package contract
- [ ] PYTHON-CI-VALIDATION: deferred to hosted-runner execution

## Sprint 32 -- Completed

## Sprint 31 -- Completed

- [x] TRACING-REFACTOR-R31: convert all remaining structured-field info!()/warn!() calls to format-string style
  - segment.rs: 22 calls converted (run() dispatch + 21 handler completions); 0 remaining = % lines
  - convert.rs: 2 calls converted (starting + complete); 0 remaining = % lines
  - resample.rs: 1 call converted (starting); 0 remaining = % lines
  - stats.rs: 1 call converted (starting); 0 remaining = % lines
  - Workspace cargo check --workspace --tests: zero errors; 173/173 CLI tests pass
- [x] STUB-SYNC-SEG-R31: add 5 missing functions to segmentation.pyi
  - binary_fill_holes(image: Image) -> Image
  - morphological_gradient(image: Image, radius: int = 1) -> Image
  - confidence_connected_segment(image, seed, initial_lower, initial_upper, multiplier=2.5, max_iterations=15) -> Image
  - neighborhood_connected_segment(image, seed, lower, upper, radius=1) -> Image
  - skeletonization(image: Image) -> Image
  - All 5 were registered in segmentation.rs register() but absent from .pyi; now fully stubbed
- [x] SMOKE-TEST-FIX-R31: correct 10 wrong function names in test_smoke.py
  - ritk.filter: canny -> canny_edge_detect
  - ritk.segmentation: connected_threshold -> connected_threshold_segment; confidence_connected -> confidence_connected_segment; kmeans -> kmeans_segment; watershed -> watershed_segment; chan_vese -> chan_vese_segment; geodesic_active_contour -> geodesic_active_contour_segment
  - ritk.statistics: image_statistics -> compute_statistics; z_score_normalize -> zscore_normalize; min_max_normalize -> minmax_normalize
  - Test assertions now check callable attributes with names that match the actual PyO3-registered function names
## Sprint 30 -- Completed

- [x] TRACING-REFACTOR: convert CLI info!()/warn!() structured-field calls to format-string style
  - `crates/ritk-cli/src/commands/filter.rs`: ~30 info! calls converted; 226 rust-analyzer diagnostics eliminated
  - `crates/ritk-cli/src/commands/register.rs`: ~11 info! calls converted; 86 rust-analyzer diagnostics eliminated
  - `crates/ritk-io/src/format/dicom/reader.rs`: 1 warn! call converted; 8 rust-analyzer diagnostics eliminated
  - `cargo check --workspace --tests`: zero real errors before and after; diagnostics eliminated are IDE-layer false positives only
- [x] STATS-STUB-SYNC-R30: add `nyul_udupa_normalize` stub to `statistics.pyi`
  - `crates/ritk-python/python/ritk/_ritk/statistics.pyi`: added `nyul_udupa_normalize(image, training_images) -> Image` with full docstring; 14/14 registered functions now stubbed
- [x] FILTER-ERROR-MSG-R30: extend filter.rs run() error message to list all 32 dispatched filters
  - `crates/ritk-cli/src/commands/filter.rs`: added grayscale-erosion, grayscale-dilation, white-top-hat, black-top-hat, hit-or-miss, label-dilation, label-erosion, label-opening, label-closing, morphological-reconstruction to the unknown-filter error message
- [x] DISCRETE-GAUSSIAN-ANALYTICAL: impulse response analytical validation
  - `crates/ritk-core/src/filter/discrete_gaussian.rs`: added `test_impulse_response_matches_analytical_gaussian`; verifies impulse response at center of 1×1×31 image matches exp(-k²/(2*4.0))/Z analytically within 1e-3 (f32 tolerance)
## Sprint 29 -- Completed

- [x] PY-DISCRETE-GAUSSIAN: expose `discrete_gaussian` in `ritk-python`
  - `crates/ritk-python/src/filter.rs`: added `discrete_gaussian(image, variance, maximum_error=0.01, use_image_spacing=true)`
  - delegates to `ritk_core::filter::DiscreteGaussianFilter` with variance parameterization, analytic truncation, and spacing-aware sigma conversion
  - registered in Python `filter` submodule export table
- [x] PY-INVERSE-CONSISTENT-DEMONS: expose inverse-consistent diffeomorphic Demons in `ritk-python`
  - `crates/ritk-python/src/registration.rs`: added `inverse_consistent_demons_register(...)`
  - delegates to `InverseConsistentDiffeomorphicDemonsRegistration` with `inverse_consistency_weight` and `n_squarings`
  - returns `(warped_moving, displacement_field)` using the existing packed displacement convention
- [x] PY-STUB-SYNC: synchronize Python type stubs with actual exported API
  - `crates/ritk-python/python/ritk/_ritk/filter.pyi`: added `discrete_gaussian` and synchronized previously exported filter functions missing from stubs
  - `crates/ritk-python/python/ritk/_ritk/registration.pyi`: added `inverse_consistent_demons_register` and `multires_demons_register`
- [x] PY-SMOKE-EXTENSION: extend Python smoke coverage for Sprint 29 surface
  - `crates/ritk-python/tests/test_smoke.py`: added callable-surface assertions for `discrete_gaussian` and `inverse_consistent_demons_register`
- [x] REPO-HOUSEKEEPING: remove stale generated artifact
  - deleted `crates/ritk-core/src/filter/morphology/label_morphology.rs.bak`
- [x] CLI-DISCRETE-GAUSSIAN: `ritk filter --filter discrete-gaussian` wiring and value-semantic CLI tests
  - `crates/ritk-cli/src/commands/filter.rs`: `run_discrete_gaussian` dispatches to `DiscreteGaussianFilter`; `--variance`, `--maximum-error`, `--use-image-spacing` args; 2 value-semantic CLI tests pass
- [x] CLI-INVERSE-CONSISTENT-DEMONS: `ritk register --method ic-demons` wiring and CLI tests
  - `crates/ritk-cli/src/commands/register.rs`: `run_inverse_consistent_demons` dispatches to `InverseConsistentDiffeomorphicDemonsRegistration`; `--inverse-consistency-weight`, `--n-squarings` args; missing-field compile errors fixed; 2 value-semantic CLI tests pass
- [x] NIFTI-SFORM-CI-REGRESSION: NIfTI sform header field regression guard
  - `crates/ritk-io/src/format/nifti/tests.rs`: `test_write_nifti_sets_sform_header_fields` extracted as standalone top-level test (was incorrectly nested); `use nifti::NiftiObject` import added; all 4 NIfTI tests pass
- [x] DICOM-NONIMAGE-INTEGRATION: synthetic DICOM integration tests for non-image SOP filtering
  - `crates/ritk-io/src/format/dicom/reader.rs`: added `write_stub_dicom` helper + 3 value-semantic tests: all-non-image returns error with UIDs; mixed CT+RTSTRUCT retains CT; RT Plan + Waveform error lists both UIDs; 5/5 reader tests pass
- [x] DICOM-MULTIFRAME-DOCS: multi-frame writer constraints and interoperability limits documented
  - `crates/ritk-io/src/format/dicom/multiframe.rs`: module header expanded with reader + writer sections covering SOP class, transfer syntax, pixel depth, global linear rescale constraint, spatial metadata absence, and interoperability limits
## Sprint 28 -- Completed

- [x] NIFTI-SFORM-FIX: persist sform/pixdim in NIfTI writer so spacing round-trips correctly
  - writer.rs: NiftiHeader{sform_code=1, qform_code=0, srow_x/y/z, pixdim[1..3]=spacing, xyzt_units=2}; reference_header chain
  - tests.rs: un-ignored test_read_write_nifti_cycle; origin/spacing within 1e-4; 3 tests pass
- [x] DICOM-MULTIFRAME-WRITE: write multi-frame DICOM objects from 3D Image<B,3>
  - write_dicom_multiframe<B,P> in multiframe.rs; global linear rescale to u16; single OW PixelData element
  - Tags: NumberOfFrames(IS) Rows/Columns(US) BitsAllocated/Stored/HighBit RescaleSlope/Intercept PixelData
  - mod.rs re-exports write_dicom_multiframe; 2 new tests (3x4x5 roundtrip + zero-dim reject)
- [x] VTK-STRUCTGRID-IO: VTK legacy reader/writer for STRUCTURED_GRID and UNSTRUCTURED_GRID datasets
  - struct_grid.rs: read/write_vtk_structured_grid; ASCII; DIMENSIONS/POINTS/SCALARS/VECTORS/NORMALS; 3 tests
  - unstruct_grid.rs: read/write_vtk_unstructured_grid; ASCII+BINARY read; CELLS/CELL_TYPES/attributes; 4 tests
  - vtk/mod.rs: pub mod struct_grid; pub mod unstruct_grid; re-exports added
- [x] DICOM-NONIMAGE-SOP: explicit accept/reject policy for non-image SOP classes
  - sop_class.rs: SopClassKind enum (31 image + 19 non-image + Other); classify_sop_class(); is_image_sop_class()
  - reader.rs: slices.retain() at scan time; tracing::warn per skipped file; bail if all filtered; permissive for unknown UIDs
  - mod.rs: pub mod sop_class; re-exports SopClassKind, classify_sop_class, is_image_sop_class
  - 22 tests: positive (CT/MR/PET/SC/US/NM/SEG), negative (RTSTRUCT/RTPLAN/SR/PR/ECG/PDF), boundary, exhaustive arrays
- [x] ITK-CONFIDENCE-CONNECTED: confidence-connected region growing (iterative mean+-k*sigma)
  - confidence_connected.rs in region_growing/; ConfidenceConnectedFilter; multiplier/max_iterations builder
  - Python: confidence_connected_segment(); CLI: run_confidence_connected(); 9 core tests + CLI tests
- [x] ITK-SKELETONIZATION: topology-preserving thinning via iterative boundary erosion
  - skeletonization.rs: Zhang-Suen 2D; is_simple_3d() 3D topology-preserving thinning
  - Python: skeletonization(); CLI: run_skeletonization(); 19 tests (1D/2D/3D invariants)
- [x] DISCRETE-GAUSSIAN-FILTER: DiscreteGaussianFilter<B> (ITK DiscreteGaussianImageFilter parity)
  - filter/discrete_gaussian.rs: variance-parameterized; maximum_error truncation r=ceil(sqrt(-2sigma^2*ln(e))); use_image_spacing
  - filter/mod.rs: pub mod discrete_gaussian; pub use DiscreteGaussianFilter; 11 tests
- [x] GAP-R02b INVERSE-CONSISTENT-DIFFEOMORPHIC-DEMONS: InverseConsistentDiffeomorphicDemonsRegistration
  - demons/exact_inverse_diffeomorphic.rs: bilateral E=(1-w)||F-M.exp(v)||^2 + w||M-F.exp(-v)||^2
  - Simultaneous phi+=exp(v) + exact phi-=exp(-v); IC residual = mean||phi+(phi-(x))-x||_2
  - demons/mod.rs + lib.rs re-exports; 9 tests (identity MSE, IC residual, MSE reduction, finiteness, w=0 equiv, error paths)
- [x] PY-CI-MATRIX: Python wheel smoke tests across supported Python versions
  - crates/ritk-python/tests/test_smoke.py: 13 tests (import, API surface, __version__, Python>=3.9)
  - .github/workflows/python_ci.yml: matrix [3.9-3.12]x[ubuntu,windows] + ubuntu/3.13; maturin develop + pytest
- [x] Verification: 22+11+9=42 new tests; 0 failed; 251 ritk-io + 701 ritk-core + 193 ritk-registration all passing

## Sprint 26 -- Completed

- [x] VTK-POLYDATA-XML: VTK XML PolyData (.vtp) reader/writer in ritk-io/src/format/vtk/polydata_xml/
  - parse_vtp: find_tag/find_section/attr_val/parse_cells/parse_attrs; no single-quote char literals
  - write_vtp_str: raw string format literals; write_cells + write_attr; correct HashMap/Vec<u32> types
  - 13 tests (triangle parse, empty, scalars, lines, vectors round-trips, error paths)
- [x] VTK-SCENE: VtkScene + VtkActor + RenderProperties in ritk-io/src/domain/vtk_scene.rs
  - RenderProperties::default(): white, opacity=1.0, point_size=2.0, line_width=1.0
  - VtkScene: add_actor, actor_by_name, remove_actor, actor_count; ordered actor list
  - 7 tests (empty, add, find, remove, defaults, ordering)
- [x] ITK-MORPHOLOGY: 3 new filters in ritk-core/src/filter/morphology/
  - HitOrMissTransform: (M erode SE1) AND (Mc erode ring-SE2); ring excludes origin; 5 tests
  - WhiteTopHatFilter: f - opening(f); BTH: closing(f) - f; both clamped >=0; 10 tests
  - LabelDilation: min-label-ID priority expansion into background voxels; 5 tests
  - All exported from filter/morphology/mod.rs and filter/mod.rs
- [x] ITKSNAP-OVERLAY: overlay composition state in ritk-core/src/annotation/overlay.rs
  - ImageOverlay, ContourOverlay, MaskOverlay, OverlayState; Serde derives; 8 tests
  - Colormap enum (Grayscale/Hot/Cool/Jet/Custom); opacity/visibility controls
- [x] ANTS-WORKFLOW: preprocessing pipeline in ritk-registration/src/preprocessing.rs
  - PreprocessingPipeline: Clamp, Masking, IntensityNormalization(ZScore/MinMax), Smoothing, N4BiasCorrection
  - Sequential execution; identity pipeline is identity; 9 tests
- [x] PY-BINDINGS: 4 new Python functions in ritk-python/src/filter.rs
  - white_top_hat, black_top_hat, hit_or_miss, label_dilation; GIL-safe allow_threads
- [x] CLI-BINDINGS: 6 new CLI filter commands in ritk-cli/src/commands/filter.rs
  - grayscale-erosion, grayscale-dilation, white-top-hat, black-top-hat, hit-or-miss, label-dilation
- [x] Verification: cargo check --workspace zero errors; 674 ritk-core + 191 ritk-io + 184 ritk-registration + 160 ritk-cli = 1209 tests, 0 failures

## Sprint 27 -- Completed

- [x] DICOM-MULTIFRAME: MultiFrameInfo + load_dicom_multiframe<B> in ritk-io/src/format/dicom/multiframe.rs; 3 tests
- [x] DICOM-WRITER-GENERAL: DicomObjectWriter (model_to_in_mem + write_object) in writer_object.rs; 5 tests
- [x] DICOM-TRANSFER-SYNTAX: TransferSyntaxKind enum (11 UIDs + Unknown), is_compressed/lossless/supported; 8 tests
- [x] ITK-RESAMPLE: Resample subcommand (ritk-cli) + resample_image Python binding; 5 CLI tests
- [x] PY-PARITY-HARNESS: 10 analytically-derived parity tests in ritk-core/tests/parity.rs; all 10 pass
- [x] VTK-STRUCT-GRID: VtkStructuredGrid + VtkUnstructuredGrid + VtkDataObject new variants; 9 tests
- [x] ITK-MORPHOLOGY-EXTENDED: LabelErosion + LabelOpening + LabelClosing + MorphologicalReconstruction; 21 tests

## Sprint 25 -- Completed

- [x] VTK-DATA-MODEL: VtkDataObject enum + VtkPolyData data type in ritk-io/src/domain/vtk_data_object.rs
  - VtkPolyData: points, vertices, lines, polygons, triangle_strips, point_data, cell_data
  - AttributeArray enum: Scalars, Vectors, Normals, TextureCoords
  - VtkPolyData::validate() enforces index bounds and attribute length invariants
  - VtkDataObject::PolyData wraps VtkPolyData; extensible for StructuredGrid, UnstructuredGrid
  - 8 unit tests (invariant checks, multi-type cell count, attribute validation)
- [x] VTK-PIPELINE: VtkSource, VtkFilter, VtkSink traits + VtkPipeline execution model in ritk-io/src/domain/vtk_pipeline.rs
  - VtkPipeline::new(source) + add_filter + set_sink + execute; composition law verified
  - Send + Sync trait bounds for safe multi-thread use
  - 5 unit tests (source-only, identity filter, sink call count, translate filter, chained cumulative)
- [x] VTK-POLYDATA-READER: VTK legacy ASCII + BINARY POLYDATA format reader in ritk-io/src/format/vtk/polydata/reader.rs
  - parse_polydata(reader) parses POINTS, VERTICES, LINES, POLYGONS, TRIANGLE_STRIPS
  - POINT_DATA and CELL_DATA with SCALARS, VECTORS, NORMALS attribute arrays
  - Big-endian binary encoding for BINARY files (f32/f64 points, i32 connectivity)
  - 8 tests: triangle parse, lines, point_data scalars, cell_data, multiple cell types, vectors/normals, error paths
- [x] VTK-POLYDATA-WRITER: VTK legacy ASCII POLYDATA writer in ritk-io/src/format/vtk/polydata/writer.rs
  - write_vtk_polydata writes all cell types and attribute arrays; omits empty sections
  - total_size = sum of (cell.len() + 1) for all cells in section
  - 7 tests: triangle round-trip, empty polydata, scalars, all cell types, bad-path error, validate after round-trip, vectors
- [x] ITK-INTENSITY-FILTERS: 5 new filters in ritk-core/src/filter/intensity/
  - RescaleIntensityFilter: (I - I_min)/(I_max - I_min) * (out_max - out_min) + out_min; 5 tests
  - IntensityWindowingFilter: clamp to [w_min, w_max] then rescale to [out_min, out_max]; 5 tests
  - ThresholdImageFilter: Below/Above/Outside modes, conditional pixel replacement; 6 tests
  - SigmoidImageFilter: (max-min)/(1+exp(-(I-alpha)/beta))+min; Sethian 1996; 5 tests
  - BinaryThresholdImageFilter: indicator function scaled to {fg, bg}; 5 tests
  - All 5 filters exported from filter/mod.rs; 26 unit tests total
- [x] ITK-INTENSITY-PYTHON: 7 new Python functions in ritk-python/src/filter.rs
  - rescale_intensity, intensity_windowing, threshold_below, threshold_above, threshold_outside, sigmoid_filter, binary_threshold
  - All use py.allow_threads; registered in filter submodule
- [x] ITK-INTENSITY-CLI: 7 new filter methods in ritk-cli/src/commands/filter.rs
  - rescale-intensity, intensity-windowing, threshold-below, threshold-above, threshold-outside, sigmoid, binary-threshold
  - 10 new FilterArgs fields; run_* functions with tracing; 7 integration tests
- [x] ITKSNAP-WORKFLOW: annotation primitives in ritk-core/src/annotation/
  - LabelTable + LabelEntry: CRUD, visibility, next_free_id, Serde; 8 tests
  - LabelMap: ZYX-layout dense label volume, mask_for_label, count_label, present_labels; 8 tests
  - AnnotationState: points, contours (>=2 pts), polylines (>=2 pts), JSON roundtrip; 9 tests
  - UndoRedoStack<S: Clone>: branching undo, push clears future, history non-empty invariant; 10 tests
  - pub mod annotation added to ritk-core/src/lib.rs; 35 tests total
- [x] Verification: cargo check --workspace zero errors; 644 ritk-core + 171 ritk-io + 160 ritk-cli = 975 tests, 0 failures

## Sprint 24 -- Completed

- [x] DICOM-OBJ-MODEL: full element preservation wired in reader.rs
  - Added tag_key, known_handled_tags, parse_sequence_item free functions
  - Full element preservation loop iterates all non-handled elements per slice
  - Private tags, text elements, sequence items, and binary elements captured in slice_meta.preservation
  - Sequence items recursively parsed into DicomValue::Sequence (depth-limited to 8)
  - Binary elements preserved as DicomPreservedElement with raw bytes
- [x] DICOM-WRITER-EMIT: preservation emission wired in write_dicom_series_with_metadata
  - Added str_to_vr, writer_tag_key, writer_exclusion_tags, sequence_item_to_dicom, emit_preservation_nodes helpers
  - emit_preservation_nodes called before PixelData to maintain Image Pixel Module ordering invariant
  - Private text tags, sequence nodes, and raw bytes all emitted from preservation set
- [x] DICOM-ROUND-TRIP-TESTS: 3 new tests in writer.rs
  - test_preservation_private_text_round_trip: private tag (0009,0010) survives write
  - test_preservation_sequence_round_trip: SQ element with nested item survives write; value verified
  - test_preservation_raw_bytes_round_trip: raw OB bytes (0xDE 0xAD 0xBE 0xEF) survive write
- [x] DICOM-SECURITY-TEST-FIX: updated dicom_security.rs to use preservation path instead of stale private_tags HashMap
- [x] Verification: cargo check --workspace zero errors; 147/147 ritk-io tests pass (144 unit + 3 integration)

## Sprint 23 -- Completed
- [x] CLI-REG-BSPLINE-FFD: run_bspline_ffd added to register.rs; BSplineFFDConfig with control_spacing/levels/learning_rate/regularization_weight
- [x] CLI-REG-MULTIRES-SYN: run_multires_syn added to register.rs; MultiResSyNConfig with levels/iterations_per_level/sigma_smooth/cc_radius/inverse_consistency
- [x] CLI-REG-BSPLINE-SYN: run_bspline_syn added to register.rs; BSplineSyNConfig with control_spacing/sigma_smooth/cc_radius/regularization_weight
- [x] CLI-REG-LDDMM: run_lddmm added to register.rs; LddmmConfig with num_time_steps/kernel_sigma/learning_rate
- [x] REGISTER-ARGS-EXT: 7 new RegisterArgs fields; all 12 existing test literals updated with new fields
- [x] CLI-STATS-MSD: run_mean_surface_distance added to stats.rs; mean_surface_distance from ritk_core::statistics; 2 tests
- [x] CLI-STATS-NOISE: run_noise_estimate added to stats.rs; estimate_noise_mad from ritk_core::statistics; 1 test
- [x] PY-STATS-NYUL: nyul_udupa_normalize added to ritk-python/src/statistics.rs; GIL-safe Arc clone; registered in module
- [x] Verification: cargo check --workspace zero errors; cargo nextest run -p ritk-cli 142/142 tests pass

## Sprint 22 — Completed
- [x] STREAM-A: Extract shared numerical helpers from GAC/Chan-Vese into helpers.rs
- [x] STREAM-A: Implement ShapeDetectionSegmentation (Malladi et al.)
- [x] STREAM-A: Implement ThresholdLevelSet (ITK-style)
- [x] STREAM-A: Python bindings for shape_detection_segment and threshold_level_set_segment (registered in segmentation module)
- [x] STREAM-A: CLI bindings for shape-detection and threshold-level-set methods (run_shape_detection, run_threshold_level_set in segment.rs)
- [x] STREAM-A: impl Default for SegmentArgs; fixed duplicate max_iterations field → level_set_max_iterations
- [x] STREAM-A: 9 new CLI tests (3 shape-detection, 6 threshold-level-set)
- [x] STREAM-A: Fixed pre-existing fill-holes CLI test (incorrect cavity geometry → hollow-sphere geometry)
- [x] STREAM-B: (Completed in Sprint 21 - BinaryFillHoles + MorphologicalGradient Python + CLI bindings)
- [x] STREAM-C: DICOM-TAG-READ — Full DICOM tag parsing in scan_dicom_directory (COMPLETED)
- [x] STREAM-C: DICOM-Z-SPACING — Z-spacing from ImagePositionPatient z-coordinates (COMPLETED)
- [x] STREAM-C: DICOM-SORT — Slice sorting by z-position, InstanceNumber, filename (COMPLETED)
- [x] STREAM-C: DICOM-ORIGIN — Origin/direction from first-slice IPP/IOP (COMPLETED)
- [x] STREAM-C: DICOM-META-WRITE — Fixed duplicate bit-depth tags in write_dicom_series_with_metadata (COMPLETED)
- [x] STREAM-C: DICOM-WRITE-REGRESSION — Added binary-level regression asserting BitsAllocated, BitsStored, and HighBit each appear exactly once and precede Pixel Data in written slices
- [x] STREAM-C: PY-WHEEL-SMOKE-LS — Extended Python wheel smoke coverage to install `pytest`, import `ritk`, construct NumPy-backed images, execute `ritk.segmentation.laplacian_level_set_segment`, assert output shape plus finite values, and run `pytest crates/ritk-python/tests -q` against the built wheel
- [x] STREAM-C: PY-SEG-INTEGRATION-TESTS — Added Python integration tests covering connected-components value semantics plus Chan-Vese, Geodesic Active Contour, Shape Detection, Threshold Level Set, and Laplacian level-set shape/finite-value invariants
- [x] STREAM-C: CLI-LAPLACIAN-LS — Added `laplacian-level-set` CLI integration with dispatch, documentation alignment, and tests covering output shape preservation, binary output invariant, and required `--initial-phi` validation
- [x] STREAM-C: REPO-CLEANUP-GENERATED — Removed temporary base64 fragments, patch scripts, generated helper scripts, and transient log/output files from the repository root
- [x] Verification: cargo check --workspace zero errors (131 ritk-cli tests, 0 failed)
- [x] Verification: cargo test -p ritk-cli 131/131 tests pass


## Sprint 21 — Completed
- [x] PY-CLI-MULTIRES: Added `multires_demons_register` Python binding with levels/use_diffeomorphic/n_squarings params
- [x] PY-CLI-MULTIRES: Added `multires-demons` CLI method with `--levels` and `--use-diffeomorphic` args
- [x] PY-CLI-MULTIRES: 2 new CLI tests (output shape, identity MSE < 1e-3 with levels=1)
- [x] SEG-MORPH-EXT: `BinaryFillHoles` (6-connected BFS, O(nz·ny·nx), encloses hollow sphere test)
- [x] SEG-MORPH-EXT: `MorphologicalGradient` (delegates dilation/erosion, boundary extraction verified)
- [x] SEG-MORPH-EXT: mod.rs updated with new module declarations and re-exports
- [x] SEG-MORPH-EXT: 10 unit tests across both operations (positive, boundary, invariant)
- [x] SEG-CLI-BINDINGS: Added `fill-holes` and `morphological-gradient` CLI methods
- [x] SEG-CLI-BINDINGS: Added tests for enclosed-hole filling and boundary extraction
- [x] DICOM-WRITE: `write_dicom_series` produces valid DICOM files (DICM magic at offset 128 verified)
- [x] DICOM-WRITE: Per-slice f32→u16 rescaling with stored rescale slope/intercept tags
- [x] DICOM-WRITE: Reader updated with `open_file` DICOM parse path + raw fallback
- [x] DICOM-WRITE: `write_dicom_series` re-exported from `ritk_io`
- [x] DICOM-WRITE: CLI DICOM write enabled (replaced error with real call)
- [x] Verification: 551 ritk-core + 132 ritk-io + 120 ritk-cli + 175 ritk-registration = 978 tests, 0 failed
- [x] Workspace: `cargo check --workspace` zero errors, zero warnings

---

## Sprint 20 — Completed
- [x] PY-BIND-FLT: Added `curvature_anisotropic_diffusion` Python binding (delegates to `CurvatureAnisotropicDiffusionFilter`, `py.allow_threads`)
- [x] PY-BIND-FLT: Added `sato_line_filter` Python binding (delegates to `SatoLineFilter`, `py.allow_threads`)
- [x] PY-BIND-SEG: Added `confidence_connected_segment` Python binding with seed validation
- [x] PY-BIND-SEG: Added `neighborhood_connected_segment` Python binding with uniform radius
- [x] PY-BIND-SEG: Added `skeletonization` Python binding (D=3 const generic)
- [x] PY-IO-EXT: Extended `read_image` to handle TIFF, VTK, MGH/MGZ, Analyze, JPEG
- [x] PY-IO-EXT: Extended `write_image` to handle TIFF, VTK, MGH/MGZ, Analyze, JPEG
- [x] CLI-FLT-EXT: Added `run_curvature` and `run_sato` functions to `commands/filter.rs`
- [x] CLI-FLT-EXT: Added `--time-step` arg (f64, default 0.0625) to FilterArgs
- [x] CLI-FLT-EXT: Added 2 output-creation tests (`test_filter_curvature_creates_output`, `test_filter_sato_creates_output`)
- [x] CLI-SEG-EXT: Added `run_confidence_connected`, `run_neighborhood_connected`, `run_skeletonization` to `commands/segment.rs`
- [x] CLI-SEG-EXT: Added `--multiplier` (f32, 2.5), `--max-iterations` (usize, 15), `--neighborhood-radius` (usize, 1) to SegmentArgs
- [x] CLI-SEG-EXT: Added 11 tests (positive, binary-invariant, boundary-missing-args)
- [x] GAP-R02b: Created `MultiResDemonsRegistration` with `MultiResDemonsConfig` (levels, use_diffeomorphic, n_squarings)
- [x] GAP-R02b: Coarse-to-fine pyramid: Gaussian downsampling + strided subsampling, trilinear upsampling with scale correction
- [x] GAP-R02b: Warm-start: upsampled coarser displacement injected as additive init at each finer level
- [x] GAP-R02b: Thirion and Diffeomorphic variants both supported
- [x] Verification: 541 ritk-core + 132 ritk-io + 118 ritk-cli + 175 ritk-registration = 966 tests, 0 failed
- [x] Workspace: `cargo check --workspace` zero errors, zero warnings

---

## Sprint 19 — Completed
- [x] PERF-R02: Added `compute_gradient_into` (zero-alloc, writes directly into caller slices; `compute_gradient` now delegates)
- [x] PERF-R02: Added `warp_image_into` (writes into caller output buffer; `warp_image` now delegates)
- [x] PERF-R02: Added `compute_mse_streaming` (streaming MSE without warped buffer; promotes to SSOT)
- [x] PERF-R02: Removed duplicate `warp_image_into` from `thirion.rs` (now imported from `deformable_field_ops`)
- [x] PERF-R02: Removed duplicate `warp_image_into`, `compute_gradient_into`, `compute_mse_streaming` from `symmetric.rs`
- [x] PERF-R02: Simplified `compute_mse_direct` in `diffeomorphic.rs` to delegate to `compute_mse_streaming`
- [x] PERF-R03: Replaced `convolve_axis` (inner-loop match) with `convolve_z`, `convolve_y`, `convolve_x` (branch-free)
- [x] Verification: 170 unit tests + 3 integration tests — 0 failed; `cargo check -p ritk-registration` zero errors

---

## Sprint 18 — Completed
- [x] PERF-DEM-01: Reduced clone-heavy scaling-and-squaring memory traffic in deformable field exponentiation
- [x] PERF-DEM-02: Streamed Demons MSE evaluation to avoid full warped-image allocation in MSE-only paths
- [x] PERF-DEM-03: Reused iteration buffers in Thirion and Symmetric Demons registration loops
- [x] PERF-DEM-04: Fused inverse displacement vector-field warping to reduce temporary allocations and repeated coordinate traversal
- [x] Verification: `cargo check -p ritk-registration` passed after optimization changes; residual warnings only

---

## Sprint 13 — Completed
- [x] IO-09: DICOM read metadata slice (series-level capture, per-slice geometry, read-only API)
- [x] IO-05: MINC2 format reader (consus-hdf5 HDF5 parsing, dimension metadata, spatial derivation, datatype conversion)
- [x] IO-05: MINC2 format writer (low-level HDF5 binary construction, contiguous f32 LE storage)
- [x] consus integration: added consus-hdf5, consus-core, consus-io, consus-compression workspace dependencies
- [x] Module wiring: minc format module registered in format/mod.rs and re-exported in lib.rs
- [x] Workspace compilation: zero errors, zero warnings
- [x] Unit tests: 27 MINC-specific tests passing (reader: data type conversion, spatial metadata, dimorder, attribute extraction; writer: validation)

---

## Sprint 11 — Completed
- [x] FLT-CAD: Curvature Anisotropic Diffusion (Alvarez et al. 1992, mean curvature motion of level sets, explicit Euler)
- [x] FLT-SATO: Sato Line Filter (Sato 1998, multi-scale Hessian curvilinear structure detection)
- [x] Hessian computation module (3-D physical-space Hessian, Cardano eigenvalue solver)
- [x] Module wiring: diffusion/curvature, vesselness/sato+hessian re-exported through filter root
- [x] Workspace compilation: zero errors
- [x] Unit tests: ritk-core 519 (unit) + 11 (integration) = 530 passing

---

## Sprint 10 — Completed
- [x] SEG-CC: Confidence Connected Region Growing (Yanowitz/Bruckstein adaptive statistics)
- [x] SEG-NC: Neighborhood Connected Region Growing (neighborhood admissibility predicate)
- [x] SEG-SK: Skeletonization (topology-preserving thinning; Zhang–Suen 2-D, directional sequential 3-D)

---

## Sprint 4 — Completed
- [x] FLT-04: RecursiveGaussianFilter (Deriche IIR, derivative orders 0/1/2)
- [x] FLT-08: CannyEdgeDetector (Gaussian + gradient + NMS + hysteresis)
- [x] FLT-10: LaplacianOfGaussianFilter (separable Gaussian + Laplacian)
- [x] FLT-09: Grayscale morphological filters (erosion, dilation)
- [x] SEG-THR: Additional thresholds (Li, Yen, Kapur, Triangle)
- [x] SEG-08: K-Means clustering segmentation (k-means++ init, Lloyd's iteration)
- [x] SEG-07: Watershed segmentation (Meyer flooding)
- [x] STA-04: Nyúl-Udupa histogram standardization (train/apply workflow)
- [x] STA-06: MAD noise estimation (unmasked + masked)
- [x] STA-08: PSNR and SSIM comparison metrics
- [x] GAP-R07: BSpline FFD registration pipeline (multi-resolution, bending energy)
- [x] Module root updates (filter, edge, segmentation, statistics, registration)
- [x] Workspace compilation: zero errors, zero warnings
- [x] Unit tests: 390 (ritk-core) + 121 (ritk-registration) = 511+ passing

## Sprint 5 — Completed
- [x] SEG-06: Level set segmentation (GAC, Chan-Vese)
- [x] FLT-09: Sobel gradient filter
- [x] PY-06: Extended Python segmentation API (16 functions)
- [x] PY-05 partial: Extended Python filter API (6 new functions, 14 total)
- [x] FLT-03/FLT-04: Native median/bilateral confirmed (already existed in core)
- Verification: workspace compiles (zero errors/warnings), 671+ tests passing

## Sprint 6 — Completed
- [x] GAP-R01: Multi-Resolution SyN (coarse-to-fine pyramid, inverse consistency enforcement)
- [x] GAP-R01b: BSplineSyN (B-spline parameterized velocity fields, bending energy regularization)
- [x] GAP-R03: LDDMM registration (geodesic shooting via EPDiff, Gaussian RKHS kernel)
- [x] GAP-R05: Composite transform serialization (JSON, TransformDescription enum, round-trip file I/O)
- [x] IO-07: TIFF/BigTIFF reader/writer (multi-page z-stack, u8/u16/u32/f32/f64 support)
- [x] PY-05: Python registration API completion (BSpline FFD, Multi-Res SyN, BSpline SyN, LDDMM — 8 total registration functions)
- [x] Module root updates (diffeomorphic, lddmm, transform, io/format, python/registration)
- [x] Workspace compilation: zero errors, zero warnings
- [x] Unit tests: ritk-core 421, ritk-registration 150+, ritk-io 50+ passing

## Sprint 7 — Completed
- [x] GAP-R04: Groupwise/Atlas Registration (iterative template building via SyN)
- [x] GAP-R06: Joint Label Fusion (Wang 2013) + Majority Voting
- [x] IO-MGH: MGZ/MGH Format Reader/Writer (FreeSurfer, gzip support)
- [x] SEG-DT: Euclidean Distance Transform (Meijster 2000, O(N) linear time)
- [x] STA-09: White Stripe Normalization (Shinohara 2014, KDE-based peak detection)
- [x] PY-STAT: Python Statistics/Normalization/Comparison API (13 functions)
- [x] Module wiring: atlas, distance_transform, white_stripe, mgh, statistics across all crates
- [x] Workspace compilation: zero errors, zero warnings
- [x] Unit tests: ritk-core 454, ritk-registration 162, ritk-io 79 passing

## Sprint 8 — Completed
- [x] IO-06: VTK image format reader/writer (legacy structured points, ASCII/BINARY, big-endian binary handling)
- [x] IO-08: JPEG 2D support (grayscale read/write, represented as shape `[1, height, width]`, write rejects `nz != 1`)
- [x] PY-07: CLI tooling completion
- [x] CLI filter coverage: median, bilateral, canny, sobel, log, recursive-gaussian
- [x] CLI segmentation coverage: li, yen, kapur, triangle, watershed, kmeans, distance-transform
- [x] CLI registration coverage: demons and syn
- [x] CLI stats subcommand: summary, dice, hausdorff, psnr, ssim
- [x] PY-08: Type stubs / py.typed packaged under `crates/ritk-python/python/ritk/`
- [x] Python registration additions: atlas building, majority-vote fusion, joint label fusion
- [x] Python IO additions: composite transform JSON read/write
- [x] Workspace verification: `cargo test --workspace` reached passing `ritk-cli` suite (107 tests) before timeout; prior attached report records full workspace pass (864 tests, 0 failed)
- [x] Bounded verification note: current local reruns hit build/cache lock and timeout during full-workspace execution; no failing diagnostics were observed in captured output

## Remaining Backlog
- [x] IO-05: MINC format reader/writer (Sprint 12 — consus pure-Rust HDF5 integration)
- [x] GAP-R02b: Diffeomorphic Demons exact inverse
- [x] FLT: Curvature anisotropic diffusion (Alvarez et al. 1992 mean curvature motion)
- [x] FLT: Sato line filter (Sato 1998 multi-scale Hessian line detection)
- [x] IO-07b: Analyze format reader/writer
- [x] SEG: Confidence connected region growing (Sprint 10)
- [x] SEG: Neighborhood connected region growing (Sprint 10)
- [x] SEG: Skeletonization (Sprint 10)
- [x] CI: nextest, clippy, fmt enforcement
- [x] CI: dependency version alignment checks
- [x] Python: wrap long-running PyO3 calls with `py.allow_threads`
- [x] CI: add maturin build + `import ritk` smoke test

## Verification Policy
- All tests must assert computed VALUES, not just Result/Option variants
- cargo check --workspace --tests: zero errors required before commit
- Targeted test runs for impacted crates before push
- Use bounded verification first; if a full-workspace run times out, record the completed subset and the blocking condition