## Sprint 28 -- Planned

- [ ] NIFTI-SFORM-FIX: persist sform/pixdim in NIfTI writer so spacing round-trips correctly
- [ ] DICOM-MULTIFRAME-WRITE: write multi-frame DICOM objects from 3D Image<B,3>
- [ ] DICOM-NONIMAGE-SOP: define explicit accept/reject policy for non-image SOP classes
- [ ] VTK-STRUCTGRID-IO: VTK legacy reader/writer for STRUCTURED_GRID and UNSTRUCTURED_GRID datasets
- [ ] ITK-CONFIDENCE-CONNECTED: confidence-connected region growing (iterative mean+-k*sigma)
- [ ] ITK-SKELETONIZATION: topology-preserving thinning via iterative boundary erosion
- [ ] PY-CI-MATRIX: Python wheel smoke tests across supported Python versions

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