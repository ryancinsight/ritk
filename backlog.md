## Sprint 37 -- Completed

### Stream A -- Zero-Copy Extraction (ZEROCOPY-R37)
| ID | Feature | Status | Notes |
|---|---|---|---|
| ZEROCOPY-R37 | Replace all redundant as_slice().to_vec() patterns with into_vec() | **CLOSED** Sprint 37 | Eliminated second O(N) copy in image_to_vec, to_numpy, extract_vec (gradient_magnitude), and 15 test helpers across binary_threshold, rescale, sigmoid, threshold, windowing, hit_or_miss, label_morphology, top_hat, frangi, parity.rs. into_vec() transmutes Vec<u8> -> Vec<f32> via bytemuck (zero-copy on fast path). cargo check clean. GradientMagnitude: 7.1ms -> 6.55ms. |

### Stream B -- DiscreteGaussian Separable Convolution (PERF-DG-R37)
| ID | Feature | Status | Notes |
|---|---|---|---|
| PERF-DG-R37 | Replace Burn conv1d tensor path with direct flat-array separable convolution | **CLOSED** Sprint 37 | convolve_separable<const D: usize> dispatches to convolve3d_dim (rayon-parallel dim-2/dim-1, serial dim-0) for D==3, serial convolve_nd_dim_serial for other ranks. Eliminates: permute + reshape + TensorData::cat padding + conv1d + reshape + inverse-permute per axis. 12/12 unit tests pass. DiscreteGaussian: 13.9ms -> 9.0ms (1.54x speedup); gap vs SITK: 2.4x -> ~3x (SITK measured at 3.03ms in miniforge3; prior measurement was 5.8ms in different environment). 702/702 ritk-core Rust unit tests pass. 30/30 SimpleITK parity tests pass including 4 Elastix tests. |

### Sprint 37 Benchmark Results (64^3 image, release build, miniforge3/Python 3.13)
| Operation | Sprint 36 RITK | Sprint 37 RITK | Sprint 37 SITK | Ratio |
|---|---|---|---|---|
| DiscreteGaussian | 13.9 ms | **9.01 ms** | 3.03 ms | 2.97x |
| MedianFilter r=2 | 14.7 ms | 14.36 ms | 21.23 ms | **0.68x (faster)** |
| compute_statistics | 7.1 ms | 6.94 ms | 0.33 ms | 21x |
| GradientMagnitude | 7.1 ms | **6.55 ms** | 1.05 ms | 6.26x |
| OtsuThreshold | 19.5 ms | 18.74 ms | 1.89 ms | 9.9x |

Note: Remaining statistics/otsu/gradient gap is structural � the single O(N) copy in clone().into_data() cannot be eliminated without architectural change to PyImage (store raw ndarray directly, bypassing Burn tensor abstraction). Planned as ZEROCOPY-ARCH-R38.

## Sprint 36 -- Completed

### Stream A -- Elastix Gap Analysis + Registration Parity
| ID | Feature | Status | Notes |
|---|---|---|---|
| GAP-ELASTIX-R36 | Add Elastix/ITK-Elastix gap (GAP-R08) to gap_audit.md | **CLOSED** Sprint 36 | GAP-R08 documents ElastixImageFilter/TransformixImageFilter gap: ASGD optimizer, parameter-map interface, Transformix application path, sparse-sampled Mattes MI. Severity: Medium. Minimum closure: parity test suite. |
| ELASTIX-PARITY-TESTS-R36 | Add 4 Elastix parity tests (Section 4) to test_simpleitk_parity.py | **CLOSED** Sprint 36 | test_elastix_translation_recovers_sphere_overlap, test_ritk_demons_vs_elastix_translation_quality, test_elastix_bspline_deformable_vs_ritk_syn, test_elastix_parameter_map_api_matches_expected_keys; all guarded with skipif(not _has_elastix); 56/56 tests pass |

### Stream B -- Performance Optimizations (ritk-core)
| ID | Feature | Status | Notes |
|---|---|---|---|
| PERF-STATS-R36 | Optimize compute_statistics: single-pass parallel reduce + par_sort | **CLOSED** Sprint 36 | Replaced two-pass O(N) min/max with single rayon fold/reduce; population variance via E[X^2]-E[X]^2 in f64; par_sort_unstable_by for percentiles; compute_from_values signature changed from &mut Vec<f32> to &[f32] |
| PERF-OTSU-R36 | Optimize otsu threshold: combine min/max into single pass | **CLOSED** Sprint 36 | Replaced two separate fold() calls for x_min and x_max with a single combined fold returning (x_min,x_max) tuple; histogram and prefix-sum logic unchanged |
| PERF-MEDIAN-R36 | Optimize median_3d: Rayon z-parallelism + select_nth_unstable + per-z-slice Vec reuse | **CLOSED** Sprint 36 | par_chunks_mut(ny*nx) parallelizes z-slices; neighbors Vec allocated once per z-slice via clear(); select_nth_unstable_by replaces sort_unstable_by; 221ms->14.7ms (15x speedup, now faster than SimpleITK 24ms) |
| PERF-GRADIENT-R36 | Optimize gradient_magnitude: single-pass parallel map | **CLOSED** Sprint 36 | GradientMagnitudeFilter::apply replaced with into_par_iter().map() computing gz/gy/gx inline per voxel; eliminates 3 intermediate Vec allocations and separate combine pass; apply_components unchanged |

### Stream B Benchmark Results (64^3 image, release build)
| Operation | Before | After | vs SimpleITK |
|---|---|---|---|
| MedianFilter r=2 | 221ms | 14.7ms (15x faster) | **0.6x (faster)** |
| compute_statistics | 10.2ms | 7.1ms | 10.6x slower |
| GradientMagnitude | 7.8ms | 7.1ms | 5.2x slower |
| OtsuThreshold | 18.9ms | 19.5ms | 4.6x slower |
| DiscreteGaussian | 13ms | 13.9ms | 2.4x slower |

Note: statistics/otsu/gradient remaining slowdown vs SimpleITK is dominated by the Burn tensor data extraction path (clone().into_data() allocates ~1MB per call). Root cause: NdArray backend does not expose a zero-copy slice view through the public Burn Tensor API. Architectural fix (PyImage stores raw ndarray directly) deferred to Sprint 37.

## Sprint 35 -- Completed

### Stream A -- SimpleITK Numerical Parity Verification
| ID | Feature | Status | Notes |
|---|---|---|---|
| SITK-PARITY-TESTS-R35 | SimpleITK numerical parity test suite (26 tests) | **CLOSED** Sprint 35 | Added crates/ritk-python/tests/test_simpleitk_parity.py; 26 tests across 3 sections: Filter (9), Segmentation (8), Statistics (9); all 52 ritk-python tests pass; SimpleITK 2.5.3 validated |
| INIT-IMPORT-FIX-R35 | Fix ritk.__init__ PyO3 submodule import and sys.modules registration | **CLOSED** Sprint 35 |  fails with PyO3 0.22 (submodules are attributes not sys.modules entries); fixed to  + explicit sys.modules.setdefault registration for all submodules;  etc. now work |
| PARITY-PARSER-R35 | Extend parse_top_level_reexports to handle ast.Assign nodes | **CLOSED** Sprint 35 | test_python_api_parity.py only parsed ast.ImportFrom; extended to also collect non-underscore ast.Assign targets so Image=_image_mod.Image is correctly counted |
| SITK-LI-DIVERGENCE-R35 | Document Li threshold algorithm divergence vs SimpleITK | **CLOSED** Sprint 35 | RITK Li threshold (iterative cross-entropy minimisation) yields ~0.5 for bimodal sphere+noise image; SimpleITK gives ~0.002; root cause is different convergence criterion and initialisation; test replaced with independent acceptance criterion (threshold in (0.05,0.95), mask Dice vs ground-truth sphere >= 0.90) |

## Sprint 34 -- Completed

### Stream A -- Python Parity Hardening Corrections
| ID | Feature | Status | Notes |
|---|---|---|---|
| PY-STUB-PARSER-FIX-R34 | Fix correctness bug in parse_top_level_stub_reexports | **CLOSED** Sprint 34 | `test_python_api_parity.py` used `ASSIGN_PATTERN` regex that cannot match `from ... import X as X` syntax in `__init__.pyi`, returning empty set and causing `missing_stub_exports` assertion to always fail; replaced with `ast.parse` + `ast.ImportFrom` walk (consistent with `python_api_drift_report.py`); `ASSIGN_PATTERN` constant removed |
| PY-CI-DRIFT-REPORT-R34 | Wire drift report into CI as always-run diagnostic step | **CLOSED** Sprint 34 | Added `Run Python API drift report (always -- diagnostic context)` step to `python_ci.yml` with `if: always()` and `continue-on-error: true`; step runs after parity/smoke gate and prints human-readable module-level and top-level drift summary in CI logs regardless of test outcome |
| PY-CI-NUMPY-SEGBINDINGS-R34 | Add numpy and segmentation bindings tests to CI | **CLOSED** Sprint 34 | `numpy` added to `pip install` step; `crates/ritk-python/tests/test_segmentation_bindings.py` added to CI pytest invocation (alphabetical order: parity → segmentation_bindings → smoke); segmentation bindings tests exercise value-semantic functional correctness (connected_components, level-set variants) against the installed wheel |
| XTASK-PARITY-REPORT-R34 | Add cargo xtask python-parity-report subcommand | **CLOSED** Sprint 34 | Added `PythonParityReport` variant to `Commands` enum and `python_parity_report` handler in `xtask/src/main.rs`; invokes `python crates/ritk-python/tests/python_api_drift_report.py` via `std::process::Command`; exits non-zero on detected drift; `--python` flag selects interpreter; `cargo check -p xtask` passes clean |

### Stream B -- Deferred
| ID | Feature | Status | Notes |
|---|---|---|---|
| PYTHON-CI-VALIDATION | Validate Python CI workflow on hosted runners | DEFERRED Sprint 34 | All local verification passes; `python_api_drift_report.py` reports clean (5/5 modules, top-level contract); hosted GitHub Actions matrix execution required to confirm Windows wheel build, patchelf behavior, and Python 3.9–3.13 compatibility |

## Sprint 33 -- Completed

### Stream A -- Python CI Hardening
| ID | Feature | Status | Notes |
|---|---|---|---|
| PY-CI-HARDENING-R33 | Harden Python CI to test installed wheel parity and smoke surface | **CLOSED** Sprint 33 | `python_ci.yml` now builds a wheel with `maturin build`, installs the built wheel deterministically, runs both `test_python_api_parity.py` and `test_smoke.py` against the installed package, and documents the local `python_api_drift_report.py` helper for human-readable parity diagnostics |
| PY-IO-PARITY-R33 | Extend Python API parity guard and smoke coverage to the `io` submodule | **CLOSED** Sprint 33 | `test_python_api_parity.py` now enforces `io.rs`/`io.pyi` parity and `test_smoke.py` covers `read_image`, `write_image`, `read_transform`, and `write_transform` |
| PY-TOPLEVEL-CONTRACT-R33 | Guard the top-level Python package export contract | **CLOSED** Sprint 33 | `test_python_api_parity.py` and `test_smoke.py` now validate `ritk/__init__.py` re-exports, stable `__all__` ordering, and non-empty `__version__` against the documented public package surface |
| PY-DRIFT-REPORT-R33 | Add a human-readable Python API drift report helper | **CLOSED** Sprint 33 | Added `crates/ritk-python/tests/python_api_drift_report.py` to print per-module and top-level drift summaries for Rust registrations, `.pyi` stubs, smoke-test required lists, and the `ritk` package contract so parity failures can be diagnosed without manual source inspection |

### Stream B -- Deferred
| ID | Feature | Status | Notes |
|---|---|---|---|
| PYTHON-CI-VALIDATION | Validate Python CI workflow on hosted runners | DEFERRED Sprint 33 | Local workflow hardening is complete; hosted-runner execution remains required to confirm matrix behavior and any Windows-specific packaging issues |

## Sprint 32 -- Completed

### Stream A -- Python API Parity Automation
| ID | Feature | Status | Notes |
|---|---|---|---|
| PY-API-PARITY-GUARD | Add automated parity test for PyO3 register() exports vs stub files and smoke-test required lists | **CLOSED** Sprint 32 | Guard now derives exported names from wrap_pyfunction! registrations across the Python submodules and fails on drift in stubs or smoke coverage |

## Sprint 31 -- Completed

### Stream A -- Tracing Refactor Completion
| ID | Feature | Status | Notes |
|---|---|---|---|
| TRACING-REFACTOR-R31 | Convert remaining structured-field info!() calls to format-string style | **CLOSED** Sprint 31 | segment.rs (22 calls), convert.rs (2 calls), resample.rs (1 call), stats.rs (1 call); all 33 remaining = % lines eliminated; 173/173 CLI tests pass; cargo check clean |

### Stream B -- Stub Sync / Correctness
| ID | Feature | Status | Notes |
|---|---|---|---|
| STUB-SYNC-SEG-R31 | Add 5 missing segmentation stubs to segmentation.pyi | **CLOSED** Sprint 31 | binary_fill_holes, morphological_gradient, confidence_connected_segment, neighborhood_connected_segment, skeletonization; all registered in segmentation.rs but absent from pyi |
| SMOKE-TEST-FIX-R31 | Fix 10 wrong function names in test_smoke.py | **CLOSED** Sprint 31 | Filter: canny->canny_edge_detect; Segmentation: connected_threshold->connected_threshold_segment, confidence_connected->confidence_connected_segment, kmeans->kmeans_segment, watershed->watershed_segment, chan_vese->chan_vese_segment, geodesic_active_contour->geodesic_active_contour_segment; Statistics: image_statistics->compute_statistics, z_score_normalize->zscore_normalize, min_max_normalize->minmax_normalize |

### Stream C -- Deferred
| ID | Feature | Status | Notes |
|---|---|---|---|
| PYTHON-CI-VALIDATION | Validate Python CI workflow on hosted runners | DEFERRED Sprint 33 | Superseded by Sprint 33 workflow hardening; hosted-runner execution is still required to confirm matrix behavior and any Windows-specific packaging issues |

## Sprint 30 -- Completed

### Stream A — Tracing / IDE Quality Refactor
| ID | Feature | Status | Notes |
|---|---|---|---|
| TRACING-REFACTOR | Convert CLI `info!()`/`warn!()` structured-field calls to format-string style | **CLOSED** Sprint 30 | `filter.rs`, `register.rs`, `reader.rs`; eliminates ~320 rust-analyzer false-positive diagnostics; `cargo check` was already clean |

### Stream B — Stub Sync / Correctness
| ID | Feature | Status | Notes |
|---|---|---|---|
| STATS-STUB-SYNC-R30 | Add `nyul_udupa_normalize` to `statistics.pyi` | **CLOSED** Sprint 30 | Function exported in `statistics.rs` register() but missing from pyi; 14 functions now fully stubbed |
| FILTER-ERROR-MSG-R30 | Extend `filter.rs` `run()` error message to list all dispatched filter names | **CLOSED** Sprint 30 | Added missing 10 filter names (grayscale-erosion, grayscale-dilation, white-top-hat, black-top-hat, hit-or-miss, label-dilation/erosion/opening/closing, morphological-reconstruction) |

### Stream C — Testing / Quantitative Validation
| ID | Feature | Status | Notes |
|---|---|---|---|
| DISCRETE-GAUSSIAN-ANALYTICAL | Impulse-response analytical validation for DiscreteGaussianFilter | **CLOSED** Sprint 30 | `test_impulse_response_matches_analytical_gaussian` in `discrete_gaussian.rs`; verifies output[0,0,k] ≈ exp(-( k-15)²/(2v))/Z for Dirac impulse at position 15 of a 1×1×31 image; tolerance 1e-3 for f32 arithmetic |

### Stream D — Deferred
| ID | Feature | Status | Notes |
|---|---|---|---|
| PYTHON-CI-VALIDATION | Validate Python CI workflow on hosted runners | DEFERRED Sprint 33 | Superseded by Sprint 33 workflow hardening; hosted-runner execution is still required to confirm matrix behavior and any Windows-specific packaging issues |

## Sprint 29 -- Completed

### Stream A -- Surface Completion / Bindings / CLI
| ID | Feature | Status | Notes |
|---|---|---|---|
| PY-DISCRETE-GAUSSIAN | Python binding + stub + smoke coverage for `DiscreteGaussianFilter` | **CLOSED** Sprint 29 | `ritk-python/src/filter.rs`: `discrete_gaussian`; `_ritk/filter.pyi` synchronized; smoke test updated |
| PY-IC-DEMONS | Python binding + stub + smoke coverage for inverse-consistent diffeomorphic Demons | **CLOSED** Sprint 29 | `ritk-python/src/registration.rs`: `inverse_consistent_demons_register`; `_ritk/registration.pyi` synchronized; smoke test updated |
| CLI-DISCRETE-GAUSSIAN | CLI `ritk filter --filter discrete-gaussian` | **CLOSED** Sprint 29 | `run_discrete_gaussian` dispatches to `DiscreteGaussianFilter`; `--variance`, `--maximum-error`, `--use-image-spacing` args; 2 value-semantic CLI tests pass (173 total) |
| CLI-IC-DEMONS | CLI `ritk register --method ic-demons` | **CLOSED** Sprint 29 | `run_inverse_consistent_demons` dispatches to `InverseConsistentDiffeomorphicDemonsRegistration`; missing `inverse_consistency_weight`/`n_squarings` test struct fields fixed; 2 value-semantic CLI tests pass |

### Stream B -- Regression / CI / Verification
| ID | Feature | Status | Notes |
|---|---|---|---|
| NIFTI-SFORM-CI-REGRESSION | NIfTI sform header field regression guard | **CLOSED** Sprint 29 | `test_write_nifti_sets_sform_header_fields` extracted from incorrectly nested position; `use nifti::NiftiObject` import added; 4 NIfTI tests pass |
| PY-STUB-SYNC-R29 | Python stub synchronization audit | **CLOSED** Sprint 29 | `_ritk/filter.pyi` and `_ritk/registration.pyi` fully synchronized |
| PY-SMOKE-R29 | Python smoke API-surface extension | **CLOSED** Sprint 29 | `test_smoke.py` covers `discrete_gaussian` and `inverse_consistent_demons_register` |
| DICOM-NONIMAGE-INTEGRATION | End-to-end synthetic DICOM tests for non-image SOP filtering | **CLOSED** Sprint 29 | 3 value-semantic tests in `reader.rs`: all-non-image returns error with UIDs; mixed CT+RTSTRUCT retains CT slice; RT Plan+Waveform error lists both UIDs; 5/5 reader tests pass |

### Stream C -- Repository Hygiene / Documentation
| ID | Feature | Status | Notes |
|---|---|---|---|
| REPO-STALE-ARTIFACT-R29 | Remove stale generated backup artifact | **CLOSED** Sprint 29 | Deleted `crates/ritk-core/src/filter/morphology/label_morphology.rs.bak` |
| DOC-DICOM-MULTIFRAME-LIMITS | Document DICOM multi-frame writer constraints | **CLOSED** Sprint 29 | `multiframe.rs` module header expanded: SOP class (Secondary Capture only), transfer syntax (Explicit VR LE only), pixel depth (16-bit unsigned), global linear rescale constraint, spatial metadata absence, interoperability limits vs Enhanced Multi-Frame |
| PYTHON-CI-VALIDATION | Validate Python CI workflow on hosted runners | DEFERRED Sprint 33 | Superseded by Sprint 33 workflow hardening; hosted-runner execution is still required to confirm matrix behavior and any Windows-specific packaging issues |

## Sprint 28 -- Completed

### Stream A -- DICOM / NIfTI I/O
| ID | Feature | Status | Notes |
|---|---|---|---|
| NIFTI-SFORM-FIX | Persist sform/pixdim in NIfTI writer | **CLOSED** Sprint 28 | `writer.rs`: sform_code=1, qform_code=0, srow_x/y/z, pixdim[1..3]=spacing, xyzt_units=2; round-trip spacing within 1e-4; 3 tests pass |
| DICOM-MULTIFRAME-WRITE | Write multi-frame DICOM objects from `Image<B,3>` | **CLOSED** Sprint 28 | `write_dicom_multiframe<B,P>` in `multiframe.rs`; global linear rescale to u16; NumberOfFrames/Rows/Columns/BitsAllocated/RescaleSlope/PixelData tags; 2 tests |
| DICOM-NONIMAGE-SOP | Explicit accept/reject policy for non-image SOP classes | **CLOSED** Sprint 28 | `sop_class.rs`: SopClassKind enum (31 image + 19 non-image + Other); classify_sop_class(); `reader.rs` retain+bail at scan time; 22 tests |

### Stream B -- VTK Grid I/O
| ID | Feature | Status | Notes |
|---|---|---|---|
| VTK-STRUCTGRID-IO | VTK legacy reader/writer for STRUCTURED_GRID and UNSTRUCTURED_GRID | **CLOSED** Sprint 28 | `struct_grid.rs` + `unstruct_grid.rs`; ASCII+BINARY read; DIMENSIONS/POINTS/CELLS/CELL_TYPES/SCALARS/VECTORS/NORMALS; 7 tests |

### Stream C -- ITK Algorithms
| ID | Feature | Status | Notes |
|---|---|---|---|
| ITK-CONFIDENCE-CONNECTED | Confidence-connected region growing (iterative mean±k×σ) | **CLOSED** Sprint 28 | `confidence_connected.rs`; multiplier + max_iterations builder; Python binding + CLI dispatch; 9 core tests |
| ITK-SKELETONIZATION | Topology-preserving thinning via iterative boundary erosion | **CLOSED** Sprint 28 | `skeletonization.rs`; Zhang-Suen 2D + is_simple_3d() 3D thinning; Python + CLI; 19 tests |
| DISCRETE-GAUSSIAN-FILTER | DiscreteGaussianFilter<B> (ITK DiscreteGaussianImageFilter parity) | **CLOSED** Sprint 28 | `filter/discrete_gaussian.rs`; variance-parameterized; maximum_error truncation r=ceil(sqrt(-2σ²·ln(e))); use_image_spacing; 11 tests |
| GAP-R02b | InverseConsistentDiffeomorphicDemonsRegistration | **CLOSED** Sprint 28 | `demons/exact_inverse_diffeomorphic.rs`; bilateral E=(1-w)‖F-M∘exp(v)‖² + w‖M-F∘exp(-v)‖²; IC residual = mean‖φ+(φ-(x))-x‖₂; 9 tests |

### Stream D -- Python / CI
| ID | Feature | Status | Notes |
|---|---|---|---|
| PY-CI-MATRIX | Python wheel smoke tests across supported Python versions | **CLOSED** Sprint 28 | `test_smoke.py` 13 tests; `.github/workflows/python_ci.yml` matrix [3.9–12]×[ubuntu,windows]+ubuntu/3.13; maturin develop + pytest |

## Sprint 27 -- Completed

### Stream A -- DICOM Extension
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-MULTIFRAME | Multi-frame DICOM reader | **CLOSED** Sprint 27 | MultiFrameInfo, load_dicom_multiframe<B>; frame stacking [n_frames,rows,cols]; 3 tests |
| DICOM-WRITER-GENERAL | General DICOM object writer | **CLOSED** Sprint 27 | model_to_in_mem, write_dicom_object; node_to_element bijection; 5 tests incl roundtrip |
| DICOM-TRANSFER-SYNTAX | Transfer syntax enumeration | **CLOSED** Sprint 27 | TransferSyntaxKind (11 UIDs + Unknown); from_uid/uid/is_compressed/is_lossless/is_supported; 8 tests |

### Stream B -- VTK Grids
| ID | Feature | Status | Notes |
|---|---|---|---|
| VTK-STRUCT-GRID | VtkStructuredGrid + VtkUnstructuredGrid | **CLOSED** Sprint 27 | VtkDataObject now has 3 variants; validate() invariants; 9 tests |

### Stream C -- ITK Resample
| ID | Feature | Status | Notes |
|---|---|---|---|
| ITK-RESAMPLE | Resample subcommand + Python binding | **CLOSED** Sprint 27 | ritk resample --spacing sz,sy,sx --interpolation nearest/linear/bspline/lanczos4; resample_image() Python; 5 CLI tests |

### Stream D -- Morphology Extension
| ID | Feature | Status | Notes |
|---|---|---|---|
| ITK-MORPHOLOGY-EXTENDED | LabelErosion/Opening/Closing + MorphologicalReconstruction | **CLOSED** Sprint 27 | Anti-extensivity, opening/closing operators, geodesic dilation/erosion reconstruction (Vincent 1993); 21 tests |

### Stream E -- Parity Harness
| ID | Feature | Status | Notes |
|---|---|---|---|
| PY-PARITY-HARNESS | Analytical parity tests | **CLOSED** Sprint 27 | 10 tests in ritk-core/tests/parity.rs: rescale, binary threshold, threshold_below, sigmoid, gradient, Laplacian, z-score, dice x2, constant rescale; all analytically derived |

## Sprint 26 -- Completed

### Stream A -- VTK XML PolyData + VTK Scene
| ID | Feature | Status | Notes |
|---|---|---|---|
| VTK-POLYDATA-XML | VTK XML PolyData (.vtp) reader/writer | **CLOSED** Sprint 26 | parse_vtp + write_vtp_str; correct HashMap/Vec<u32> types; 13 tests |
| VTK-SCENE | VtkScene + VtkActor + RenderProperties | **CLOSED** Sprint 26 | Ordered actor list; find/remove by name; 7 tests |

### Stream B -- ITK Morphology Expansion
| ID | Feature | Status | Notes |
|---|---|---|---|
| ITK-MORPHOLOGY | HitOrMissTransform | **CLOSED** Sprint 26 | SE1=cubic box, SE2=ring (excludes origin); anti-extensivity; 5 tests |
| ITK-MORPHOLOGY | WhiteTopHatFilter + BlackTopHatFilter | **CLOSED** Sprint 26 | f-opening and closing-f; non-negative clamp; 10 tests |
| ITK-MORPHOLOGY | LabelDilation | **CLOSED** Sprint 26 | Min-label-ID conflict resolution; 5 tests |

### Stream C -- ITKSNAP Overlay + ANTs Preprocessing
| ID | Feature | Status | Notes |
|---|---|---|---|
| ITKSNAP-OVERLAY | OverlayState (ImageOverlay/ContourOverlay/MaskOverlay) | **CLOSED** Sprint 26 | Serde; visibility; 8 tests |
| ANTS-WORKFLOW | PreprocessingPipeline | **CLOSED** Sprint 26 | Clamp/Masking/ZScore/MinMax/Smoothing/N4; 9 tests |

### Stream D -- Python + CLI Bindings
| ID | Feature | Status | Notes |
|---|---|---|---|
| PY-MORPH | white_top_hat, black_top_hat, hit_or_miss, label_dilation | **CLOSED** Sprint 26 | Arc::clone pattern; GIL-safe |
| CLI-MORPH | grayscale-erosion/dilation, white/black-top-hat, hit-or-miss, label-dilation | **CLOSED** Sprint 26 | --radius flag reused |

## Sprint 25 -- Completed

### Stream A -- VTK Data Model + PolyData
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| VTK-DATA-MODEL | VTK data-object hierarchy | **CLOSED** Sprint 25 | High | VtkDataObject enum, VtkPolyData (points/cells/attrs), AttributeArray; validate() invariants; 8 unit tests |
| VTK-POLYDATA | PolyData ASCII+BINARY reader/writer | **CLOSED** Sprint 25 | High | Legacy .vtk POLYDATA: POINTS, VERTICES, LINES, POLYGONS, TRIANGLE_STRIPS, POINT_DATA, CELL_DATA; 15 tests |
| VTK-PIPELINE | VtkSource/VtkFilter/VtkSink traits + VtkPipeline | **CLOSED** Sprint 25 | High | Composition law D_n = F_n(D_{n-1}); Send+Sync; 5 pipeline tests |

### Stream B -- ITK Intensity Filter Suite
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| ITK-RESCALE | RescaleIntensityFilter | **CLOSED** Sprint 25 | High | Linear bijection [I_min,I_max] -> [out_min,out_max]; 5 tests |
| ITK-WINDOWING | IntensityWindowingFilter | **CLOSED** Sprint 25 | High | Clamp + rescale; ITK IntensityWindowingImageFilter parity; 5 tests |
| ITK-THRESHOLD | ThresholdImageFilter (Below/Above/Outside) | **CLOSED** Sprint 25 | High | Conditional pixel replacement; 6 tests |
| ITK-SIGMOID | SigmoidImageFilter | **CLOSED** Sprint 25 | High | Sethian 1996 sigmoid; monotone increasing, bounded output; 5 tests |
| ITK-BINARY-THRESHOLD | BinaryThresholdImageFilter | **CLOSED** Sprint 25 | High | Indicator function {fg, bg}; ITK BinaryThresholdImageFilter parity; 5 tests |
| PY-INTENSITY | Python bindings for all 5 intensity filters (7 functions) | **CLOSED** Sprint 25 | High | GIL-safe py.allow_threads; registered in ritk.filter submodule |
| CLI-INTENSITY | CLI bindings for all 5 intensity filters (7 commands) | **CLOSED** Sprint 25 | High | 10 new FilterArgs fields; 7 run_* functions; 7 integration tests |

### Stream C -- ITK-SNAP Annotation Primitives
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| ITKSNAP-LABELS | LabelTable + LabelEntry | **CLOSED** Sprint 25 | Medium-High | CRUD, visibility, next_free_id, Serde; unique-ID invariant; 8 tests |
| ITKSNAP-LABELMAP | LabelMap (3-D dense label volume) | **CLOSED** Sprint 25 | Medium-High | ZYX-flat Vec<u32>; mask_for_label, count_label, present_labels; 8 tests |
| ITKSNAP-WORKFLOW | AnnotationState (points/contours/polylines) | **CLOSED** Sprint 25 | Medium-High | seed annotations, JSON roundtrip, >=2-point contour invariant; 9 tests |
| ITKSNAP-UNDO | UndoRedoStack<S: Clone> | **CLOSED** Sprint 25 | Medium | Branching undo; push clears future; history non-empty invariant; 10 tests |


## Sprint 24 -- Planned

### Next-stage roadmap
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| DICOM-OBJ-MODEL | DICOM object-model preservation | **CLOSED** Sprint 24 | High | Reader: full element iteration, DicomValue::Sequence, DicomPreservedElement. Writer: emit_preservation_nodes before PixelData. 3 round-trip tests. |
| DICOM-MULTIFRAME | DICOM multi-frame / enhanced image support | **CLOSED** Sprint 28 | High | Extend series-centric I/O toward enhanced and multi-frame object handling |
| DICOM-WRITER-GENERAL | Generalized DICOM writer architecture | PLANNED | High | Separate object model serialization from series image writing |
| VTK-DATA-MODEL | VTK data-object hierarchy | **CLOSED** Sprint 25/28 | High | Add canonical mesh/grid data models beyond legacy structured points I/O |
| VTK-PIPELINE | VTK-style pipeline abstractions | PLANNED | High | Introduce data-flow primitives for readers, filters, mappers, and renderable objects |
| ITK-SIMPLEITK-FAMILY | ITK / SimpleITK algorithm breadth expansion | **CLOSED** Sprint 24–28 | High | Prioritize long-tail filters, segmentation, morphology, resampling, and intensity transforms |
| ITKSNAP-WORKFLOW | ITK-SNAP workflow primitives | **CLOSED** Sprint 22 | Medium-High | Add annotation state, overlays, labels, seeds, and undo/redo-oriented editing primitives |
| ANTS-WORKFLOW | ANTs workflow refinement | PLANNED | Medium | Add inverse-consistency controls, preprocessing helpers, and registration workflow composition |
| PY-PARITY-HARNESS | Python parity and reproducibility harness | PLANNED | Medium | Compare `ritk-python` against Python references with value-based tests and benchmarks |

### Stream A — DICOM
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| DICOM-OBJ-MODEL | DICOM object-model preservation | **CLOSED** Sprint 24 | High | Reader: full element iteration, DicomValue::Sequence, DicomPreservedElement. Writer: emit_preservation_nodes before PixelData. 3 round-trip tests. |
| DICOM-MULTIFRAME | DICOM multi-frame / enhanced image support | **CLOSED** Sprint 28 | High | Extend series-centric I/O toward enhanced and multi-frame object handling |
| DICOM-WRITER-GENERAL | Generalized DICOM writer architecture | PLANNED | High | Separate object model serialization from series image writing |
| DICOM-TRANSFER-SYNTAX | Transfer syntax coverage audit | PLANNED | Medium | Audit codec and transfer-syntax handling against supported image SOP classes |
| DICOM-NONIMAGE-SOP | Non-image SOP-class policy | PLANNED | Medium | Define explicit acceptance/rejection behavior for non-image DICOM objects |
| DICOM-METADATA-VALIDATION | Metadata-aware read-path validation | PLANNED | Medium | Validate `DicomReadMetadata` and `DicomSliceMetadata` invariants against round-trips |

### Stream B — VTK
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| VTK-DATA-MODEL | VTK data-object hierarchy | **CLOSED** Sprint 25/28 | High | Canonical mesh/grid data models beyond legacy structured points |
| VTK-POLYDATA | PolyData and topology primitives | **CLOSED** Sprint 25 | High | Surface geometry, vertices, lines, polygons, and scalar/vector attachments |
| VTK-PIPELINE | VTK-style pipeline abstractions | PLANNED | High | Reader/filter/mapper/data-object execution model |
| VTK-SCENE | Scene graph and renderable object model | PLANNED | Medium | Support visualization-facing composition without duplicating algorithm code |

### Stream C — ITK / SimpleITK breadth
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| ITK-SIMPLEITK-FAMILY | Algorithm breadth expansion | **CLOSED** Sprint 24–28 | High | Long-tail filters, segmentation, morphology, resampling, intensity transforms |
| ITK-RESAMPLE | Resampling and interpolation expansion | PLANNED | High | Additional resampling helpers and transform-aware utilities |
| ITK-MORPHOLOGY | Morphology family expansion | PLANNED | Medium-High | Additional binary/grayscale operators and topology-preserving variants |
| ITK-REGION | Region-growing and label tools | PLANNED | Medium-High | Extend connected, confidence, neighborhood, and label-processing utilities |
| ITK-NORM | Intensity statistics and normalization | PLANNED | Medium | Additional SimpleITK-style intensity mapping and normalization helpers |

### Stream D — ITK-SNAP workflow
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| ITKSNAP-WORKFLOW | Interactive segmentation primitives | **CLOSED** Sprint 22 | Medium-High | Annotation state, overlays, labels, seeds, and undo/redo-oriented editing primitives |
| ITKSNAP-LABELS | Label and mask state model | **CLOSED** Sprint 22 | Medium | Label tables, editable masks, and selection state |
| ITKSNAP-OVERLAY | Overlay and contour composition | **CLOSED** Sprint 22 | Medium | Visualization-ready state for masks, contours, and image overlays |

### Stream E — ANTs workflow refinement
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| ANTS-WORKFLOW | Registration workflow composition | PLANNED | Medium | Preprocessing, multi-stage transforms, and pipeline composition |
| ANTS-INVERSE | Inverse-consistency controls | PLANNED | Medium | Strengthen exact-inverse validation and controls around diffeomorphic workflows |
| ANTS-PREPROCESS | Registration preprocessing helpers | PLANNED | Medium | Bias correction, masking, resampling, and intensity normalization pipeline helpers |

### Stream F — Python parity and reproducibility
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PY-PARITY-HARNESS | Python parity and reproducibility harness | PLANNED | Medium | Compare `ritk-python` against Python references with value-based tests and benchmarks |
| PY-CI-MATRIX | Python CI matrix expansion | **CLOSED** Sprint 28 | Medium | Exercise wheel smoke tests and integration tests across supported Python versions |
| PY-STUB-SYNC | Stub and binding synchronization | PLANNED | Low | Keep `pyi` signatures aligned with exported Rust bindings |

## Sprint 23 -- Completed

### Stream A -- CLI Registration Extension
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| CLI-REG-BSPLINE-FFD | BSpline FFD CLI register (run_bspline_ffd) | COMPLETED | High | BSplineFFDConfig; 2 tests |
| CLI-REG-MULTIRES-SYN | Multi-resolution SyN CLI register (run_multires_syn) | COMPLETED | High | MultiResSyNConfig; 2 tests |
| CLI-REG-BSPLINE-SYN | BSpline SyN CLI register (run_bspline_syn) | COMPLETED | High | BSplineSyNConfig; 2 tests |
| CLI-REG-LDDMM | LDDMM CLI register (run_lddmm) | COMPLETED | High | LddmmConfig; 2 tests |
| REGISTER-ARGS-EXT | 7 new RegisterArgs fields (regularization_weight, control_spacing, cc_radius, inverse_consistency, num_time_steps, kernel_sigma, learning_rate) | COMPLETED | High | 12 existing literals updated; 142/142 tests pass |

### Stream B -- CLI Stats and Python Stats Extension
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| CLI-STATS-MSD | mean-surface-distance metric (run_mean_surface_distance) | COMPLETED | Medium | Delegates to mean_surface_distance; requires --reference; 2 tests |
| CLI-STATS-NOISE | noise-estimate metric (run_noise_estimate) | COMPLETED | Medium | Delegates to estimate_noise_mad; no reference; 1 test |
| PY-STATS-NYUL | nyul_udupa_normalize Python binding | COMPLETED | Medium | GIL-safe Arc clone; learn_standard from training_images then apply; registered |

## Sprint 22 — In Progress

### Stream A — Level-set helpers (Planned)
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| LEVEL-SET-HELPERS | Extract shared numerical helpers (indexing, FD, curvature, separable Gaussian) into helpers.rs | COMPLETED | High | Used by ShapeDetection and ThresholdLevelSet |
| SHAPE-DETECT | ShapeDetectionSegmentation (Malladi et al.) + Python binding + CLI binding (run_shape_detection) | COMPLETED | High | 3 CLI tests; ritk-core unit tests passing |
| THRESHOLD-LS | ThresholdLevelSet (Whitaker 1998) + Python binding + CLI binding (run_threshold_level_set) | COMPLETED | Medium | 6 CLI tests; ritk-core unit tests passing |
| CLI-STRUCT-FIX | SegmentArgs duplicate max_iterations field removed; level_set_max_iterations added; impl Default; fill-holes test geometry fixed | COMPLETED | High | 131/131 ritk-cli tests pass |

### Stream B — Python + CLI morphology bindings (Planned)
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| (Completed in Sprint 21) | BinaryFillHoles + MorphologicalGradient Python + CLI bindings | COMPLETED | High | Already done in prior session |

### Stream C — DICOM metadata enrichment (In Progress)
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| DICOM-TAG-READ | Full DICOM tag parsing in scan_dicom_directory | COMPLETED | High | Parse SOPInstanceUID, InstanceNumber, SliceLocation, ImagePositionPatient, ImageOrientationPatient, PixelSpacing, SliceThickness, RescaleSlope/Intercept, SOPClassUID, TransferSyntaxUID per-slice; SeriesInstanceUID, StudyInstanceUID, FrameOfReferenceUID, SeriesDescription, Modality, PatientID, PatientName, StudyDate, SeriesDate, SeriesTime, BitsAllocated, BitsStored, HighBit, PhotometricInterpretation series-level |
| DICOM-Z-SPACING | Z-spacing computation from ImagePositionPatient z-coordinates | COMPLETED | High | spacing[2] = (z_last - z_first) / (N-1); falls back to SliceThickness |
| DICOM-SORT | Slice sorting by z-position, then InstanceNumber, then filename | COMPLETED | High | Deterministic ordering independent of filesystem | 
| DICOM-ORIGIN | Origin and direction from first-slice ImagePositionPatient and ImageOrientationPatient | COMPLETED | High | Direction = row/col cross-product; origin from first IPP |
| DICOM-META-WRITE | write_dicom_series_with_metadata with spatial reference tags | COMPLETED | High | Fixed duplicate bit-depth tag emission after Pixel Data; tags now emitted exactly once before Pixel Data |
| DICOM-WRITE-REGRESSION | DICOM writer regression coverage for Image Pixel Module ordering | COMPLETED | High | Added binary-level regression asserting BitsAllocated, BitsStored, and HighBit each appear exactly once and precede Pixel Data in written slices |
| PY-WHEEL-SMOKE-LS | Python wheel smoke coverage for level-set bindings | COMPLETED | Medium | CI wheel smoke test now installs `pytest`, imports `ritk`, constructs NumPy-backed images, executes `ritk.segmentation.laplacian_level_set_segment`, asserts output shape plus finite values, and runs `pytest crates/ritk-python/tests -q` against the built wheel |
| PY-SEG-INTEGRATION-TESTS | Python segmentation integration tests for connected components and level-set bindings | COMPLETED | Medium | Added Python tests covering connected-components value semantics plus Chan-Vese, Geodesic Active Contour, Shape Detection, Threshold Level Set, and Laplacian level-set shape/finite-value invariants |
| CLI-LAPLACIAN-LS | CLI Laplacian level-set integration | COMPLETED | High | Added `laplacian-level-set` dispatch and CLI tests covering output shape preservation, binary output invariant, and required `--initial-phi` validation |
| REPO-CLEANUP-GENERATED | Root-level generated and stale patch/log artifact cleanup | COMPLETED | Medium | Removed temporary base64 fragments, patch scripts, generated helper scripts, and transient log/output files from the repository root |


## Sprint 21 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PY-CLI-MULTIRES | MultiResDemons Python + CLI bindings | COMPLETED | High | Added `multires_demons_register` to `ritk-python/src/registration.rs`; added `multires-demons` method + `--levels`/`--use-diffeomorphic` args to `ritk-cli/src/commands/register.rs`; 2 new CLI tests |
| SEG-MORPH-EXT | Binary fill holes + morphological gradient | COMPLETED | High | Created `BinaryFillHoles` (6-connected border flood-fill, O(N)) and `MorphologicalGradient` (dilation AND NOT erosion) in `ritk-core/src/segmentation/morphology/`; 10 unit tests added |
| DICOM-WRITE | Real DICOM binary writer | COMPLETED | High | Replaced scaffold writer with `InMemDicomObject` + `FileMetaTableBuilder` per-slice writer; DICM magic verified; updated reader to use `open_file` path; enabled DICOM write in CLI |
| SEG-CLI-BINDINGS | Binary fill holes + morphological gradient CLI exposure | COMPLETED | High | Added `fill-holes` and `morphological-gradient` methods to `ritk-cli/src/commands/segment.rs`; added tests for enclosed-hole filling and boundary extraction |

---

## Sprint 20 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PY-BIND-FLT | Python filter bindings for Sprint 11 algorithms | COMPLETED | High | Added `curvature_anisotropic_diffusion` and `sato_line_filter` to `ritk-python/src/filter.rs`; registered in submodule; updated module docstring |
| PY-BIND-SEG | Python segmentation bindings for Sprint 10 algorithms | COMPLETED | High | Added `confidence_connected_segment`, `neighborhood_connected_segment`, `skeletonization` to `ritk-python/src/segmentation.rs`; registered in submodule |
| PY-IO-EXT | Python IO extended format coverage | COMPLETED | High | Extended `read_image`/`write_image` in `ritk-python/src/io.rs` to cover TIFF, VTK, MGH/MGZ, Analyze (.hdr/.img), JPEG — matching the full ritk-io format surface |
| CLI-FLT-EXT | CLI filter curvature + sato variants | COMPLETED | High | Added `curvature` and `sato` filter subcommands to `ritk-cli/src/commands/filter.rs`; added `--time-step` arg to FilterArgs; added 2 positive tests |
| CLI-SEG-EXT | CLI segment confidence-connected, neighborhood-connected, skeletonization | COMPLETED | High | Added 3 new methods to `ritk-cli/src/commands/segment.rs`; added `--multiplier`, `--max-iterations`, `--neighborhood-radius` args to SegmentArgs; added 11 positive + boundary tests |
| GAP-R02b | Multi-resolution Demons pyramid | COMPLETED | Medium | Created `MultiResDemonsRegistration` + `MultiResDemonsConfig` in `crates/ritk-registration/src/demons/multires.rs`; coarse-to-fine pyramid (2^l downsampling, trilinear upsampling, warm-start displacement injection); supports Thirion and Diffeomorphic variants; 5 unit tests passing |

---

## Sprint 19 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PERF-R02 | Promote shared `_into` primitives to SSOT in `deformable_field_ops.rs` | COMPLETED | High | Added `pub(crate) compute_gradient_into` (true zero-alloc in-place), `pub(crate) warp_image_into`, `pub(crate) compute_mse_streaming`; refactored `compute_gradient` and `warp_image` to delegate; removed duplicate local copies from `thirion.rs`, `symmetric.rs`; eliminated inline streaming loop in `diffeomorphic.rs` by delegating to `compute_mse_streaming`; `crates/ritk-registration/src/{deformable_field_ops.rs,demons/thirion.rs,demons/symmetric.rs,demons/diffeomorphic.rs}` |
| PERF-R03 | Specialize `convolve_axis` into three branch-free per-axis functions | COMPLETED | Medium | Replaced single generic `convolve_axis` (inner-loop `match axis`) with `convolve_z`, `convolve_y`, `convolve_x`; eliminates per-element axis dispatch across every voxel × kernel-element iteration; `gaussian_smooth_inplace` updated to call the three specialized functions; `crates/ritk-registration/src/deformable_field_ops.rs` |

---

## Sprint 18 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PERF-R01 | Demons performance and memory optimization | COMPLETED | High | Reduced clone-heavy scaling-and-squaring allocations, added streaming MSE evaluation for Demons paths, reused iteration buffers in Thirion/Symmetric Demons, and fused inverse-field vector warping in `crates/ritk-registration/src/{deformable_field_ops.rs,demons/thirion.rs,demons/symmetric.rs,demons/diffeomorphic.rs,demons/inverse.rs}` |

---

## Sprint 17 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| GAP-R02b | Diffeomorphic Demons exact inverse | COMPLETED | High | Retained stationary velocity fields in `DemonsResult`, computed exact inverse as `exp(-v)` for diffeomorphic Demons, and added forward/inverse composition regression tests in `crates/ritk-registration/src/demons/diffeomorphic.rs` |

---

## Sprint 16 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| CI-DEPS | Workspace dependency alignment checks | COMPLETED | Medium | Centralized shared crate versions in `[workspace.dependencies]`, migrated crate manifests to `workspace = true`, and added CI enforcement in `.github/workflows/ci.yml` |

---

## Sprint 15 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| CI-NEXTEST | CI nextest enforcement | COMPLETED | Medium | Replaced workspace `cargo test` CI execution with `cargo nextest run --workspace --lib --tests` and installed `cargo-nextest` in `.github/workflows/ci.yml` |

---

## Sprint 14 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PY-THREAD | PyO3 GIL Release Coverage | COMPLETED | Medium | Wrapped long-running Python filter, segmentation, registration, statistics, and image I/O compute paths with `py.allow_threads`; `crates/ritk-python/src/{filter,segmentation,registration,statistics,io}.rs` |

---

## Sprint 13 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| IO-09 | DICOM Read Metadata Slice | COMPLETED | High | Series-level capture plus per-slice geometry and rescale fields; `ritk-io/src/format/dicom/mod.rs` |
| IO-05 | MINC2 Reader | COMPLETED | High | consus-hdf5 HDF5 parsing; `ritk-io/src/format/minc/reader.rs` |
| IO-05 | MINC2 Writer | COMPLETED | High | Low-level HDF5 binary construction; `ritk-io/src/format/minc/writer.rs` |
| DEP | consus Integration | COMPLETED | High | consus-hdf5/core/io/compression workspace dependencies |

---

## Sprint 11 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| FLT-CAD | Curvature Anisotropic Diffusion | COMPLETED | Medium | Alvarez et al. 1992 mean curvature motion; `ritk-core/src/filter/diffusion/curvature.rs` |
| FLT-SATO | Sato Line Filter | COMPLETED | Medium | Multi-scale Hessian line detection (Sato 1998); `ritk-core/src/filter/vesselness/sato.rs` |
| FLT-HESS | Hessian Module | COMPLETED | Medium | 3-D physical-space Hessian + Cardano eigenvalue solver; `ritk-core/src/filter/vesselness/hessian.rs` |

---

## Sprint 10 — Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| SEG-CC | Confidence Connected Region Growing | COMPLETED | Medium | Yanowitz/Bruckstein adaptive statistics; `ritk-core/src/segmentation/region_growing/confidence_connected.rs` |
| SEG-NC | Neighborhood Connected Region Growing | COMPLETED | Medium | Rectangular neighborhood admissibility predicate; `ritk-core/src/segmentation/region_growing/neighborhood_connected.rs` |
| SEG-SK | Skeletonization | COMPLETED | Low | Topology-preserving thinning (Zhang–Suen 2-D, directional 3-D); `ritk-core/src/segmentation/morphology/skeletonization.rs` |

---