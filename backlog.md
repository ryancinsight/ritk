
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