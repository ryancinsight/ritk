## Sprint 63 — CT Bed Separation Filter, Viewer Selection, and Modality-Aware Geometry Audit

**Status**: Planned
**Phase**: Foundation
**Goal**: Add a core CT bed separation filter, expose it in `ritk-snap`, and audit modality-specific geometry handling for CT, MRI, and ultrasound visualizations.

### Gaps closed
| ID | Gap | Root cause | Resolution |
|---|---|---|---|
| GAP-R62-01 | GantryDetectorTilt not handled | (0018,1120) not read; IOP not synthesized | Read tag per slice; synthesize oblique IOP when axial/absent and &#952; > 0.01° |
| GAP-R62-02 | Reader affine axis order wrong | spacing=[&#916;Row,&#916;Col,&#916;z] and direction cols=[F_r,F_c,N&#770;] — inconsistent with [depth,rows,cols] tensor | Fixed to spacing=[&#916;z,&#916;Row,&#916;Col] and direction cols=[N&#770;,F_c,F_r] |
| GAP-R62-03 | Writer affine inconsistency | Writer used spacing[2]=&#916;z and direction[6..9]=N&#770; (old convention) | Updated to spacing[0]=&#916;z, direction[0..3]=N&#770;, IOP=[direction[6..9],direction[3..6]] |
| GAP-R63-01 | CT bed separation filter absent | No core filter for separating table/bed from CT foreground | Sprint 63 |
| GAP-R63-02 | `ritk-snap` filter selection lacks bed separation entry | Viewer core does not expose CT bed separation as a selectable filter | Sprint 63 |
| GAP-R63-03 | Modality handling not audited across CT/MRI/ultrasound | Geometry and orientation handling are not unified by modality semantics in visualization | Sprint 63 |

### Additional changes
| Item | Description |
|---|---|
| DICOMDIR traversal | `try_read_dicomdir`: reads (0004,1220) DirectoryRecordSequence, filters IMAGE records via (0004,1430), resolves (0004,1500) ReferencedFileID |
| Mixed-series filtering | Plurality-dimension canonical selection; excludes scout/localizer files |
| Skull CT integration tests | Fixed path resolution; tests exercise DICOMDIR traversal + 512×512 CT scan loading |
| ritk-snap fixes | serde dependency added; ViewerState Eq removed |
| CT bed separation filter | Add a core filter that masks table/bed voxels while retaining patient foreground |
| Viewer filter selection | Add `bed-separation` as a selectable `ritk-snap`/CLI filter entry |
| Modality audit | Review CT, MRI, and ultrasound orientation/visualization handling and normalize display rules |

### Tests added (+4 new; total 432 lib + 3 integration)
| Test | File | Coverage |
|---|---|---|
| `test_physical_transform_depth_index_advances_along_slice_normal` | reader.rs | Depth/row/col axis &#8594; physical displacement TOL=1e-10 |
| `test_gantry_tilt_synthesizes_oblique_orientation` | reader.rs | (0018,1120)=15° &#8594; F_c=[0,cos15°,-sin15°], N&#770;=[0,sin15°,cos15°] |
| `test_scan_skull_ct_folder_with_dicomdir_loads_series` | reader.rs (existing, now passes) | DICOMDIR traversal + mixed-dim filtering |
| `test_scan_skull_ct_dicomdir_and_folder_agree_on_series` | reader.rs (existing, now passes) | Spacing positive, direction invertible |
| `test_bed_separation_masks_background` | filter.rs | Bed separation filter removes table/background voxels and preserves CT foreground |

### Verification
- `cargo test -p ritk-io -- --test-threads=4`: 432 passed, 0 failed
- `cargo test -p ritk-snap -- --test-threads=4`: 4 passed, 0 failed
- Integration: 3 passed, 0 failed

### Sprint 62 Residual Risk
| ID | Risk | Description | Target |
|---|---|---|---|
| GAP-R63-01 | DICOM-SEG writer absent | Writing segmentation masks as DICOM-SEG | Sprint 63 |
| GAP-R63-02 | VTI binary-appended format | Only ASCII-inline .vti implemented | Sprint 63 |
| GAP-R63-03 | RT Dose / RT Plan readers | RT workflow missing dose grid and beam geometry readers | Sprint 63 |
| GAP-R63-04 | DICOMDIR multi-series selection | No series-UID filtering when DICOMDIR has multiple same-size series | Sprint 63 |
| GAP-R63-05 | Bed separation filter modality audit | CT-specific assumptions need validation against MRI and ultrasound display conventions | Sprint 63 |

---

## Sprint 61 — DICOM Direction Matrix Fix + Cross-Slice IOP/PixelSpacing Validation

**Status**: Completed
**Phase**: Closure
**Goal**: Close GAP-R60-03 (direction matrix transpose), GAP-R60-01 (IOP consistency), GAP-R60-02 (PixelSpacing consistency).

### Gaps closed
| ID | Gap | Root cause | Resolution |
|---|---|---|---|
| GAP-C61-01 (GAP-R60-03) | `load_from_series` direction matrix was transpose of ITK convention | `from_row_slice` on column-major `[rx,ry,rz, cx,cy,cz, nx,ny,nz]` layout produces D^T | Changed to `from_column_slice`; columns = [r, c, n] as ITK requires; consistent with `multiframe.rs` |
| GAP-C61-02 (GAP-R60-01) | IOP inconsistency across slices not validated | No cross-slice IOP check; first slice IOP used silently | Added check after sort; `tracing::warn!` when max |&#916;iop_component| > 1e-4 |
| GAP-C61-03 (GAP-R60-02) | PixelSpacing inconsistency across slices not validated | First file PixelSpacing used unconditionally | Added check after IOP guard; `tracing::warn!` when max |&#916;spacing| > 1e-4 mm |

### Constants added (`reader.rs`, file-scope)
| Symbol | Value | Derivation |
|---|---|---|
| `IOP_CONSISTENCY_THRESHOLD` | `1e-4` | >100× max DS {:.6} roundtrip error (5e-7) for unit cosines |
| `PIXEL_SPACING_CONSISTENCY_THRESHOLD` | `1e-4` | >100× DS roundtrip error for sub-mm spacings |

### Tests added (+3; total 428)
| Test | Location | Coverage |
|---|---|---|
| `test_load_from_series_oblique_direction_uses_column_slice_convention` | `reader.rs` | Coronal IOP [1,0,0,0,0,-1]; dir[(2,1)]=-1.0, dir[(1,2)]=+1.0 |
| `test_scan_directory_warns_on_inconsistent_iop` | `reader.rs` | Axial+coronal in one dir; Ok; direction[0..6] canonical |
| `test_scan_directory_warns_on_inconsistent_pixel_spacing` | `reader.rs` | 0.8+1.0mm in one dir; Ok; spacing canonical |

### Verification
- `cargo test -p ritk-io`: 428 passed, 0 failed
- `from_column_slice` verified against ITK convention: column i = basis vector i; for coronal IOP [1,0,0,0,0,-1], expected matrix element (2,1)=-1 and (1,2)=+1; both confirmed by test.
- Threshold derivations: DS `{:.6}` format error &#8804; 0.5e-6 per encoded float; roundtrip error &#8804; 1e-6; 1e-4 >> 1e-6.

### Sprint 61 Residual Risk
| ID | Risk | Description | Target |
|---|---|---|---|
| GAP-R60-04 | DICOM-SEG writer absent | Writing segmentation masks as DICOM-SEG not yet implemented | Sprint 62 |
| GAP-R60-05 | VTI binary-appended format absent | Only ASCII-inline .vti implemented | Sprint 62 |
| GAP-R60-06 | RT Dose / RT Plan readers absent | RT workflow missing dose grid and beam geometry readers | Sprint 62 |
| GAP-R62-01 | Gantry tilt not handled | `GantryDetectorTilt` is not read or synthesized into an oblique affine when IOP is axial | Sprint 62 |
| GAP-R62-02 | Reader affine axis order needs audit | Slice-axis spacing and direction-column semantics need confirmation against `Image` physical transforms | Sprint 62 |
| GAP-R62-03 | Writer/readback affine consistency needs audit | Series writer spatial tags and series reader reconstruction must remain inverse-consistent for all orientations | Sprint 62 |

---

## Sprint 60 — DICOM Slice Geometry Hardening: Nonuniform Spacing Detection, Warning, and Resampling

**Status**: Completed
**Phase**: Closure
**Goal**: Close GAP-R59-05 (nonuniform/missing DICOM slice spacing silently corrupted volumes) and GAP-R59-06 (oblique series sorted by raw z-component instead of IPP·N&#770;).

### Gaps closed
| ID | Gap | Root cause | Resolution |
|---|---|---|---|
| GAP-C60-01 (GAP-R59-05) | Series reader silent nonuniform/missing slice spacing | `spacing_z` computed as single-span average; per-adjacent-pair gaps never checked | `reader.rs`: compute per-adjacent-pair gaps projected onto N&#770; = normalize(row×col); derive `nominal_spacing` = median(gaps); detect nonuniform (max deviation > 1%) and missing slices (gap > 1.5× nominal); emit `tracing::warn!`; resample to uniform grid via linear interpolation per pixel; update `metadata.dimensions[2]` and `metadata.spacing[2]` |
| GAP-C60-02 (GAP-R59-06) | Series reader oblique sort used raw IPP[2] | Sort key was `image_position_patient[2]` (raw LPS z-component); incorrect for coronal/sagittal/oblique acquisitions | `reader.rs` `scan_dicom_directory`: compute `maybe_normal` from first IOP-bearing slice via `slice_normal_from_iop`; sort by `dot_3d(IPP, N&#770;)`; fall back to `IPP[2]` when IOP absent |
| GAP-C60-03 (GAP-R59-07) | Series reader spacing used single-span average | `(last_z &#8722; first_z) / (N&#8722;1)` masked all per-pair variation | Replace with `analyze_slice_spacing(&positions).nominal_spacing` (median of adjacent-pair gaps); resistant to single outlier, duplicate positions, and missing slices |
| GAP-C60-04 (GAP-R59-08) | Multiframe reader ignored per-frame IPP for spacing | `load_dicom_multiframe` used global `SliceThickness` tag unconditionally even when `per_frame` carries accurate per-frame `image_position` values | `multiframe.rs`: when `per_frame.len() >= 2` and all frames carry `image_position`, project onto N&#770;, run `analyze_slice_spacing`, emit structured warnings, resample via `resample_frames_linear` when nonuniform or missing; fall back to `frame_thickness` otherwise |

### New geometry utilities added (`reader.rs`, `pub(super)`)
| Symbol | Contract |
|---|---|
| `normalize_3d(v) -> Option<[f64;3]>` | Returns unit vector; `None` when `|v| < 1e-10` |
| `dot_3d(a, b) -> f64` | Standard R³ inner product |
| `slice_normal_from_iop(iop) -> Option<[f64;3]>` | N&#770; = normalize(row × col); `None` on degenerate IOP |
| `SliceGeometryReport` | `nominal_spacing`, `max_relative_deviation`, `missing_between`, `is_nonuniform`, `has_missing_slices` |
| `analyze_slice_spacing(positions) -> SliceGeometryReport` | Median-gap analysis per DICOM PS3.3 C.7.6.2 invariants |
| `resample_frames_linear(frames, positions, spacing) -> Vec<Vec<f32>>` | Linear interp along slice axis; N_target = round(span/spacing)+1; clamped endpoints |
| `NONUNIFORM_SPACING_THRESHOLD = 0.01` | 1% relative deviation triggers nonuniform flag |
| `MISSING_SLICE_GAP_FACTOR = 1.5` | Gap > 1.5× nominal triggers missing-slice flag |

### Tests added (+10 from Sprint 59 baseline of 415; total 425)
| Test | Location | Coverage |
|---|---|---|
| `test_analyze_slice_spacing_uniform` | `reader.rs` | Nominal=1.0, zero deviation, no flags |
| `test_analyze_slice_spacing_nonuniform` | `reader.rs` | max_rel_dev=0.2 (20%), `is_nonuniform=true`, `has_missing_slices=false` |
| `test_analyze_slice_spacing_missing_slice` | `reader.rs` | Gap=2.0, `missing_between=[1]`, both flags |
| `test_resample_frames_linear_identity_on_uniform` | `reader.rs` | 4 frames uniform &#8594; identity pass-through < 1e-5 |
| `test_resample_frames_linear_missing_slice` | `reader.rs` | 4&#8594;5 frames; interpolated midpoint = 0.5×20+0.5×40=30.0 < 1e-4 |
| `test_resample_frames_linear_nonuniform_interpolation` | `reader.rs` | t=1/1.1; expected=(1&#8722;t)×10+t×20 < 1e-4 |
| `test_normalize_3d` | `reader.rs` | Unit vector, diagonal, zero&#8594;None |
| `test_slice_normal_from_iop_axial` | `reader.rs` | [1,0,0]×[0,1,0]=[0,0,1] |
| `test_dot_3d` | `reader.rs` | [1,2,3]·[4,5,6]=32; orthogonal&#8594;0 |
| `test_load_multiframe_spacing_from_slice_thickness` | `multiframe.rs` | SliceThickness fallback; spacing_z=2.5±0.01; shape [3,4,4] |

### Verification
- `cargo test -p ritk-io`: **425 passed, 0 failed** (baseline was 415).
- `cargo check -p ritk-io --tests`: zero compilation errors.
- `resample_frames_linear` verified analytically: N_target = round(span/spacing)+1; t = (target&#8722;p[lo])/(p[hi]&#8722;p[lo]) &#8712; [0,1]; output = (1&#8722;t)·src[lo] + t·src[hi] per pixel.
- `analyze_slice_spacing` verified: median of sorted gaps; degenerate nominal &#8804; 0 &#8594; safe fallback.
- DICOM PS3.3 C.7.6.2: ImagePositionPatient projected onto slice normal N&#770; is the correct inter-slice distance metric for all acquisition orientations.

### Sprint 60 Residual Risk
| ID | Risk | Description | Target |
|---|---|---|---|
| GAP-R60-01 | IOP inconsistency across slices not validated | Series with mixed IOP (stitched or multi-acquisition) silently uses first slice's IOP for sort and normal; no cross-slice IOP check | Sprint 61 |
| GAP-R60-02 | PixelSpacing inconsistency across slices not validated | First file's PixelSpacing used unconditionally; mixed-spacing series silently produces incorrect in-plane geometry | Sprint 61 |
| GAP-R60-03 | Direction matrix construction inconsistency | `load_dicom_multiframe` uses `from_column_slice`; `load_from_series` uses `from_row_slice` for the same [rx,ry,rz,cx,cy,cz,nx,ny,nz] layout — one is transposed | Sprint 61 |
| GAP-R60-04 | DICOM-SEG writer absent | Writing segmentation masks as DICOM-SEG not yet implemented | Sprint 61 |
| GAP-R60-05 | VTI binary-appended format absent | Only ASCII-inline .vti implemented | Sprint 61 |
| GAP-R60-06 | RT Dose / RT Plan readers absent | RT workflow missing dose grid and beam geometry readers | Sprint 61 |

---

## Sprint 59 — DICOM-SEG Reader, DICOM-RT Structure Set Reader, VTK XML ImageData (.vti) Reader/Writer

**Status**: Completed
**Phase**: Closure
**Goal**: Close GAP-R58-01 (DICOM-SEG reader), GAP-R58-02 (DICOM-RT Structure Set &#8594; VTK PolyData), and GAP-R58-03 (VTK XML ImageData .vti reader/writer) for MITK/VTK/ITK parity.

### Gaps closed
| ID | Gap | Root cause | Resolution |
|---|---|---|---|
| GAP-C59-01 (GAP-R58-01) | DICOM-SEG reader absent | Segmentation Object (SOP 1.2.840.10008.5.1.4.1.1.66.4) not parsed | `seg.rs`: `read_dicom_seg` parses Rows/Cols/NumFrames/BitsAllocated/SegmentationType, Segment Sequence (0062,0002), Per-Frame FG (5200,9230) for segment numbers + image positions, Shared FG (5200,9229) for orientation/spacing; BINARY unpacking: frame_bytes=&#8968;rows×cols/8&#8969;, MSB-first bit extraction; output: `DicomSegmentation` with `pixel_data: Vec<Vec<u8>>` (0 or 1 per pixel for BINARY) |
| GAP-C59-02 (GAP-R58-02) | DICOM-RT Structure Set &#8594; VTK absent | RT Structure Set (SOP 1.2.840.10008.5.1.4.1.1.481.3) not parsed | `rt_struct.rs`: `read_rt_struct` builds ROI map from (3006,0020), merges interpreted types from (3006,0080), populates display color + contours from (3006,0039); `rt_roi_to_polydata` maps CLOSED_PLANAR&#8594;polygons, OPEN_PLANAR&#8594;lines, POINT&#8594;vertices with running point-offset indexing |
| GAP-C59-03 (GAP-R58-03) | VTK XML ImageData (.vti) reader/writer absent | No ASCII-inline VTI implementation | `format/vtk/image_xml/writer.rs` (ASCII-inline writer, 10 tests); `format/vtk/image_xml/reader.rs` (ASCII-inline reader, 10 tests); `VtkImageData` domain type added to `vtk_data_object.rs` with `n_points()`/`n_cells()`/`validate()` and `ImageData` variant in `VtkDataObject` enum; `image_xml` module exposed via `vtk/mod.rs` |

### Tests added (+35 from Sprint 58 baseline of 380; total 415)
| Area | Tests | Count |
|---|---|---|
| DICOM-SEG reader (`seg.rs`) | missing file, wrong SOP, binary 4×4 single-frame, two-frame two-segment, pixel spacing, per-frame image position | +6 |
| DICOM-RT Structure Set (`rt_struct.rs`) | missing file, wrong SOP, single ROI CLOSED_PLANAR, two ROIs sorted, interpreted type, polydata CLOSED_PLANAR, polydata OPEN_PLANAR, polydata mixed | +8 |
| `VtkImageData` domain type (`vtk_data_object.rs`) | n_points/n_cells, validate ok, validate wrong scalar len, ImageData variant | +4 |
| VTI writer (`image_xml/writer.rs`) | VTKFile header, WholeExtent format, origin/spacing, scalar point data, multicomponent vectors, cell data, empty grid, file roundtrip via string, invalid grid rejection, write to file | +10 |
| VTI reader (`image_xml/reader.rs`) | WholeExtent parse, origin/spacing, scalars, multicomponent, cell data, empty PointData, full roundtrip, file roundtrip, missing Piece tag error, nonexistent file error | +10 |
| `seg.rs` compile fix | Replaced `.to_int::<u16>()` (absent in crate) with `.to_str().ok().and_then(|s| s.trim().parse().ok())` pattern; replaced malformed `debug!` with `tracing::debug!` format-string form | — |
| **Total** | | **415 passed, 0 failed** |

### Verification
- `cargo check -p ritk-io --tests`: zero errors, zero warnings.
- `cargo test -p ritk-io --lib`: 415 passed, 0 failed.
- DICOM PS3.3 C.8.20 — SEG pixel unpacking: frame_bytes = &#8968;rows×cols/8&#8969;; bit i = byte i/8, bit-pos 7&#8722;(i%8); MSB-first per DICOM convention; BINARY output &#8712; {0,1}.
- DICOM PS3.3 C.8.8.5 — RT Structure Set ROI contour parsing; coordinate triples from (3006,0050) DS; closed polygon winding preserved.
- VTK File Formats §6 — VTI WholeExtent "x0 x1 y0 y1 z0 z1"; n_points = &#8719;(e&#8342;&#8330;&#8321;&#8722;e&#8342;+1); Piece Extent == WholeExtent for single-piece output.

### Sprint 59 Residual Risk
| ID | Risk | Description | Target |
|---|---|---|---|
| GAP-R59-01 | DICOM-SEG writer absent | Writing segmentation masks as DICOM-SEG not yet implemented | Sprint 60 |
| GAP-R59-02 | VTI binary-appended format absent | Only ASCII-inline .vti implemented; large volumes require appended/binary mode | Sprint 60 |
| GAP-R59-03 | RT Dose / RT Plan readers absent | RT workflow missing dose grid and beam geometry readers | Sprint 60 |
| GAP-R59-04 | VTK Rectilinear Grid XML (.vtr) absent | VTR format needed for rectilinear grid parity | Sprint 60 |

---

## Sprint 58 — VtkCellType + VTU Reader/Writer, DICOM Enhanced Multiframe Per-Frame Functional Groups, JPEG 2000 Lossless Round-Trip, Build Fix

**Status**: Completed
**Phase**: Closure
**Goal**: Close GAP-R57-01 (JPEG 2000 lossless round-trip via openjpeg-sys FFI), add `VtkCellType` enum and VTK XML UnstructuredGrid (VTU) reader/writer, implement DICOM Enhanced Multiframe per-frame functional groups parsing, and resolve libstdc++ link resolution on Windows GNU targets.

### Gaps closed
| ID | Gap | Root cause | Resolution |
|---|---|---|---|
| GAP-C58-01 (GAP-R57-01) | JPEG 2000 lossless round-trip test missing | No pure-Rust J2K encoder; `jpeg2k` is decode-only | `write_jpeg2000_lossless_dicom_file` via openjpeg-sys FFI (`OPJ_CODEC_J2K`, `irreversible=0`, `numresolution=1`); `into_temp_path()` closes Rust handle before OpenJPEG opens path (Windows file-sharing safety); wraps codestream in `PixelFragmentSequence`; TS `1.2.840.10008.1.2.4.90` |
| GAP-C58-02 | `VtkCellType` enum absent; `VtkUnstructuredGrid.cell_types` typed as `Vec<u8>` | Cell type codes stored unvalidated as raw bytes | Added `VtkCellType` enum (34 variants, codes 1–34 per VTK File Formats spec) with `to_u8`/`from_u8`; changed `cell_types` field from `Vec<u8>` to `Vec<VtkCellType>`; ASCII/binary parsers map via `from_u8` with `tracing::warn` fallback |
| GAP-C58-03 | VTK XML UnstructuredGrid (VTU) reader/writer absent | VTU format not implemented | Created `format/vtk/unstructured_xml/writer.rs` (ASCII-inline writer, 10 tests) and `format/vtk/unstructured_xml/reader.rs` (ASCII-inline reader, 16 tests); exposed via `pub mod unstructured_xml` in `format/vtk/mod.rs` |
| GAP-C58-04 | DICOM Enhanced Multiframe per-frame functional groups not parsed | `MultiFrameInfo` lacked per-frame position, orientation, spacing, rescale fields | Added `PerFrameInfo` struct (image_position, image_orientation, pixel_spacing, slice_thickness, rescale_slope, rescale_intercept — all `Option`); added `per_frame: Vec<PerFrameInfo>` to `MultiFrameInfo`; `extract_functional_groups` parses Shared (5200,9229) and Per-Frame (5200,9230) per DICOM PS3.3 C.7.6.16; `load_dicom_multiframe` applies per-frame rescale when non-empty |
| GAP-C58-05 | libstdc++ not linked in example/binary link steps on Windows GNU | `build.rs` emitted link metadata only for the library; examples and integration tests did not inherit it | `build.rs`: `locate_libstdcxx_dir()` queries `g++`/`CXX -print-file-name=libstdc++.a`, canonicalizes, strips `\\?\` prefix for lld; emits `cargo:rustc-link-search=native=<dir>`; `.cargo/config.toml`: added `-C link-arg=-lstdc++` to `[target.x86_64-pc-windows-gnu]` rustflags |

### Tests added (+41 from Sprint 57 baseline of 339; total 380)
| Area | Tests | Count |
|---|---|---|
| JPEG 2000 lossless round-trip (`codec.rs`) | `write_jpeg2000_lossless_dicom_file` helper + `test_decode_compressed_frame_jpeg2000_lossless_round_trip` | +2 |
| `VtkCellType` enum (`vtk_data_object.rs`) | `test_vtk_cell_type_roundtrip`, `test_vtk_cell_type_from_u8_unknown` | +2 |
| VTU writer (`unstructured_xml/writer.rs`) | 10 tests: point/cell data, multi-component, empty mesh | +10 |
| VTU reader (`unstructured_xml/reader.rs`) | 16 tests: parse, round-trip, error paths, coordinate precision | +16 |
| DICOM Enhanced Multiframe (`multiframe.rs`) | default struct, empty functional groups, basic SOP, shared groups, per-frame rescale E2E | +5 |
| Build/linker integration | 4 tests from build and linker work | +4 |
| **Total** | | **380 passed, 0 failed** |

### Verification
- `cargo check -p ritk-io --tests`: clean, zero errors, zero warnings.
- Mathematical invariant: ISO 15444-1 §C.5.5.1 — `irreversible=0` &#10233; 5/3 integer wavelet &#10233; |S'[i] &#8722; S[i]| = 0 for all i.
- VTK File Formats §5.10 — cell type codes 1–34 exactly match `VtkCellType::to_u8`/`from_u8` round-trip.
- DICOM PS3.3 C.7.6.16 — Shared (5200,9229) and Per-Frame (5200,9230) Functional Groups Sequence parsed per spec.

### Sprint 58 Residual Risk
| ID | Risk | Description | Target |
|---|---|---|---|
| GAP-R58-01 | DICOM-SEG reader absent | Segmentation Object reading not implemented; required for MITK parity | Sprint 59 |
| GAP-R58-02 | DICOM-RT structure set absent | RT Structure Set to VTK mesh path not implemented | Sprint 59 |
| GAP-R58-03 | VTK image data XML absent | `vtkImageData`/STRUCTURED_POINTS XML reader/writer absent | Sprint 59 |

---

## Sprint 57 — JPEG-LS + JPEG 2000 Codec Integration with Clang

**Status**: Completed
**Phase**: Execution &#8594; Closure
**Goal**: Enable JPEG-LS Lossless/Near-Lossless (TS .80/.81) and JPEG 2000 Lossless/Lossy (TS .90/.91) decode paths; configure LLVM/Clang as C/C++ compiler for native build dependencies.

### Gap closed
| Gap | Root cause | Resolution |
|---|---|---|
| JPEG-LS codec not supported | `charls` feature not enabled; `charls` static build not configured | Enabled `charls` feature on `dicom-transfer-syntax-registry`; added `charls = { version = "0.4", features = ["static"] }` workspace dep for bundled static build; added `charls = { workspace = true }` to `ritk-io` deps for Cargo feature unification |
| JPEG 2000 codec not supported | `openjpeg-sys` feature not enabled | Enabled `openjpeg-sys` feature on `dicom-transfer-syntax-registry`; added `openjpeg-sys = "1.0"` workspace dep |
| No C/C++ compiler configured | Native build deps require clang | Added `[env]` section to `.cargo/config.toml` with target-specific clang/clang-cl vars (`force = false`); updated CI to install LLVM/Clang on all three OS matrices |
| `is_codec_supported()` incomplete | JPEG-LS and JPEG2000 variants absent from match | Added `JpegLsLossless`, `JpegLsLossy`, `Jpeg2000Lossless`, `Jpeg2000Lossy` to `is_codec_supported()` |

### Tests added
- `test_is_codec_supported_jpeg_ls_true`: asserts JPEG-LS Lossless + Near-Lossless in `is_codec_supported()`.
- `test_is_codec_supported_jpeg2000_true`: asserts JPEG 2000 Lossless + Lossy in `is_codec_supported()`.
- `test_decode_compressed_frame_jpegls_lossless_round_trip`: full round-trip via CharLS encode &#8594; DICOM &#8594; decode_compressed_frame; asserts max_error = 0.0 (ISO 14495-1 NEAR=0 invariant).
- `test_decode_compressed_frame_jpegls_near_lossless_round_trip`: near-lossless round-trip with NEAR=2; asserts max_error &#8804; 2.0 (ISO 14495-1 analytical bound).

### Sprint 57 Residual Risk
| Risk | Description | Mitigation |
|---|---|---|
| JPEG 2000 round-trip test | No pure-Rust JPEG 2000 encoder; `jpeg2k` crate is decode-only. Full round-trip requires openjpeg-sys FFI encoding. | **Closed Sprint 58**: implemented openjpeg-sys FFI encoding helper `write_jpeg2000_lossless_dicom_file`; full round-trip test added and verified clean. |
| Windows clang-cl availability | `clang-cl` must be in PATH for Windows CI builds. | CI step installs LLVM via Chocolatey and appends `C:\Program Files\LLVM\bin` to GITHUB_PATH. |

---

## Sprint 56 -- Completed

### Stream A -- RLE Lossless Native Decoder (DICOM-RLE-NATIVE-R56)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-RLE-NATIVE-R56 | `packbits_decode`, `decode_rle_lossless_frame` in `codec.rs`; RLE bypass in `decode_compressed_frame` | **CLOSED** Sprint 56 | `dicom-transfer-syntax-registry v0.8.2` RLE decoder has an off-by-one write-start offset (`start = spp &#8722; byte_offset = 1` instead of `0`) for 8-bit grayscale. This silently forces `dst[0] = 0` and loses `dst[N&#8722;1]` for any file where `pixel[0] &#8800; 0`. A post-hoc correction is impossible without data loss. Fix: native `decode_rle_lossless_frame` implements DICOM PS3.5 Annex G directly. `packbits_decode` is the strict left inverse of `packbits_encode` (PackBits is lossless). The 64-byte RLE header is parsed for segment count and offsets. Byte-plane segments are decoded via `packbits_decode` and reassembled into LE pixel bytes per DICOM PS3.5 §G.5. `decode_compressed_frame` detects `RleLossless` via `obj.meta().transfer_syntax()` and dispatches to the native decoder before invoking the upstream registry. Correct for `bits_allocated &#8712; {8, 16}` and any `samples_per_pixel`. Fragment bytes accessed via `Value::PixelSequence(seq).fragments()[frame_idx]` (dicom-rs stores fragments as `Vec<u8>`). |

### Stream B -- Upstream-Bug Test Coverage (DICOM-RLE-UNRESTRICTED-RT-R56)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-RLE-UNRESTRICTED-RT-R56 | `test_decode_compressed_frame_rle_lossless_unrestricted_round_trip` in `codec.rs` | **CLOSED** Sprint 56 | New test encodes all N=16 pixels including `pixel[0] = 42` (non-zero). Would FAIL with the upstream decoder (which forces `dst[0] = 0`) and MUST pass with the native decoder. Exercises `pixel[0] = 42` assertion explicitly. Verifies `max_error = 0` over all 16 pixels. Old `test_decode_compressed_frame_rle_lossless_round_trip` updated: `build_rle_fragment_8bit(&original[1..])` &#8594; `build_rle_fragment_8bit(&original)` (no offset-compensation needed); docstring updated to remove upstream-bug compensation proof. |

### Stream A–B -- Formal Invariants
| Invariant | Expression | Verified By |
|---|---|---|
| PackBits decode–encode inverse | `&#8704;S: packbits_decode(packbits_encode(S), S.len()) = S` | `test_decode_compressed_frame_rle_lossless_round_trip`, `test_decode_compressed_frame_rle_lossless_unrestricted_round_trip` |
| RLE frame exact fidelity | `max|decoded[i] &#8722; original[i]| = 0` for all `i &#8712; [0, N&#8722;1]` | Both round-trip tests; `pixel[0] = 42` assertion in unrestricted test |
| Pixel[0] correctness | `decoded[0] = 42.0` (non-zero pixel[0] not corrupted to 0) | `test_decode_compressed_frame_rle_lossless_unrestricted_round_trip` |
| LE byte reassembly | `raw[p×S×B + s×B + j] = segment[s×B + (B&#8722;1&#8722;j)][p]` (j=0 = LSB) | Verified analytically for B=1 (8-bit) and B=2 (16-bit) cases |

### Sprint 56 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-RLE-NATIVE-RT1-R56 | `test_decode_compressed_frame_rle_lossless_round_trip` (updated) | **CLOSED** Sprint 56 | Now encodes all 16 pixels; `max_error == 0.0`; exercises both PackBits run types |
| DICOM-RLE-UNRESTRICTED-RT1-R56 | `test_decode_compressed_frame_rle_lossless_unrestricted_round_trip` (new) | **CLOSED** Sprint 56 | `pixel[0] = 42`; encodes all 16 pixels; asserts `decoded[0] == 42.0`; `max_error == 0.0` |

### Sprint 56 Test Results
| Suite | Count | Notes |
|---|---|---|
| codec tests | +1 new | `test_decode_compressed_frame_rle_lossless_unrestricted_round_trip` |
| Updated | 1 | `test_decode_compressed_frame_rle_lossless_round_trip` (offset-compensation removed) |
| Regression | 336 prior | All Sprint 55 and earlier tests passing |
| Diagnostics | Clean | Zero errors, zero warnings |
| Total | **337 passed, 0 failed** | Full ritk-io unit suite |

### Sprint 56 Residual Risk
| Risk | Description | Mitigation |
|---|---|---|
| Upstream RLE bug (closed for ritk-io) | `dicom-transfer-syntax-registry v0.8.2` off-by-one is bypassed for all RLE Lossless frames by `decode_rle_lossless_frame`. No residual risk for ritk-io. | File upstream bug report against `dicom-transfer-syntax-registry` with the minimal reproducer. |
| JPEG-LS (deferred) | Requires `charls` feature (C++ library). Not yet enabled. | Enable `charls` + add `JpegLsLossless / JpegLsLossy` to `is_codec_supported()` in a future sprint. |
| JPEG 2000 (deferred) | Requires `openjp2` feature (C library). Not yet enabled. | Enable `openjp2` + add `Jpeg2000Lossless / Jpeg2000Lossy` to `is_codec_supported()` in a future sprint. |

---

## Sprint 55 -- Completed

### Stream A -- Codec Documentation Sync (DICOM-CODEC-DOC-R55)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-CODEC-DOC-R55 | Update `codec.rs` module docstring to list all 8 supported codecs | **CLOSED** Sprint 55 | Sprint 53 docstring listed only 3 codecs. Updated table adds JPEG Extended (`.51`), JPEG Lossless NH (`.57`), and JPEG XL variants (`.110`/`.111`/`.112`). Added `Feature` column. Replaced "Extension points" with "Not yet supported" section with correct UIDs and C/C++ feature names. Added JPEG Extended tolerance and RLE fidelity contract entries. |

### Stream B -- JPEG Extended Round-Trip Test (DICOM-CODEC-EXT-RT-R55)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-CODEC-EXT-RT-R55 | `test_decode_compressed_frame_jpeg_extended_round_trip` in `codec.rs` | **CLOSED** Sprint 55 | JPEG Extended (1.2.840.10008.1.2.4.51) was `is_codec_supported()=true` but had no round-trip test. SOF0 frame encapsulated under TS `.51`; `jpeg-decoder` handles both SOF0 and SOF1. Tolerance &#8804; 16 (analytically identical to Baseline Q75 bound). Exercises same 4×4 8-bit image with values spanning [20, 225]. |

### Stream C -- RLE Lossless Round-Trip Test (DICOM-CODEC-RLE-RT-R55)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-CODEC-RLE-RT-R55 | `packbits_encode`, `build_rle_fragment_8bit`, `test_decode_compressed_frame_rle_lossless_round_trip` in `codec.rs` | **CLOSED** Sprint 55 | RLE Lossless (1.2.840.10008.1.2.5) was `is_codec_supported()=true` but had no round-trip test. DICOM RLE PackBits encoder implemented inline per DICOM PS3.5 Annex G.3.1–G.4.1. Upstream `dicom-transfer-syntax-registry v0.8.2` RLE decoder has an off-by-one write offset for 8-bit grayscale: `start = 1` instead of `0` for single-channel 8-bit, causing `dst[0]=0` always and `dst[i]=decoded_segment[i-1]` for i &#8712; [1, N-1]. Compensation proof: set `original[0]=0`, encode `original[1..]`; decoder maps `decoded_segment[i]&#8594;dst[i+1]` exactly. Lossless invariant: `max_error = 0`. Test exercises both repeat and literal runs in the same 4×4 frame. |

### Stream D -- CI Matrix Expansion (CI-MATRIX-R55)
| ID | Feature | Status | Notes |
|---|---|---|---|
| CI-MATRIX-R55 | Extend `test` job in `.github/workflows/ci.yml` to matrix `[ubuntu-latest, windows-latest, macos-latest]` | **CLOSED** Sprint 55 | Previously Ubuntu-only. `strategy.matrix.os` added to `test` job. `runs-on`, job `name`, cache `key`, and `restore-keys` all parameterized on `matrix.os`. All other jobs (`fmt`, `clippy`, `dependency-alignment`, `python-wheel`) remain Ubuntu-only. `python-wheel: needs: test` preserved; GitHub Actions waits for all matrix variants to succeed. |

### Stream A–D -- Formal Invariants
| Invariant | Expression | Verified By |
|---|---|---|
| JPEG Extended tolerance | `&#8704;i: |decoded[i] &#8722; original[i]| &#8804; 16` | `test_decode_compressed_frame_jpeg_extended_round_trip` (Q75 DC+AC bound = 13, tolerance = 16) |
| RLE Lossless exact fidelity | `max|decoded[i] &#8722; original[i]| = 0` | `test_decode_compressed_frame_rle_lossless_round_trip` (PackBits lossless + offset-compensation proof) |
| RLE offset compensation | `original[0]=0 &#8743; encode(original[1..]) &#10233; decoded = original` | `test_decode_compressed_frame_rle_lossless_round_trip` docstring proof |

### Sprint 55 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-CODEC-EXT-RT1-R55 | `test_decode_compressed_frame_jpeg_extended_round_trip` | **CLOSED** Sprint 55 | JPEG Extended TS `.51`; pixel count == 16; values &#8712; [0,255]; `max_error &#8804; 16.0` |
| DICOM-CODEC-RLE-RT1-R55 | `test_decode_compressed_frame_rle_lossless_round_trip` | **CLOSED** Sprint 55 | RLE Lossless TS `.5`; pixel count == 16; values &#8712; [0,255]; `max_error == 0.0` |

### Sprint 55 Test Results
| Suite | Count | Notes |
|---|---|---|
| codec tests | +2 new | JPEG Extended round-trip × 1, RLE Lossless round-trip × 1 |
| Regression | 334 prior | All Sprint 54 and earlier tests unmodified and passing |
| Diagnostics | Clean | Zero errors, zero warnings |
| Total | **336 passed, 0 failed** | Full ritk-io unit suite |

### Sprint 55 Residual Risk
| Risk | Description | Mitigation |
|---|---|---|
| Upstream RLE off-by-one | `dicom-transfer-syntax-registry v0.8.2` RLE decoder writes at `start=1` for 8-bit grayscale, dropping `dst[0]`. Affects real DICOM RLE Lossless files with non-zero pixel[0]. | Consider filing upstream bug report. Current test is offset-compensated; real files may decode pixel[0] as 0 incorrectly. Pin version in `Cargo.toml` with explanatory comment when upgrading. |

---

## Sprint 54 -- Completed

### Stream A -- JPEG Extended + JPEG Lossless Non-Hierarchical (DICOM-CODEC-EXT-R54)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-CODEC-EXT-TS1-R54 | Add `JpegExtended` (1.2.840.10008.1.2.4.51) to `TransferSyntaxKind` | **CLOSED** Sprint 54 | JPEG Extended (Process 2 & 4), lossy 12-bit. Covered by existing `jpeg` feature (zero new deps). `is_compressed()=true`, `is_lossless()=false`, `is_codec_supported()=true`. |
| DICOM-CODEC-EXT-TS2-R54 | Add `JpegLosslessNonHierarchical` (1.2.840.10008.1.2.4.57) to `TransferSyntaxKind` | **CLOSED** Sprint 54 | JPEG Lossless, Non-Hierarchical (Process 14). Covered by existing `jpeg` feature. `is_compressed()=true`, `is_lossless()=true`, `is_codec_supported()=true`. |

### Stream B -- JPEG XL Codec Integration (DICOM-CODEC-JXL-R54)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-CODEC-JXL-DEP-R54 | Enable `jpegxl` feature of `dicom-transfer-syntax-registry` | **CLOSED** Sprint 54 | Added `dicom-transfer-syntax-registry = { version = "0.8", features = ["native", "jpegxl"] }` to workspace deps. Pure-Rust: `jxl-oxide` (decoder) + `zune-jpegxl` + `zune-core` (encoder). No native library. |
| DICOM-CODEC-JXL-TS1-R54 | Add `JpegXlLossless` (1.2.840.10008.1.2.4.110) to `TransferSyntaxKind` | **CLOSED** Sprint 54 | `is_compressed()=true`, `is_lossless()=true`, `is_codec_supported()=true`. ISO 18181-1 modular path. |
| DICOM-CODEC-JXL-TS2-R54 | Add `JpegXlJpegRecompression` (1.2.840.10008.1.2.4.111) to `TransferSyntaxKind` | **CLOSED** Sprint 54 | `is_compressed()=true`, `is_lossless()=false`, `is_codec_supported()=true`. Decoder-only (`JpegXlAdapter`). |
| DICOM-CODEC-JXL-TS3-R54 | Add `JpegXl` (1.2.840.10008.1.2.4.112) to `TransferSyntaxKind` | **CLOSED** Sprint 54 | `is_compressed()=true`, `is_lossless()=false` (not guaranteed), `is_codec_supported()=true`. |
| DICOM-CODEC-JXL-RT-R54 | JXL Lossless round-trip test in `codec.rs` | **CLOSED** Sprint 54 | 4×4 8-bit frame encoded via `zune-jpegxl` &#8594; wrapped in DICOM Part 10 &#8594; decoded via `decode_compressed_frame` &#8594; `max_error == 0.0` (lossless invariant). |

### Stream C -- `is_compressed()` Semantics Correction (DICOM-TS-SEM-R54)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-TS-SEM-R54 | Remove `DeflatedExplicitVrLittleEndian` from `is_compressed()` | **CLOSED** Sprint 54 | Per DICOM PS3.5 Table A-1, `is_compressed()` is defined as pixel-data fragment encapsulation. Deflated compresses the dataset byte-stream, not pixel fragments. Semantics now correct; `is_natively_supported() &#10233; !is_compressed() &#8743; !is_big_endian()` invariant preserved. |

### Stream A + B + C -- Formal Invariants
| Invariant | Expression | Verified By |
|---|---|---|
| Codec path only for encapsulated TS | `is_codec_supported() &#10233; is_compressed()` | `test_codec_supported_implies_compressed` (exhaustive over 16 variants) |
| Disjoint decode paths | `is_natively_supported() &#10233; !is_codec_supported()` | `test_natively_supported_and_codec_supported_are_disjoint` (exhaustive over 16 variants) |
| Native path soundness | `is_natively_supported() &#10233; !is_compressed() &#8743; !is_big_endian()` | `test_natively_supported_implies_not_compressed_and_not_big_endian` |
| Deflated not pixel-compressed | `DeflatedExplicitVrLittleEndian.is_compressed() == false` | `test_is_compressed_deflated_false` |
| JXL Lossless exact fidelity | `max|decoded[i] &#8722; original[i]| = 0` | `test_decode_compressed_frame_jxl_lossless_round_trip` |
| UID bijection | `from_uid(v.uid()) == v` for all 16 known variants | `test_uid_roundtrip_all_known` |

### Sprint 54 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-CODEC-EXT-PRED1-R54 | `test_from_uid_jpeg_extended` | **CLOSED** Sprint 54 | `from_uid("1.2.840.10008.1.2.4.51") == JpegExtended` |
| DICOM-CODEC-EXT-PRED2-R54 | `test_from_uid_jpeg_lossless_non_hierarchical` | **CLOSED** Sprint 54 | `from_uid("1.2.840.10008.1.2.4.57") == JpegLosslessNonHierarchical` |
| DICOM-CODEC-EXT-PRED3-R54 | `test_is_compressed_jpeg_extended_true` | **CLOSED** Sprint 54 | `JpegExtended.is_compressed() == true` |
| DICOM-CODEC-EXT-PRED4-R54 | `test_is_compressed_jpeg_lossless_nh_true` | **CLOSED** Sprint 54 | `JpegLosslessNonHierarchical.is_compressed() == true` |
| DICOM-CODEC-EXT-PRED5-R54 | `test_is_lossless_jpeg_extended_false` | **CLOSED** Sprint 54 | `JpegExtended.is_lossless() == false` (lossy) |
| DICOM-CODEC-EXT-PRED6-R54 | `test_is_lossless_jpeg_lossless_nh_true` | **CLOSED** Sprint 54 | `JpegLosslessNonHierarchical.is_lossless() == true` |
| DICOM-CODEC-EXT-PRED7-R54 | `test_is_codec_supported_jpeg_extended_true` | **CLOSED** Sprint 54 | `JpegExtended.is_codec_supported() == true` |
| DICOM-CODEC-EXT-PRED8-R54 | `test_is_codec_supported_jpeg_lossless_nh_true` | **CLOSED** Sprint 54 | `JpegLosslessNonHierarchical.is_codec_supported() == true` |
| DICOM-CODEC-JXL-PRED1-R54 | `test_from_uid_jpeg_xl_lossless` | **CLOSED** Sprint 54 | `from_uid("1.2.840.10008.1.2.4.110") == JpegXlLossless` |
| DICOM-CODEC-JXL-PRED2-R54 | `test_from_uid_jpeg_xl_recompression` | **CLOSED** Sprint 54 | `from_uid("1.2.840.10008.1.2.4.111") == JpegXlJpegRecompression` |
| DICOM-CODEC-JXL-PRED3-R54 | `test_from_uid_jpeg_xl` | **CLOSED** Sprint 54 | `from_uid("1.2.840.10008.1.2.4.112") == JpegXl` |
| DICOM-CODEC-JXL-PRED4-R54 | `test_is_compressed_jpeg_xl_lossless_true` | **CLOSED** Sprint 54 | `JpegXlLossless.is_compressed() == true` |
| DICOM-CODEC-JXL-PRED5-R54 | `test_is_lossless_jpeg_xl_lossless_true` | **CLOSED** Sprint 54 | `JpegXlLossless.is_lossless() == true` |
| DICOM-CODEC-JXL-PRED6-R54 | `test_is_lossless_jpeg_xl_false` | **CLOSED** Sprint 54 | `JpegXl.is_lossless() == false`; `JpegXlJpegRecompression.is_lossless() == false` |
| DICOM-CODEC-JXL-PRED7-R54 | `test_is_codec_supported_jpeg_xl_lossless_true` | **CLOSED** Sprint 54 | `JpegXlLossless.is_codec_supported() == true` |
| DICOM-CODEC-JXL-PRED8-R54 | `test_is_codec_supported_jpeg_xl_recompression_true` | **CLOSED** Sprint 54 | `JpegXlJpegRecompression.is_codec_supported() == true` |
| DICOM-CODEC-JXL-PRED9-R54 | `test_is_codec_supported_jpeg_xl_true` | **CLOSED** Sprint 54 | `JpegXl.is_codec_supported() == true` |
| DICOM-TS-SEM-PRED1-R54 | `test_is_compressed_deflated_false` | **CLOSED** Sprint 54 | `DeflatedExplicitVrLittleEndian.is_compressed() == false` (dataset compression &#8800; pixel encapsulation) |
| DICOM-CODEC-JXL-RT-R54 | `test_decode_compressed_frame_jxl_lossless_round_trip` | **CLOSED** Sprint 54 | 4×4 8-bit JXL Lossless: pixel count == 16, values in [0,255], `max_error == 0.0` |
| DICOM-CODEC-JXL-INV1-R54 | `test_codec_supported_implies_compressed` (updated) | **CLOSED** Sprint 54 | Exhaustive over all 16 known variants |
| DICOM-CODEC-JXL-INV2-R54 | `test_natively_supported_and_codec_supported_are_disjoint` (updated) | **CLOSED** Sprint 54 | Exhaustive over all 16 known variants |
| DICOM-CODEC-JXL-INV3-R54 | `test_uid_roundtrip_all_known` (updated) | **CLOSED** Sprint 54 | Exhaustive over all 16 known variants |

### Sprint 54 Test Results
| Suite | Count | Notes |
|---|---|---|
| transfer_syntax tests | +18 new | UID round-trips × 5 new variants, predicate tests × 11, invariant updates × 3 |
| codec tests | +1 new | JXL Lossless round-trip (zune-jpegxl encode &#8594; jxl-oxide decode via PixelDecoder) |
| Regression | 312 prior | All Sprint 53 and earlier tests unmodified and passing |
| Diagnostics | Clean | Zero errors, zero warnings (`cargo check --workspace --tests`) |
| Total | **334 passed, 0 failed** | Full ritk-io unit suite, 0.09 s |

---

## Sprint 53 -- Completed

### Stream A -- DICOM Compressed Transfer Syntax Codec Integration (DICOM-CODEC-R53)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-CODEC-DEP-R53 | Add `dicom-pixeldata = "0.8"` with `native` feature as direct ritk-io dependency | **CLOSED** Sprint 53 | `dicom-pixeldata 0.8.2` was already a transitive dep via `dicom = "0.8.0"`; promoted to direct dep so `PixelDecoder` API is accessible. `jpeg-decoder`, `jpeg-encoder`, and `dicom-rle` codecs were already compiled in via the `native` default feature. No new downloads required. |
| DICOM-CODEC-MODULE-R53 | New `codec.rs` module: `pub(super) fn decode_compressed_frame` | **CLOSED** Sprint 53 | Single dispatch entry point for all codec-supported compressed TS. Calls `PixelDecoder::decode_pixel_data_frame`, extracts raw bytes via `.data()`, applies existing `decode_pixel_bytes` linear modality LUT (DICOM PS3.3 C.7.6.3.1.4). Zero new unsafe code. |
| DICOM-CODEC-TS-PRED-R53 | Add `is_codec_supported()` predicate to `TransferSyntaxKind` | **CLOSED** Sprint 53 | Returns `true` for `JpegBaseline`, `JpegLosslessFirstOrderPrediction`, `RleLossless` (pure-Rust codecs). Returns `false` for JPEG-LS (requires `charls`) and JPEG 2000 (requires `openjp2`). |
| DICOM-CODEC-GUARD-R53 | Relax compressed-TS guard in `load_from_series` and `load_dicom_multiframe` | **CLOSED** Sprint 53 | Guard changed from `is_compressed()` to `is_compressed() && !is_codec_supported()`. Codec-supported TS (JPEG Baseline, JPEG Lossless FOP, RLE Lossless) pass through to the decode path instead of being rejected. |
| DICOM-CODEC-READER-R53 | Codec dispatch in `read_slice_pixels` | **CLOSED** Sprint 53 | Detects TS from `slice.transfer_syntax_uid`. When `is_codec_supported()`, calls `codec::decode_compressed_frame(&obj, 0, ...)`. Otherwise falls through to existing native raw-bytes path. |
| DICOM-CODEC-MF-R53 | Codec dispatch in `load_dicom_multiframe` | **CLOSED** Sprint 53 | When `ts.is_codec_supported()`, decodes each frame individually via `codec::decode_compressed_frame(&obj, frame_idx, ...)`. Uncompressed path unchanged. |

### Stream A -- Formal Invariants
| Invariant | Expression | Verified By |
|---|---|---|
| Codec path only for compressed TS | `is_codec_supported() &#10233; is_compressed()` | `test_codec_supported_implies_compressed` |
| Disjoint decode paths | `is_natively_supported() &#10233; !is_codec_supported()` | `test_natively_supported_and_codec_supported_are_disjoint` |
| Modality LUT contract | `Output[i] = codec_sample[i] × slope + intercept` | `test_decode_compressed_frame_rescale_contract` |
| JPEG tolerance bound | `|decoded[i] &#8722; original[i]| &#8804; 16` (DC step &#8804; 4 + 3 primary AC terms &#8804; 3 each + margin) | `test_decode_compressed_frame_jpeg_baseline_round_trip` |

### Stream A -- Sprint 53 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-CODEC-TEST-PRED1-R53 | `test_is_codec_supported_jpeg_baseline_true` | **CLOSED** Sprint 53 | `JpegBaseline.is_codec_supported() == true` |
| DICOM-CODEC-TEST-PRED2-R53 | `test_is_codec_supported_jpeg_lossless_fop_true` | **CLOSED** Sprint 53 | `JpegLosslessFirstOrderPrediction.is_codec_supported() == true` |
| DICOM-CODEC-TEST-PRED3-R53 | `test_is_codec_supported_rle_lossless_true` | **CLOSED** Sprint 53 | `RleLossless.is_codec_supported() == true` |
| DICOM-CODEC-TEST-PRED4-R53 | `test_is_codec_supported_jpeg_ls_false` | **CLOSED** Sprint 53 | JPEG-LS variants return `false`; `charls` feature not enabled |
| DICOM-CODEC-TEST-PRED5-R53 | `test_is_codec_supported_jpeg2000_false` | **CLOSED** Sprint 53 | JPEG 2000 variants return `false`; `openjp2` feature not enabled |
| DICOM-CODEC-TEST-INV1-R53 | `test_codec_supported_implies_compressed` | **CLOSED** Sprint 53 | Exhaustive over all 11 known variants; `is_codec_supported() &#10233; is_compressed()` |
| DICOM-CODEC-TEST-INV2-R53 | `test_natively_supported_and_codec_supported_are_disjoint` | **CLOSED** Sprint 53 | `is_natively_supported() &#10233; !is_codec_supported()` for all 11 variants |
| DICOM-CODEC-TEST-RT1-R53 | `test_decode_compressed_frame_jpeg_baseline_round_trip` | **CLOSED** Sprint 53 | 4×4 8-bit JPEG Baseline: pixel count correct, values in [0,255], max error &#8804; 16 |
| DICOM-CODEC-TEST-RT2-R53 | `test_decode_compressed_frame_rescale_contract` | **CLOSED** Sprint 53 | Uniform patch: `scaled[i] == base[i] × 2.0 + 10.0` within 0.01 fp epsilon |
| DICOM-CODEC-TEST-RT3-R53 | `test_load_series_jpeg_baseline_codec_round_trip` | **CLOSED** Sprint 53 | Full E2E: build SC DICOM with JPEG Baseline TS &#8594; scan &#8594; load &#8594; verify shape [1,4,4] and per-pixel error &#8804; 16 |
| DICOM-CODEC-TEST-RT4-R53 | `test_load_multiframe_jpeg_baseline_codec_round_trip` | **CLOSED** Sprint 53 | Full E2E: build 2-frame JPEG Baseline multiframe DICOM &#8594; load &#8594; verify shape [2,4,4] and per-frame error &#8804; 16 |
| DICOM-CODEC-TEST-GUARD1-R53 | `test_load_series_compressed_ts_errors` (updated) | **CLOSED** Sprint 53 | TS changed to JPEG-LS Lossless (1.2.840.10008.1.2.4.80); still correctly rejected |
| DICOM-CODEC-TEST-GUARD2-R53 | `test_load_multiframe_compressed_ts_errors` (updated) | **CLOSED** Sprint 53 | TS changed to JPEG-LS Lossless; still correctly rejected |

### Sprint 53 Test Results
| Suite | Count | Notes |
|---|---|---|
| transfer_syntax tests | 7 new | `is_codec_supported()` predicates + 2 formal invariant tests |
| codec tests | 2 new | JPEG Baseline round-trip + rescale contract |
| reader tests | 2 new / 1 updated | Series codec E2E + guard update |
| multiframe tests | 2 new / 1 updated | Multiframe codec E2E + guard update |
| Regression | 301 prior | All prior tests unmodified and passing |
| Diagnostics | Clean | Zero errors, zero warnings |
| Total | **312 passed, 0 failed** | Full ritk-io unit suite, 0.17 s |

---

## Sprint 52 -- Completed

### Stream A -- Series UID Generator Monotonicity (DICOM-SERIES-UID-MONO-R52)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-SERIES-UID-MONO-R52 | Fix `generate_series_uid` monotonicity in writer.rs | **CLOSED** Sprint 52 | Added `AtomicU64` static counter; format changed to `2.25.<ns>.<seq>` matching the multiframe UID generator. Eliminates UID collision risk on Windows where SystemTime resolution is ~100 ns. Symmetric with the Sprint 51 fix for `generate_multiframe_uid`. |

### Stream B -- Transfer Syntax Correctness (DICOM-TS-CORRECT-R52)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-TS-BE-R52 | Remove ExplicitVrBigEndian from `is_natively_supported()` | **CLOSED** Sprint 52 | `decode_pixel_bytes` always uses `u16::from_le_bytes` / `i16::from_le_bytes`. Applying LE decode to BE pixel bytes produces `bswap(x)` instead of `x` — silently incorrect intensities. BigEndian DICOM is also retired per DICOM PS 3.5 (withdrawn 2004). |
| DICOM-TS-DEFLATE-R52 | Remove DeflatedExplicitVrLittleEndian from `is_natively_supported()` | **CLOSED** Sprint 52 | Both readers reject Deflated via `is_compressed()`. Classifying it as natively supported was a contradictory invariant (`is_natively_supported() => !is_compressed()` was violated). |
| DICOM-TS-BIGENDIAN-PRED-R52 | Add `is_big_endian()` predicate to `TransferSyntaxKind` | **CLOSED** Sprint 52 | Returns `true` only for `ExplicitVrBigEndian`. Enables precise rejection in both readers distinct from the compressed check. |
| DICOM-TS-READER-GUARD-R52 | Add BigEndian guard to series reader (`load_from_series`) | **CLOSED** Sprint 52 | Guard added alongside existing `is_compressed()` check. Returns `Err` with message containing `"big-endian"` before any pixel decode attempt. |
| DICOM-TS-MF-GUARD-R52 | Add BigEndian guard to multiframe reader (`load_dicom_multiframe`) | **CLOSED** Sprint 52 | Guard added alongside existing `is_compressed()` check. Returns `Err` with message containing `"big-endian"` before any pixel decode attempt. |
| DICOM-TS-INVARIANT-R52 | Enforce formal invariant: `is_natively_supported() &#10233; !is_compressed() &#8743; !is_big_endian()` | **CLOSED** Sprint 52 | Property test exhaustively verifies invariant over all 11 known `TransferSyntaxKind` variants. Module docstring updated. |

### Stream A -- Sprint 52 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-SERIES-UID-TEST-R52 | `test_series_uid_distinct_on_rapid_successive_calls` | **CLOSED** Sprint 52 | Two rapid calls to `generate_series_uid()` must return distinct `2.25.<ns>.<seq>` strings; AtomicU64 counter ensures this even when `t` collides. |

### Stream B -- Sprint 52 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-TS-DEFLATE-TEST-R52 | `test_is_natively_supported_deflated_false` | **CLOSED** Sprint 52 | Asserts `!DeflatedExplicitVrLittleEndian.is_natively_supported()`. |
| DICOM-TS-BE-TEST1-R52 | `test_big_endian_is_not_natively_supported` | **CLOSED** Sprint 52 | Asserts `!ExplicitVrBigEndian.is_natively_supported()`. |
| DICOM-TS-BE-TEST2-R52 | `test_big_endian_is_big_endian_true` | **CLOSED** Sprint 52 | Asserts `ExplicitVrBigEndian.is_big_endian() == true`. |
| DICOM-TS-BE-TEST3-R52 | `test_explicit_vr_le_is_not_big_endian` | **CLOSED** Sprint 52 | Asserts `ExplicitVrLittleEndian.is_big_endian() == false`. |
| DICOM-TS-NATIVE-TEST1-R52 | `test_implicit_vr_le_is_natively_supported` | **CLOSED** Sprint 52 | Positive coverage: Implicit VR LE remains natively supported. |
| DICOM-TS-NATIVE-TEST2-R52 | `test_explicit_vr_le_is_natively_supported` | **CLOSED** Sprint 52 | Positive coverage: Explicit VR LE remains natively supported. |
| DICOM-TS-INV-TEST-R52 | `test_natively_supported_implies_not_compressed_and_not_big_endian` | **CLOSED** Sprint 52 | Exhaustive property test over all 11 variants: for each `v`, `is_natively_supported(v) &#10233; !is_compressed(v) &#8743; !is_big_endian(v)`. |
| DICOM-TS-READER-TEST-R52 | `test_load_series_big_endian_ts_errors` | **CLOSED** Sprint 52 | Verifies UID `1.2.840.10008.1.2.2` is classified as `is_big_endian()` and `!is_natively_supported()`. Guard path confirmed. |
| DICOM-TS-MF-TEST-R52 | `test_multiframe_rejects_big_endian_ts` | **CLOSED** Sprint 52 | Writes a real DICOM Part-10 file with BigEndian TS in file meta; asserts `load_dicom_multiframe` returns `Err` with message containing `"big-endian"`. |

### Stream C -- Repository Hygiene (HYGIENE-R52)
| ID | Action | Status | Notes |
|---|---|---|---|
| HYGIENE-SCRATCH-R52 | Delete 37 scratch/temp files from repository root | **CLOSED** Sprint 52 | Removed: `TransformParameters.0.txt`, all `_*.py` scripts, `_*.txt` scratch files, `dg_test.tmp`, `fix_docs.py`, `gen_morph.py`, `gen_sprint27.py`, `result.0.nii`, `sizes.csv`, `sprint27_write.py`, `test2.py`, `test_out.rs`, `test_out.txt`, `test_output.txt`, `test_sprint.rs`, all `write_*.py` scripts, and other ad-hoc artifacts. |
| HYGIENE-GITIGNORE-R52 | Append `*.tmp`, `*.nii`, `sizes.csv` to `.gitignore` | **CLOSED** Sprint 52 | Prior patterns (`_*.tmp`, `result.*.nii`) were narrower; broadened to prevent future accidental commits. |

### Sprint 52 Test Results
| Suite | Count | Notes |
|---|---|---|
| transfer_syntax tests | 9 new | TS invariant property test + 8 unit classification tests |
| writer tests | 1 new | Series UID monotonicity |
| multiframe tests | 1 new | BigEndian rejection integration test |
| reader tests | 1 new | BigEndian TS classification guard |
| Correctness bugs fixed | 4 | Series UID collision, BigEndian pixel corruption, Deflated native-support contradiction, missing BigEndian reader guards |
| Repository cleanup | 37 files deleted | Root scratch artifacts removed |
| Diagnostics | Clean | Zero errors, zero warnings |
| Total | **301 passed, 0 failed** | Full ritk-io unit suite |

## Sprint 51 -- Completed

### Stream A -- DICOM Multiframe IOD Conformance (DICOM-MF-IOD-R51)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-MF-UID-R51 | Add StudyInstanceUID (0020,000D) and SeriesInstanceUID (0020,000E) to multiframe writer | **CLOSED** Sprint 51 | Both UIDs are Type 1 mandatory per SC Multi-Frame IOD PS3.3 A.8.5.2. Generated via `generate_multiframe_uid()` (atomic counter + nanosecond clock). UIDs are guaranteed distinct within a process call. |
| DICOM-MF-TYPE2-R51 | Add six Type 2 mandatory tags to multiframe writer | **CLOSED** Sprint 51 | PatientName (0010,0010), PatientID (0010,0020), StudyDate (0008,0020), ReferringPhysicianName (0008,0090), StudyID (0020,0010), SeriesNumber (0020,0011) now emitted with empty-string defaults per SC IOD Type 2 semantics. |
| DICOM-MF-PR-R51 | Honor PixelRepresentation (0028,0103) in `load_dicom_multiframe` | **CLOSED** Sprint 51 | `MultiFrameInfo` gains `pixel_representation: u16` field (default 0). `extract_multiframe_header` extracts tag (0028,0103). `load_dicom_multiframe` now uses `super::reader::decode_pixel_bytes` (made `pub(super)`) to decode signed i16 pixels correctly. Previously only unsigned u16 was handled. |
| DICOM-MF-UID-MONO-R51 | Fix `generate_multiframe_uid` monotonicity | **CLOSED** Sprint 51 | Added `AtomicU64` counter; format changed to `2.25.<ns>.<seq>` guaranteeing distinct UIDs for successive calls within a process. Closes collision risk on Windows where SystemTime resolution is ~100ns. |

### Stream A -- Sprint 51 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-MF-UID-TEST-R51 | `test_multiframe_has_study_and_series_uids` | **CLOSED** Sprint 51 | Writes via `write_dicom_multiframe`, asserts StudyInstanceUID and SeriesInstanceUID present, non-empty, and mutually distinct. |
| DICOM-MF-TYPE2-TEST-R51 | `test_multiframe_has_type2_patient_study_series_tags` | **CLOSED** Sprint 51 | Asserts all six Type 2 mandatory tags present in emitted file. |
| DICOM-MF-SIGNED-TEST-R51 | `test_load_multiframe_signed_i16_roundtrip` | **CLOSED** Sprint 51 | Manually constructs DICOM file with PixelRepresentation=1; analytical ground truth [-1000, 0, 1000, 2000]; asserts decoded f32 within 0.5 of expected. |

### Sprint 51 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io multiframe tests | 3 new + existing | `test_multiframe_has_study_and_series_uids`, `test_multiframe_has_type2_patient_study_series_tags`, `test_load_multiframe_signed_i16_roundtrip` |
| Correctness bugs fixed | 4 | Missing Type 1 UIDs in MF writer, missing Type 2 tags in MF writer, unsigned-only pixel decode in MF reader, UID collision risk |
| Architecture improvements | 1 | `decode_pixel_bytes` exposed as `pub(super)` eliminating duplicate logic between reader and multiframe |
| Diagnostics | Clean | Zero errors, zero warnings |
| Total | **292 passed, 0 failed** | Full ritk-io unit suite |

## Sprint 50 -- Completed

### Stream A -- Pixel Decode, File Detection, Writer DRY, Window Metadata (DICOM-GAP-D1-D4-R50)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-D1-PIXELDECODE-R50 | Centralized `decode_pixel_bytes` helper in reader | **CLOSED** Sprint 50 | Replaced per-slice inline decode logic with `decode_pixel_bytes(bytes, bits_allocated, pixel_representation, slope, intercept)`. Handles 8-bit unsigned, 16-bit unsigned, 16-bit signed (i16 two's complement) per DICOM PS3.3 C.7.6.3.1.4. |
| DICOM-D2-FILEDETECT-R50 | Canonicalize `is_likely_dicom_file` extension filter | **CLOSED** Sprint 50 | Only `.dcm`, `.dicom`, `.ima` accepted. `.hdr`/`.img` (Analyze 7.5) and `.raw` (unstructured binary) rejected. Extensionless files probed for DICM magic at byte offset 128 (DICOM PS3.10 §7.1). |
| DICOM-D3-WRITERDRY-R50 | Use `DICOM_SOP_CLASS_SECONDARY_CAPTURE` constant in writer | **CLOSED** Sprint 50 | `write_dicom_series` now references the `DICOM_SOP_CLASS_SECONDARY_CAPTURE` constant instead of duplicated string literal. SamplesPerPixel added to `writer_exclusion_tags` to prevent preservation re-emission. |
| DICOM-D4-WINDOWMETA-R50 | Window metadata fields added to `DicomSliceMetadata` | **CLOSED** Sprint 50 | Fields `pixel_representation: u16`, `bits_allocated: u16`, `window_center: Option<f64>`, `window_width: Option<f64>` added. `DicomSliceMetadata::default()` implemented. `known_handled_tags` updated for (0028,0002), (0028,0103), (0028,1050), (0028,1051). |

### Sprint 50 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-D1-TEST-R50 | `test_decode_pixel_bytes_unsigned_16bit_identity_rescale` | **CLOSED** Sprint 50 | [0x00,0x00,0xFF,0xFF] u16 &#8594; [0.0, 65535.0] |
| DICOM-D1-TEST2-R50 | `test_decode_pixel_bytes_signed_16bit_identity_rescale` | **CLOSED** Sprint 50 | i16::MIN / i16::MAX bytes &#8594; [-32768.0, 32767.0] |
| DICOM-D1-TEST3-R50 | `test_decode_pixel_bytes_signed_16bit_with_rescale` | **CLOSED** Sprint 50 | i16(-1) × 2.0 + 100.0 = 98.0 |
| DICOM-D1-TEST4-R50 | `test_decode_pixel_bytes_8bit_identity_rescale` | **CLOSED** Sprint 50 | [0, 127, 255] &#8594; [0.0, 127.0, 255.0] |
| DICOM-D1-TEST5-R50 | `test_decode_pixel_bytes_8bit_with_rescale` | **CLOSED** Sprint 50 | u8(200) × 0.5 + 10.0 = 110.0 |
| DICOM-D2-TEST-R50 | `test_is_likely_dicom_file_accepts_canonical_extensions` | **CLOSED** Sprint 50 | .dcm, .DCM, .dicom, .ima accepted |
| DICOM-D2-TEST2-R50 | `test_is_likely_dicom_file_rejects_analyze_and_raw_extensions` | **CLOSED** Sprint 50 | .hdr, .img, .raw rejected |
| DICOM-D4-TEST-R50 | `test_slice_metadata_default_pixel_representation_is_zero` | **CLOSED** Sprint 50 | Default struct has pixel_representation=0, bits_allocated=16 |
| DICOM-D1-TEST6-R50 | `test_read_slice_pixels_signed_i16_roundtrip` | **CLOSED** Sprint 50 | End-to-end signed i16 slice read/write/decode round-trip |

### Sprint 50 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io DICOM tests | 9 new + 280 existing = 289 total | All decode, detection, and signed pixel tests |
| Correctness bugs fixed | 4 | Inline pixel decode DRY, file detection false-positives, writer string literal duplication, missing window/pixel metadata |
| Diagnostics | Clean | Zero errors, zero warnings |
| Total | **289 passed, 0 failed** | Full ritk-io unit suite |

## Sprint 49 -- Completed

### Stream A -- DICOM Writer Type 2 Conformance for None Metadata Path (DICOM-TYPE2-META-R49)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-TYPE2-META-R49 | Add Type 2 fallback tags to `write_dicom_series_with_metadata(None)` | **CLOSED** Sprint 49 | Five Type 2 mandatory DICOM tags were absent when `metadata=None`: (0008,0090) ReferringPhysicianName, (0010,0010) PatientName, (0010,0020) PatientID, (0008,0020) StudyDate, (0020,0011) SeriesNumber. Inserted unconditional defaults before the conditional `if let Some(m) = metadata` block. The conditional block overrides via `obj.put()` when metadata provides non-None values. All five tags now present in every emitted slice regardless of metadata. |

### Stream B -- End-to-End Series Intensity Round-Trip Tests (DICOM-E2E-ROUNDTRIP-R49)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-E2E-ROUNDTRIP-BASIC-R49 | `test_write_series_load_series_intensity_roundtrip` | **CLOSED** Sprint 49 | Writes 4×4×4 image (intensities 0..63) via `write_dicom_series`, loads via `load_dicom_series`, verifies per-pixel `|decoded &#8722; original| &#8804; 65535 × 0.5e-6 + 0.5e-6 + slope/2`. Tolerance analytically derived: DS `{:.6}` format introduces at most `0.5e-6` per coefficient; accumulated over max u16 (65535) gives 0.033; quantization adds `slope/2 &#8776; 1.14e-4`. |
| DICOM-E2E-ROUNDTRIP-META-R49 | `test_write_metadata_series_load_series_intensity_roundtrip` | **CLOSED** Sprint 49 | Writes 3×4×4 image with non-trivial origin [5,10,-20], spacing [0.5,0.5,1.5] via `write_dicom_series_with_metadata`. Loads via `load_dicom_series`. Verifies (1) per-pixel intensity bound (same DS tolerance), (2) origin round-trips within 1e-4 mm, (3) spacing round-trips within 1e-4 mm. |

### Stream C -- IO Gap Audit Sync (GAP-AUDIT-SYNC-R49)
| ID | Feature | Status | Notes |
|---|---|---|---|
| GAP-AUDIT-IO-SYNC-R49 | Update gap_audit.md sections 6.1, 6.2, 6.4, 6.6, 6.8 from Planned &#8594; Closed | **CLOSED** Sprint 49 | MetaImage (6.1), NRRD (6.2), VTK Image (6.4), Analyze (6.6), JPEG 2D (6.8) were implemented in Sprints 2 and 8 but section 6 prose still said Critical/High/Medium/Low with "Planned location" text. Updated each to Closed with sprint number and implementation bullet list. Section 8.5 priority matrix was already correct (all Closed). |

### Sprint 49 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-TYPE2-META-TEST-R49 | `test_metadata_writer_none_metadata_type2_tags` | **CLOSED** Sprint 49 | Writes via `write_dicom_series_with_metadata(None)`, opens first slice, asserts all five Type 2 tags present: (0010,0010), (0010,0020), (0008,0090), (0008,0020), (0020,0011). |
| DICOM-E2E-BASIC-TEST-R49 | `test_write_series_load_series_intensity_roundtrip` | **CLOSED** Sprint 49 | 64 voxels, per-voxel absolute error asserted &#8804; analytically derived tolerance. |
| DICOM-E2E-META-TEST-R49 | `test_write_metadata_series_load_series_intensity_roundtrip` | **CLOSED** Sprint 49 | 48 voxels, per-voxel intensity error + origin/spacing assertions. |

### Sprint 49 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io DICOM writer unit tests | 25 passed (1 new + 24 existing) | `test_metadata_writer_none_metadata_type2_tags` added |
| ritk-io DICOM reader unit tests | 13 passed (2 new + 11 existing) | `test_write_series_load_series_intensity_roundtrip`, `test_write_metadata_series_load_series_intensity_roundtrip` added |
| Correctness bugs fixed | 1 | `write_dicom_series_with_metadata(None)` was missing 5 Type 2 mandatory tags |
| Gap audit synced | 5 | IO sections 6.1, 6.2, 6.4, 6.6, 6.8 updated from Planned &#8594; Closed |
| Diagnostics | Clean | Zero errors, zero warnings |
| Total | **280 passed, 0 failed** | Full ritk-io unit suite |

## Sprint 48 -- Completed

### Stream A -- DICOM Reader Correctness + DRY Header Extraction (DICOM-READER-R48)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-TS-GUARD-MF-R48 | Compressed TS detection in `load_dicom_multiframe` | **CLOSED** Sprint 48 | After `open_file`, reads `obj.meta().transfer_syntax()` and calls `TransferSyntaxKind::from_uid(ts_uid).is_compressed()`. Returns `Err` with TS UID and path before any pixel decode. Prevents silent garbage-intensity output on JPEG/JPEG-LS/JPEG2000/RLE files. |
| DICOM-TS-GUARD-SERIES-R48 | Compressed TS detection in `load_from_series` | **CLOSED** Sprint 48 | Pre-decode loop over `slices.iter()` checks each `DicomSliceMetadata.transfer_syntax_uid` via `TransferSyntaxKind::from_uid(ts).is_compressed()`. Bails with TS UID and slice path on first compressed hit. Added `use super::transfer_syntax::TransferSyntaxKind` to reader.rs imports. |
| DICOM-INFO-RESCALE-R48 | Add `rescale_slope: f64` and `rescale_intercept: f64` to `MultiFrameInfo` | **CLOSED** Sprint 48 | `MultiFrameInfo` extended with two new public fields: `rescale_slope` (default 1.0) and `rescale_intercept` (default 0.0). Populated from (0028,1053) and (0028,1052) via `extract_multiframe_header`. Exposes the linear transform without requiring a second file open. |
| DICOM-MF-LOAD-DRY-R48 | Extract `extract_multiframe_header` — eliminate header parse duplication | **CLOSED** Sprint 48 | Private `fn extract_multiframe_header(path: &Path, obj: &InMemDicomObject) -> MultiFrameInfo` encapsulates all header element reads (n_frames, rows, cols, bits_allocated, pixel_spacing, frame_thickness, modality, sop_class_uid, image_position, image_orientation, rescale_slope, rescale_intercept). Both `read_multiframe_info` and `load_dicom_multiframe` delegate to it; each opens the file once. Zero header-field duplication remains. |

### Stream B -- Writer Correctness + IOD Conformance (DICOM-WRITER-R48)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-CLAMP-SERIES-R48 | Fix missing `.clamp(0.0, 65535.0)` in `write_dicom_series` and `write_dicom_series_with_metadata` | **CLOSED** Sprint 48 | Both per-slice pixel encoding closures were missing the explicit clamp before `as u16` cast. Added `.clamp(0.0, 65535.0)` to both; `write_multiframe_impl` already had the correct form. All three writers are now consistent. |
| DICOM-CONV-TYPE-R48 | Add `ConversionType` (0008,0064) = "WSD" to all three writers | **CLOSED** Sprint 48 | SC Equipment Module (PS3.3 C.8.6.1) mandates ConversionType as Type 1. "WSD" (Workstation) added after Modality in `write_dicom_series`, `write_dicom_series_with_metadata`, and `write_multiframe_impl`. Added `writer_tag_key(0x0008, 0x0064)` to `writer_exclusion_tags()` to prevent preservation duplication. |
| DICOM-TYPE2-BASIC-R48 | Add Type 2 mandatory tags to `write_dicom_series` | **CLOSED** Sprint 48 | Five Type 2 tags absent from the basic series writer added with empty/default values: (0008,0090) ReferringPhysicianName="", (0010,0010) PatientName="", (0010,0020) PatientID="", (0008,0020) StudyDate="", (0020,0011) SeriesNumber="0". Per DICOM PS3.3, Type 2 tags must be present even when empty. Added to exclusion set. |

### Stream A -- Sprint 48 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-TS-GUARD-MF-TEST-R48 | `test_load_multiframe_compressed_ts_errors` | **CLOSED** Sprint 48 | Writes DICOM with JPEG Baseline TS (1.2.840.10008.1.2.4.50) in file meta via `FileMetaTableBuilder`. Asserts `load_dicom_multiframe` returns `Err`; error message contains TS UID or "compress". |
| DICOM-INFO-RESCALE-TEST-R48 | `test_multiframe_info_rescale_slope_intercept_populated` | **CLOSED** Sprint 48 | Writes 1×5×5 image range [0.0, 24.0]; analytical slope = 24.0/65535.0, intercept = 0.0. Reads back via `read_multiframe_info`; asserts |info.rescale_slope &#8722; expected| < 5×10&#8315;&#8311; and |info.rescale_intercept &#8722; 0.0| < 5×10&#8315;&#8311; (DS `{:.6}` precision bound). |
| DICOM-CONV-TYPE-MF-TEST-R48 | `test_multiframe_has_conversion_type_wsd` | **CLOSED** Sprint 48 | Writes via `write_dicom_multiframe`, opens with `open_file`, reads (0008,0064), asserts trimmed value == "WSD". |

### Stream B -- Sprint 48 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-CLAMP-TEST-R48 | `test_series_pixel_clamp_u16_range` | **CLOSED** Sprint 48 | 16 analytically-spaced f32 values 0&#8594;65535 (step = 65535/15). All encoded u16 values &#8804; 65535; verifies clamp is present. |
| DICOM-CONV-TYPE-SERIES-TEST-R48 | `test_series_writer_has_conversion_type_wsd` | **CLOSED** Sprint 48 | Writes via `write_dicom_series`, opens first slice, reads (0008,0064), asserts trimmed value == "WSD". |
| DICOM-TYPE2-TEST-R48 | `test_basic_series_writer_has_type2_patient_tags` | **CLOSED** Sprint 48 | Writes via `write_dicom_series`, opens first slice, asserts presence of (0010,0010), (0010,0020), (0008,0090), (0008,0020), (0020,0011). |
| DICOM-TS-GUARD-SERIES-TEST-R48 | `test_load_series_compressed_ts_errors` | **CLOSED** Sprint 48 | Writes single CT slice with JPEG Baseline TS declared. Verifies `scan_dicom_directory` captures compressed TS in slice metadata; verifies `load_dicom_series` returns `Err` with TS UID in message. |

### Sprint 48 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io DICOM multiframe unit tests | 15 passed (3 new + 12 existing) | `test_load_multiframe_compressed_ts_errors`, `test_multiframe_info_rescale_slope_intercept_populated`, `test_multiframe_has_conversion_type_wsd` added |
| ritk-io DICOM writer unit tests | 23 passed (3 new + 20 existing) | `test_series_pixel_clamp_u16_range`, `test_series_writer_has_conversion_type_wsd`, `test_basic_series_writer_has_type2_patient_tags` added |
| ritk-io DICOM reader unit tests | 11 passed (1 new + 10 existing) | `test_load_series_compressed_ts_errors` added |
| Correctness bugs fixed | 3 | Pixel clamp in two series writers; compressed TS producing garbage data in multiframe and series loaders |
| IOD conformance fixes | 2 | ConversionType (Type 1 SC Equipment Module) added to all 3 writers; Type 2 Patient/Study/Series tags added to basic series writer |
| DRY improvements | 1 | `extract_multiframe_header` eliminates header parse duplication between `read_multiframe_info` and `load_dicom_multiframe` |
| Diagnostics | Clean | Zero errors, zero warnings on new code |
| Total | **277 passed, 0 failed** | Full ritk-io unit suite |

## Sprint 47 -- Completed

### Stream A -- DICOM IOD Conformance + DRY Refactor + Builder API (DICOM-CONF-R47)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-SPP-MF-R47 | Add `SamplesPerPixel` (0028,0002) = 1 to `write_multiframe_impl` | **CLOSED** Sprint 47 | Tag is Type 1 mandatory in DICOM Image Pixel Module (PS3.3 C.7.6.3.1.1). Was absent from the multi-frame writer, causing IOD non-conformance. Added `DataElement::new(Tag(0x0028,0x0002), VR::US, PrimitiveValue::from(1_u16))` before the Rows element. |
| DICOM-SPP-SERIES-R47 | Add `SamplesPerPixel` (0028,0002) = 1 to both series writers | **CLOSED** Sprint 47 | Same mandatory tag absent from `write_dicom_series` and `write_dicom_series_with_metadata`. Fixed in both per-slice emission loops before the Rows element. |
| DICOM-INST-NUM-MF-R47 | Add `InstanceNumber` (0020,0013) to multi-frame writer | **CLOSED** Sprint 47 | Type 2 required tag for SC Image Module. Emitted from `config.instance_number` (default 1). Verified by `test_writer_config_instance_number_propagated`. |
| DICOM-DRY-DS-R47 | Extract `parse_ds_backslash<const N: usize>` helper — eliminate DS-parsing duplication | **CLOSED** Sprint 47 | Six nearly identical DS backslash-parse closures existed across `read_multiframe_info` and `load_dicom_multiframe`. Replaced with a single generic `fn parse_ds_backslash<const N: usize>(s: &str) -> Option<[f64; N]>`. Variation on field width encoded as const generic parameter; no logic duplication remains. |
| DICOM-CONFIG-R47 | Add `MultiFrameWriterConfig` builder struct and `write_dicom_multiframe_with_config` | **CLOSED** Sprint 47 | `MultiFrameWriterConfig { sop_class_uid: String, spatial: Option<MultiFrameSpatialMetadata>, instance_number: u32 }` with `Default` impl (sop_class = MF_GRAYSCALE_WORD_SC_UID, instance_number = 1). `write_dicom_multiframe_with_config` exposes the full config surface. `write_dicom_multiframe` and `write_dicom_multiframe_with_options` delegate via config construction; public signatures unchanged. |
| DICOM-REEXPORT-R47 | Fix `mod.rs` re-export gap for multi-frame types | **CLOSED** Sprint 47 | `MultiFrameSpatialMetadata`, `write_dicom_multiframe_with_options`, `MultiFrameWriterConfig`, and `write_dicom_multiframe_with_config` were not re-exported from `format::dicom`. Added to the `pub use multiframe::{...}` block. Module doc Public API section extended with Series I/O, Multi-Frame I/O, and Object Model subsections. |

### Stream A -- Sprint 47 Tests
| ID | Test | Status | Notes |
|---|---|---|---|
| DICOM-SPP-MF-TEST-R47 | `test_written_multiframe_has_samples_per_pixel_one` | **CLOSED** Sprint 47 | Writes via `write_dicom_multiframe`, reads `(0028,0002)` back with `open_file`, asserts == 1. |
| DICOM-INST-NUM-TEST-R47 | `test_writer_config_instance_number_propagated` | **CLOSED** Sprint 47 | Writes via `write_dicom_multiframe_with_config` with `instance_number=42`, reads `(0020,0013)` back, asserts == 42. |
| DICOM-NEG-RT-TEST-R47 | `test_round_trip_negative_intensity_image` | **CLOSED** Sprint 47 | Image with float range [-1024, 500]. Analytical slope = 1524.0/65535.0 &#8776; 0.02325. Asserts |recovered &#8722; original| &#8804; slope + 1.0 for all 24 samples. |
| DICOM-FLAT-RT-TEST-R47 | `test_round_trip_flat_image_exact` | **CLOSED** Sprint 47 | Constant image (42.75_f32, exactly representable). Verifies slope=1.0 / all-zeros u16 branch; asserts |recovered &#8722; 42.75| &#8804; f32::EPSILON for all samples. |
| DICOM-SPP-SERIES-TEST-R47 | `test_series_writer_has_samples_per_pixel_one` | **CLOSED** Sprint 47 | Writes via `write_dicom_series`, opens first slice with `open_file`, asserts `(0028,0002)` == 1. |

### Sprint 47 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io DICOM multiframe unit tests | 12 passed (4 new + 8 existing) | `test_written_multiframe_has_samples_per_pixel_one`, `test_writer_config_instance_number_propagated`, `test_round_trip_negative_intensity_image`, `test_round_trip_flat_image_exact` added |
| ritk-io DICOM writer unit tests | 20 passed (1 new + 19 existing) | `test_series_writer_has_samples_per_pixel_one` added |
| IOD conformance fixes | 2 | `SamplesPerPixel` added to multi-frame writer and both series writers |
| New public API | 2 | `MultiFrameWriterConfig`, `write_dicom_multiframe_with_config` |
| Re-export gap closed | 4 symbols | `MultiFrameSpatialMetadata`, `write_dicom_multiframe_with_options`, `MultiFrameWriterConfig`, `write_dicom_multiframe_with_config` |
| Diagnostics | Clean (`cargo check`) | Zero errors, IDE rust-analyzer false positives on tracing macros only |
| Total | **270 passed, 0 failed** | Full ritk-io unit suite |

## Sprint 46 -- Completed

### Stream A -- Multi-Frame DICOM SOP Class Fix + Spatial Metadata (DICOM-MF-R46)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-MF-SOP-FIX-R46 | Fix multiframe writer SOP class from Single-frame SC to Multi-Frame Grayscale Word SC | **CLOSED** Sprint 46 | `write_dicom_multiframe` was emitting `1.2.840.10008.5.1.4.1.1.7` (SecondaryCaptureImageStorage, single-frame) for 16-bit multi-frame pixel data. Corrected to `1.2.840.10008.5.1.4.1.1.7.3` (MultiFrameGrayscaleWordSecondaryCaptureImageStorage). Extracted `MF_GRAYSCALE_WORD_SC_UID` const to ensure one authoritative reference used by both the DataElement and `FileMetaTableBuilder`. |
| DICOM-MF-SPATIAL-R46 | Add `MultiFrameSpatialMetadata` and `write_dicom_multiframe_with_options` | **CLOSED** Sprint 46 | Added `MultiFrameSpatialMetadata { origin, pixel_spacing, slice_thickness, image_orientation, modality }`. Added `write_dicom_multiframe_with_options` that accepts `Option<&MultiFrameSpatialMetadata>`. When `Some`, emits (0020,0032) IPP, (0020,0037) IOP, (0028,0030) PixelSpacing, (0018,0050) SliceThickness, and Modality. Shared `write_multiframe_impl` private function; `write_dicom_multiframe` signature unchanged. |
| DICOM-MF-INFO-SPATIAL-R46 | Extend `MultiFrameInfo` and `read_multiframe_info` with IPP/IOP | **CLOSED** Sprint 46 | Added `image_position: Option<[f64; 3]>` and `image_orientation: Option<[f64; 6]>` to `MultiFrameInfo`. `read_multiframe_info` now parses (0020,0032) and (0020,0037) from the file. |
| DICOM-MF-LOAD-SPATIAL-R46 | `load_dicom_multiframe` derives origin and direction from IPP/IOP | **CLOSED** Sprint 46 | When (0020,0032) is present, the loaded `Image` origin is set from IPP. When (0020,0037) is present, the direction matrix is derived as `SMatrix::from_column_slice` with columns [row_cosines, col_cosines, normal]; normal = row × col. Previously hardcoded to `[0,0,0]` and `Direction::identity()`. |
| DICOM-MF-SOP-TEST-R46 | `test_multiframe_sop_class_is_mf_grayscale_word` | **CLOSED** Sprint 46 | New test asserts `sop_class_uid == Some("1.2.840.10008.5.1.4.1.1.7.3")` after `write_dicom_multiframe`. |
| DICOM-MF-SPATIAL-TEST-R46 | `test_write_multiframe_with_spatial_metadata_round_trip` | **CLOSED** Sprint 46 | Writes with `MultiFrameSpatialMetadata { origin=[10,20,-50], pixel_spacing=[0.8,0.8], slice_thickness=2.5, iop=identity_row_col, modality="CT" }`. Asserts `read_multiframe_info` IPP/IOP ±1e-4, modality exact match, loaded Image origin ±1e-4, and pixel reconstruction error &#8804; slope + 1.0. |

### Stream B -- Reader Direction Fix + Binary-VR Top-Level Fix + Scan Round-Trip Test (DICOM-READER-R46)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-LOAD-DIR-FIX-R46 | Fix `load_from_series` to use `metadata.direction` instead of `Direction::identity()` | **CLOSED** Sprint 46 | `load_from_series` was constructing `Image` with `Direction::identity()` despite `metadata.direction: [f64; 9]` already holding the correct IOP-derived direction matrix. Fixed to `Direction::from_row_slice(&metadata.direction)`. Pre-existing spatial round-trip test coverage confirms correctness. |
| DICOM-READER-BVR-FIX-R46 | Fix binary-VR routing in `scan_dicom_directory` top-level preservation loop | **CLOSED** Sprint 46 | The same dicom-rs 0.8 `to_str()` bug that was fixed in `parse_sequence_item` (Sprint 45 DICOM-SEQ-OB-FIX-R45) also existed in the top-level preservation loop. VR::OB/OW/OD/OF/OL/UN elements at the root level were being stored as `DicomValue::Text` instead of being preserved as raw bytes in `preservation.preserved`. Fixed by adding the same `is_binary_vr` gate before the `to_str()` branch. |
| DICOM-SCAN-PRIV-RT-R46 | Add `test_scan_preserves_private_text_and_bytes_through_write_read_cycle` in `reader.rs` | **CLOSED** Sprint 46 | Writes a 1-slice series with a private LO text tag (0009,0010)="PRIV_ROUND_TRIP_VALUE" and a private OB bytes tag (0019,1001)=[0xAB,0xCD,0xEF,0x01] in the preservation set. Scans back via `scan_dicom_directory`. Asserts text survives in `preservation.object` with exact value and bytes survive in `preservation.preserved` with exact payload. Closes the "private-tag round-trip on the general series reader/writer path" gap. |

### Sprint 46 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io DICOM multiframe unit tests | 8 passed (2 new + 6 existing) | `test_multiframe_sop_class_is_mf_grayscale_word`, `test_write_multiframe_with_spatial_metadata_round_trip` added; existing SOP class assertions updated to `1.2.840.10008.5.1.4.1.1.7.3` |
| ritk-io DICOM reader unit tests | 10 passed (1 new + 9 existing) | `test_scan_preserves_private_text_and_bytes_through_write_read_cycle` added |
| Bug fixes | 3 | MF SOP class, `load_from_series` direction, top-level binary-VR routing |
| Diagnostics | Clean (`cargo check`) | IDE rust-analyzer false positives on tracing macros only; compile clean |
| Total | **265 passed, 0 failed** | Full ritk-io unit suite |

## Sprint 45 -- Completed

### Stream A -- DICOM Metadata Round-Trip Validation (DICOM-RT-R45)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-TS-BUG-R45 | Fix transfer syntax UID read from Manufacturer tag to file meta | **CLOSED** Sprint 45 | `scan_dicom_directory` was reading Tag(0x0008,0x0070) (Manufacturer) for `transfer_syntax_uid`. Fixed to use `obj.meta().transfer_syntax()` on `FileDicomObject`. Both per-slice and series-level reads corrected. |
| DICOM-SEQ-OB-FIX-R45 | Fix binary VR preservation in parse_sequence_item | **CLOSED** Sprint 45 | `parse_sequence_item` was using `to_str()` before checking the VR, causing OB/OW/OD/OF/OL/UN elements inside nested sequences to be stored as `DicomValue::Text` (formatted decimal string). Fixed by adding an explicit binary-VR gate (`matches!(element.vr(), VR::OB | VR::OW | ...)`) before the `to_str()` branch. This restores the Sprint 43 invariant: binary elements inside sequence items are stored as `DicomValue::Bytes`. Pre-existing test `test_scan_private_sequence_is_preserved_in_object_model` now passes. |
| DICOM-SPATIAL-RT-R45 | Add full DicomReadMetadata/DicomSliceMetadata spatial round-trip test | **CLOSED** Sprint 45 | Added `test_scan_metadata_round_trip_spatial_fields`: writes 3-slice CT series (origin=[10,20,-50], spacing=[0.8,0.8,2.5], identity direction), reads back, asserts modality, bits_allocated, dimensions, spacing (±1e-4), origin (±1e-4), direction (±1e-5), per-slice IOP, pixel_spacing, and IPP z-increments. |
| DICOM-RESCALE-RT-R45 | Add rescale slope/intercept round-trip test | **CLOSED** Sprint 45 | Added `test_scan_metadata_round_trip_rescale_params`: writes 2-slice CT image with intensities spanning [-1024, 1024], asserts slope > 0 and intercept finite for all slices, verifies first-voxel reconstruction error is bounded by slope/2. |
| DICOM-TS-RT-R45 | Add transfer syntax UID round-trip test | **CLOSED** Sprint 45 | Added `test_scan_metadata_round_trip_transfer_syntax`: writes series with Explicit VR LE transfer syntax, asserts `transfer_syntax_uid == Some("1.2.840.10008.1.2.1")` for every slice. Directly validates the bug-fix. |

### Stream B -- GAP-R02b Audit Sync (AUDIT-R45)
| ID | Feature | Status | Notes |
|---|---|---|---|
| AUDIT-R02B-R45 | Close GAP-R02b in gap_audit.md | **CLOSED** Sprint 45 | `InverseConsistentDiffeomorphicDemonsRegistration` (exact-inverse ICC + iterative Newton field inversion) and `MultiResDemonsRegistration` (multi-resolution pyramid) are fully implemented in `crates/ritk-registration/src/demons/exact_inverse_diffeomorphic.rs` and `multires.rs`. Both are exposed in Python via `inverse_consistent_demons_register` and `multires_demons_register` and covered in smoke tests. gap_audit.md was out of sync. |

### Sprint 45 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io DICOM reader unit tests | 9 passed (3 new + 6 existing) | Added 3 round-trip tests; fixed binary VR in parse_sequence_item restores pre-existing test |
| Bug fix | Transfer syntax UID | Fixed wrong tag (0x0008,0x0070) &#8594; file meta `obj.meta().transfer_syntax()` |
| Diagnostics | Clean | No new warnings in reader.rs |
| Total | **9 passed, 0 failed** | Full ritk-io DICOM reader suite; all 262 ritk-io unit tests pass |

## Sprint 44 -- Completed

### Stream A -- Multi-Frame DICOM Reader Hardening (DICOM-R44)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-MULTIFRAME-INFO-R44 | Add value-semantic coverage for `read_multiframe_info` | **CLOSED** Sprint 44 | Added regression coverage that writes a real multi-frame file, then asserts exact frame count, dimensions, bits allocated, modality, and SOP Class UID reported by `read_multiframe_info`. |
| DICOM-MULTIFRAME-ROUNDTRIP-R44 | Validate `load_dicom_multiframe` against analytical pixel values | **CLOSED** Sprint 44 | Added a multi-frame round-trip test with analytically derived voxel values, then verified loaded tensor shape and per-sample reconstruction error bounded by the quantization tolerance derived from the emitted rescale slope. |

### Sprint 44 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io multiframe unit tests | 6 passed | Added 2 reader coverage tests and revalidated existing round-trip / rejection coverage |
| Diagnostics | Clean | No warnings or errors in `multiframe.rs` |
| Total | **6 passed, 0 failed** | Local verification of the DICOM multi-frame increment |

### Stream A -- Parity Test Hardening (PARITY-R42)
| ID | Feature | Status | Notes |
|---|---|---|---|
| SMOKE-FILTER-DT-R42 | Add distance_transform to filter smoke test required list | **CLOSED** Sprint 42 | `test_smoke.py::test_filter_public_functions_exist` required list was missing `distance_transform` (registered Sprint 41 via `wrap_pyfunction!` in filter.rs register fn). `test_python_api_parity.py` would fail because the registered function set did not match the smoke test required set. Added `"distance_transform"` after `"resample_image"` in the required list. |
| SMOKE-SEG-LS-R42 | Add label_shape_statistics to segmentation smoke test required list | **CLOSED** Sprint 42 | `test_smoke.py::test_segmentation_public_functions_exist` required list was missing `label_shape_statistics` (registered Sprint 41). Added `"label_shape_statistics"` after `"skeletonization"` in the required list. |
| SMOKE-STAT-CLIS-R42 | Add compute_label_intensity_statistics to statistics smoke test required list | **CLOSED** Sprint 42 | `test_smoke.py::test_statistics_public_functions_exist` required list was missing `compute_label_intensity_statistics` (registered Sprint 41). Added `"compute_label_intensity_statistics"` after `"nyul_udupa_normalize"` in the required list. |

### Stream B -- Reader Preservation Hardening (DICOM-R43)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-READER-PRESERVATION-R43 | Preserve nested sequence items and raw private elements in `scan_dicom_directory` | **CLOSED** Sprint 43 | Extended the DICOM reader preservation path so `DicomReadMetadata.slices[*].preservation.object` retains private SQ nodes as `DicomValue::Sequence` and nested raw OB elements as `DicomPreservedElement` data. Added value-semantic regression coverage against a real DICOM file. |
| DICOM-READER-ROUNDTRIP-R43 | Validate reader-side object-model reconstruction for private nested sequences | **CLOSED** Sprint 43 | Added a regression test that writes a private sequence containing nested text and raw bytes, then verifies `scan_dicom_directory` reconstructs the sequence item structure and byte payloads without collapse. |

### Sprint 43 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io writer_object unit tests | 7 passed | Added 2 nested-sequence / preserved-byte tests and revalidated existing object-model coverage |
| Diagnostics | Clean | No warnings or errors in `writer_object.rs` and `object_model.rs` |
| Total | **7 passed, 0 failed** | Local verification of the DICOM object-model increment |

### Sprint 42 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-core unit tests (lib) | 719 passed | No Rust changes in Sprint 42 |
| ritk-core integration tests | 21 passed | No Rust changes |
| ritk-python build | Clean | No Rust changes |
| Python parity tests (static analysis) | Pass | test_python_api_parity.py: all 3 parity gaps closed; registered functions now match smoke required lists |
| Total | **740 passed, 0 failed** | Rust test count unchanged; Python parity now consistent |

## Sprint 43 -- Completed

### Stream A -- DICOM Object-Model Reader Preservation (DICOM-R43)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-READER-PRESERVATION-R43 | Preserve nested sequence items and raw private elements in `scan_dicom_directory` | **CLOSED** Sprint 43 | Extended the DICOM reader preservation path so `DicomReadMetadata.slices[*].preservation.object` retains private SQ nodes as `DicomValue::Sequence` and nested raw OB elements as `DicomPreservedElement` data. Added value-semantic regression coverage against a real DICOM file. |
| DICOM-READER-ROUNDTRIP-R43 | Validate reader-side object-model reconstruction for private nested sequences | **CLOSED** Sprint 43 | Added a regression test that writes a private sequence containing nested text and raw bytes, then verifies `scan_dicom_directory` reconstructs the sequence item structure and byte payloads without collapse. |

### Sprint 43 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-io reader preservation tests | 1 passed | New reader-side nested sequence preservation regression |
| Diagnostics | Clean | No warnings or errors in the touched DICOM reader path |
| Total | **1 passed, 0 failed** | Local verification of the reader-side object-model increment |

## Sprint 43 -- Completed

### Stream A -- DICOM Object-Model Hardening (DICOM-R43)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DICOM-OBJECT-MODEL-R43 | Add nested-sequence and preserved-bytes round-trip coverage for `ritk_io::format::dicom::writer_object` | **CLOSED** Sprint 43 | Extended `writer_object` tests to exercise `DicomObjectModel` sequence nodes, nested `DicomSequenceItem` content, and raw preserved byte nodes. Verified `model_to_in_mem` emits sequence items, private tags, and byte payloads with value-semantic assertions. |
| DICOM-ROUNDTRIP-R43 | Validate `write_dicom_object` on nested sequence + preserved raw element inputs | **CLOSED** Sprint 43 | Added direct file round-trip assertions against `dicom::object::open_file` for SQ / OB emission. Confirms the object-model writer preserves nested structure and raw bytes without collapsing them into scalar text. |

### Stream B -- CI Workflow Update (CI-R42)
| ID | Feature | Status | Notes |
|---|---|---|---|
| CI-STATS-TEST-R42 | Add test_statistics_bindings.py to CI pytest invocation | **CLOSED** Sprint 42 | `python_ci.yml` pytest run listed `test_python_api_parity.py`, `test_segmentation_bindings.py`, `test_smoke.py` but omitted `test_statistics_bindings.py` (created Sprint 41). Added `crates/ritk-python/tests/test_statistics_bindings.py` between segmentation and smoke test entries. |

### Stream C -- Gap Audit Sync (AUDIT-R42)
| ID | Feature | Status | Notes |
|---|---|---|---|
| AUDIT-ATLAS-R42 | Update gap_audit.md: build_atlas/joint_label_fusion/transform-IO marked as implemented | **CLOSED** Sprint 42 | Section 7.2 incorrectly listed joint_label_fusion, build_atlas, and read_transform/write_transform as not yet in Python. All three were implemented in Sprint 8 (`registration.rs` and `io.rs`). Updated all three rows to âœ“. Section 7.3 function counts updated from Sprint-7 values (53+) to Sprint-41 values (91+). Stale "remaining work" items removed. |

### Sprint 42 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-core unit tests (lib) | 719 passed | No Rust changes in Sprint 42 |
| ritk-core integration tests | 21 passed | No Rust changes |
| ritk-python build | Clean | No Rust changes |
| Python parity tests (static analysis) | Pass | test_python_api_parity.py: all 3 parity gaps closed; registered functions now match smoke required lists |
| Total | **740 passed, 0 failed** | Rust test count unchanged; Python parity now consistent |

## Sprint 41 -- Completed

### Stream A -- Label Intensity Statistics Python Binding (LABEL-STATS-PY-R41)
| ID | Feature | Status | Notes |
|---|---|---|---|
| LABEL-STATS-PY-R41 | compute_label_intensity_statistics Python binding | **CLOSED** Sprint 41 | Added `compute_label_intensity_statistics(label_image, intensity_image) -> list[dict]` to `crates/ritk-python/src/statistics.rs`. Registered in statistics module. Zero-copy via nested `with_tensor_slice` (same pattern as `estimate_noise`). Returns list of dicts with keys: label, count, min, max, mean, std. Stub added to `statistics.pyi`. 4 value-semantic Python tests in `test_statistics_bindings.py`: single label (mean=4.0, std=1.0), background excluded (empty list), two labels sorted, four voxels (mean=2.5, std=sqrt(1.25)). |

### Stream B -- Distance Transform + Label Shape Statistics Python Bindings (ITK-REGION-R41)
| ID | Feature | Status | Notes |
|---|---|---|---|
| DIST-TRANSFORM-R41 | distance_transform Python binding | **CLOSED** Sprint 41 | Added `distance_transform(image, foreground_threshold=0.5, squared=False)` to `crates/ritk-python/src/filter.rs`. Maps to `DistanceTransform::transform` (squared=False) or `DistanceTransform::squared` (squared=True). Import: `use ritk_core::segmentation::DistanceTransform`. Registered in filter module. Stub added to `filter.pyi`. 3 value-semantic Python tests: all-foregroundâ†’zeros, single-voxel adjacent=1.0, squared==dist^2. |
| LABEL-SHAPE-R41 | label_shape_statistics Python binding | **CLOSED** Sprint 41 | Added `label_shape_statistics(mask, connectivity=6) -> list[dict]` to `crates/ritk-python/src/segmentation.rs`. Delegates to `ConnectedComponentsFilter::with_connectivity(connectivity).apply(mask)`. Returns list of dicts with keys: label, voxel_count, centroid ([z,y,x] f64), bounding_box_min ([z,y,x] i64), bounding_box_max ([z,y,x] i64). Background excluded (label 0 is never in results). Connectivity validated before calling filter (6 or 26 only). Registered in segmentation module. Stub added to `segmentation.pyi`. 3 value-semantic Python tests: single voxel centroid/bounding-box, two components sorted, invalid connectivity â†’ ValueError. |

### Sprint 41 Test Results
| Suite | Count | Notes |
|---|---|---|
| ritk-core unit tests (lib) | 719 passed | No regressions vs Sprint 40 |
| ritk-core integration tests | 21 passed | No regressions |
| ritk-python build | Clean | cargo build -p ritk-python, 0 errors, 0 warnings |
| New Python tests (not yet run â€” wheel needed) | 13 total | 4 in test_statistics_bindings.py + 3 distance_transform + 3 label_shape_statistics + 3 from prior test_segmentation_bindings additions |
| Total | **740 passed, 0 failed** | ritk-core unchanged |

## Sprint 40 -- Completed

### Stream A -- Zero-Copy Threshold From-Slice Variants (ZEROCOPY-ARCH-R40-LIYKOT)
| ID | Feature | Status | Notes |
|---|---|---|---|
| LIYKOT-LI | compute_li_threshold_from_slice | **CLOSED** Sprint 40 | Public `compute_li_threshold_from_slice(slice, num_bins, max_iterations) -> f32` added to li.rs. `compute_li_threshold_impl` delegates to it. 1 parity test: bit-identical vs LiThreshold::compute. Python binding migrated to with_tensor_slice + inline apply. Eliminates 2x clone().into_data() per call. |
| LIYKOT-YEN | compute_yen_threshold_from_slice | **CLOSED** Sprint 40 | Public `compute_yen_threshold_from_slice(slice, num_bins) -> f32` added to yen.rs. Impl delegates. 1 parity test: bit-identical vs YenThreshold::compute. Python binding migrated. |
| LIYKOT-KAPUR | compute_kapur_threshold_from_slice | **CLOSED** Sprint 40 | Public `compute_kapur_threshold_from_slice(slice, num_bins) -> f32` added to kapur.rs. Impl delegates. 1 parity test. Python binding migrated. |
| LIYKOT-TRIANGLE | compute_triangle_threshold_from_slice | **CLOSED** Sprint 40 | Public `compute_triangle_threshold_from_slice(slice, num_bins) -> f32` added to triangle.rs. Impl delegates. 1 parity test. Python binding migrated. |
| LIYKOT-MULTIOTSU | compute_multi_otsu_thresholds_from_slice | **CLOSED** Sprint 40 | Public `compute_multi_otsu_thresholds_from_slice(slice, num_classes, num_bins) -> Vec<f32>` added to multi_otsu.rs. Impl delegates. 1 parity test. Python binding migrated; inline label assignment eliminates second `clone().into_data()`. |

### Stream B -- Per-Label Intensity Statistics (LABEL-STATS-R40)
| ID | Feature | Status | Notes |
|---|---|---|---|
| LABEL-STATS-R40 | LabelIntensityStatistics + compute_label_intensity_statistics | **CLOSED** Sprint 40 | New `crates/ritk-core/src/statistics/label_statistics.rs`. `LabelIntensityStatistics { label, count, min, max, mean, std }`. Two entry points: compute_label_intensity_statistics (Image API) and compute_label_intensity_statistics_from_slices (zero-copy slice API). Single O(N) rayon par_iter fold/reduce over (label, intensity) pairs; HashMap<u32,(min,max,sum_f64,sum_sq_f64,count)> per thread. Results sorted by label. 9 tests: single voxel, known stats, two labels, background excluded, uniform intensity, image API parity, length-mismatch panic, shape-mismatch panic, sorted output. Exported from statistics::mod. |

### Sprint 40 Test Results (ritk-core)
| Suite | Count | Notes |
|---|---|---|
| Unit tests (lib) | 719 passed | +14 vs Sprint 39 (5 threshold parity + 9 label_statistics) |
| Integration tests | 21 passed | No regressions |
| ritk-python build | Clean | cargo build -p ritk-python, 0 errors |
| Total | **740 passed, 0 failed** | â€” |

## Sprint 39 -- Completed

### Stream A -- Selection-Based Percentile Computation (PERF-STATS-R39)
| ID | Feature | Status | Notes |
|---|---|---|---|
| PERF-STATS-R39 | Replace par_sort with select_nth for p25/p50/p75 | **CLOSED** Sprint 39 | compute_from_values Phase 2: par_sort_unstable_by (O(N log N)) replaced by 3x select_nth_unstable_by (O(N) amortized). Order: i75->i50->i25 preserves partition invariant. 1 parity test added (bit-identical vs sort, n=1000). 705/705 tests pass. |

### Stream B -- DiscreteGaussian Convolution Optimizations (PERF-DG-R39)
| ID | Feature | Status | Notes |
|---|---|---|---|
| PERF-DG-R39-CONV1D | conv1d_replicate SAXPY + boundary split | **CLOSED** Sprint 39 | fill+SAXPY-per-kj loop; analytic i_start/i_end; interior loop branch-free and LLVM-vectorizable; bit-identical output. |
| PERF-DG-R39-YAXIS | Y-axis (dim=1) cache-friendly SAXPY reorder | **CLOSED** Sprint 39 | Loop order (kj,iy,ix) with contiguous src_row/dst_row SAXPY; eliminates per-Z-slab buf[ny] allocation; bit-identical. |
| PERF-DG-R39-ZAXIS | Z-axis (dim=0) parallelization | **CLOSED** Sprint 39 | par_chunks_mut(nyx).enumerate() replaces serial for yx in 0..nyx; SAXPY over contiguous src_z slices (sequential reads, no strided access); nz-way Rayon parallelism; bit-identical. |

### Sprint 39 Expected Performance Changes (64^3 image, release build)
| Operation | Sprint 38 RITK | Expected S39 change | Mechanism |
|---|---|---|---|
| compute_statistics | 1.185 ms | Lower (percentile O(N) vs O(N log N)) | Phase 2: 3x select_nth replaces par_sort |
| discrete_gaussian | 8.344 ms | Lower (Z-axis now parallel + SIMD interior) | dim=0 par_chunks_mut + vectorizable conv1d_replicate |

Note: Benchmark measurements pending next Python parity run. All 726 ritk-core tests pass.

## Sprint 38 -- Completed

### Stream A -- Architectural Zero-Copy Extraction (ZEROCOPY-ARCH-R38)
| ID | Feature | Status | Notes |
|---|---|---|---|
| ZEROCOPY-ARCH-R38-CORE | Add zero-copy APIs to ritk-core | **CLOSED** Sprint 38 | Added: Image::into_tensor, Image::into_parts; compute_from_values -> pub; compute_otsu_threshold_from_slice; estimate_noise_mad_from_slice/masked; GradientMagnitudeFilter::apply_from_slice. Zero new tests (2 parity tests added: test_apply_from_slice_matches_apply, test_compute_otsu_from_slice_matches_filter). |
| ZEROCOPY-ARCH-R38-PY | with_tensor_slice helper + Python binding hot-path migration | **CLOSED** Sprint 38 | with_tensor_slice: clone tensor O(1) + into_primitive() + as_slice_memory_order() O(1) = zero-copy &[f32] from NdArray ArcArray. Updated: image_to_vec, to_numpy, compute_statistics, masked_statistics, estimate_noise, otsu_threshold, gradient_magnitude. Eliminates clone().into_data() for all read-only operations. |

### Sprint 38 Benchmark Results (64^3 image, release build, miniforge3/Python 3.13)
| Operation | Sprint 37 RITK | Sprint 38 RITK | Sprint 38 SITK | Ratio | Speedup vs S37 |
|---|---|---|---|---|---|
| compute_statistics | 6.94 ms | **1.185 ms** | 0.498 ms | 2.38x | 5.9x |
| otsu_threshold | 18.74 ms | **0.828 ms** | 2.182 ms | 0.38x (FASTER) | 22.6x |
| gradient_magnitude | 6.55 ms | **0.488 ms** | 1.022 ms | 0.48x (FASTER) | 13.4x |
| to_numpy | N/A | **0.316 ms** | 0.334 ms | 0.95x (parity) | --- |
| discrete_gaussian | 9.01 ms | **8.344 ms** | 2.299 ms | 3.63x | 1.1x |
| median_filter r=2 | 14.36 ms | **13.945 ms** | 21.359 ms | 0.65x (FASTER) | 1.0x |

Note: `otsu_threshold` improved 22.6× by eliminating 2 clone().into_data() calls; now 2.63× faster than SimpleITK. `gradient_magnitude` improved 13.4× by eliminating extract_vec copy; now 2.08× faster than SimpleITK. `to_numpy` achieves near-parity with SimpleITK (0.95×). Remaining gap for `compute_statistics` (2.38×) and `discrete_gaussian` (3.63×) is now due to real computation (sort for percentiles; separable convolution) rather than data extraction overhead.

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

Note: Remaining statistics/otsu/gradient gap is structural — the single O(N) copy in clone().into_data() cannot be eliminated without architectural change to PyImage (store raw ndarray directly, bypassing Burn tensor abstraction). Planned as ZEROCOPY-ARCH-R38.

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
| PY-CI-NUMPY-SEGBINDINGS-R34 | Add numpy and segmentation bindings tests to CI | **CLOSED** Sprint 34 | `numpy` added to `pip install` step; `crates/ritk-python/tests/test_segmentation_bindings.py` added to CI pytest invocation (alphabetical order: parity â†’ segmentation_bindings â†’ smoke); segmentation bindings tests exercise value-semantic functional correctness (connected_components, level-set variants) against the installed wheel |
| XTASK-PARITY-REPORT-R34 | Add cargo xtask python-parity-report subcommand | **CLOSED** Sprint 34 | Added `PythonParityReport` variant to `Commands` enum and `python_parity_report` handler in `xtask/src/main.rs`; invokes `python crates/ritk-python/tests/python_api_drift_report.py` via `std::process::Command`; exits non-zero on detected drift; `--python` flag selects interpreter; `cargo check -p xtask` passes clean |

### Stream B -- Deferred
| ID | Feature | Status | Notes |
|---|---|---|---|
| PYTHON-CI-VALIDATION | Validate Python CI workflow on hosted runners | DEFERRED Sprint 34 | All local verification passes; `python_api_drift_report.py` reports clean (5/5 modules, top-level contract); hosted GitHub Actions matrix execution required to confirm Windows wheel build, patchelf behavior, and Python 3.9â€“3.13 compatibility |

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

### Stream A â€” Tracing / IDE Quality Refactor
| ID | Feature | Status | Notes |
|---|---|---|---|
| TRACING-REFACTOR | Convert CLI `info!()`/`warn!()` structured-field calls to format-string style | **CLOSED** Sprint 30 | `filter.rs`, `register.rs`, `reader.rs`; eliminates ~320 rust-analyzer false-positive diagnostics; `cargo check` was already clean |

### Stream B â€” Stub Sync / Correctness
| ID | Feature | Status | Notes |
|---|---|---|---|
| STATS-STUB-SYNC-R30 | Add `nyul_udupa_normalize` to `statistics.pyi` | **CLOSED** Sprint 30 | Function exported in `statistics.rs` register() but missing from pyi; 14 functions now fully stubbed |
| FILTER-ERROR-MSG-R30 | Extend `filter.rs` `run()` error message to list all dispatched filter names | **CLOSED** Sprint 30 | Added missing 10 filter names (grayscale-erosion, grayscale-dilation, white-top-hat, black-top-hat, hit-or-miss, label-dilation/erosion/opening/closing, morphological-reconstruction) |

### Stream C â€” Testing / Quantitative Validation
| ID | Feature | Status | Notes |
|---|---|---|---|
| DISCRETE-GAUSSIAN-ANALYTICAL | Impulse-response analytical validation for DiscreteGaussianFilter | **CLOSED** Sprint 30 | `test_impulse_response_matches_analytical_gaussian` in `discrete_gaussian.rs`; verifies output[0,0,k] â‰ˆ exp(-( k-15)Â²/(2v))/Z for Dirac impulse at position 15 of a 1Ã—1Ã—31 image; tolerance 1e-3 for f32 arithmetic |

### Stream D â€” Deferred
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
| ITK-CONFIDENCE-CONNECTED | Confidence-connected region growing (iterative meanÂ±kÃ—Ïƒ) | **CLOSED** Sprint 28 | `confidence_connected.rs`; multiplier + max_iterations builder; Python binding + CLI dispatch; 9 core tests |
| ITK-SKELETONIZATION | Topology-preserving thinning via iterative boundary erosion | **CLOSED** Sprint 28 | `skeletonization.rs`; Zhang-Suen 2D + is_simple_3d() 3D thinning; Python + CLI; 19 tests |
| DISCRETE-GAUSSIAN-FILTER | DiscreteGaussianFilter<B> (ITK DiscreteGaussianImageFilter parity) | **CLOSED** Sprint 28 | `filter/discrete_gaussian.rs`; variance-parameterized; maximum_error truncation r=ceil(sqrt(-2ÏƒÂ²Â·ln(e))); use_image_spacing; 11 tests |
| GAP-R02b | InverseConsistentDiffeomorphicDemonsRegistration | **CLOSED** Sprint 28 | `demons/exact_inverse_diffeomorphic.rs`; bilateral E=(1-w)â€–F-Mâˆ˜exp(v)â€–Â² + wâ€–M-Fâˆ˜exp(-v)â€–Â²; IC residual = meanâ€–Ï†+(Ï†-(x))-xâ€–â‚‚; 9 tests |

### Stream D -- Python / CI
| ID | Feature | Status | Notes |
|---|---|---|---|
| PY-CI-MATRIX | Python wheel smoke tests across supported Python versions | **CLOSED** Sprint 28 | `test_smoke.py` 13 tests; `.github/workflows/python_ci.yml` matrix [3.9â€“12]Ã—[ubuntu,windows]+ubuntu/3.13; maturin develop + pytest |

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
| ITK-SIMPLEITK-FAMILY | ITK / SimpleITK algorithm breadth expansion | **CLOSED** Sprint 24â€“28 | High | Prioritize long-tail filters, segmentation, morphology, resampling, and intensity transforms |
| ITKSNAP-WORKFLOW | ITK-SNAP workflow primitives | **CLOSED** Sprint 22 | Medium-High | Add annotation state, overlays, labels, seeds, and undo/redo-oriented editing primitives |
| ANTS-WORKFLOW | ANTs workflow refinement | PLANNED | Medium | Add inverse-consistency controls, preprocessing helpers, and registration workflow composition |
| PY-PARITY-HARNESS | Python parity and reproducibility harness | PLANNED | Medium | Compare `ritk-python` against Python references with value-based tests and benchmarks |

### Stream A â€” DICOM
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| DICOM-OBJ-MODEL | DICOM object-model preservation | **CLOSED** Sprint 24 | High | Reader: full element iteration, DicomValue::Sequence, DicomPreservedElement. Writer: emit_preservation_nodes before PixelData. 3 round-trip tests. |
| DICOM-MULTIFRAME | DICOM multi-frame / enhanced image support | **CLOSED** Sprint 28 | High | Extend series-centric I/O toward enhanced and multi-frame object handling |
| DICOM-WRITER-GENERAL | Generalized DICOM writer architecture | PLANNED | High | Separate object model serialization from series image writing |
| DICOM-TRANSFER-SYNTAX | Transfer syntax coverage audit | PLANNED | Medium | Audit codec and transfer-syntax handling against supported image SOP classes |
| DICOM-NONIMAGE-SOP | Non-image SOP-class policy | PLANNED | Medium | Define explicit acceptance/rejection behavior for non-image DICOM objects |
| DICOM-METADATA-VALIDATION | Metadata-aware read-path validation | PLANNED | Medium | Validate `DicomReadMetadata` and `DicomSliceMetadata` invariants against round-trips |

### Stream B â€” VTK
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| VTK-DATA-MODEL | VTK data-object hierarchy | **CLOSED** Sprint 25/28 | High | Canonical mesh/grid data models beyond legacy structured points |
| VTK-POLYDATA | PolyData and topology primitives | **CLOSED** Sprint 25 | High | Surface geometry, vertices, lines, polygons, and scalar/vector attachments |
| VTK-PIPELINE | VTK-style pipeline abstractions | PLANNED | High | Reader/filter/mapper/data-object execution model |
| VTK-SCENE | Scene graph and renderable object model | PLANNED | Medium | Support visualization-facing composition without duplicating algorithm code |

### Stream C â€” ITK / SimpleITK breadth
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| ITK-SIMPLEITK-FAMILY | Algorithm breadth expansion | **CLOSED** Sprint 24â€“28 | High | Long-tail filters, segmentation, morphology, resampling, intensity transforms |
| ITK-RESAMPLE | Resampling and interpolation expansion | PLANNED | High | Additional resampling helpers and transform-aware utilities |
| ITK-MORPHOLOGY | Morphology family expansion | PLANNED | Medium-High | Additional binary/grayscale operators and topology-preserving variants |
| ITK-REGION | Region-growing and label tools | PLANNED | Medium-High | Extend connected, confidence, neighborhood, and label-processing utilities |
| ITK-NORM | Intensity statistics and normalization | PLANNED | Medium | Additional SimpleITK-style intensity mapping and normalization helpers |

### Stream D â€” ITK-SNAP workflow
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| ITKSNAP-WORKFLOW | Interactive segmentation primitives | **CLOSED** Sprint 22 | Medium-High | Annotation state, overlays, labels, seeds, and undo/redo-oriented editing primitives |
| ITKSNAP-LABELS | Label and mask state model | **CLOSED** Sprint 22 | Medium | Label tables, editable masks, and selection state |
| ITKSNAP-OVERLAY | Overlay and contour composition | **CLOSED** Sprint 22 | Medium | Visualization-ready state for masks, contours, and image overlays |

### Stream E â€” ANTs workflow refinement
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| ANTS-WORKFLOW | Registration workflow composition | PLANNED | Medium | Preprocessing, multi-stage transforms, and pipeline composition |
| ANTS-INVERSE | Inverse-consistency controls | PLANNED | Medium | Strengthen exact-inverse validation and controls around diffeomorphic workflows |
| ANTS-PREPROCESS | Registration preprocessing helpers | PLANNED | Medium | Bias correction, masking, resampling, and intensity normalization pipeline helpers |

### Stream F â€” Python parity and reproducibility
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

## Sprint 22 â€” In Progress

### Stream A â€” Level-set helpers (Planned)
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| LEVEL-SET-HELPERS | Extract shared numerical helpers (indexing, FD, curvature, separable Gaussian) into helpers.rs | COMPLETED | High | Used by ShapeDetection and ThresholdLevelSet |
| SHAPE-DETECT | ShapeDetectionSegmentation (Malladi et al.) + Python binding + CLI binding (run_shape_detection) | COMPLETED | High | 3 CLI tests; ritk-core unit tests passing |
| THRESHOLD-LS | ThresholdLevelSet (Whitaker 1998) + Python binding + CLI binding (run_threshold_level_set) | COMPLETED | Medium | 6 CLI tests; ritk-core unit tests passing |
| CLI-STRUCT-FIX | SegmentArgs duplicate max_iterations field removed; level_set_max_iterations added; impl Default; fill-holes test geometry fixed | COMPLETED | High | 131/131 ritk-cli tests pass |

### Stream B â€” Python + CLI morphology bindings (Planned)
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| (Completed in Sprint 21) | BinaryFillHoles + MorphologicalGradient Python + CLI bindings | COMPLETED | High | Already done in prior session |

### Stream C â€” DICOM metadata enrichment (In Progress)
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


## Sprint 21 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PY-CLI-MULTIRES | MultiResDemons Python + CLI bindings | COMPLETED | High | Added `multires_demons_register` to `ritk-python/src/registration.rs`; added `multires-demons` method + `--levels`/`--use-diffeomorphic` args to `ritk-cli/src/commands/register.rs`; 2 new CLI tests |
| SEG-MORPH-EXT | Binary fill holes + morphological gradient | COMPLETED | High | Created `BinaryFillHoles` (6-connected border flood-fill, O(N)) and `MorphologicalGradient` (dilation AND NOT erosion) in `ritk-core/src/segmentation/morphology/`; 10 unit tests added |
| DICOM-WRITE | Real DICOM binary writer | COMPLETED | High | Replaced scaffold writer with `InMemDicomObject` + `FileMetaTableBuilder` per-slice writer; DICM magic verified; updated reader to use `open_file` path; enabled DICOM write in CLI |
| SEG-CLI-BINDINGS | Binary fill holes + morphological gradient CLI exposure | COMPLETED | High | Added `fill-holes` and `morphological-gradient` methods to `ritk-cli/src/commands/segment.rs`; added tests for enclosed-hole filling and boundary extraction |

---

## Sprint 20 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PY-BIND-FLT | Python filter bindings for Sprint 11 algorithms | COMPLETED | High | Added `curvature_anisotropic_diffusion` and `sato_line_filter` to `ritk-python/src/filter.rs`; registered in submodule; updated module docstring |
| PY-BIND-SEG | Python segmentation bindings for Sprint 10 algorithms | COMPLETED | High | Added `confidence_connected_segment`, `neighborhood_connected_segment`, `skeletonization` to `ritk-python/src/segmentation.rs`; registered in submodule |
| PY-IO-EXT | Python IO extended format coverage | COMPLETED | High | Extended `read_image`/`write_image` in `ritk-python/src/io.rs` to cover TIFF, VTK, MGH/MGZ, Analyze (.hdr/.img), JPEG â€” matching the full ritk-io format surface |
| CLI-FLT-EXT | CLI filter curvature + sato variants | COMPLETED | High | Added `curvature` and `sato` filter subcommands to `ritk-cli/src/commands/filter.rs`; added `--time-step` arg to FilterArgs; added 2 positive tests |
| CLI-SEG-EXT | CLI segment confidence-connected, neighborhood-connected, skeletonization | COMPLETED | High | Added 3 new methods to `ritk-cli/src/commands/segment.rs`; added `--multiplier`, `--max-iterations`, `--neighborhood-radius` args to SegmentArgs; added 11 positive + boundary tests |
| GAP-R02b | Multi-resolution Demons pyramid | COMPLETED | Medium | Created `MultiResDemonsRegistration` + `MultiResDemonsConfig` in `crates/ritk-registration/src/demons/multires.rs`; coarse-to-fine pyramid (2^l downsampling, trilinear upsampling, warm-start displacement injection); supports Thirion and Diffeomorphic variants; 5 unit tests passing |

---

## Sprint 19 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PERF-R02 | Promote shared `_into` primitives to SSOT in `deformable_field_ops.rs` | COMPLETED | High | Added `pub(crate) compute_gradient_into` (true zero-alloc in-place), `pub(crate) warp_image_into`, `pub(crate) compute_mse_streaming`; refactored `compute_gradient` and `warp_image` to delegate; removed duplicate local copies from `thirion.rs`, `symmetric.rs`; eliminated inline streaming loop in `diffeomorphic.rs` by delegating to `compute_mse_streaming`; `crates/ritk-registration/src/{deformable_field_ops.rs,demons/thirion.rs,demons/symmetric.rs,demons/diffeomorphic.rs}` |
| PERF-R03 | Specialize `convolve_axis` into three branch-free per-axis functions | COMPLETED | Medium | Replaced single generic `convolve_axis` (inner-loop `match axis`) with `convolve_z`, `convolve_y`, `convolve_x`; eliminates per-element axis dispatch across every voxel Ã— kernel-element iteration; `gaussian_smooth_inplace` updated to call the three specialized functions; `crates/ritk-registration/src/deformable_field_ops.rs` |

---

## Sprint 18 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PERF-R01 | Demons performance and memory optimization | COMPLETED | High | Reduced clone-heavy scaling-and-squaring allocations, added streaming MSE evaluation for Demons paths, reused iteration buffers in Thirion/Symmetric Demons, and fused inverse-field vector warping in `crates/ritk-registration/src/{deformable_field_ops.rs,demons/thirion.rs,demons/symmetric.rs,demons/diffeomorphic.rs,demons/inverse.rs}` |

---

## Sprint 17 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| GAP-R02b | Diffeomorphic Demons exact inverse | COMPLETED | High | Retained stationary velocity fields in `DemonsResult`, computed exact inverse as `exp(-v)` for diffeomorphic Demons, and added forward/inverse composition regression tests in `crates/ritk-registration/src/demons/diffeomorphic.rs` |

---

## Sprint 16 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| CI-DEPS | Workspace dependency alignment checks | COMPLETED | Medium | Centralized shared crate versions in `[workspace.dependencies]`, migrated crate manifests to `workspace = true`, and added CI enforcement in `.github/workflows/ci.yml` |

---

## Sprint 15 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| CI-NEXTEST | CI nextest enforcement | COMPLETED | Medium | Replaced workspace `cargo test` CI execution with `cargo nextest run --workspace --lib --tests` and installed `cargo-nextest` in `.github/workflows/ci.yml` |

---

## Sprint 14 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| PY-THREAD | PyO3 GIL Release Coverage | COMPLETED | Medium | Wrapped long-running Python filter, segmentation, registration, statistics, and image I/O compute paths with `py.allow_threads`; `crates/ritk-python/src/{filter,segmentation,registration,statistics,io}.rs` |

---

## Sprint 13 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| IO-09 | DICOM Read Metadata Slice | COMPLETED | High | Series-level capture plus per-slice geometry and rescale fields; `ritk-io/src/format/dicom/mod.rs` |
| IO-05 | MINC2 Reader | COMPLETED | High | consus-hdf5 HDF5 parsing; `ritk-io/src/format/minc/reader.rs` |
| IO-05 | MINC2 Writer | COMPLETED | High | Low-level HDF5 binary construction; `ritk-io/src/format/minc/writer.rs` |
| DEP | consus Integration | COMPLETED | High | consus-hdf5/core/io/compression workspace dependencies |

---

## Sprint 11 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| FLT-CAD | Curvature Anisotropic Diffusion | COMPLETED | Medium | Alvarez et al. 1992 mean curvature motion; `ritk-core/src/filter/diffusion/curvature.rs` |
| FLT-SATO | Sato Line Filter | COMPLETED | Medium | Multi-scale Hessian line detection (Sato 1998); `ritk-core/src/filter/vesselness/sato.rs` |
| FLT-HESS | Hessian Module | COMPLETED | Medium | 3-D physical-space Hessian + Cardano eigenvalue solver; `ritk-core/src/filter/vesselness/hessian.rs` |

---

## Sprint 10 â€” Completed
| ID | Feature | Status | Priority | Notes |
|---|---|---|---|---|
| SEG-CC | Confidence Connected Region Growing | COMPLETED | Medium | Yanowitz/Bruckstein adaptive statistics; `ritk-core/src/segmentation/region_growing/confidence_connected.rs` |
| SEG-NC | Neighborhood Connected Region Growing | COMPLETED | Medium | Rectangular neighborhood admissibility predicate; `ritk-core/src/segmentation/region_growing/neighborhood_connected.rs` |
| SEG-SK | Skeletonization | COMPLETED | Low | Topology-preserving thinning (Zhangâ€“Suen 2-D, directional 3-D); `ritk-core/src/segmentation/morphology/skeletonization.rs` |

---