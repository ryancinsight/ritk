import pathlib, re

# ── checklist.md: mark Sprint 27 as Completed, add Sprint 28 Planned ────────
cl = pathlib.Path('checklist.md')
c = cl.read_text(encoding='utf-8')

old_hdr = '## Sprint 27 -- Planned'
new_hdr = '## Sprint 27 -- Completed'
c = c.replace(old_hdr, new_hdr, 1)

old_items = '''- [ ] DICOM-MULTIFRAME: extend series-centric I/O toward enhanced and multi-frame object handling
- [ ] DICOM-WRITER-GENERAL: separate object-model serialization from series image writing
- [ ] DICOM-TRANSFER-SYNTAX: audit codec and transfer-syntax coverage
- [ ] ITK-RESAMPLE: add B-spline and Lanczos CLI/Python exposure via interpolation mode selection
- [ ] PY-PARITY-HARNESS: compare ritk-python against Python references with value-based tests
- [ ] VTK-STRUCT-GRID: VtkStructuredGrid and VtkUnstructuredGrid data objects
- [ ] ITK-MORPHOLOGY-EXTENDED: label erosion, label opening/closing, morphological reconstruction'''

new_items = '''- [x] DICOM-MULTIFRAME: MultiFrameInfo + load_dicom_multiframe<B> in ritk-io/src/format/dicom/multiframe.rs; 3 tests
- [x] DICOM-WRITER-GENERAL: DicomObjectWriter (model_to_in_mem + write_object) in writer_object.rs; 5 tests
- [x] DICOM-TRANSFER-SYNTAX: TransferSyntaxKind enum (11 UIDs + Unknown), is_compressed/lossless/supported; 8 tests
- [x] ITK-RESAMPLE: Resample subcommand (ritk-cli) + resample_image Python binding; 5 CLI tests
- [x] PY-PARITY-HARNESS: 10 analytically-derived parity tests in ritk-core/tests/parity.rs; all 10 pass
- [x] VTK-STRUCT-GRID: VtkStructuredGrid + VtkUnstructuredGrid + VtkDataObject new variants; 9 tests
- [x] ITK-MORPHOLOGY-EXTENDED: LabelErosion + LabelOpening + LabelClosing + MorphologicalReconstruction; 21 tests'''

c = c.replace(old_items, new_items, 1)

sprint28 = '''## Sprint 28 -- Planned

- [ ] NIFTI-SFORM-FIX: persist sform/pixdim in NIfTI writer so spacing round-trips correctly
- [ ] DICOM-MULTIFRAME-WRITE: write multi-frame DICOM objects from 3D Image<B,3>
- [ ] DICOM-NONIMAGE-SOP: define explicit accept/reject policy for non-image SOP classes
- [ ] VTK-STRUCTGRID-IO: VTK legacy reader/writer for STRUCTURED_GRID and UNSTRUCTURED_GRID datasets
- [ ] ITK-CONFIDENCE-CONNECTED: confidence-connected region growing (iterative mean+-k*sigma)
- [ ] ITK-SKELETONIZATION: topology-preserving thinning via iterative boundary erosion
- [ ] PY-CI-MATRIX: Python wheel smoke tests across supported Python versions

'''

c = sprint28 + c
cl.write_text(c, encoding='utf-8')
print("checklist.md patched")

# ── backlog.md: add Sprint 27 Completed block at top ────────────────────────
bl = pathlib.Path('backlog.md')
b = bl.read_text(encoding='utf-8')

sprint27_block = '''## Sprint 27 -- Completed

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

'''

b = sprint27_block + b
bl.write_text(b, encoding='utf-8')
print("backlog.md patched")
