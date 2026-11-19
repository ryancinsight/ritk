# RITK Gap Audit

<a id="RITK-REGISTRY-PUBLICATION-001"></a>
## RITK-REGISTRY-PUBLICATION-001: Git validation does not establish registry publication [patch]
- risk: a corrected Git revision passes native checks while the registry resolves older incompatible dependencies or lacks required packages.
- evidence: main's release package verification fails in registry Leto 0.42.0; an isolated Coeus 0.11 probe compiles but its registry plan holds registration on unpublished Consus and missing Coeus component versions.
- next: preserve active provider owners, validate the complete required registry graph, and complete approved first publications before claiming a released registration version.
- review: [RITK-REGISTRY-PUBLICATION-001](backlog/RITK-REGISTRY-PUBLICATION-001.md) records exact failure, ownership and credential boundaries.

<a id="PERF-432-01"></a>
## PERF-432-01: Revalidate the legacy MSE performance claim
- risk: June profiling may no longer describe the current backend and can select the wrong optimization target.
- evidence: prior profile claims predate backend and interpolation changes; no current-main reproduction is recorded.
- reopen: before changing MSE or B-spline performance, collect a new exact-main profile and replace or remove the stale claim.
- basis: 961e2bc62737367c1bd125dab0c3d35526a1feef

<a id="RITK-MIF-OFFSET-CONVENTION-001"></a>
## RITK-MIF-OFFSET-CONVENTION-001: `.mif` `file: . N` offset may not match MRtrix3
- risk: the reader counts N from the `END` line and the writer emits `file: . 0`; if MRtrix3 counts from file start and refuses 0, real MRtrix files misread here and ours fail there.
- evidence: raised from recall by the ritk#874 judge at 8081bafb6; no MRtrix source or fixture was available offline to confirm.
- reopen: before the next `.mif` reader or writer change, read MRtrix3 `core/formats/mrtrix.cpp` at a pinned commit and file a fix item with an MRtrix-written fixture if it differs.

<a id="RITK-ORIENT-RELABEL-DOCS-001"></a>
## RITK-ORIENT-RELABEL-DOCS-001: orient relabeling docs and tests overstate what is pinned
- risk: `OrientImageFilter` documents the physical object as unchanged, false for an oblique input that snaps; `SignedAxisPermutation::reexpress` documents "the input voxel that becomes output (0, 0, 0)", but a zero-extent grid has none and the corner index saturates to 0; no test orients an oblique input successfully and `orient_invalid_codes_error` asserts bare `is_err()`.
- evidence: filed findings A-F1 and B-F1 to B-F4 of the judges of ritk#887 and ritk#888, read at f4474d00c and 4eddccdd1; behaviour was not shown wrong, only unpinned or misdescribed.
- reopen: at the next change to `crates/ritk-filter/src/transform/orient.rs` or `crates/ritk-image/src/axis_permutation.rs`, or when the backlog has room, file the item: zero-extent origin test first, then the doc corrections.

<a id="RITK-J2K-TILE-ALLOCATION-001"></a>
## RITK-J2K-TILE-ALLOCATION-001: a budget-admitted JPEG 2000 frame can still abort on allocation
- risk: after ritk#876 the read budget bounds the `f32` output, but the tile buffers in `jpeg_2000/packet/reader.rs` allocate infallibly and the decode working set is about three times the output, so a frame the budget admits can end in an out-of-memory abort instead of a typed error.
- evidence: filed findings of the ritk#876 judge at fa9c3d733, by reading; no near-cap decode was run. The same verdict filed, in the files that PR touched: `decode_jpeg2000_fragment` omits the invalid-layout `Codestream` error under `# Errors`, `mod.rs` keeps an `# Arguments` section, `jpeg_2000/tests.rs` is 1206 lines with a banner comment, the encoder tests call the two-argument decode past the budget, and the regression test is 64-bit only.
- reopen: at the next change under `crates/ritk-codecs/src/jpeg_2000/`, or when the backlog has room, file the item: measure one admitted frame's peak allocation with dhat, then make the tile buffers reserve with `try_reserve_exact`.

<a id="RITK-DICOM-VERIFY-PACKED-BITS-001"></a>
## RITK-DICOM-VERIFY-PACKED-BITS-001: the export gate sizes a 1-bit frame at one byte per sample
- risk: `check_geometry` in `anonymize/verify/geometry.rs` expects `Rows * Columns * SamplesPerPixel * ceil(BitsAllocated / 8)` bytes, so a single-frame object with BitsAllocated 1 and bit-packed PixelData of three or more samples is reported short and removes a whole de-identified export.
- evidence: filed by the judge of ritk#931 from reading; the packing rule was recalled from PS3.5, not looked up, and no fixture reproduces it.
- reopen: at the next change to `check_geometry`; read PS3.5 section 8.1.1, then export one 1-bit single-frame object and file the correctness item if it is rejected.

<a id="RITK-DICOM-ANONYMIZE-STUDY-002"></a>
## RITK-DICOM-ANONYMIZE-STUDY-002: study export failures lose their cause, and filed judge findings are open
- risk: an unreadable DICOM file counts as non-DICOM without its path; a `create_dir_all` failure returns `Tree`; `exists()` hides metadata errors; `Retained` has no `#[source]`; `Instance` boxes `dyn Error`; emptied folders stay after a failed run; the verifier module doc claims a profile-completeness check no code performs; `examples/anonymize_pacs_export.rs` line 12 says FrameOfReferenceUID must be present and its geometry line omits SamplesPerPixel.
- evidence: filed findings of the judges of ritk#914, #917, #919, #920 and #921, in each PR's verdict comment.
- reopen: at the next change under `crates/ritk-io/src/format/dicom/anonymize/` or `export_tree.rs`, or when the backlog has room (998 of 1,000 lines at d247fada9).

<a id="RITK-DICOM-ANONYMIZE-DIRECTORY-NAMES-001"></a>
## RITK-DICOM-ANONYMIZE-DIRECTORY-NAMES-001: directory-mode de-identification copies source file names
- risk: `anonymize_dicom_directory` and its verified wrapper write each instance under its source file name, so a patient name or source UID in a file name reaches the output; the Python binding calls this mode. `anonymize_dicom_study` numbers its outputs since ritk#933.
- evidence: read in `anonymize/mod.rs` (`output_dir.join(file_name)`) and `crates/ritk-python/src/io/anonymize.rs`; no fixture reproduces it.
- reopen: before the Python de-identification API is called complete; file the correctness item with a failing test, then number the outputs or route directory mode through the study export.

<a id="RITK-DICOM-ANONYMIZE-NESTED-META-001"></a>
## RITK-DICOM-ANONYMIZE-NESTED-META-001: group 0002 elements inside sequence items survive de-identification
- risk: `file_meta::remove_origin` removes top-level data set elements of group 0002 only; one nested in a sequence item stays. A file whose group length ends before (0002,0012) is read with dicom-rs default implementation identifiers, and the misplaced real ones are then deleted.
- evidence: filed by the judges of ritk#932 from reading ritk-io and dicom-object 0.10.1; neither input was built.
- reopen: at the next change to `anonymize/file_meta.rs` or `anonymize_dataset`; build both inputs as byte streams and file the correctness item for each that leaks or loses a value.

<a id="RITK-DICOM-ANONYMIZE-EXPORT-003"></a>
## RITK-DICOM-ANONYMIZE-EXPORT-003: filed judge findings on the numbered export and the gate are open
- risk: `anonymize/study.rs` keeps a `map_or(target, ..)` fallback no input reaches; export names past 999,999 stop sorting numerically; the source-path mapping is tested for `PatientLeak` only; the non-image study test does not assert the report's StudyInstanceUID and SOPInstanceUID were replaced; the root README says a failed run removes everything it wrote, but a created output folder stays; `tests_verify.rs` line 6 still lists geometry drift, moved to `tests_verify_geometry.rs`; the misplaced-element fixture uses Implicit VR and does not assert the stray values.
- evidence: filed findings in the judge verdict comments on ritk#931, #932, #933 and #934.
- reopen: at the next change under `crates/ritk-io/src/format/dicom/anonymize/`.

<a id="RITK-DICOM-REPAIR-STUDY-002"></a>
## RITK-DICOM-REPAIR-STUDY-002: filed judge findings on the repair export and viewer tools are open
- risk: `anonymize/file_meta.rs` says the dicom-rs reader refuses a file meta (0002,0003) that is not a UID, but it accepts `1..2` and `01.2`; `repair/study.rs` `failed` duplicates the anonymize helper and its `!=` against the default summary has a surviving `==` mutant; no test reaches `Instance`, `Write` or `Retained`; the viewer ends held gestures before a dialog the reader may cancel, and repeats the left-out sentence in two modules.
- evidence: filed findings in the judge verdict comments on ritk#926, #927, #928 and #929.
- reopen: at the next change under `crates/ritk-io/src/format/dicom/repair/`, `anonymize/file_meta.rs` or `crates/ritk-snap/src/presentation/native_session/{repair,deidentify}.rs`, or when the backlog has room (1,000 of 1,000 lines once ritk#928 lands).

## DICOM geometry and isotope quantities [major]

Series and slice geometry and PET activity now have dimensional storage. The
viewer gates SUV calculation on classic PET activity-concentration pixels.
Compound time metadata outside spatial Enhanced PET quantification remains open.

## Enhanced PET quantitative voxels [major]

The spatial multi-frame reader applies frame-scoped real-world mappings,
retains diagnostic values separately, and routes body-weight SUV to the
pointer, linked cursor, and fusion. The native Metal suite exercises the
viewer alongside the Part 10 quantification regression. The temporal and gated reader is recorded below.


## Temporal and gated Enhanced PET [major]

The volume-set reader groups explicit nonspatial DICOM indices and keeps
the original frame identity for quantitative mapping. A Part 10 fixture
interleaves time, separate cardiac and respiratory gates, and slice position. It verifies
per-frame decay, lookup tables, patient origin, and the decoded-byte limit.
The eframe selector and the Windows native View menu use the shared
selection operation. Metal tests verify the reader and eframe viewer. The
Windows cross-check requires SDK headers for ring. DICOM concatenations
that span several SOP instances remain a separate loading contract.

Locked all-target compilation passes on macOS with the eframe shell enabled.
The combined I/O and viewer gate passes 1,482 cases on Metal.

## Enhanced PET concatenations [minor]

Select a concatenation by its UID when opening one of its files. The
volume-set reader reconstructs time points and gates across members and
preserves source-local calibration and patient geometry. A fixture compares
a complete object with a shuffled two-member concatenation and verifies
display samples and SUV. It also rejects frame overlap, a missing first
member, and aggregate budget overruns. Retained bytes remain decodable after
source files change. A missing final member is detectable when the DICOM
objects declare the total.

The macOS reader and viewer suite passes 1,483 cases, including eight
separately rerun Metal projection cases. The Windows 2025 runner uses its
installed SDK and builds the default Métis viewer. The first native
capture reached the Windows client area at 1028 × 749. The request was
1280 × 800. The test now compares the PNG with the actual client area.

[Windows CI](https://github.com/ryancinsight/ritk/actions/runs/44623546473)
builds candidate `a14251cd8123119ded76ff72bbacca65b19579fe` with Windows SDK
10.0.26100.0 and executes four reader tests and two viewer tests. The native
test opens the two-instance PET study, verifies the captured client size
and overlay, then checks the selected gate's samples and body-weight SUV.
The executable reports version 0.2.0.

## Published providers and spatial PET slice gaps [patch]

The RITK lock selects crates.io Aequitas 0.4.0 and Eunomia 0.12.0.
The reader and viewer pass 1,484 tests, and the eight GPU projection
cases pass a separate Metal rerun.

A complete two-member concatenation can contain an unmeasured spatial plane.
The reader reports irregular spacing through tracing and linearly
interpolates display and calibrated SUV samples inside their observed
spatial range. The regression uses measured planes at 4, 7, 13, and 16 mm
with nominal 3 mm spacing. The plane at 10 mm carries the arithmetic mean
of the adjacent display samples and the corresponding SUV.

Missing SOP members or unavailable temporal gates remain input coverage
errors when identified. Interpolated-plane provenance is available as
geometry diagnostics. Exposing per-plane provenance in the viewer remains open.
