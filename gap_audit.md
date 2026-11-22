# RITK Gap Audit

## DICOM multiframe geometry

Generic scalar and color headers and writer inputs carry typed lengths.
The scalar loader takes the first decoded frame's position and uses
the existing millimetre interpolation boundary. The oblique fixture
checks a missing plane and prevents a malformed coordinate token from
shifting later axes.

## DICOM series writer geometry

Native, Coeus, and metadata-preserving series writers share typed image
plane geometry. Writer regressions inspect signed patient coordinates,
anisotropic spacing, and depth progression in the emitted Part 10 tags
and reject an unrepresentable positive spacing before output.

<a id="ritk-native-dicom"></a>
## RITK-NATIVE-DICOM — Application cutover verified

- The RITK dataset, writer, frame dispatch, and DIMSE wrappers form one
  production graph. The `dicom-ul` test peer retains `dicom-core` and
  Chrono only in RITK’s development graph.
- Frame assembly uses Basic or Extended Offset Tables when supplied.
  An empty table supports one frame or one fragment per frame. Ambiguous
  multi-fragment layouts and unclassified transfer syntaxes report an error.
  The parity scope covers the former backend’s supported pixel layouts.
- Explicit big-endian Part 10 parsing, writing, and stored-sample decode
  are covered at the provider boundary. Higher-level RITK I/O series
  loaders retain their existing explicit big-endian policy.
- Borrowed catalog selection records at most one span per requested tag.
  With the encoded input resident, 32 frames (16,777,690 encoded bytes)
  use zero heap bytes for this selection and 16,790,724 bytes for an owned
  parse. The checks cover both explicit byte orders, implicit little
  endian, nested attributes, and truncation.
- The reconciled workspace resolves from an empty lockfile state with one
  registry Aequitas 0.4.0, Eunomia 0.12.1, Leto 0.51.0, and Coeus 0.13.0.
  Across all targets and features, Chrono appears only through the independent
  DIMSE test peer. The 650 focused DICOM, I/O, and Diffusion Scheme
  tests pass, including scanner bounds, numeric VR boundaries, and real
  DIMSE peers. Two opt-in tests remain outside this run.
- A fresh crates.io consumer writes a Part 10 CT slice and recovers all
  four calibrated voxel values exactly through the published I/O crate.
  Its dependency graph contains no Chrono.

<a id="RITK-REGISTRY-PUBLICATION-001"></a>
## RITK-REGISTRY-PUBLICATION-001 — Registry graph published

- All sixteen cutover versions are indexed with matching archive checksums
  and release tags. DICOM 0.5.0 and I/O 0.6.0 use the native writer released
  from merge 49dc944ed862. [Trusted publishing](https://github.com/ryancinsight/ritk/actions/runs/44667442964)
- Registration 0.56.0 publishes after Parcellation 0.2.0 and the other
  registry providers. [Registration release](https://github.com/ryancinsight/ritk/releases/tag/ritk-registration-v0.56.0)

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
## RITK-DICOM-ANONYMIZE-DIRECTORY-NAMES-001 — Number directory exports [major]
- risk: Rust directory export and its Python binding copied patient names and source UIDs embedded in filenames. Concurrent calls could overwrite an earlier export while reporting success.
- evidence: the installed-wheel fixture reproduces both failures on `762f0e92`. The proposed correction shares the study export's numbering and publishes each encoded copy without replacing an occupied name. The Python fixture checks source bytes, collisions, per-file statistics, and remapped UID relationships after reopening. The Rust directory verification test reopens the numbered export through the existing gate.
- status: [ADR 0068](docs/adr/0068-number-dicom-directory-exports.md) accepted. The fix is committed on the implementation branch. Hosted integration and release verification are pending.

<a id="RITK-DICOM-ANONYMIZE-REAL-001"></a>
## RITK-DICOM-ANONYMIZE-REAL-001 — Preserve real DICOM export structure [major]
- risk: the profile added absent attributes to sequence items, retained CT creation and contribution metadata and irradiation-event identifiers, encoded UUIDs outside their 128-bit range, and published a DICOMDIR whose paths did not resolve.
- evidence: the installed wheel at `84de2d4ae` preserves all 94 MRI and 303 CT images, but exposes these regressions. The real CT file-set index contains 304 references, one of which is already missing in the source fixture.
- status: [ADR 0069](docs/adr/0069-preserve-dicom-export-structure.md) records the selected contract. The opt-in Python suite covers both fixtures and the file-set root. RITK Python volume reopening of the source compressed CT is a separate gap. The independent decoder verifies all CT slices.
- verification: the macOS arm64 wheel passes both profiles for the 94-instance MRI and 303-instance CT fixtures and rejects the CT DICOMDIR. The focused Rust suite passes 93 tests.
- next: review the stacked change after PR #1104 lands and rerun the required main-branch checks.

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
errors when identified. The decoder records the source of each resampled
plane in the same branch that writes the output values. The viewer
labels interpolation and copied edges on the depth view, using the
currently selected gate. The decoded-byte budget includes provenance.
The separate GPU rerun exercises the Metal projection paths on macOS.
[Windows run 44641586577](https://github.com/ryancinsight/ritk/actions/runs/44641586577)
builds the native viewer with SDK 10.0.26100.0 and passes the exact
provenance display-command test, five reader cases, and two viewer cases.
The Windows candidate uses published Aequitas 0.5.0 and two bundled provider
revisions pending public lock alignment. The same candidate passes
21 PET and projection tests on macOS with Metal selected.

The public compatibility candidate resolves one registry Aequitas 0.5.0
and one each of Eunomia and Eunomia Derive 0.12.1. It uses GitHub revisions
for the Atlas providers, including open Gaia, Metis, Apollo, and Coeus
compatibility PRs. Replace the PR references with their integrated commits
before promoting the lock to the stack.

The reconciled AsterSlice source passes 27 PET, DICOM geometry, and Metal
projection tests. [Windows run 44651506776](https://github.com/ryancinsight/ritk/actions/runs/44651506776)
builds that source with Windows SDK 10.0.26100.0 and passes five reader tests,
two viewer tests, and the exact native provenance overlay test. These runs
use bundled compatibility revisions for Leto and Gaia.

## Enhanced PET on native DICOM datasets [patch]

The AsterSlice rollout preserves the upstream native dataset parser,
source-qualified measurements, and frame codecs. PET dimension and
concatenation validation now read the native dataset. Stored pixel
calibration stays independent of display rescale. The decay path
validates full civil timestamps and computes offsets before converting
elapsed time into Aequitas time.

The 0.5 lock uses public compatibility commits for Apollo, Coeus, and
Hephaestus. Locked all-feature metadata resolves one registry Aequitas
0.5.0. The 58 selected PET, geometry, measurement, and Metal projection
tests pass. Registry package promotion requires the published provider
archives, including a Gaia version that admits Aequitas 0.5.0.

## 2026-11-21 · Series geometry quantities

The directory and stored-instance readers share series finalization. Series
spacing follows tensor-axis order and patient origin follows XYZ. The scan
keeps lengths through assembly, and the tensor interface converts them to
millimetres. The oblique Part 10 fixture exercises measured and fallback depth.
