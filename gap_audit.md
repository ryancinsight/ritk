# RITK Gap Audit

## 1.0 candidate verification

The [release backlog](backlog.md#milestones) records the remaining milestone criteria and their owners.

- [patch] MGH 0.6.1 and AsterSlice 0.2.1 use FreeSurfer's voxel center and open both gzip suffixes. Three regressions fail on the baseline. All 629 MGH and viewer library tests pass on macOS. NiBabel independently confirms all 12 samples and the full affine through MGZ → NIfTI → MGH → compressed MGH → MGZ.
- [minor] CLI 0.3.0 preserves all decompressed bytes when repacking a supported NIfTI document. The boundary tests cover a calibrated two-volume NIfTI-2 integer file and truncated input with an existing destination.
- [patch] Verify Python VTK and Analyze physical coordinates against SimpleITK and reject unrepresentable Analyze stored headers. All 306 Analyze/VTK library tests pass.
- [patch] Preserve NIfTI gzip member boundaries on read by joining the expanded data and validating every document trailer. Two regressions failed on the baseline and all 106 NIfTI tests pass.
- [patch] Inventory MetaImage conversion fields and correct big-endian reads using the alternate header key. All 25 library tests pass, including 16 packaging/key combinations.
- [minor] Transfer decoded scalar DICOM pixels into the viewer with their resolved grid and release encoded inputs after decoding. Across seven affected packages, 1,666 Rust tests pass. All 196 applicable minor-release checks pass against the 0.8.0 source on main.
- [patch] Qualify the optimized macOS ARM64 Python wheel against SimpleITK 2.5.8 for patch denoising and ternary magnitude on the original numeric bounds.
- [minor] Verify the native Display menu half-turn and its inverse on Windows. The rectangular fixture compares every presented pixel. [Passing Windows job](https://github.com/ryancinsight/ritk/actions/runs/45063679762/job/137338722471)
- [minor] MGH 0.7.0 and the CLI repack FreeSurfer documents with exact header, frame, and footer bytes. Multi-frame signed integers, gzip trailer validation and in-place replacement are covered.
- [patch] Verify complete MGZ framing on scalar and multi-frame reads. All 2,853 affected Rust tests pass across eight packages.
- [patch] Apply NIfTI calibration before binary32 narrowing. Compare all 42,991,616 voxels in the four shared OpenNeuro MRIs with NiBabel and SimpleITK.
- [patch] Verify manual feature-branch documentation runs against the default-branch merge base, including when both branches have advanced.
- [patch] Remove the skull CT and paired MRI/CT collections from the candidate tree after verifying all 2,642 retained files against their Git objects.
- [patch] Verify the OpenNeuro ds000102 source, license and checksum and run the registration consumer from the shared fixture cache.
- [major] Record the four 1.0 compatibility surfaces in [ADR 0059](docs/adr/0059-release-scope-1-0.md). Compatibility qualification remains pending.

## AsterSlice browser workspace

The previous browser page implemented an independent HTML toolbar, sidebar and pane
layout. The standalone application now runs the Windows workstation compositor
and event reducer. Chromium reads the 94-image public MRI, navigates slices,
measures length, rotates the view and advances cine. The native series dialog
opens MR and CT volumes in independent panels. Sixteen displayed samples per
plane match the embedding renderer and each source plane matches the stored
RGBA digest. Raster and WebGPU produce byte-identical desktop screenshots.
The transformed embedding check selects voxel (47, 340, 173) and steps to
(46, 340, 173). All 1,139 macOS viewer and I/O library tests pass. The original
root failed the single-workstation acceptance test.

## Shared fixture inventory

The pre-retention RITK inventory records 3,191 tracked test-data files totaling 1,490,441,937 bytes. A local inventory records path, size, Git blob, available provenance, code references, and the names of selected populated DICOM fields. It parsed 1,980 DICOM headers. Identifying-field presence does not establish a subject’s identity or permission to redistribute.

The independently downloaded OpenNeuro ds000102 T1 volume agrees with the tracked copy. Its source record declares PDDL. The shared catalog resolves it into a local checksum-verified cache, and the inter-subject registration test runs from that cache. License and source records for the skull CT and paired MRI/CT collections remain unknown.

The owner authorized local retention and removal from new checkouts for the skull CT and paired MRI/CT collections. Before removing their index entries we verified 308 and 2,334 retained files respectively against their tracked objects, identical in the base tree at `71f94fc74`. Their total size is 1,134,385,072 bytes. The owner holds a private per-file SHA-256 inventory alongside the files. Prior Git history still contains the original objects.

## Documentation check range

A manually dispatched development-branch run supplied no event base. The job compared the candidate with the repository root and attributed historical images to the current branch. The job now uses the common commit with the default branch for this event. A diverged-branch regression reproduces the old failure and checks the default-branch, push, and PR comparisons.

## Browser GPU pixel verification

The gallery slice and window samplers requested a 2D context on a canvas already bound to WebGPU. On a real Chromium adapter this failed before sampling. Directly drawing the live GPU canvas into a second context also returned a cleared frame. Serializing its presented bitmap as lossless PNG yields the native-resolution RGBA samples. A six-pixel fixture with distinct color channels produced the same SHA-256 in both presentation modes and matched the independent input bytes.

The saved 94-slice MRI passes 50 slider actions with the Métis rejection-order correction. Each anatomical plane returns to its starting pixel digest. Native RITK validation accepts the three-canvas trace. The earlier macOS Chromium window-preset action timed out with the previous selection. The workspace’s keyboard-selectable preset list now passes the raster and WebGPU journeys on that MRI. Hosted Chromium offered no WebGPU adapter, and hosted WebKit could not read the selected files. These acceptance checks remain open.

## Scalar DICOM browser memory

Loading the same 94-slice, 512 × 512 MRI four times on Chromium increased WebAssembly memory from 403,832,832 to 502,464,512 bytes on the fourth load. The scalar viewer previously transferred the entire reconstructed volume into a tensor and copied it back. It now takes ownership of the decoder’s pixel allocation. The native image loader and viewer use the same decoder and read budget.

On the changed decoder, Chromium completed all four loads with 401,604,608 bytes after the first transfer. All three canvases passed native trace validation for each load. A separate attempt stopped when the coronal canvas reported an unexpected input type. Windows and hosted browser results remain qualification gates.

## Python denoising parity

Six failures in the macOS ARM64 optimized wheel traced to three numeric differences from SimpleITK 2.5.8. ITK converts floored Gaussian draws to unsigned coordinates before testing the boundary. Apple ARM64 saturates negative draws to zero. Apple libc++ selects float powers for spline weights, and Clang contracts two multiply-adds in the squared norm. Correcting these operations preserves the existing bounds. All 1,301 Python tests that ran as ordinary passing cases now pass, with eight skips and one unexpected pass. The eight affected Rust packages pass 2,851 tests. All 223 applicable patch compatibility checks pass against published RITK Filter 0.6.0.

## MGZ compute reader

The scalar and series readers previously stopped at the first gzip member. A member split inside a signed short returned a truncation error, and a footer extending past the declared read limit was accepted. Both regressions now pass. The compatibility check also verifies that the additional error variants preserve existing discriminants.

## MetaImage sample byte order

The reader decoded an `ElementByteOrderMSB = True` file containing −513 as −3.
Both spellings now select the declared order, including for zlib and detached
data. The [field inventory](docs/book/metaimage_format.md#conversion-inventory)
records remaining sample, calibration, geometry, and packaging work.

## Analyze stored header fidelity

The former stored writer admitted a header that rounded the source spacing,
origin or intensity scale. Its preflight now rejects that conversion before
opening either destination file. The independent VTK and Analyze Python
geometry checks exercise the representable signed-permutation case. The
legacy Analyze compute writer retains its documented quantization behavior.

## NIfTI compute intensities

In OpenNeuro ds000208 the first background sample decoded as −32,768 while NiBabel and SimpleITK both returned zero. The compute reader discarded the slope and intercept. Two independent regressions exercise signed-zero identity, invalid coefficients and integer cancellation above 2²⁴. Corrected compute reads agree with both libraries at every one of the 42,991,616 voxels in the four cataloged MRIs.

## NIfTI document conversion

Two gzip regressions reproduced a split header failing to read and a damaged
second member being accepted. The document reader now drains all members within
the expanded limit and the compute reader can decode samples spanning members.

The CLI previously sent same-format conversion through the single-volume
binary32 compute reader. This rejected a two-volume NIfTI-2 input and could
round wide integers in a single volume. The CLI now delegates this route
to the document transcoder. The [cross-format item](backlog/RITK-NRRD-NIFTI-001.md)
tracks the distinct requirements for NRRD interchange.

## FreeSurfer document repacking

The CLI rejected a two-frame MRI_INT input during a pure compression change.
The document path checks the declared payload and copies every expanded byte
through a temporary output. Coverage includes the four supported scalar types,
nonzero DOF, an unset RAS flag, padding, acquisition footer, corrupt trailers,
and the exact expanded-byte boundary.

## FreeSurfer volume geometry

The MGH 0.6.0 reader and writer use the same displaced center and therefore pass a mutual round trip. The independent viewer fixture, hand-built header and emitted-header regressions all fail on `533f2e077`. Version 0.6.1 corrects both transforms. The [inventory](docs/book/mgh_format.md#conversion-inventory) identifies the remaining stored-sample and metadata work.

## Viewer dependency audit [patch]

The locked all-feature audit of the 2026-11-22 candidate clears the URL and RSA vulnerabilities. Updating the xtask progress bar to indicatif 0.18.6 removed number_prefix. The executor resolves to unyanked async-executor 1.14.1. The native chooser uses rfd 0.18.0, removing the three older portal findings on zbus_names 3.0.0, zvariant 4.2.0 and zvariant_derive 4.2.0. The chooser-enabled build passes on macOS.

`cargo deny --locked --all-features check advisories` still reports nine unmaintained package versions. Each remaining path has an owner and a verification requirement.

| Owner and entry point | Affected packages | Replacement and acceptance |
| --- | --- | --- |
| RITK I/O networking tests, independent dicom-ul 0.10.1 peer | chrono 0.4.45, encoding 0.2.33 | Move to a peer using maintained time and character-set implementations. Preserve real loopback C-ECHO, C-FIND and C-MOVE exchanges with an independent peer. |
| RITK native viewer, eframe accessibility through AccessKit and AT-SPI | derivative 2.2.0, instant 0.1.13, zbus_names 2.6.1, zvariant 3.15.2, zvariant_derive 3.15.2 | Qualify a maintained accessibility stack. zbus 6 owns the names, wire and derive APIs. Exercise Linux assistive technology and retain Windows accessibility. An eframe 0.37 resolution still includes the three standalone D-Bus crates. |
| RITK native viewer text, ab_glyph and owned_ttf_parser | ttf-parser 0.25.1 | Qualify a font stack using skrifa. Preserve text layout, glyph selection and exact input interactions across supported hosts. |
| RITK native rendering and image codecs | paste 1.0.15 | Coordinate all three paths through Windows AccessKit, wgpu’s Metal backend and the EXR codec’s pulp dependency. Retest accessible windows, Metal rendering and EXR decoding before retiring the final edge. |

## Saved-study browser qualification [patch]

[Run 45079835823](https://github.com/ryancinsight/ritk/actions/runs/45079835823) exercised all eight scenarios on RITK `54140426b` with Metis `94b698dcd`. Seven passed. Hosted Chromium WebGPU read the 94-file study, presented all three anatomical planes, passed the native cine-rate trace validator, removed its observer and closed the session. Both responsive scenarios, Chromium raster, the Chromium window and projection scenarios, and Firefox raster passed.

Safari 27.2 recorded a trusted selection of 49,807,236 bytes across all 94 files. macOS denied WebContent access to the first file before decoding. Byte-for-byte verification and filesystem permissions are recorded in the [Safari job](https://github.com/ryancinsight/ritk/actions/runs/45079835823/job/137389340300). The [qualification checklist](backlog/ritk-stable-qualification.md) keeps Safari and the post-merge replay open.

The [replay at `a9bd623aa`](https://github.com/ryancinsight/ritk/actions/runs/45087220365) passed six scenarios, including hosted WebGPU. Safari again denied the first file read. Chromium raster completed four loads and cleaned up its listeners and file handles, but its WebAssembly memory grew from 403,832,832 to 502,464,512 bytes on the fourth load. The [existing decoder change in PR #1108](https://github.com/ryancinsight/ritk/commit/f701208ea2873d01fd89eb8b02789fe5700831b6) transfers ownership of the reconstructed pixels and records a passing four-load replay. Repeat the original capacity check on the integrated browser candidate.

## Classic PET SUV timing

The classic PET reader admits a common activity reference across instances
before the shared pointer and compatibility fusion paths display SUVbw.
The baseline disagrees with the decimal reference or reports SUV for
unavailable timing in 31 of the initial 50 Part 10 cases. Seven further
cases exercise single-valued Frame Reference Time and Decay Factor,
valid padding in decay mode, UTC offset and frame duration, and the
DICOM Integer String range. Coverage includes verified zero, one-hour
and dated midnight intervals, offset-aware clocks, and conflicting
frame timing.

Classic series with distinct frame-scale factors retain activity display
and need frame-specific normalization before interpolation to support
quantitative SUV. Native Windows UI verification remains part of
[RITK-SNAP-NATIVE-FUSION-001](backlog/RITK-SNAP-NATIVE-FUSION-001.md).

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
- verification: the macOS arm64 wheel passes both profiles for the 94-instance MRI and 303-instance CT fixtures and rejects the CT DICOMDIR. All 564 `ritk-io` library tests pass on the integrated source.
- target: `ritk-io` 0.8.0 and `ritk-python` 0.13.0.
- integration: PRs #1104 and #1105 merged after their required checks passed. The crates.io release cohort tracks the `ritk-io` 0.8.0 manifest.

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
