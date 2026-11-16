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

<a id="RITK-DICOM-ANONYMIZE-OUTSIDE-DATASET-001"></a>
## RITK-DICOM-ANONYMIZE-OUTSIDE-DATASET-001: a de-identified export keeps identifiers outside the data set
- risk: `anonymize_dicom_study` mirrors source folder and file names, so `DOE^JOHN/<StudyUID>/<SeriesUID>_a.dcm` keeps the patient name and source UIDs in output paths the leak scan never reads; file meta (0002,0016), (0002,0017), (0002,0018), (0002,0100) and (0002,0102) keep their source values.
- evidence: mirroring is the documented layout of ritk#920; both witnesses cited by the judge of ritk#922; no test reproduces it yet.
- reopen: before the viewer's de-identified export is called complete; file the correctness item with a failing test, then choose neutral output names or a path-component scan in the export gate.

<a id="RITK-DICOM-ANONYMIZE-NONIMAGE-001"></a>
## RITK-DICOM-ANONYMIZE-NONIMAGE-001: a study holding SR, KO or PR objects may fail the export gate whole
- risk: the default geometry check rejects an instance without image geometry, and one rejected instance removes the entire de-identified export.
- evidence: raised by the judge of ritk#920 from reading `anonymize/verify.rs`; not reproduced with a fixture.
- reopen: with the next change to `anonymize/study.rs` or `anonymize/verify.rs`, export one image plus one SR without pixel data; file the correctness item if it is rejected, delete this risk if it passes.

<a id="RITK-DICOM-ANONYMIZE-STUDY-002"></a>
## RITK-DICOM-ANONYMIZE-STUDY-002: study export failures lose their cause, and filed judge findings are open
- risk: an unreadable DICOM file counts as non-DICOM without its path; a `create_dir_all` failure returns `Tree`; `exists()` hides metadata errors; `Retained` has no `#[source]`; `Instance` boxes `dyn Error`; emptied folders stay after a failed run; the verifier module doc claims a profile-completeness check no code performs; `examples/anonymize_pacs_export.rs` lines 12 to 15 are stale.
- evidence: filed findings of the judges of ritk#914, #917, #919, #920 and #921, in each PR's verdict comment.
- reopen: at the next change under `crates/ritk-io/src/format/dicom/anonymize/` or `export_tree.rs`, or when the backlog has room (998 of 1,000 lines at d247fada9).

<a id="RITK-DICOM-REPAIR-STUDY-002"></a>
## RITK-DICOM-REPAIR-STUDY-002: filed judge findings on the repair export and viewer tools are open
- risk: `anonymize/file_meta.rs` says the dicom-rs reader refuses a file meta (0002,0003) that is not a UID, but it accepts `1..2` and `01.2`; `repair/study.rs` `failed` duplicates the anonymize helper and its `!=` against the default summary has a surviving `==` mutant; no test reaches `Instance`, `Write` or `Retained`; the viewer ends held gestures before a dialog the reader may cancel, and repeats the left-out sentence in two modules.
- evidence: filed findings in the judge verdict comments on ritk#926, #927, #928 and #929.
- reopen: at the next change under `crates/ritk-io/src/format/dicom/repair/`, `anonymize/file_meta.rs` or `crates/ritk-snap/src/presentation/native_session/{repair,deidentify}.rs`, or when the backlog has room (1,000 of 1,000 lines once ritk#928 lands).
