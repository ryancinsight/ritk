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
