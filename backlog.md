# RITK execution backlog

Target: AsterSlice and RITK 1.0.0. The milestone and item blocks below carry the acceptance checklist. [Executed verification](gap_audit.md#10-candidate-verification) records completed checks.

# Milestones
<a id="M-viewer-reading"></a>
## M-viewer-reading — 0.2.0 — read a single study on the native shell
- exit: `cargo nextest run -p ritk-snap -E 'test(pointer_bindings) or test(modifier_gesture) or test(native_annotation) or test(native_display) or test(native_view_transform) or test(native_overlay_text) or test(measure_set)'`
- [patch] One workstation serves Windows and the web. Chromium verifies real MR and CT comparison, measurements and the three MRI planes. See [the reading workspace checklist](backlog/RITK-SNAP-RADIANT-UI-001.md).
- needs: none
- non-goals: PACS, archive, 3D rendering
<a id="M-viewer-compare"></a>
## M-viewer-compare — 0.3.0 — compare series by patient position
- exit: `cargo nextest run -p ritk-snap -E 'test(native_link) or test(native_load) or test(native_projection) or test(oblique)'`; `metis-browser-dicom.yml` green on chromium jobs
- [minor] The scalar reader transfers its pixel allocation to the viewer. Browser qualification includes four loads of the same 94-slice MRI.
- needs: M-viewer-reading
- non-goals: fusion
<a id="M-viewer-access"></a>
## M-viewer-access — 0.4.0 — get studies in and images out
- exit: `cargo nextest run -p ritk-snap -E 'test(native_open) or test(native_tags) or test(native_export) or test(native_session_round_trip) or test(native_pacs) or test(native_archive)'`
- exit: `cargo nextest run -p ritk-snap -E 'test(=presentation::native_session::tests::anonymize::native_anonymize_basic_replaces_identifiers_and_instance_uids)'` (export a de-identified study copy)
- exit: `cargo nextest run -p ritk-snap -E 'test(=presentation::native_session::tests::repair::native_repair_writes_a_corrected_copy_and_leaves_the_source)'` (write a repaired study copy)
- needs: M-viewer-compare
- non-goals: cloud storage
<a id="M-viewer-advanced"></a>
## M-viewer-advanced — 1.0.0 — advanced reading tools
- exit: `cargo nextest run -p ritk-snap -E 'test(native_filter) or test(native_vr) or test(native_fusion) or test(native_dynamic) or test(native_documents) or test(native_movie) or test(native_shell)'`
- exit: `cargo semver-checks` for `ritk-snap`; `cargo publish --dry-run` for the viewer's closure
- exit: `cargo doc` with no new warnings; zero open correctness items and verified compatibility for AsterSlice controls, the CLI, Python and published Rust crate APIs ([ADR 0059](docs/adr/0059-release-scope-1-0.md))
- [patch] The macOS ARM64 Python wheel passes the SimpleITK denoising and squared-magnitude comparisons. Cross-platform Python qualification remains open.
- needs: M-viewer-access
- non-goals: encrypted ZIP, optical disc burning, unlisted RadiAnt add-ons
- [patch] Manual documentation qualification compares a development branch against its merge base with the default branch.

# Items

<a id="ritk-stable-qualification"></a>- [RITK-STABLE-QUALIFICATION](backlog/ritk-stable-qualification.md) — ritk stable-release qualification [patch] — todo
<a id="RITK-FORMAT-RESCUE-RECONCILE-001"></a>- [RITK-FORMAT-RESCUE-RECONCILE-001](backlog/RITK-FORMAT-RESCUE-RECONCILE-001.md) — Rehome format work from stale rescue branches — blocked
<a id="RITK-GAP-2026-08-20-02"></a>- [RITK-GAP-2026-08-20-02](backlog/RITK-GAP-2026-08-20-02.md) — Fuzz supported format parsers — todo
<a id="RITK-PYTHON-VTK-DIRECTION-001"></a>- [RITK-PYTHON-VTK-DIRECTION-001](backlog/RITK-PYTHON-VTK-DIRECTION-001.md) — Write permuted images to VTK and Analyze — done [patch]
<a id="RITK-METAIMAGE-CONVERSION-INVENTORY-001"></a>- [RITK-METAIMAGE-CONVERSION-INVENTORY-001](backlog/RITK-METAIMAGE-CONVERSION-INVENTORY-001.md) — Map MetaImage volume semantics — done [patch]
<a id="RITK-MGH-CONVERSION-INVENTORY-001"></a>- [RITK-MGH-CONVERSION-INVENTORY-001](backlog/RITK-MGH-CONVERSION-INVENTORY-001.md) — Map MGH volume semantics — done [patch]
<a id="RITK-VTK-IMAGE-CONTRACT-001"></a>- [RITK-VTK-IMAGE-CONTRACT-001](backlog/RITK-VTK-IMAGE-CONTRACT-001.md) — Separate VTK image semantics — todo
<a id="SEC-446-05"></a>- [SEC-446-05](backlog/SEC-446-05.md) — Bound parser header counts by ImageReadBudget — todo
<a id="RITK-NRRD-NIFTI-001"></a>- [RITK-NRRD-NIFTI-001](backlog/RITK-NRRD-NIFTI-001.md) — Convert NRRD and NIfTI volumes — blocked
<a id="RITK-METAIMAGE-CONVERSION-001"></a>- [RITK-METAIMAGE-CONVERSION-001](backlog/RITK-METAIMAGE-CONVERSION-001.md) — Preserve MetaImage volume semantics — todo
<a id="RITK-MINC-CONVERSION-001"></a>- [RITK-MINC-CONVERSION-001](backlog/RITK-MINC-CONVERSION-001.md) — Preserve MINC volume semantics — blocked
<a id="RITK-MIF-CONVERSION-001"></a>- [RITK-MIF-CONVERSION-001](backlog/RITK-MIF-CONVERSION-001.md) — Preserve MRtrix image semantics — blocked
<a id="RITK-MGH-CONVERSION-001"></a>- [RITK-MGH-CONVERSION-001](backlog/RITK-MGH-CONVERSION-001.md) — Preserve MGH and MGZ volume semantics — todo
<a id="RITK-VTK-VOLUME-CONVERSION-001"></a>- [RITK-VTK-VOLUME-CONVERSION-001](backlog/RITK-VTK-VOLUME-CONVERSION-001.md) — Preserve VTK image-volume semantics — blocked
<a id="RITK-SNAP-OBLIQUE-SESSION-WIRING-001"></a>- [RITK-SNAP-OBLIQUE-SESSION-WIRING-001](backlog/RITK-SNAP-OBLIQUE-SESSION-WIRING-001.md) — Wire the oblique session pane — blocked
<a id="RITK-DICOM-CALIBRATION-UNITS-001"></a>- [RITK-DICOM-CALIBRATION-UNITS-001](backlog/RITK-DICOM-CALIBRATION-UNITS-001.md) — Preserve DICOM values and units — blocked
<a id="RITK-CI-MERGE-GATE-001"></a>- [RITK-CI-MERGE-GATE-001](backlog/RITK-CI-MERGE-GATE-001.md) — Gate pull requests through one CI check — todo
<a id="RITK-GIFTI-SURFACE-001"></a>- [RITK-GIFTI-SURFACE-001](backlog/RITK-GIFTI-SURFACE-001.md) — Preserve complete GIFTI surface documents — todo
<a id="RITK-MESH-CONVERSION-001"></a>- [RITK-MESH-CONVERSION-001](backlog/RITK-MESH-CONVERSION-001.md) — Convert supported mesh formats — todo
<a id="RITK-TRACTOGRAM-CONVERSION-001"></a>- [RITK-TRACTOGRAM-CONVERSION-001](backlog/RITK-TRACTOGRAM-CONVERSION-001.md) — Convert supported tractogram formats — todo
<a id="RITK-GAP-2026-08-20-01"></a>- [RITK-GAP-2026-08-20-01](backlog/RITK-GAP-2026-08-20-01.md) — Retire duplicate X and X_native APIs — todo
<a id="RITK-RASTER-CONVERSION-001"></a>- [RITK-RASTER-CONVERSION-001](backlog/RITK-RASTER-CONVERSION-001.md) — Convert PNG TIFF and JPEG rasters — blocked
<a id="RITK-SNAP-OBLIQUE-SESSION-MODULES-001"></a>- [RITK-SNAP-OBLIQUE-SESSION-MODULES-001](backlog/RITK-SNAP-OBLIQUE-SESSION-MODULES-001.md) — Separate native oblique session concerns — blocked
<a id="RITK-DICOM-CONVERSION-001"></a>- [RITK-DICOM-CONVERSION-001](backlog/RITK-DICOM-CONVERSION-001.md) — Convert DICOM series through RITK — blocked
<a id="RITK-FORMAT-CONVERSION-001"></a>- [RITK-FORMAT-CONVERSION-001](backlog/RITK-FORMAT-CONVERSION-001.md) — Convert supported formats through RITK — blocked
<a id="RITK-SNAP-OBLIQUE-NATIVE-001"></a>- [RITK-SNAP-OBLIQUE-NATIVE-001](backlog/RITK-SNAP-OBLIQUE-NATIVE-001.md) — Native oblique MPR — blocked
<a id="RITK-BROWSER-READ-001"></a>- [RITK-BROWSER-READ-001](backlog/RITK-BROWSER-READ-001.md) — Resolve WebKit study-file reads [patch] — in progress
<a id="RITK-BROWSER-WEBGPU-001"></a>- [RITK-BROWSER-WEBGPU-001](backlog/RITK-BROWSER-WEBGPU-001.md) — Verify browser WebGPU presentation on main — todo
<a id="RITK-GAP-2026-08-20-04"></a>- [RITK-GAP-2026-08-20-04](backlog/RITK-GAP-2026-08-20-04.md) — Reconcile tracked imaging payload provenance — todo
<a id="RITK-GAP-2026-08-20-06"></a>- [RITK-GAP-2026-08-20-06](backlog/RITK-GAP-2026-08-20-06.md) — Raise lint and documentation floors — todo
<a id="RITK-GAP-2026-08-20-08"></a>- [RITK-GAP-2026-08-20-08](backlog/RITK-GAP-2026-08-20-08.md) — Complete registration and dispatch chapters — todo
<a id="RITK-SNAP-OBLIQUE-APP-TESTS-001"></a>- [RITK-SNAP-OBLIQUE-APP-TESTS-001](backlog/RITK-SNAP-OBLIQUE-APP-TESTS-001.md) — Verify oblique adapter boundaries — blocked
<a id="RITK-FIXTURE-RETENTION-001"></a>- [RITK-FIXTURE-RETENTION-001](backlog/RITK-FIXTURE-RETENTION-001.md) — Decide imaging fixture retention — two collections retained locally, remaining records blocked [patch]
<a id="RITK-TEST-DATASET-RELAY-001"></a>- [RITK-TEST-DATASET-RELAY-001](backlog/RITK-TEST-DATASET-RELAY-001.md) — Externalize large public test datasets — blocked
<a id="RITK-SNAP-OBLIQUE-SESSION-TESTS-001"></a>- [RITK-SNAP-OBLIQUE-SESSION-TESTS-001](backlog/RITK-SNAP-OBLIQUE-SESSION-TESTS-001.md) — Verify oblique rendering and startup — blocked
<a id="RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001"></a>- [RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001](backlog/RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001.md) — Verify complete native gestures — blocked
<a id="RITK-SNAP-OBLIQUE-MANUAL-001"></a>- [RITK-SNAP-OBLIQUE-MANUAL-001](backlog/RITK-SNAP-OBLIQUE-MANUAL-001.md) — Demonstrate oblique MPR in the user manual — blocked
<a id="RITK-FORMAT-BULK-DECODE-001"></a>- [RITK-FORMAT-BULK-DECODE-001](backlog/RITK-FORMAT-BULK-DECODE-001.md) — Decode sample buffers through one bulk path — todo
<a id="RITK-SNAP-INTERACTION-REGIONS-001"></a>- [RITK-SNAP-INTERACTION-REGIONS-001](backlog/RITK-SNAP-INTERACTION-REGIONS-001.md) — Separate interaction test families — todo
<a id="RITK-GAP-2026-08-20-05"></a>- [RITK-GAP-2026-08-20-05](backlog/RITK-GAP-2026-08-20-05.md) — Derive test budgets and remove dead filters — todo
<a id="RITK-GAP-2026-08-20-07"></a>- [RITK-GAP-2026-08-20-07](backlog/RITK-GAP-2026-08-20-07.md) — Restore the changelog version axis — todo
<a id="RITK-SNAP-OBLIQUE-APP-ADAPTER-001"></a>- [RITK-SNAP-OBLIQUE-APP-ADAPTER-001](backlog/RITK-SNAP-OBLIQUE-APP-ADAPTER-001.md) — Route oblique viewer actions — todo
<a id="RITK-SNAP-RADIANT-UI-001"></a>- [RITK-SNAP-RADIANT-UI-001](backlog/RITK-SNAP-RADIANT-UI-001.md) — Organize a multi-series DICOM viewer workspace — todo
<a id="RITK-SNAP-OBLIQUE-ROUTING-001"></a>- [RITK-SNAP-OBLIQUE-ROUTING-001](backlog/RITK-SNAP-OBLIQUE-ROUTING-001.md) — Route native oblique input — blocked
<a id="RITK-BROWSER-TOOL-TRACE-001"></a>- [RITK-BROWSER-TOOL-TRACE-001](backlog/RITK-BROWSER-TOOL-TRACE-001.md) — Publish ROI results in the browser journey — todo
<a id="RITK-SNAP-EFRAME-REMOVAL-001"></a>- [RITK-SNAP-EFRAME-REMOVAL-001](backlog/RITK-SNAP-EFRAME-REMOVAL-001.md) — Remove the eframe shell — blocked
<a id="RITK-SNAP-NATIVE-ARCHIVE-001"></a>- [RITK-SNAP-NATIVE-ARCHIVE-001](backlog/RITK-SNAP-NATIVE-ARCHIVE-001.md) — Keep a local study archive — blocked
<a id="RITK-SNAP-NATIVE-DOCUMENTS-001"></a>- [RITK-SNAP-NATIVE-DOCUMENTS-001](backlog/RITK-SNAP-NATIVE-DOCUMENTS-001.md) — Display reports, PDFs and video objects — todo
<a id="RITK-SNAP-NATIVE-DYNAMIC-001"></a>- [RITK-SNAP-NATIVE-DYNAMIC-001](backlog/RITK-SNAP-NATIVE-DYNAMIC-001.md) — Add time-intensity curves and DSA — todo
<a id="RITK-SNAP-NATIVE-EXPORT-001"></a>- [RITK-SNAP-NATIVE-EXPORT-001](backlog/RITK-SNAP-NATIVE-EXPORT-001.md) — Export and copy displayed images — todo
<a id="RITK-SNAP-NATIVE-FILTERS-001"></a>- [RITK-SNAP-NATIVE-FILTERS-001](backlog/RITK-SNAP-NATIVE-FILTERS-001.md) — Apply image filters natively — todo
<a id="RITK-SNAP-NATIVE-FUSION-001"></a>- [RITK-SNAP-NATIVE-FUSION-001](backlog/RITK-SNAP-NATIVE-FUSION-001.md) — Fuse series and read SUV natively — blocked on native comparison controls
<a id="RITK-SNAP-NATIVE-LINK-001"></a>- [RITK-SNAP-NATIVE-LINK-001](backlog/RITK-SNAP-NATIVE-LINK-001.md) — Link panes by patient position — todo
<a id="RITK-SNAP-NATIVE-LOAD-001"></a>- [RITK-SNAP-NATIVE-LOAD-001](backlog/RITK-SNAP-NATIVE-LOAD-001.md) — Browse a study while it loads — todo
<a id="RITK-SNAP-NATIVE-INPUT-PARITY-001"></a>- [RITK-SNAP-NATIVE-INPUT-PARITY-001](backlog/RITK-SNAP-NATIVE-INPUT-PARITY-001.md) — Finish native input parity — todo
<a id="RITK-SNAP-NATIVE-OPEN-001"></a>- [RITK-SNAP-NATIVE-OPEN-001](backlog/RITK-SNAP-NATIVE-OPEN-001.md) — Open files, archives and drops natively — todo
<a id="RITK-SNAP-NATIVE-OVERLAY-TEXT-001"></a>- [RITK-SNAP-NATIVE-OVERLAY-TEXT-001](backlog/RITK-SNAP-NATIVE-OVERLAY-TEXT-001.md) — Draw clinical image overlays natively — todo
<a id="RITK-SNAP-NATIVE-PACS-001"></a>- [RITK-SNAP-NATIVE-PACS-001](backlog/RITK-SNAP-NATIVE-PACS-001.md) — Query and receive PACS studies natively — todo
<a id="RITK-SNAP-NATIVE-PROJECTION-001"></a>- [RITK-SNAP-NATIVE-PROJECTION-001](backlog/RITK-SNAP-NATIVE-PROJECTION-001.md) — Select projections from the native menu — todo
<a id="RITK-SNAP-NATIVE-RT-001"></a>- [RITK-SNAP-NATIVE-RT-001](backlog/RITK-SNAP-NATIVE-RT-001.md) — Show RT structures and dose natively — todo
<a id="RITK-SNAP-NATIVE-SHELL-001"></a>- [RITK-SNAP-NATIVE-SHELL-001](backlog/RITK-SNAP-NATIVE-SHELL-001.md) — Finish native shell integration — todo
<a id="RITK-SNAP-NATIVE-TAGS-001"></a>- [RITK-SNAP-NATIVE-TAGS-001](backlog/RITK-SNAP-NATIVE-TAGS-001.md) — Show searchable DICOM tags natively — todo
<a id="RITK-SNAP-NATIVE-VR-001"></a>- [RITK-SNAP-NATIVE-VR-001](backlog/RITK-SNAP-NATIVE-VR-001.md) — Render volumes in the native shell — todo
<a id="RITK-SNAP-NATIVE-SEGMENTATION-001"></a>- [RITK-SNAP-NATIVE-SEGMENTATION-001](backlog/RITK-SNAP-NATIVE-SEGMENTATION-001.md) — Draw and edit label maps natively — todo
<a id="SEC-MIF-DETACHED-DEVICE-NAME-001"></a>- [SEC-MIF-DETACHED-DEVICE-NAME-001](backlog/SEC-MIF-DETACHED-DEVICE-NAME-001.md) — Refuse device names as .mif detached data and pin the check's order — todo
<a id="RITK-MGH-ERROR-TEST-PINS-001"></a>- [RITK-MGH-ERROR-TEST-PINS-001](backlog/RITK-MGH-ERROR-TEST-PINS-001.md) — Pin the MGH error fields and limits that mutants survive — todo
<a id="RITK-SNAP-EFRAME-DELETE-001"></a>- [RITK-SNAP-EFRAME-DELETE-001](backlog/RITK-SNAP-EFRAME-DELETE-001.md) — Delete the eframe entry points — todo
<a id="RITK-SNAP-EFRAME-PORT-001"></a>- [RITK-SNAP-EFRAME-PORT-001](backlog/RITK-SNAP-EFRAME-PORT-001.md) — Port smoothing and segmentation filter ranges — todo
<a id="RITK-SNAP-EFRAME-PORT-002"></a>- [RITK-SNAP-EFRAME-PORT-002](backlog/RITK-SNAP-EFRAME-PORT-002.md) — Port morphology, geometry, CPR and pointwise filter ranges — todo
<a id="RITK-SNAP-EFRAME-PORT-003"></a>- [RITK-SNAP-EFRAME-PORT-003](backlog/RITK-SNAP-EFRAME-PORT-003.md) — Port filter labels and starting parameters — todo
<a id="RITK-SNAP-DICOM-GRAPHICS-001"></a>- [RITK-SNAP-DICOM-GRAPHICS-001](backlog/RITK-SNAP-DICOM-GRAPHICS-001.md) — Display DICOM graphic overlay planes — todo
<a id="RITK-SNAP-ROI-EDIT-001"></a>- [RITK-SNAP-ROI-EDIT-001](backlog/RITK-SNAP-ROI-EDIT-001.md) — Move and resize image regions — todo
<a id="RITK-SNAP-MARK-EDIT-001"></a>- [RITK-SNAP-MARK-EDIT-001](backlog/RITK-SNAP-MARK-EDIT-001.md) — Edit completed angles and drawn marks — todo
<a id="RITK-SNAP-ANNOTATION-TEXT-001"></a>- [RITK-SNAP-ANNOTATION-TEXT-001](backlog/RITK-SNAP-ANNOTATION-TEXT-001.md) — Position readable annotation labels — todo
<a id="RITK-SNAP-ANNOTATION-COPY-001"></a>- [RITK-SNAP-ANNOTATION-COPY-001](backlog/RITK-SNAP-ANNOTATION-COPY-001.md) — Copy and duplicate image annotations — todo
<a id="RITK-SNAP-US-REGION-001"></a>- [RITK-SNAP-US-REGION-001](backlog/RITK-SNAP-US-REGION-001.md) — Measure calibrated ultrasound regions — todo
<a id="RITK-SNAP-FORMAT-VIEWER-001"></a>- [RITK-SNAP-FORMAT-VIEWER-001](backlog/RITK-SNAP-FORMAT-VIEWER-001.md) — Exercise encoded studies in the viewer — todo
<a id="RITK-SNAP-CUSTOM-WINDOW-001"></a>- [RITK-SNAP-CUSTOM-WINDOW-001](backlog/RITK-SNAP-CUSTOM-WINDOW-001.md) — Retain precise custom window settings — todo
<a id="RITK-SNAP-LARGE-STUDY-001"></a>- [RITK-SNAP-LARGE-STUDY-001](backlog/RITK-SNAP-LARGE-STUDY-001.md) — Measure native study resource bounds — todo
<a id="RITK-SNAP-NATIVE-DISPLAY-001"></a>- [RITK-SNAP-NATIVE-DISPLAY-001](backlog/RITK-SNAP-NATIVE-DISPLAY-001.md) — Route display controls in the native shell — todo
<a id="RITK-SNAP-NAVIGATION-UNITS-001"></a>- [RITK-SNAP-NAVIGATION-UNITS-001](backlog/RITK-SNAP-NAVIGATION-UNITS-001.md) — Keep navigation in display coordinates — todo
<a id="RITK-SNAP-IMAGE-FIT-001"></a>- [RITK-SNAP-IMAGE-FIT-001](backlog/RITK-SNAP-IMAGE-FIT-001.md) — Share continuous sample-aspect fitting — todo
<a id="RITK-SNAP-EFRAME-NAVIGATION-001"></a>- [RITK-SNAP-EFRAME-NAVIGATION-001](backlog/RITK-SNAP-EFRAME-NAVIGATION-001.md) — Paint pan and retain the press origin — todo
<a id="RITK-SNAP-ARROW-PROPERTIES-001"></a>- [RITK-SNAP-ARROW-PROPERTIES-001](backlog/RITK-SNAP-ARROW-PROPERTIES-001.md) — Edit arrow text and appearance — todo
<a id="RITK-SNAP-CALIBRATION-INPUT-001"></a>- [RITK-SNAP-CALIBRATION-INPUT-001](backlog/RITK-SNAP-CALIBRATION-INPUT-001.md) — Expose calibration sources and properties — todo
<a id="RITK-SNAP-MEASUREMENT-UNITS-001"></a>- [RITK-SNAP-MEASUREMENT-UNITS-001](backlog/RITK-SNAP-MEASUREMENT-UNITS-001.md) — Retain missing-spacing measurement units — todo
<a id="RITK-SNAP-ELLIPSE-CONSTRAINT-001"></a>- [RITK-SNAP-ELLIPSE-CONSTRAINT-001](backlog/RITK-SNAP-ELLIPSE-CONSTRAINT-001.md) — Constrain ellipse drawing to a circle — todo
<a id="RITK-SNAP-ELLIPSE-PIXEL-MASK-001"></a>- [RITK-SNAP-ELLIPSE-PIXEL-MASK-001](backlog/RITK-SNAP-ELLIPSE-PIXEL-MASK-001.md) — Show ellipse contributing-pixel outlines — todo
<a id="RITK-SNAP-ROI-NATIVE-001"></a>- [RITK-SNAP-ROI-NATIVE-001](backlog/RITK-SNAP-ROI-NATIVE-001.md) — Test native ROI creation and labels — todo
<a id="RITK-SNAP-ROI-LABEL-STATISTICS-001"></a>- [RITK-SNAP-ROI-LABEL-STATISTICS-001](backlog/RITK-SNAP-ROI-LABEL-STATISTICS-001.md) — Show complete ROI label statistics — todo

- [x] [major] Carry typed DICOM geometry and isotope activity into the viewer.
  Validate file round trips, isotope units by SOP class, and SUV pixel units.

- [x] [major] Quantify Enhanced PET spatial voxels using the declared real-world
  mapping and frame-appropriate isotope activity.

- [x] [major] Reconstruct and select temporal and gated Enhanced PET volumes within one object.

- [x] [minor] Reconstruct Enhanced PET concatenations from the exact scanned
  instances. Validate ordinal continuity, geometry, and per-frame SUV.

- [x] [patch] Use the published Aequitas and Eunomia releases in RITK.
- [x] [patch] Verify geometry-based interpolation of an interior Enhanced
  PET plane between complete concatenation members.
- [x] [minor] Label synthesized Enhanced PET planes on the storage-depth
  view. Preserve each selected reconstruction's measured, interpolated,
  and copied-edge provenance.
- [x] [patch] Resolve one published Aequitas 0.5.0 and one Eunomia 0.12.1
  in the public compatibility candidate.
- [ ] [patch] Advance the lock to merged provider revisions and repeat native
  viewer and registry package verification on the integration head.

- [x] [major] Approve [numbered DICOM directory exports](docs/adr/0068-number-dicom-directory-exports.md) for `ritk-io` 0.8.0 and `ritk-python` 0.13.0. See [RITK-DICOM-ANONYMIZE-DIRECTORY-NAMES-001](gap_audit.md#RITK-DICOM-ANONYMIZE-DIRECTORY-NAMES-001) for regression evidence.
- [x] [major] Correct [real-data DICOM export structure](docs/adr/0069-preserve-dicom-export-structure.md). See [RITK-DICOM-ANONYMIZE-REAL-001](gap_audit.md#RITK-DICOM-ANONYMIZE-REAL-001).
