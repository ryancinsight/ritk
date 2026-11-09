# RITK execution backlog

> ## Vocabulary policy
>
> New migration text uses provider/native names directly (Coeus,
> MoiraiBackend, Leto, Eunomia, native) and does not introduce new
> Atlas-* migration labels. Historical PM entries retain their original
> wording unless touched by the current slice. Domain medical-atlas terms are
> preserved.

# Milestones

<a id="M-viewer-reading"></a>
## M-viewer-reading — 0.2.0 — read a single study on the native shell
- exit: `cargo nextest run -p ritk-snap -E 'test(pointer_bindings) or test(modifier_gesture) or test(native_annotation) or test(native_display) or test(native_view_transform) or test(native_overlay_text) or test(measure_set)'`
- needs: none
- non-goals: PACS, archive, 3D rendering

<a id="M-viewer-compare"></a>
## M-viewer-compare — 0.3.0 — compare series by patient position
- exit: `cargo nextest run -p ritk-snap -E 'test(native_link) or test(native_load) or test(native_projection) or test(oblique)'`; `metis-browser-dicom.yml` green on chromium jobs
- needs: M-viewer-reading
- non-goals: fusion

<a id="M-viewer-access"></a>
## M-viewer-access — 0.4.0 — get studies in and images out
- exit: `cargo nextest run -p ritk-snap -E 'test(native_open) or test(native_tags) or test(native_export) or test(native_session_round_trip) or test(native_pacs) or test(native_archive)'`
- needs: M-viewer-compare
- non-goals: cloud storage

<a id="M-viewer-advanced"></a>
## M-viewer-advanced — 1.0.0 — advanced reading tools
- exit: `cargo nextest run -p ritk-snap -E 'test(native_filter) or test(native_vr) or test(native_fusion) or test(native_dynamic) or test(native_documents) or test(native_movie) or test(native_shell)'`
- exit: `cargo semver-checks` for `ritk-snap`; `cargo publish --dry-run` for the viewer's closure
- needs: M-viewer-access
- non-goals: encrypted ZIP, optical disc burning, unlisted RadiAnt add-ons

# Items

<a id="RITK-FORMAT-RESCUE-RECONCILE-001"></a>- [RITK-FORMAT-RESCUE-RECONCILE-001](backlog/RITK-FORMAT-RESCUE-RECONCILE-001.md) — Rehome format work from stale rescue branches — blocked
<a id="RITK-SNAP-METIS-SHARED-TEXT-001"></a>- [RITK-SNAP-METIS-SHARED-TEXT-001](backlog/RITK-SNAP-METIS-SHARED-TEXT-001.md) — Build display text from shared strings — todo
<a id="RITK-GAP-2026-08-20-02"></a>- [RITK-GAP-2026-08-20-02](backlog/RITK-GAP-2026-08-20-02.md) — Fuzz supported format parsers — todo
<a id="RITK-GAP-2026-08-20-09"></a>- [RITK-GAP-2026-08-20-09](backlog/RITK-GAP-2026-08-20-09.md) — Derive mutual-information subsample stride — todo
<a id="RITK-PYTHON-VTK-DIRECTION-001"></a>- [RITK-PYTHON-VTK-DIRECTION-001](backlog/RITK-PYTHON-VTK-DIRECTION-001.md) — Python images cannot be written to a no-direction format (legacy VTK, Analyze 7.5) — todo
<a id="RITK-METAIMAGE-CONVERSION-INVENTORY-001"></a>- [RITK-METAIMAGE-CONVERSION-INVENTORY-001](backlog/RITK-METAIMAGE-CONVERSION-INVENTORY-001.md) — Map MetaImage volume semantics — todo
<a id="RITK-MGH-CONVERSION-INVENTORY-001"></a>- [RITK-MGH-CONVERSION-INVENTORY-001](backlog/RITK-MGH-CONVERSION-INVENTORY-001.md) — Map MGH volume semantics — todo
<a id="RITK-VTK-IMAGE-CONTRACT-001"></a>- [RITK-VTK-IMAGE-CONTRACT-001](backlog/RITK-VTK-IMAGE-CONTRACT-001.md) — Separate VTK image semantics — todo
<a id="SEC-446-05"></a>- [SEC-446-05](backlog/SEC-446-05.md) — Bound format parser allocations by input — todo
<a id="RITK-NRRD-NIFTI-001"></a>- [RITK-NRRD-NIFTI-001](backlog/RITK-NRRD-NIFTI-001.md) — Convert NRRD and NIfTI volumes — blocked
<a id="RITK-METAIMAGE-CONVERSION-001"></a>- [RITK-METAIMAGE-CONVERSION-001](backlog/RITK-METAIMAGE-CONVERSION-001.md) — Preserve MetaImage volume semantics — blocked
<a id="RITK-MINC-CONVERSION-001"></a>- [RITK-MINC-CONVERSION-001](backlog/RITK-MINC-CONVERSION-001.md) — Preserve MINC volume semantics — blocked
<a id="RITK-MIF-CONVERSION-001"></a>- [RITK-MIF-CONVERSION-001](backlog/RITK-MIF-CONVERSION-001.md) — Preserve MRtrix image semantics — blocked
<a id="RITK-MGH-CONVERSION-001"></a>- [RITK-MGH-CONVERSION-001](backlog/RITK-MGH-CONVERSION-001.md) — Preserve MGH and MGZ volume semantics — blocked
<a id="RITK-VTK-VOLUME-CONVERSION-001"></a>- [RITK-VTK-VOLUME-CONVERSION-001](backlog/RITK-VTK-VOLUME-CONVERSION-001.md) — Preserve VTK image-volume semantics — blocked
<a id="RITK-SNAP-OBLIQUE-SESSION-WIRING-001"></a>- [RITK-SNAP-OBLIQUE-SESSION-WIRING-001](backlog/RITK-SNAP-OBLIQUE-SESSION-WIRING-001.md) — Wire the oblique session pane — blocked
<a id="RITK-DICOM-CALIBRATION-UNITS-001"></a>- [RITK-DICOM-CALIBRATION-UNITS-001](backlog/RITK-DICOM-CALIBRATION-UNITS-001.md) — Preserve DICOM values and units — blocked
<a id="RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001"></a>- [RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001](backlog/RITK-SNAP-PATIENT-MEASUREMENT-OVERLAY-001.md) — Present patient-space lengths — todo
<a id="RITK-CI-MERGE-GATE-001"></a>- [RITK-CI-MERGE-GATE-001](backlog/RITK-CI-MERGE-GATE-001.md) — Gate pull requests through one CI check — todo
<a id="RITK-REGISTRY-PUBLICATION-001"></a>- [RITK-REGISTRY-PUBLICATION-001](backlog/RITK-REGISTRY-PUBLICATION-001.md) — Publish the required registry dependency graph — blocked
<a id="RITK-GIFTI-SURFACE-001"></a>- [RITK-GIFTI-SURFACE-001](backlog/RITK-GIFTI-SURFACE-001.md) — Preserve complete GIFTI surface documents — todo
<a id="RITK-MESH-CONVERSION-001"></a>- [RITK-MESH-CONVERSION-001](backlog/RITK-MESH-CONVERSION-001.md) — Convert supported mesh formats — todo
<a id="RITK-TRACTOGRAM-CONVERSION-001"></a>- [RITK-TRACTOGRAM-CONVERSION-001](backlog/RITK-TRACTOGRAM-CONVERSION-001.md) — Convert supported tractogram formats — todo
<a id="RITK-GAP-2026-08-20-01"></a>- [RITK-GAP-2026-08-20-01](backlog/RITK-GAP-2026-08-20-01.md) — Retire duplicate X and X_native APIs — todo
<a id="RITK-RASTER-CONVERSION-001"></a>- [RITK-RASTER-CONVERSION-001](backlog/RITK-RASTER-CONVERSION-001.md) — Convert PNG TIFF and JPEG rasters — blocked
<a id="RITK-SNAP-OBLIQUE-SESSION-MODULES-001"></a>- [RITK-SNAP-OBLIQUE-SESSION-MODULES-001](backlog/RITK-SNAP-OBLIQUE-SESSION-MODULES-001.md) — Separate native oblique session concerns — blocked
<a id="RITK-DICOM-CONVERSION-001"></a>- [RITK-DICOM-CONVERSION-001](backlog/RITK-DICOM-CONVERSION-001.md) — Convert DICOM series through RITK — blocked
<a id="RITK-FORMAT-CONVERSION-001"></a>- [RITK-FORMAT-CONVERSION-001](backlog/RITK-FORMAT-CONVERSION-001.md) — Convert supported formats through RITK — blocked
<a id="RITK-SNAP-OBLIQUE-NATIVE-001"></a>- [RITK-SNAP-OBLIQUE-NATIVE-001](backlog/RITK-SNAP-OBLIQUE-NATIVE-001.md) — Native oblique MPR — blocked
<a id="RITK-BROWSER-READ-001"></a>- [RITK-BROWSER-READ-001](backlog/RITK-BROWSER-READ-001.md) — Resolve WebKit study-file reads — blocked
<a id="RITK-BROWSER-WEBGPU-001"></a>- [RITK-BROWSER-WEBGPU-001](backlog/RITK-BROWSER-WEBGPU-001.md) — Demonstrate browser WebGPU presentation — blocked
<a id="RITK-GAP-2026-08-20-04"></a>- [RITK-GAP-2026-08-20-04](backlog/RITK-GAP-2026-08-20-04.md) — Reconcile tracked imaging payload provenance — todo
<a id="RITK-GAP-2026-08-20-06"></a>- [RITK-GAP-2026-08-20-06](backlog/RITK-GAP-2026-08-20-06.md) — Raise lint and documentation floors — todo
<a id="RITK-GAP-2026-08-20-08"></a>- [RITK-GAP-2026-08-20-08](backlog/RITK-GAP-2026-08-20-08.md) — Complete registration and dispatch chapters — todo
<a id="RITK-SNAP-OBLIQUE-APP-TESTS-001"></a>- [RITK-SNAP-OBLIQUE-APP-TESTS-001](backlog/RITK-SNAP-OBLIQUE-APP-TESTS-001.md) — Verify oblique adapter boundaries — blocked
<a id="RITK-FIXTURE-RETENTION-001"></a>- [RITK-FIXTURE-RETENTION-001](backlog/RITK-FIXTURE-RETENTION-001.md) — Decide imaging fixture retention — blocked
<a id="RITK-TEST-DATASET-RELAY-001"></a>- [RITK-TEST-DATASET-RELAY-001](backlog/RITK-TEST-DATASET-RELAY-001.md) — Externalize large public test datasets — blocked
<a id="RITK-SNAP-OBLIQUE-SESSION-TESTS-001"></a>- [RITK-SNAP-OBLIQUE-SESSION-TESTS-001](backlog/RITK-SNAP-OBLIQUE-SESSION-TESTS-001.md) — Verify oblique rendering and startup — blocked
<a id="RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001"></a>- [RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001](backlog/RITK-SNAP-OBLIQUE-INTERACTION-TESTS-001.md) — Verify complete native gestures — blocked
<a id="RITK-SNAP-OBLIQUE-MANUAL-001"></a>- [RITK-SNAP-OBLIQUE-MANUAL-001](backlog/RITK-SNAP-OBLIQUE-MANUAL-001.md) — Demonstrate oblique MPR in the user manual — blocked
<a id="RITK-FORMAT-BULK-DECODE-001"></a>- [RITK-FORMAT-BULK-DECODE-001](backlog/RITK-FORMAT-BULK-DECODE-001.md) — Decode sample buffers through one bulk path — todo
<a id="RITK-SNAP-INTERACTION-REGIONS-001"></a>- [RITK-SNAP-INTERACTION-REGIONS-001](backlog/RITK-SNAP-INTERACTION-REGIONS-001.md) — Separate region interaction tests — todo
<a id="RITK-GAP-2026-08-20-05"></a>- [RITK-GAP-2026-08-20-05](backlog/RITK-GAP-2026-08-20-05.md) — Derive test budgets and remove dead filters — todo
<a id="RITK-GAP-2026-08-20-07"></a>- [RITK-GAP-2026-08-20-07](backlog/RITK-GAP-2026-08-20-07.md) — Restore the changelog version axis — todo
<a id="RITK-SNAP-INTERACTION-MEASUREMENTS-001"></a>- [RITK-SNAP-INTERACTION-MEASUREMENTS-001](backlog/RITK-SNAP-INTERACTION-MEASUREMENTS-001.md) — Separate measurement tests — blocked
<a id="RITK-SNAP-INTERACTION-STATE-001"></a>- [RITK-SNAP-INTERACTION-STATE-001](backlog/RITK-SNAP-INTERACTION-STATE-001.md) — Separate interaction state tests — blocked
<a id="RITK-SNAP-INTERACTION-WINDOW-LEVEL-001"></a>- [RITK-SNAP-INTERACTION-WINDOW-LEVEL-001](backlog/RITK-SNAP-INTERACTION-WINDOW-LEVEL-001.md) — Separate window-level tests — blocked
<a id="RITK-SNAP-OBLIQUE-APP-ADAPTER-001"></a>- [RITK-SNAP-OBLIQUE-APP-ADAPTER-001](backlog/RITK-SNAP-OBLIQUE-APP-ADAPTER-001.md) — Route oblique viewer actions — todo
<a id="RITK-SNAP-RADIANT-UI-001"></a>- [RITK-SNAP-RADIANT-UI-001](backlog/RITK-SNAP-RADIANT-UI-001.md) — Organize a multi-series DICOM viewer workspace — todo
<a id="RITK-SNAP-OBLIQUE-ROUTING-001"></a>- [RITK-SNAP-OBLIQUE-ROUTING-001](backlog/RITK-SNAP-OBLIQUE-ROUTING-001.md) — Route native oblique input — blocked
<a id="RITK-BROWSER-TOOL-TRACE-001"></a>- [RITK-BROWSER-TOOL-TRACE-001](backlog/RITK-BROWSER-TOOL-TRACE-001.md) — Publish ROI results in the browser journey — todo
<a id="RITK-DEPS-MOIRAI-09-COEVOLUTION-001"></a>- [RITK-DEPS-MOIRAI-09-COEVOLUTION-001](backlog/RITK-DEPS-MOIRAI-09-COEVOLUTION-001.md) — Advance ritk past the moirai 0.9 / metis 0.4 provider bumps — blocked
<a id="RITK-SNAP-EFRAME-REMOVAL-001"></a>- [RITK-SNAP-EFRAME-REMOVAL-001](backlog/RITK-SNAP-EFRAME-REMOVAL-001.md) — Remove the eframe shell — todo
<a id="RITK-SNAP-NATIVE-ARCHIVE-001"></a>- [RITK-SNAP-NATIVE-ARCHIVE-001](backlog/RITK-SNAP-NATIVE-ARCHIVE-001.md) — Keep a local study archive — blocked
<a id="RITK-SNAP-NATIVE-DISPLAY-001"></a>- [RITK-SNAP-NATIVE-DISPLAY-001](backlog/RITK-SNAP-NATIVE-DISPLAY-001.md) — Route display controls in the native shell — todo
<a id="RITK-SNAP-NATIVE-DOCUMENTS-001"></a>- [RITK-SNAP-NATIVE-DOCUMENTS-001](backlog/RITK-SNAP-NATIVE-DOCUMENTS-001.md) — Display reports, PDFs and video objects — todo
<a id="RITK-SNAP-NATIVE-DYNAMIC-001"></a>- [RITK-SNAP-NATIVE-DYNAMIC-001](backlog/RITK-SNAP-NATIVE-DYNAMIC-001.md) — Add time-intensity curves and DSA — blocked
<a id="RITK-SNAP-NATIVE-EXPORT-001"></a>- [RITK-SNAP-NATIVE-EXPORT-001](backlog/RITK-SNAP-NATIVE-EXPORT-001.md) — Export and copy displayed images — todo
<a id="RITK-SNAP-NATIVE-FILTERS-001"></a>- [RITK-SNAP-NATIVE-FILTERS-001](backlog/RITK-SNAP-NATIVE-FILTERS-001.md) — Apply image filters natively — todo
<a id="RITK-SNAP-NATIVE-FUSION-001"></a>- [RITK-SNAP-NATIVE-FUSION-001](backlog/RITK-SNAP-NATIVE-FUSION-001.md) — Fuse series and read SUV natively — blocked
<a id="RITK-SNAP-NATIVE-LINK-001"></a>- [RITK-SNAP-NATIVE-LINK-001](backlog/RITK-SNAP-NATIVE-LINK-001.md) — Link panes by patient position — todo
<a id="RITK-SNAP-NATIVE-LOAD-001"></a>- [RITK-SNAP-NATIVE-LOAD-001](backlog/RITK-SNAP-NATIVE-LOAD-001.md) — Browse a study while it loads — todo
<a id="RITK-SNAP-NATIVE-MEASURE-SET-001"></a>- [RITK-SNAP-NATIVE-MEASURE-SET-001](backlog/RITK-SNAP-NATIVE-MEASURE-SET-001.md) — Complete the measurement and annotation set — blocked
<a id="RITK-SNAP-NATIVE-OPEN-001"></a>- [RITK-SNAP-NATIVE-OPEN-001](backlog/RITK-SNAP-NATIVE-OPEN-001.md) — Open files, archives and drops natively — todo
<a id="RITK-SNAP-NATIVE-OVERLAY-TEXT-001"></a>- [RITK-SNAP-NATIVE-OVERLAY-TEXT-001](backlog/RITK-SNAP-NATIVE-OVERLAY-TEXT-001.md) — Draw clinical image overlays natively — todo
<a id="RITK-SNAP-NATIVE-PACS-001"></a>- [RITK-SNAP-NATIVE-PACS-001](backlog/RITK-SNAP-NATIVE-PACS-001.md) — Query and receive PACS studies natively — todo
<a id="RITK-SNAP-NATIVE-POINTER-CAPTURE-001"></a>- [RITK-SNAP-NATIVE-POINTER-CAPTURE-001](backlog/RITK-SNAP-NATIVE-POINTER-CAPTURE-001.md) — End a drag released off the window — todo
<a id="RITK-SNAP-NATIVE-PROJECTION-001"></a>- [RITK-SNAP-NATIVE-PROJECTION-001](backlog/RITK-SNAP-NATIVE-PROJECTION-001.md) — Select projections from the native menu — todo
<a id="RITK-SNAP-NATIVE-RT-001"></a>- [RITK-SNAP-NATIVE-RT-001](backlog/RITK-SNAP-NATIVE-RT-001.md) — Show RT structures and dose natively — todo
<a id="RITK-SNAP-NATIVE-SHELL-001"></a>- [RITK-SNAP-NATIVE-SHELL-001](backlog/RITK-SNAP-NATIVE-SHELL-001.md) — Finish native shell integration — todo
<a id="RITK-SNAP-NATIVE-TAGS-001"></a>- [RITK-SNAP-NATIVE-TAGS-001](backlog/RITK-SNAP-NATIVE-TAGS-001.md) — Show searchable DICOM tags natively — todo
<a id="RITK-SNAP-NATIVE-VR-001"></a>- [RITK-SNAP-NATIVE-VR-001](backlog/RITK-SNAP-NATIVE-VR-001.md) — Render volumes in the native shell — blocked
<a id="RITK-SNAP-RADIANT-INPUT-001"></a>- [RITK-SNAP-RADIANT-INPUT-001](backlog/RITK-SNAP-RADIANT-INPUT-001.md) — Let readers reassign mouse buttons — todo
<a id="RITK-SNAP-NATIVE-SEGMENTATION-001"></a>- [RITK-SNAP-NATIVE-SEGMENTATION-001](backlog/RITK-SNAP-NATIVE-SEGMENTATION-001.md) — Draw and edit label maps natively — todo
