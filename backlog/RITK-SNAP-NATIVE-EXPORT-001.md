<a id="RITK-SNAP-NATIVE-EXPORT-001"></a>
## RITK-SNAP-NATIVE-EXPORT-001 — Export and copy displayed images — todo
- outcome: Native menus export the displayed image to PNG, JPEG or BMP and original-format DICOM, and copy the image to the clipboard. Native users can export a cine loop to an MP4 movie at the chosen frame rate. Native users can save and restore annotations, layout and display state, with UI preferences persisted between runs. Native users can export annotation CSV and the clinical report bundle, save and load label maps as NIfTI or DICOM SEG, and export label surfaces as VTK.
- acceptance: Tests decode each export and assert its pixels equal the presented frame; the clipboard test asserts the written bitmap. A test decodes the movie and asserts frame count, rate and first-frame pixels within the codec's derived error. A round-trip test asserts restored state equals saved state and rejects a tampered session.
- scope: `crates/ritk-snap/src/presentation/native_session/{composition,window_actions}.rs`, metis clipboard image write, `crates/ritk-snap/src/presentation/native_session/window_actions.rs`, encoder provider decision in an ADR, `crates/ritk-snap/src/session/`, `crates/ritk-snap/src/presentation/native_session/window_actions.rs`, `crates/ritk-snap/src/ui/annotation_csv.rs`, `crates/ritk-snap/src/app/clinical_distribution*`, `crates/ritk-snap/src/app/surface_export*`, `crates/ritk-snap/src/app/io_ops/`
- next: Route the capture path through a save dialog.
- basis: 30b8ba098
- needs: none
- milestone: M-viewer-access
- priority: feature
