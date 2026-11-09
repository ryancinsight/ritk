# Migrate DICOM viewer contracts

The unreleased viewer changes implement [ADR 0026](adr/0026-viewer-presentation-migration.md)
and [ADR 0052](adr/0052-native-oblique-mpr.md). They change public viewer
carriers, patient-space measurement types, and session serialization; they do
not publish a new version.

Use `ritk_io::scan_dicom_path` for a selected instance, directory, or DICOMDIR.
A selected instance chooses its SeriesInstanceUID before collecting neighbouring
members. An unselected directory or byte batch must contain one image series;
mixed series now return an error instead of selecting the largest population
or combining equal populations. Use discovery followed by explicit selection
when a file set contains several acquisitions.

`ritk_io::scan_dicom_files` validates an exact member list. The existing
`ritk_io::scan_dicom_directory` remains the multi-series discovery API; its
result is distinct from the reconstructed-volume scan descriptor. An existing
DICOMDIR restricts discovery to its references, including when browsing a
parent directory. Invalid indices fail instead of enabling folder fallback.

| Public caller | Required change |
| --- | --- |
| `scan_folder_for_series` | Handle its `Result`; discovery failures no longer become an empty or partial successful tree. |
| `SeriesEntry` / `SeriesNode` struct construction | Retain `Arc<ritk_io::DicomSeriesInfo>` in `acquisition`; use `SeriesEntry::from_dicom_series_info` for a discovery result. Display fields derive from this descriptor. |
| `SeriesNode<'a>` | Remove the lifetime argument; the node owns its shared acquisition descriptor. |
| `SeriesTree::find_by_folder` | Select by UID with `find_by_uid`, then retain the returned node's acquisition. A shared folder cannot identify one series. |
| `run_native_viewer` | Pass `None` for the series selection to preserve unambiguous-path loading, or pass `Some(series_instance_uid)` with the startup path to select one discovered acquisition before the Métis host starts. |
| `ViewerSessionSnapshot` struct construction | Supply `SessionFormat::default()` and `Option<StudySource>`; filesystem sources use `StudySource::Path`, selected series use `StudySource::Dicom { series_uid, files }`. |
| `AppLaunchOptions` struct construction | Supply `capture: None` and `initial_series_uid: None`, or an optional initial path with an optional `initial_series_uid`. The `metis_native` field is gone because Métis is the only shell. |
| `Annotation` | Keep a wildcard arm for the non-exhaustive enum; handle `PatientLength` in patient millimetres and retain its full `f64` distance. |
| `AnnotationSummary::primary_value()` | Consume `f64` so patient-space distances are not narrowed to `f32`. |

New sessions serialize format 3 and preserve selected DICOM UID and member
paths. Unversioned and version-1 path sessions remain readable, and version-2
exact-acquisition sessions retain their source; all normalize to format 3.
Format 3 patient-space lengths serialize validated millimetre endpoints and
derive their three-dimensional Euclidean distance from those endpoints. Legacy
directory sources containing several series require a new explicit selection.
Unknown versions fail.
Restoring a session validates its source before replacing either the displayed
study or presentation controls; a failure preserves the current viewer state.
The native shell reaches this path once `RITK-SNAP-NATIVE-EXPORT-001` lands.

The [user workflow](manual/dicom-workflow.md) contains runnable commands,
synthetic pixel/coordinate oracles, and the distinction between software
slice captures and the rendered application window.

The native series-selection increment adds `initial_series_uid` to
`AppLaunchOptions`, a breaking public-struct change. Struct literals set it
explicitly: `None`, or a selected UID with an optional initial path. When the
path is absent, the Windows Métis shell obtains one from the native folder
picker. The eframe shell and its widgets are deleted; Métis is the only shell.
`load_volume_from_series_uid` is the
RITK-owned path API for non-interactive selection and never chooses an
arbitrary series.
