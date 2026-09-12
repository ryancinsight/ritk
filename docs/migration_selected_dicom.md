# Migrate DICOM acquisition selection

The unreleased selection change implements
[ADR 0026](adr/0026-viewer-presentation-migration.md). It changes public viewer
carriers and directory-opening behavior; it does not publish a new version.

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
| `SidebarPanel::new` | Pass the last successfully loaded acquisition as `Option<&DicomSeriesInfo>`. |
| `SidebarPanel::with_tag_search` | Use `new`; the removed alias ignored its tag-search argument. Struct literals use `selected_acquisition` instead of `selected_path`. |
| `SidebarPanel::show` | Consume the returned `Arc<DicomSeriesInfo>`, keeping both UID and files. Update highlighting only after successful loading. |
| `ViewerSessionSnapshot` struct construction | Supply `SessionFormat::default()` and `Option<StudySource>`; filesystem sources use `StudySource::Path`, selected series use `StudySource::Dicom { series_uid, files }`. |
| `AppLaunchOptions` struct construction | Supply `capture: None`, `initial_series_uid: None`, and `metis_native: false` for the eframe shell, or set `metis_native: true` with an initial path and optional `initial_series_uid` to use the Windows Métis host. `Default` keeps all optional startup inputs disabled. |
| `OverlayRenderer::draw` | Consume its `Option<String>` result. Pass returned overflow metadata to `OverlayRenderer::show_details` with the viewport UI and rectangle. |
| `OverlayRenderer::draw_orientation_labels` | Remove the separate call; `draw` now lays out orientation and corner annotations together. |

New sessions serialize format 2 and preserve selected DICOM UID and member
paths. Unversioned and version-1 sessions remain readable at the persisted
format boundary and normalize to format 2. Legacy directory sources containing
several series require a new explicit selection. Unknown versions fail.
Restoring a session validates its source before replacing either the displayed
study or presentation controls; a failure preserves the current viewer state.

The [user workflow](manual/dicom-workflow.md) contains runnable commands,
synthetic pixel/coordinate oracles, and the distinction between software
slice captures and the rendered application window.

The Métis native-host increment adds `metis_native` to `AppLaunchOptions` and
is therefore a breaking public-struct change. The native series-selection
increment adds `initial_series_uid` to the same public struct. Downstream
struct literals must set both fields explicitly (`initial_series_uid: None` and
`metis_native: false` for the eframe shell, or a selected UID with
`metis_native: true` and an initial path for the Windows Métis shell); callers
that use `Default` need no source change. `load_volume_from_series_uid` is the
RITK-owned path API for non-interactive selection and never chooses an
arbitrary series.
