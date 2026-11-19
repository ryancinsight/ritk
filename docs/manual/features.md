# AsterSlice feature comparison

This chapter compares AsterSlice with the supplied RadiAnt DICOM Viewer 2021.2
manual. Page numbers refer to that PDF. The Windows column describes controls
a reader can reach in the Métis viewer. Evidence links point to assertions for
the stated behavior. Image-processing tests and native-session tests cover
different boundaries.

Use the [user guide](dicom-workflow.md) for instructions and the
[developer reference](development.md) for capture and browser procedures.

## Reading and annotations

The [source geometry tests](../../crates/ritk-io/src/format/dicom/reader/parse/geometry_tests.rs)
check exact component counts, finite values, and unchanged anisotropic spacing
through metadata extraction and Part-10 file scanning. These reader checks do
not establish physical units for missing spacing or certify reconstructed planes;
the [measurement-unit work](../../backlog/RITK-SNAP-MEASUREMENT-UNITS-001.md) remains open.

| RadiAnt capability and PDF pages | Current Windows behavior | Test coverage and remaining work |
| --- | --- | --- |
| Open folders and choose a series, 43–46 | Folder picker and startup path/series selection | [Catalog tests](../../crates/ritk-snap/src/presentation/native_session/tests/series_browser/catalog.rs) assert selected series and pixels. File-picker, ZIP, and drop workflows remain [opening work](../../backlog/RITK-SNAP-NATIVE-OPEN-001.md). |
| Asynchronous loading, 9 | Startup/replacement opening is synchronous; selected catalog series decode in workers | [Series-load tests](../../crates/ritk-snap/src/presentation/native_session/tests/series_loads.rs) assert independent publication, cancellation, and pixels equal synchronous decoding. [Loading](../../backlog/RITK-SNAP-NATIVE-LOAD-001.md) still needs interactive initial opening. |
| Concurrent large studies, memory, multicore execution, 9 | Input and decoded-workspace budgets; selected-series workers | [Reader budgets](../../crates/ritk-io/src/format/dicom/reader/budget.rs) and [series worker tests](../../crates/ritk-snap/src/presentation/native_session/tests/series_loads.rs) cover bounded admission and publication. Sustained native memory, concurrent large-volume, and controlled scaling/performance measurements remain unverified; see [resource measurements](../../backlog/RITK-SNAP-LARGE-STUDY-001.md). |
| Scroll, zoom, pan, cine, 62–64, 73–74, 105 | Wheel, keys, draggable slice rails, assigned buttons, and cine | [Navigation](../../crates/ritk-snap/src/presentation/native_session/tests.rs), [scrollbars](../../crates/ritk-snap/src/presentation/native_session/tests/slice_scrollbar.rs), and [interaction](../../crates/ritk-snap/src/presentation/native_session/tests/interaction.rs) assert index/frame/tool changes. [Session capture](../../crates/ritk-snap/src/presentation/native_session/tests/pointer_capture.rs) asserts equal pan/zoom results across event batches and drags beyond image bounds; the [Windows host test](../../crates/ritk-snap/src/presentation/native_session/tests/host_capture.rs) asserts real HWND capture and a signed outside release through the session. Navigation still converts inverse-image deltas into display offsets: pan distance and zoom/window sensitivity depend on fit and orientation; see the [navigation correction](../../backlog/RITK-SNAP-NAVIGATION-UNITS-001.md). [Zoom 1:1 geometry](../../crates/ritk-snap/src/presentation/native_session/tests/display_controls/zoom.rs) asserts normalized physical scale before and after rotation. |
| Window/level, presets, inversion, rotation, 65–74 and 103–104 | Display menu and fixed Windows shortcuts | [Display controls](../../crates/ritk-snap/src/presentation/native_session/tests/display_controls.rs) assert values and pixels; [custom presets](../../crates/ritk-snap/src/presentation/native_session/tests/display_custom_presets.rs) cover in-memory save/apply. Pointer-driven sensitivity has the [navigation unit defect](../../backlog/RITK-SNAP-NAVIGATION-UNITS-001.md). [Exact-value dialogs and retained presets](../../backlog/RITK-SNAP-CUSTOM-WINDOW-001.md) need native tests. [Direct 180-degree rotation](../../backlog/RITK-SNAP-NATIVE-DISPLAY-001.md) also remains. |
| Pointer assignments/modifiers, 60–61 | In-memory button assignments | [Input parity](../../backlog/RITK-SNAP-NATIVE-INPUT-PARITY-001.md) needs native Control/Shift press and restart/restoration tests. |
| Length/calibration/angle/Cobb/deviation, 75–80 and 91–96 | Tools menu; marks belong to their source image | [Length/calibration](../../crates/ritk-snap/src/presentation/native_session/tests/measurement_overlay.rs), [Cobb](../../crates/ritk-snap/src/presentation/native_session/tests/cobb_overlay.rs), and [adapters](../../crates/ritk-snap/src/app/tests/action_adapter) assert geometry, scaling, cancellation, and slice ownership. [Native frame composition](../../crates/ritk-snap/src/presentation/native_session/tests/manual_capture.rs) asserts that measurement labels clear completed marks, edit handles, other labels, and a live pointer preview. Manual calibration currently rescales ordinary Length labels only; [other measurement families](../../backlog/RITK-SNAP-CALIBRATION-MEASURES-001.md), [inch/French inputs and calibration-source colors](../../backlog/RITK-SNAP-CALIBRATION-INPUT-001.md), and [missing-spacing units](../../backlog/RITK-SNAP-MEASUREMENT-UNITS-001.md) remain. [Completed Length marks](../../crates/ritk-snap/src/presentation/native_session/tests/length_edit.rs) support endpoint editing and whole-line translation with Pan; [label dragging](../../backlog/RITK-SNAP-ANNOTATION-TEXT-001.md), constrained lines, and [ultrasound-region calibration](../../backlog/RITK-SNAP-US-REGION-001.md) remain to implement. |
| Rectangular/elliptical ROI and PET units, 81–84 | ROI menus and mean/standard-deviation/area rendering exist; the application also computes min/max | [ROI arithmetic tests](../../crates/ritk-snap/src/tools/interaction/tests.rs) assert computed statistics. [Native menu, pointer and overlay coverage](../../backlog/RITK-SNAP-ROI-NATIVE-001.md) remains unverified. [Moving/resizing](../../backlog/RITK-SNAP-ROI-EDIT-001.md), [on-image min/max and pixel count](../../backlog/RITK-SNAP-ROI-LABEL-STATISTICS-001.md), [Shift-circle drawing](../../backlog/RITK-SNAP-ELLIPSE-CONSTRAINT-001.md), [Ctrl pixel outlines](../../backlog/RITK-SNAP-ELLIPSE-PIXEL-MASK-001.md), ultrasound regions, and native PET/SUV ROI workflows remain. [Fusion/SUV](../../backlog/RITK-SNAP-NATIVE-FUSION-001.md) tracks PET. |
| Open/closed polygons, 85–90 | Enter/double-click completion; Pan selection, vertex movement/insertion, deletion | [Completion](../../crates/ritk-snap/src/presentation/native_session/tests/polygon_double_click.rs), [insertion](../../crates/ritk-snap/src/presentation/native_session/tests/polygon_edge_insert.rs), and [persistence](../../crates/ritk-snap/src/presentation/native_session/tests/polygon_edit_session.rs) assert coordinates, recomputed values, cancellation, and saved geometry. The persistence test requires eframe-shell. |
| Arrow/pencil, 97–102 | Tools menu, Y and P | [Arrow](../../crates/ritk-snap/src/presentation/native_session/tests/arrow_overlay.rs) and [pencil](../../crates/ritk-snap/src/presentation/native_session/tests/pencil_overlay.rs) assert direction/path, source slice, cancellation, bounded capture. [Completed-mark editing](../../backlog/RITK-SNAP-MARK-EDIT-001.md) remains to implement. Arrow comment text, style/size/color/opacity, named presets, and default presets from pages 98–101 need [property controls and tests](../../backlog/RITK-SNAP-ARROW-PROPERTIES-001.md). |
| Copy/paste, duplication, fonts, 164–166 | Polygon vertex editing and deletion | Other mark selection, whole-mark movement, [label/font controls](../../backlog/RITK-SNAP-ANNOTATION-TEXT-001.md), and [copy/paste/duplication/bulk deletion](../../backlog/RITK-SNAP-ANNOTATION-COPY-001.md) need implementation and native tests. |
| Clinical corners/orientation/ruler/cursor, 106–107 | Orthogonal panes have readouts and overlay/anonymized toggles; comparison panes currently have reduced image/window/dimension text | [Text tests](../../crates/ritk-snap/src/presentation/native_session/tests/overlay_text.rs) assert directions, physical scale, privacy, cursor value. [Display overlays](../../crates/ritk-snap/src/presentation/native_session/tests/display_overlays.rs) assert colorbar/ruler separation. [Comparison-pane clinical overlays](../../backlog/RITK-SNAP-NATIVE-OVERLAY-TEXT-001.md) remain to implement and test. |
| Stored DICOM graphic overlays, 10 | Stored overlay planes need an implementation distinct from clinical text | [Stored-overlay work](../../backlog/RITK-SNAP-DICOM-GRAPHICS-001.md) needs decode, image-transform, toggle, and exact-pixel coverage. |
| Full-screen/distraction-free/touch, 108–114 and 295–304 | Resizable Windows workspace | Native full-screen/touch controls and host tests remain [shell work](../../backlog/RITK-SNAP-NATIVE-SHELL-001.md). |
| Archive/import/search/keywords/databases/CSV, 116–143 | Series browser for a loaded study | Persistent import/query/index/keywords/CSV and database-isolation tests remain [archive work](../../backlog/RITK-SNAP-NATIVE-ARCHIVE-001.md). |
| Images/original DICOM/movies/clipboard, 110–111 and 144–147 | Developer frame capture; native anonymized/repaired study copies | Reader-facing image/source/movie export, clipboard, and session controls remain [export work](../../backlog/RITK-SNAP-NATIVE-EXPORT-001.md). Artifact tests must decode and compare pixels, rates, or SOP instances. |
| Multiple series/windows, 148–154 | Series picker and panel grid with maximize/close | [Browser tests](../../crates/ritk-snap/src/presentation/native_session/tests/series_browser) assert identity and panel lifetime. Multiple top-level windows and duplication remain shell work. |
| Synchronization/cross-reference/3D cursor, 155–160 | Linked cursor in orthogonal MPR | The [native Crosshair test](../../crates/ritk-snap/src/presentation/native_session/tests/crosshair_tool.rs) asserts the exact linked voxel, six projected cursor segments, and Pan visibility preservation. Patient-position synchronization, plane intersections, and comparison-grid cursor remain [linking work](../../backlog/RITK-SNAP-NATIVE-LINK-001.md). |
| Add studies/split multi-sequence series, 161–163 | Opening selects a study/series | Retaining multiple studies and splitting dynamic/echo volumes need native catalog and exact-volume tests. |

## Advanced workflows

| RadiAnt capability and PDF pages | Current Windows behavior | Test coverage and remaining work |
| --- | --- | --- |
| Sharpen/smooth/edge/emboss, 167–168 | Image-processing code exists | [Filters](../../backlog/RITK-SNAP-NATIVE-FILTERS-001.md) need native controls and known-matrix tests, including restoration of original pixels. |
| Orthogonal MPR, 169–170 | Window → Three-plane MPR | [Reslice](../../crates/ritk-snap/src/render/tests_reslice.rs) asserts sampled values and geometry; [native tests](../../crates/ritk-snap/src/presentation/native_session/tests.rs) assert linked presentation. |
| Oblique planes/linked axes/thickness, 171–194 | Physical reslice computation exists | [Reslice tests](../../crates/ritk-snap/src/render/tests_reslice.rs) assert rotated anisotropic and trilinear reference values. Interactive controls/intersection tests remain [oblique work](../../backlog/RITK-SNAP-OBLIQUE-NATIVE-001.md). |
| MIP/MinIP/average, 182–183 | Launch-selected fourth projection pane | [Projection tests](../../crates/ritk-snap/src/presentation/native_session/projection.rs) assert explicit reductions. Runtime menus/thickness remain [projection work](../../backlog/RITK-SNAP-NATIVE-PROJECTION-001.md). |
| Volume rendering, 195–219 and 230–232 | orthogonal-with-vr startup layout shows a fixed axial projection | [VR projection](../../crates/ritk-snap/src/presentation/native_session/projection.rs) asserts RGBA, dimensions, spacing, label. Camera/shading/presets/scalpel/undo/snapshots/annotations and interactive GPU tests remain [VR work](../../backlog/RITK-SNAP-NATIVE-VR-001.md). |
| VR snapshots/image/movie/STL export, 218–229 | Rendering and label-map surface computation exist | Reader-facing VR exports need camera/geometry/pixel and decoded-artifact tests in [VR](../../backlog/RITK-SNAP-NATIVE-VR-001.md) and [exports](../../backlog/RITK-SNAP-NATIVE-EXPORT-001.md). |
| PET/CT and MR fusion/SUV, 233–236 | Patient-coordinate fusion computation exists | [Fusion tests](../../crates/ritk-snap/src/render/tests_fusion.rs) assert alpha endpoints, PET units, landmarks, geometry rejection. Native controls/SUV ROI remain [fusion work](../../backlog/RITK-SNAP-NATIVE-FUSION-001.md). |
| Time-intensity curves, 237–241 | Missing from native viewer | [Dynamic imaging](../../backlog/RITK-SNAP-NATIVE-DYNAMIC-001.md) needs time/ROI plots with exact sample/time-coordinate tests. |
| DSA/masks/automatic and manual shift, 242–251 | Missing from native viewer | [Dynamic imaging](../../backlog/RITK-SNAP-NATIVE-DYNAMIC-001.md) needs translated angiography, shift/residual, split mask, and Magic Mask tests. |
| PACS config/search/retrieve/receive, 252–269 | PACS library exists | [PACS unit tests](../../crates/ritk-snap/src/pacs/tests.rs) assert config/query values. [PACS workflows](../../backlog/RITK-SNAP-NATIVE-PACS-001.md) need a native panel, known retrieved instances, cancellation, transport errors. |
| Structured reports/MPEG/PDF, 270–273 | Missing from native viewer | [Documents](../../backlog/RITK-SNAP-NATIVE-DOCUMENTS-001.md) need bounded valid/malformed fixtures and displayed-content tests. |
| Searchable tags/raw-value saving, 274–276 | Metadata parsing exists | Native search/rows/raw-byte tests remain [tag work](../../backlog/RITK-SNAP-NATIVE-TAGS-001.md). |
| Keyboard customization/URL and CLI integration, 279–281 and 305–307 | Fixed shortcuts and startup CLI arguments | Rebinding/restart tests and a defined URL grammar remain [shell work](../../backlog/RITK-SNAP-NATIVE-SHELL-001.md). |
| Program/version information and shortcut help, 282–283 and 286–294 | CLI help/version are available; native Help/About controls are incomplete | A native version/about surface and working shortcut-help display need [shell implementation and tests](../../backlog/RITK-SNAP-NATIVE-SHELL-001.md). |
| Source preservation/privacy tools | Anonymize menu and File → Repair Study Copy | [Anonymize](../../crates/ritk-snap/src/presentation/native_session/tests/anonymize.rs) and [repair](../../crates/ritk-snap/src/presentation/native_session/tests/repair.rs) assert replaced identifiers, preserved source, corrected metadata, visible errors. |

## Format coverage and verification

A codec test establishes its tested sample layout and transfer syntax. A
native-session test also establishes that the viewer opens the study and
presents expected pixels. The modality/transfer-syntax inventory on PDF page 9
still needs a [complete native matrix](../../backlog/RITK-SNAP-FORMAT-VIEWER-001.md) for monochrome, color, multiframe, RLE,
JPEG lossy/lossless, JPEG 2000, and JPEG-LS. SR, PDF, and MPEG objects are
tracked separately above.

Run the Windows viewer tests from the repository root.

    cargo nextest run --locked -p ritk-snap
    cargo nextest run --locked -p ritk-snap --features eframe-shell

The second command includes current persistence and multiframe/RGB cross-host
cases. Source-linked assertions describe the covered values. GPU paths need
a suitable device; device absence leaves those execution paths unverified.

## Release exclusions

The [viewer scope](../adr/0057-viewer-release-scope.md) excludes encrypted ZIP,
optical import beyond reading a mounted folder, the DCMTK sending add-on,
Windows 7/8 and 32-bit builds, and non-English translations. Product activation,
RadiAnt installation, and RadiAnt-branded autorun belong to the reference
product.

Configurable overlays, URL integration, WMV export, ARM64 packaging, and touch
also need implementation contracts and tests. Their reference-product file
names and URL spelling are distinct from the behavior the reader needs.
