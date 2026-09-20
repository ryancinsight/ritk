# ritk-snap

`ritk-snap` is the RITK medical-image viewer. On Windows the Métis host is the
default desktop shell. The default package has no eframe, egui, or rfd active
dependency; the complete compatibility shell is the separately named
`ritk-snap-eframe` package. The migration is described in
[ADR 0026](../../docs/adr/0026-viewer-presentation-migration.md).

RITK owns DICOM opening, decoding, geometry, and medical display semantics.
The `presentation` module exposes a validated format-neutral RGBA frame for a
Métis host and translates native input into format-neutral events; the host
does not parse DICOM or retain viewer state.
The Métis-facing path renders through RITK's neutral RGBA carrier and applies
orientation before the host boundary; `egui::ColorImage` remains only the
legacy eframe slice adapter.
Each `PresentationFrame` also carries a validated `PresentationSpacing` value,
so native placement and browser aspect semantics consume one axis-ordered
geometry value while Metis remains format-neutral.

For downstream VTK work, `LoadedVolume` implements
`TryFrom<&LoadedVolume> for ritk_vtk::VtkImageVolume`. The conversion reorders
the RITK `[depth, row, column]` geometry into VTK `[x, y, z]` order without
copying the scalar allocation. DICOM parsing and clinical semantics remain in
RITK; `to_vtk_image_data` is an explicit copy boundary only for legacy VTK
serializers and filters.

From a standalone RITK checkout, open a study directory with:

```console
cargo run --locked -p ritk-snap -- path/to/study
```

The [synthetic DICOM workflow](../../docs/manual/dicom-workflow.md) explains
the deterministic study used to check real file decoding, coordinates, and
rendered slice pixels. It includes reproducible captures and the current
verification limits.

Select a DICOM instance to choose its acquisition in a mixed-series folder.
An unselected mixed directory requires selection in the series browser;
DICOMDIR file sets use their referenced members. Session restore retains the
selected UID and files and validates them before replacing the current study.
Public caller changes are in the
[selection migration guide](../../docs/migration_selected_dicom.md).

For scripted or Métis-native launches, select the acquisition explicitly with
its SeriesInstanceUID. RITK discovers the path, verifies the requested UID and
re-scans the exact member list before decoding; an unknown UID fails without a
fallback:

```console
cargo run --locked -p ritk-snap -- path/to/study \
  --series-instance-uid 2.25.20260905001 --metis-native
```

To capture the rendered eframe compatibility window and exit, use the
dedicated compatibility package:

```console
cargo run --locked -p ritk-snap-eframe -- path/to/study --capture window.png
```

The compatibility binary accepts `--viewport-size WIDTHxHEIGHT` in logical
points for controlled comparison captures. The physical PNG dimensions remain
display-scale dependent and belong in the capture provenance; the default is
`1280x800` logical points.

The existing source-level capture harness can still select the complete shell
with the feature explicitly enabled:

```console
cargo run --locked -p ritk-snap --features eframe-shell -- \
  path/to/study --eframe --capture window.png
```

For a mixed folder, keep `--series-instance-uid` on the same command so the
capture waits for that selected RITK acquisition to load. A supplied study must
load successfully; capture failure returns an error. To run the same loaded
study through the Windows Métis host, use:

```console
cargo run --locked -p ritk-snap -- path/to/study --metis-native
```

On Windows the same command works without `--metis-native`; the flag remains an
explicit spelling for scripts and existing workflows. The separately named
`ritk-snap-eframe` executable selects the compatibility shell without activating
the legacy graph in the default package.

The Métis session owns the HWND, bounded event wait, framebuffer presentation,
resize/minimize and terminal cleanup. RITK owns DICOM opening, decoded volume
state, slice rendering, window/level and viewer actions. A deterministic hidden
host capture closes after its first idle event batch:

```console
cargo run --locked -p ritk-snap -- path/to/study --metis-native --capture window.png
```

The PNG is the complete 1280 × 800 RITK content framebuffer: axial, coronal,
and sagittal panels from left to right. RITK performs DICOM opening, slice
rendering, physical-aspect placement, and input routing; Métis owns the native
surface and receives only the bounded framebuffer. The visible session remains
open until the user closes it.

To show the same loaded scalar study with a RITK scalar projection, select one
of the native four-panel presentations:

```console
cargo run --locked -p ritk-snap -- path/to/study \
  --metis-native --metis-native-layout orthogonal-with-mip \
  --capture mip-frame.png --capture-application
```

Use `orthogonal-with-minip` for the minimum statistic or
`orthogonal-with-average` for the arithmetic mean. RITK remains responsible
for DICOM decoding, window/level, colormap and scalar reduction. Métis receives
only the bounded RGBA framebuffer: axial, coronal, and sagittal occupy the
first three panels and the display-only projection is in the lower-right
panel. Color volumes reject every projection mode because the typed RITK
contract is scalar-only.

The reviewed capture is shown in the [DICOM workflow manual](../../docs/manual/dicom-workflow.md#present-validated-ritk-views-through-metis).

For host-neutral scalar slab work, use the validated RITK projection contract:

```rust
use ritk_snap::render::{ProjectionStatistic, SlabProjection};

let request = SlabProjection::try_new(&volume, 0, centre, half_width)?;
let plane = request.compute(&volume, ProjectionStatistic::Maximum)?;
```

The request validates the axis-aligned inclusive range and rejects malformed
or RGB volumes. `ProjectionPlane::dimensions` and `pixels` follow the existing
slice order, so each host can apply its own presentation carrier without
duplicating voxel indexing. Oblique resampling and GPU slab dispatch are not
implied by this contract.

API reference: `cargo doc --locked -p ritk-snap --no-deps`.

## Browser host

The WASM entrypoint composes the generic Métis HTML5/CSS host with the RITK
canvas workflow. The page must provide both a `#metis-app` element for the
Métis host and the canvas element whose ID is passed to `start_web`. Métis owns
the browser `File` handles and transfers one bounded named-byte batch; RITK
then classifies, scans and decodes those bytes through its DICOM loader.
The checked-in DICOM consumer page is
[`web/gallery`](web/gallery); it owns the chooser wording, DICOM filter, slice
controls and three-canvas layout. The browser workflow passes that directory
explicitly to Metis together with the generated RITK package.

```javascript
import init, { start_web } from "./ritk_snap.js";

await init();
await start_web("ritk-canvas");
```

`start_web` retains the asynchronous JavaScript contract while delegating to
the same single-canvas workflow as `start_web_canvas`. Browser DICOM opening
therefore uses the same RITK path as desktop pathless input. The host never
receives a filesystem path and never decides whether a payload is DICOM.
After each browser presentation, RITK publishes bounded `data-ritk-*`
attributes on the named canvas for consumer workflow assertions. They expose
load/frame state, axis, slice bounds and pixel dimensions without patient
identifiers, DICOM metadata or pixels.

The direct canvas migration slice uses the same host and byte handoff without
starting eframe. It renders one selected RITK slice through the borrowed Métis
canvas seam:

```javascript
import init, { start_web_canvas, stop_web_canvas } from "./ritk_snap.js";

await init();
start_web_canvas("ritk-snap-canvas");
// Call stop_web_canvas() when the page or route is torn down.
```

`start_web_canvas` is a presentation increment, not a second DICOM
implementation: RITK owns classification, parsing, metadata, geometry,
window/level and viewer state. The default path uses the reviewed raster
surface; `start_web_canvas_gpu` is an explicit asynchronous WebGPU variant and
returns setup errors without falling back. A real browser GPU visual run and
resource profile remain open evidence work.

The direct three-view browser entrypoint uses three canvases ordered axial,
coronal, sagittal:

```javascript
import init, {
  start_web_orthogonal_canvases,
  stop_web_canvas,
} from "./ritk_snap.js";

await init();
start_web_orthogonal_canvases(
  "ritk-snap-axial",
  "ritk-snap-coronal",
  "ritk-snap-sagittal",
);
// Call stop_web_canvas() when the page or route is torn down.
```

The gallery can opt into the asynchronous WebGPU surface with
`start_web_orthogonal_canvases_gpu` (or the single-canvas
`start_web_canvas_gpu`). These entrypoints wait for adapter/device setup and
surface a typed JavaScript error when WebGPU is unavailable; they never switch
to raster implicitly. The RITK gallery uses this mode only for
`?renderer=webgpu`, so the default real-study capture remains reproducible.

The three canvases receive RITK-owned axial, coronal and sagittal
`PresentationFrame` values from one bounded drop batch. Métis remains the
format-neutral browser host and receives no DICOM state. The reviewed
three-canvas runtime capture is in the [DICOM workflow manual](../../docs/manual/dicom-workflow.md#inspect-the-browser-orthogonal-visual-capture).

Consumers that want the native scalar projection panel can opt into the
four-canvas entrypoint. The first three canvases retain their interactive
listeners; the fourth is display-only. The statistic index is `0` for MIP, `1`
for MinIP and `2` for arithmetic average:

```javascript
import init, {
  start_web_orthogonal_canvases_with_projection,
  stop_web_canvas,
} from "./ritk_snap.js";

await init();
start_web_orthogonal_canvases_with_projection(
  "ritk-snap-axial",
  "ritk-snap-coronal",
  "ritk-snap-sagittal",
  "ritk-snap-projection",
  0,
);
// Call stop_web_canvas() when the page or route is torn down.
```

The projection canvas publishes `data-ritk-role="projection"`,
`data-ritk-projection-statistic`, load/frame state, dimensions and physical
display aspect. RITK computes the typed scalar slab and applies the same
window/level and colormap policy as the orthogonal planes; Métis receives only
the resulting borrowed RGBA frame. The asynchronous
`start_web_orthogonal_canvases_gpu_with_projection` entrypoint selects WebGPU
explicitly and reports setup errors without raster fallback.
Trusted pointer and wheel positions use the measured local canvas content box,
excluding borders and padding and accounting for invertible 2D ancestor CSS
transforms. Fractional positions and event-time dimensions preserve voxel
selection through resizing. See [the embedding contract](../../docs/adr/0032-browser-semantic-snapshot.md)
for supported geometry and browser regression coverage.

Browser controls can call `select_web_slice(axis, index)` with zero-based axis
(`0` axial, `1` coronal, `2` sagittal) and slice index after loading a study.
RITK rejects invalid coordinates without changing the selection. A changed
selection renders on the next animation frame; controls observe the canvas
`data-ritk-slice-index` and `data-ritk-slice-count` attributes to synchronize their
position with selection, wheel navigation and cine playback. The RITK consumer
gallery provides one labeled range control per plane using this entrypoint.

To produce the browser module, build the library target and run the pinned
`wasm-bindgen 0.2.128` CLI over
`target/wasm32-unknown-unknown/release/ritk_snap.wasm`:

```powershell
cargo build --locked -p ritk-snap --lib --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/ritk_snap.wasm `
  --target web --no-typescript --out-dir target/wasm-bindgen/ritk-snap
```

The generated web module exports the RITK-owned launch functions; DICOM bytes
and viewer state remain inside RITK.
