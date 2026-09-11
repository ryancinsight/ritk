# ritk-snap

`ritk-snap` is the RITK medical-image viewer. The current desktop shell uses
egui/eframe, and the Windows Métis host is available for the migrated native
session described in [ADR 0026](../../docs/adr/0026-viewer-presentation-migration.md).

RITK owns DICOM opening, decoding, geometry, and medical display semantics.
The `presentation` module exposes a validated format-neutral RGBA frame for a
Métis host and translates native input into format-neutral events; the host
does not parse DICOM or retain viewer state.
The Métis-facing path renders through RITK's neutral RGBA carrier and applies
orientation before the host boundary; `egui::ColorImage` remains only the
legacy eframe slice adapter.

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

To capture the rendered eframe window and exit, append `--capture window.png`.
A supplied study must load successfully; capture failure returns an error. To
run the same loaded study through the Windows Métis host, use:

```console
cargo run --locked -p ritk-snap -- path/to/study --metis-native
```

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

The reviewed capture is shown in the [DICOM workflow manual](../../docs/manual/dicom-workflow.md#present-validated-ritk-views-through-metis).

API reference: `cargo doc --locked -p ritk-snap --no-deps`.

## Browser host

The WASM entrypoint composes the generic Métis HTML5/CSS host with the RITK
canvas workflow. The page must provide both a `#metis-app` element for the
Métis host and the canvas element whose ID is passed to `start_web`. Métis owns
the browser `File` handles and transfers one bounded named-byte batch; RITK
then classifies, scans and decodes those bytes through its DICOM loader.

```javascript
import init, { start_web } from "./ritk_snap.js";

await init();
await start_web("ritk-canvas");
```

`start_web` retains the asynchronous JavaScript contract while delegating to
the same single-canvas workflow as `start_web_canvas`. Browser DICOM opening
therefore uses the same RITK path as desktop pathless input. The host never
receives a filesystem path and never decides whether a payload is DICOM.

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
window/level and viewer state. Browser pointer actions, GPU upload and runtime
visual capture remain open migration work.

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

The three canvases receive RITK-owned axial, coronal and sagittal
`PresentationFrame` values from one bounded drop batch. Métis remains the
format-neutral browser host and receives no DICOM state. The reviewed
three-canvas runtime capture is in the [DICOM workflow manual](../../docs/manual/dicom-workflow.md#inspect-the-browser-orthogonal-visual-capture);
physical browser input and GPU upload remain open migration work.

To produce the browser module, build the library target and run the pinned
`wasm-bindgen 0.2.128` CLI over
`target/wasm32-unknown-unknown/release/ritk_snap.wasm`:

```powershell
cargo build --locked -p ritk-snap --lib --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/ritk_snap.wasm `
  --target web --out-dir target/wasm-bindgen/ritk-snap
```

The generated web module exports the RITK-owned launch functions; DICOM bytes
and viewer state remain inside RITK.
