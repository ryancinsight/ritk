# ADR 0037: Opt-in browser WebGPU presentation

Status: Accepted

Date: 2026-09-16

Driver: [RITK-SNAP-METIS-001](../../backlog.md#RITK-SNAP-METIS-001)

Upstream contract: [Metis ADR 0036](../../../metis/docs/adr/0036-browser-webgpu-canvas.md)
and [Moirai ADR 0061](../../../moirai/docs/adr/0061-browser-webgpu-canvas.md)

## Context

RITK owns DICOM scanning, decoding, geometry, slice state and clinical image
semantics. Metis owns the format-neutral browser surface and Moirai owns the
browser WebGPU device. The current RITK browser viewer presents actual study
frames through Metis's two-dimensional canvas path. A GPU presentation run is
needed for the migration comparison, but selecting a device must remain
explicit so unsupported browsers and setup failures cannot be mistaken for a
successful raster run.

## Decision

`WebCanvasPresenter` gains asynchronous GPU constructors that select Metis's
WebGPU `CanvasSurface` while retaining the existing input and borrowed-frame
contract. `BrowserCanvas` and the single/orthogonal viewer launchers gain
matching asynchronous constructors. The WASM API exports opt-in
`start_web_canvas_gpu` and `start_web_orthogonal_canvases_gpu` entrypoints;
the existing 2D exports remain unchanged.

The RITK gallery selects the GPU launchers only when its URL contains
`renderer=webgpu`. The default gallery remains the reviewed 2D real-study
workflow. GPU setup errors are written to the gallery status and stop the
viewer; no CPU fallback is added. The query choice and status belong to the
RITK consumer page, not Metis. RITK's image oracle remains the decoded
`PresentationFrame` and its canvas dimensions/aspect semantics.

### Revision 2026-09-16

The browser evidence workflow adds a Chromium entry that captures the
consumer's real WebGPU element PNGs through Metis's named-context mode. It
derives a dimensions-and-attributes oracle from the existing RITK study
oracle, leaving raster RGBA verification unchanged. The consumer lock and
workflow default now pin the merged Metis revision
`29812898870042665010c6534dce111ea1966925`; the resolved Moirai provider is
`95275651722583f52e098c0d30b1a53ec82c1fc7`.

## Alternatives rejected

1. Replacing the default gallery path would invalidate the existing real-study
   evidence and make browser capability differences implicit.
2. Falling back to 2D after GPU setup failure would hide device or permission
   errors and invalidate a GPU comparison.
3. Parsing DICOM or choosing a renderer in Metis would cross the established
   consumer ownership boundary.

## Failure modes and limits

Missing WebGPU, adapter/device denial, context loss and upload rejection are
observable setup or presentation failures. The async launch is cancelled by
the existing viewer teardown path, which drops listeners, decoded frames and
the animation-frame task. This increment does not claim a GPU device is
available on every browser, visual equivalence with the 2D path, compositor
latency or lower memory use; those claims require a real configured browser run
and revision-bound image/resource artifacts.

## Verification

The locked native `ritk-snap` suite, strict native Clippy and formatting cover
the unchanged viewer state and compile the new WASM entrypoints. The locked
WASM library check and Clippy cover the async constructor path. The RITK
browser workflow records the explicit `renderer=webgpu` query, exact
Metis/Moirai revisions, named `webgpu` canvas-context checks, element PNG
dimensions and RITK semantic attributes. The existing raster matrix retains
the RGBA oracle; a WebGPU PNG does not claim 2D readback equivalence or
hardware acceleration. When a browser lacks WebGPU the workflow records the
typed failure instead of reporting a raster result as GPU evidence.
