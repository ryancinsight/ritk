# Verify a synthetic DICOM study

This workflow generates three small Part 10 instances, opens them through
the viewer's directory and dropped-byte loaders, checks their decoded values
and physical coordinates, and captures three orthogonal scalar and RGB slice
buffers plus a patient-coordinate fusion buffer.
It needs no downloaded datasets or patient information.

RITK owns the DICOM boundary used by presentation hosts. A host that receives
files or browser drops passes the named Part 10 byte batch to
`ritk_io::scan_dicom_part10_bytes` (or its budgeted form), then passes the
validated descriptor to `ritk_io::load_dicom_from_series`. The scanner and
loader retain the validated bytes, enforce the parser, retained-study, and
decoded-workspace budgets, and return the RITK `Image` plus
`DicomReadMetadata`. No GUI framework type or parser object crosses this
boundary; the host owns only input and presentation lifecycle.

This workflow is the DICOM opening demonstration for both the current eframe
shell and the migrated Windows Métis shell. The code, fixtures, visual goldens,
and rejection tests remain in RITK so a framework migration cannot fork
medical-data semantics.

Build from a standalone RITK checkout, then run the bounded demonstration:

```console
cargo build --locked -p ritk-snap --example dicom_workflow
python scripts/viewer.py target/debug/examples/dicom_workflow
```

On Windows the binary ends in `.exe`. Under Atlas, use the configured shared
target directory (`D:/atlas/target/debug/examples/dicom_workflow.exe`). Atlas's
development overlay changes dependency resolution; verification against the
committed standalone lock must run outside that overlay while retaining the
shared target and build settings.

The runner requires Python 3.11 or newer, enforces a 60-second deadline per process and replaces the fixed
artifact set in `scratch/viewer/`. Its `workflow.json` records the actual
decoded voxels, geometry, capture dimensions and SHA-256 hashes of the binary
and images. A failed run invalidates its previous success report.
It compares the generated PNG bytes exactly against the reviewed manual
images; the deterministic software renderer needs no tolerance. Use
`--update-goldens` only to regenerate deliberately changed captures, then
inspect the images and independent pixel tests before accepting them.

The example and required loader tests share the
[fixture source](../../crates/ritk-snap/src/dicom/loader/tests/fixtures.rs).
The fixture uses unsigned 8-bit, uncompressed, single-frame images with
MONOCHROME2 photometric interpretation. CT and MR modality labels are tested;
this small Secondary Capture object is not a complete CT or MR acquisition IOD.
Filenames and instance numbers run backwards relative to spatial position,
so sorting them by name would give the wrong answer.

| Quantity | Expected value |
| --- | --- |
| Volume shape `[depth, row, column]` | `[3, 2, 4]` |
| Stored samples in spatial order | `0, 10, …, 230` |
| Modality rescale | `decoded = 2 × stored − 20` |
| Decoded voxels | `−20, 0, …, 440` |
| Spacing `[depth, row, column]`, mm | `[2, 1.5, 0.5]` |
| First voxel LPS position, mm | `[10, 20, 30]` |
| Voxel `[2, 1, 3]` LPS position, mm | `[14, 21.5, 31.5]` |

Position and orientation give the independent coordinate equation
`P(d,r,c) = [10 + 2d, 20 + 0.5c, 30 + 1.5r]` mm. DICOM defines the row/column
spacing and orientation mapping in
[PS3.3 C.7.6.2.1.1, Equation C.7.6.2.1-1](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1).

## Open a multi-frame object

RITK inspects `NumberOfFrames` on each scanned member before choosing the
single-frame series decoder. A multi-frame object is decoded through
`ritk_io::load_dicom_multiframe_flat` (or its byte-payload counterpart), so
the resulting `LoadedVolume` has one depth entry for every decoded frame.
The path and dropped-byte entry points share the same frame buffer, modality
rescale, spatial orientation, and spacing checks. RGB multi-frame objects use
RITK's interleaved color reader and retain three channels per voxel.

The viewer rejects a multi-frame object that declares more than one temporal
position or names `TemporalPositionIndex` in its dimension organization. It
also rejects a batch that mixes a multi-frame object with conventional
single-frame members; neither case is silently presented as a spatial stack.
The regression fixtures use two distinct 2 × 2 frames with decoded values
`[-8,-6,-4,-2]` and `[12,14,16,18]`, assert both file and byte workflows, and
render each axial frame through `SliceRenderer` with independent pixel
oracles. A declared three-frame object with only two frames is rejected during
decode.

## Preserve RGB presentation

The same workflow writes a two-frame RGB object with interleaved unsigned
8-bit samples. RITK's color reader admits only `PhotometricInterpretation=RGB`,
`SamplesPerPixel=3`, `PlanarConfiguration=0`, and supported uncompressed or
codec-backed transfer syntaxes. Scalar, palette, YBR, CMYK, planar, signed, or
unsupported codec inputs fail at the DICOM boundary; they are never reduced to
the first channel.

The color object has shape `[depth, rows, columns] = [2, 2, 2]` and channel
samples `[red, green, blue, white]` in frame zero followed by
`[cyan, magenta, yellow, neutral]` in frame one. The filesystem and dropped-byte
loaders must return the same interleaved samples. `SliceRenderer` copies these
decoded channels directly into the display image for axial, coronal, and
sagittal views. Scalar window/level and colormap parameters are intentionally
not applied to RGB data, so a red source voxel remains red in the submitted
texture.

The images below are the actual RGB slice buffers enlarged with nearest-neighbor
sampling. The color-channel tests compare every RGBA pixel, and `viewer.py`
compares these grid bytes with the reviewed manual images.

![RGB axial slice pixel grid](images/dicom-color-depth.png)

![RGB coronal slice pixel grid](images/dicom-color-row.png)

![RGB sagittal slice pixel grid](images/dicom-color-column.png)

## Present signed grayscale data

The scalar viewer follows the DICOM grayscale presentation contract after RITK
IO has decoded the stored samples and applied modality rescale. `MONOCHROME2`
keeps the mapped value and `MONOCHROME1` inverts it once after the VOI mapping.
When `VOI LUT Function (0028,1056)` is absent, the viewer uses DICOM's default
`LINEAR` function. Explicit `LINEAR_EXACT` and `SIGMOID` values select their
specified equations; an unsupported value or table-based `VOI LUT Sequence`
fails at load instead of being approximated as a window.

The workflow fixture stores signed samples `−10, 0, 10, 20`, applies slope `2`
and intercept `−10`, and declares `MONOCHROME1` with `LINEAR_EXACT`. The
decoded modality values are therefore `−30, −10, 10, 30`; with centre `0` and
width `40`, the displayed bytes are `255, 191, 64, 0`. The image is generated
by `SliceRenderer` and checked byte-for-byte by `scripts/viewer.py`.

![Signed grayscale presentation pixel grid](images/dicom-grayscale.png)

The same presentation value is shared by scalar slices, MIP, volume rendering,
fused comparison, and native GPU paths. This keeps CPU and GPU output aligned
and prevents a second inversion or a renderer-specific window equation.

The images below come from `SliceRenderer`, which uses Iris's grayscale map.
They show the decoded pixel grid enlarged by nearest-neighbour sampling.
These are software slice-buffer captures before texture upload; they do not
show the application window or physical-aspect-ratio display. Axis labels refer
to storage dimensions because this study has a nonidentity orientation.

Depth index 1, size 4 × 2: expected bytes `80,90,100,110 / 120,130,140,150`.

![Depth slice pixel grid](images/dicom-depth.png)

Row index 1, size 4 × 3: expected bytes
`40,50,60,70 / 120,130,140,150 / 200,210,220,230`.

![Row slice pixel grid](images/dicom-row.png)

Column index 2, size 2 × 3: expected bytes `20,60 / 100,140 / 180,220`.

![Column slice pixel grid](images/dicom-column.png)

The chosen center 235 and width 510 cancel the fixture's modality rescale under
explicit `LINEAR_EXACT` metadata, so each grayscale byte equals its stored
sample. The tests also check the scratch-buffer rendering
path and reject malformed and truncated byte inputs. They separately verify
exact NIfTI file/byte roundtrips for `.nii` and `.nii.gz`.

## Open, select, and restore a study

Use **File → Open DICOM file…** to select a particular acquisition in a folder
containing several series. The selected instance's SeriesInstanceUID determines
which neighbouring image files load. **Open DICOM folder…** discovers the series
browser; if the folder contains several series, choose a series there instead of
accepting an arbitrary largest series. The highlighted series changes after
successful loading. Selecting a secondary series retains its own exact files.

**Open DICOMDIR…** uses the index's referenced image set. Missing references or
an invalid index report an error; unreferenced subdirectories do not supply a
replacement study. The reader follows the linked PATIENT/STUDY/SERIES/IMAGE
record tree, excludes inactive or unreachable records, rejects cycles and
out-of-sequence offsets, and verifies each IMAGE record's SOP class, SOP
instance, and transfer syntax against its referenced file. Dropped byte batches
must identify one image series.

Saving a session records the primary study's UID and exact files along with the
presentation controls. Restore validates those members before replacing the
current volume or controls. A missing member, changed UID, or member that is no
longer an image fails explicitly and leaves the current study displayed.
Older path-based sessions remain readable, but an ambiguous folder requires
selection. See the [public API migration guide](../migration_selected_dicom.md).
This session format does not persist the secondary comparison acquisition.

Native tests exercise actual egui series-row pointer events, primary/secondary
loads, failed replacement, and session restore with deterministic Part 10 files.
The IO tests load a complete synthetic linked PATIENT/STUDY/SERIES/IMAGE index,
then exercise inactive and unreachable records, malformed links, identity
mismatches, and final-component symlinks. They compare active member paths and
exact pixel values and geometry with explicit member loading. DICOMDIR member
reads use `moirai_pal::fs::open_file_within_root`, which walks from the selected
root directory handle and returns the handle RITK reads. The Moirai PAL tests
reject parent traversal and intermediate/final links; browser file entries use
the DOM provider because the native path contract is unsupported on WebAssembly.

## Resource-bounded DICOM ingress

The RITK DICOM boundary runs a structural Part 10 preflight before dicom-rs
materializes an object. `ritk_dicom::ParseBudget` limits the encoded byte span,
structural element count, and sequence nesting depth. The scanner checks
declared value spans, sequence and item delimiters, implicit sequence tags, and
encapsulated pixel fragments without copying their values. The reader exposes
matching `scan_dicom_*_with_budget` entry points and the series loaders expose
`load_*_with_budget` counterparts. They accept the typed `DicomReadBudget`,
which combines the parser budget with separate retained-study and
decoded-workspace ceilings. The default forms use finite shared ceilings;
constrained hosts can construct a smaller budget with
`DicomReadBudget::try_new(ParseBudget::new(...), retained_bytes, decoded_bytes)`.

The preflight accepts implicit little-endian, explicit little-endian, and
explicit big-endian datasets, plus the encapsulated pixel syntaxes used by the
native RITK codecs. Deflated dataset input is rejected with an explicit error
because the locked dicom-rs registry does not provide a bounded deflate decoder.
The DICOMDIR index uses the same parser preflight. Filesystem reads resolve and
open one path handle, compare its metadata with the resolved path, and read from
that handle; DICOMDIR member reads additionally use Moirai PAL's root-confined
directory-handle walk. Scanned slices retain the exact validated bytes for pixel decode.
The reader charges retained bytes before storing each member. The loader plans
the peak decoded frame, resample, and contiguous-volume workspace before any of
those buffers are allocated, including the temporary frame-vector handles.
Deterministic tests cover parser, DICOMDIR, retained-byte, decoded-workspace,
and replacement cases. Unix reads reject a final symlink with `O_NOFOLLOW`;
Windows reads request a reparse-point handle and reject a final reparse point.
The DICOMDIR record-tree and referenced-identity contract is delivered by
[RITK-SNAP-DIRECTORY-001](../../backlog.md#RITK-SNAP-DIRECTORY-001); the
root-confined filesystem contract is provided by Moirai PAL.

## Capture the eframe application

Build the binary alongside the example and run:

```console
cargo build --locked -p ritk-snap --bins --example dicom_workflow
python scripts/viewer.py target/debug/examples/dicom_workflow --native-binary target/debug/ritk-snap
```

Use `.exe` suffixes on Windows and the shared Atlas target paths when applicable.
The optional eframe workflow launches the real viewer with the generated study,
saves its rendered root viewport to `scratch/viewer/window.png`, and exits. It
then launches with a missing study and requires an explicit failure without a
screenshot. Each of the three processes has a 60-second limit; the complete
native workflow therefore has a maximum 180-second subprocess budget.

Capture uses the normal viewer update and egui/eframe screenshot response.
For your own local study, run `ritk-snap path/to/study --capture window.png`.
The supplied study must load and the PNG must save before success is reported.
Native window images depend on the host renderer and fonts; the exact pixel
goldens above remain the deterministic software-rendering check.

For the migrated Windows host, run the same generated study through Métis:

```console
target/debug/ritk-snap.exe scratch/viewer/study --metis-native
```

The Métis host owns the window handle, finite event wait, retained framebuffer,
resize/minimize handling, DPI updates and terminal cleanup. RITK owns the file
open, DICOM decode, selected volume, window/level, colormap, slice navigation
and action reduction. The host receives only a bounded RGBA
`PresentationFrame`; no DICOM identifier, path, parser object or decoded volume
crosses the seam. Plain vertical wheel input steps the active slice, and
Ctrl/Command plus vertical wheel applies the existing zoom policy. Focus loss
cancels an in-progress pointer gesture.

The deterministic Métis capture path is hidden and bounded: it exits after the
first idle event batch and writes the complete three-panel RITK composition as
PNG. The fixed capture surface is 1280 × 800 pixels; the panels are axial,
coronal, and sagittal from left to right, separated by four-pixel black gaps.

```console
target/debug/ritk-snap.exe scratch/viewer/study --metis-native --capture scratch/viewer/metis-frame.png
```

This capture checks real DICOM opening, RITK rendering, orthogonal slice
selection, physical-aspect placement, and Métis framebuffer transfer. It is a
content capture without OS chrome or text overlays; the native host owns the
surface while RITK owns every DICOM and clinical display decision.

The complete synthetic workflow can run this Métis check; it records the
executable hash, invalid-study rejection and `metis-frame.png` hash in
`scratch/viewer/workflow.json`:

```console
python scripts/viewer.py target/debug/examples/dicom_workflow.exe --native-binary target/debug/ritk-snap.exe --metis-native
```

![Métis native three-panel DICOM capture](images/dicom-metis-native.png)

The reviewed image is generated by the command above and compared byte-for-byte
by the workflow script. It is a deterministic application-content snapshot,
not a screenshot of Windows decorations or host fonts.

## Build the RITK SNAP executable and installer

RITK owns the application manifest at
[`metis.json`](../../metis.json). It declares
the single `ritk-snap` Cargo binary and contains no DICOM parser, decoded study,
or clinical resource. Métis supplies the packaging tool and Windows Installer
authoring; the viewer binary remains the RITK-owned DICOM implementation.

From a Windows x64 checkout of RITK and a local checkout of Métis, build the
packager and create a new output directory:

```powershell
cargo build --release --locked --manifest-path path\to\metis\Cargo.toml -p metis-cli
New-Item -ItemType Directory output -Force | Out-Null
path\to\metis\target\release\metis.exe package `
  metis.json output\ritk-snap-installer
```

The output contains `app\ritk-snap.exe`, the exact `inventory.json`, and the
Windows per-user MSI. Launch the portable executable with a study path, for
example `output\ritk-snap-installer\app\ritk-snap.exe study\`. The installer
does not bundle patient data or choose a study; DICOM opening still follows the
RITK workflow above. A package output directory is create-new and must not
already exist.

The manifest and package command are local integration evidence. The
[`metis-package.yml`](../../.github/workflows/metis-package.yml) workflow
repeats the same lock-pinned build on a Windows runner and uploads the
executable, inventory and MSI as a reviewable artifact. Registry publication,
signing, and release promotion remain separate release-authority decisions.

## Open dropped DICOM files in the browser host

The browser build uses the same RITK byte loader as the native dropped-input
path. The HTML page supplies a Métis mount point and an eframe canvas:

```html
<main id="metis-app" aria-label="Métis browser host"></main>
<canvas id="ritk-canvas"></canvas>
<script type="module">
  import init, { start_web } from "./ritk_snap.js";

  await init();
  await start_web("ritk-canvas");
</script>
```

Métis registers the browser drag-and-drop listeners and transfers one bounded
batch of named bytes. The RITK adapter copies those payloads into egui's input
carrier; `ui::dropped_input` then detects Part 10 content or a DICOM suffix,
and `dicom::loader` performs the existing scan, series selection, budgeted
preflight, and decode. No filesystem path, browser handle, DICOM tag, or
parser object crosses into Métis. A malformed or oversized payload follows the
same typed rejection path as a native byte drop.

The browser acceptance checks are the locked `wasm32-unknown-unknown` build and
warning-denied Clippy for `ritk-snap`, plus the RITK byte-routing and loader
tests. The standalone consumer lock and graph now use the merged [Mnemosyne
#141](https://github.com/ryancinsight/Mnemosyne/pull/141), [Coeus
#393](https://github.com/ryancinsight/Coeus/pull/393), [Apollo
#386](https://github.com/ryancinsight/apollo/pull/386), and [Leto
#187](https://github.com/ryancinsight/leto/pull/187) portability fixes. The
reviewed visual oracle remains the native Métis content capture above until a
browser runtime bundle is generated on a host with the wasm-bindgen packaging
tool; the browser check must not be represented by a static host screenshot.

The direct Métis canvas workflow is also available for the first browser
presentation slice. It keeps the canvas outside Métis's `#metis-app` mount and
uses the RITK-owned browser loop:

```html
<main id="metis-app" aria-label="Métis browser host"></main>
<canvas id="ritk-snap-canvas" width="512" height="512"></canvas>
<script type="module">
  import init, { start_web_canvas, stop_web_canvas } from "./ritk_snap.js";

  await init();
  start_web_canvas("ritk-snap-canvas");
  // Call stop_web_canvas() when the page or route is torn down.
</script>
```

`start_web_canvas` mounts Métis, consumes its bounded named-byte drop batch,
and passes the files to RITK's existing DICOM classifier and byte-series
loader. A valid drop replaces the RITK study and presents the selected slice on
the named canvas; malformed, oversized, or non-DICOM input follows RITK's
typed error and status paths. The browser host receives only the borrowed
RGBA frame. DICOM parsing, metadata, geometry, window/level and viewer state
remain in RITK. This increment presents one selected slice; browser pointer
actions, the three-view layout, GPU upload, and a runtime visual capture are
separate acceptance work and are not represented by a fabricated screenshot.

## Present validated RITK views through Métis

RITK remains the only DICOM owner. After RITK has opened the study, decoded the
selected frame, applied modality rescale, window/level, and colormap rules, the
`ritk-snap::presentation::PresentationFrame` boundary copies bounded, row-major
RGBA slices. The native session composes three such frames using the same
window/level, colormap, orientation, and physical-spacing rules as the RITK
viewer. The host receives one bounded framebuffer; DICOM identifiers, paths,
codec state, geometry, and volume storage stay in RITK.

On Windows, `run_native_viewer` loads the selected study in RITK, presents the
three orthogonal views through Métis's native surface, routes pointer events to
the panel under the pointer, translates each bounded native event batch to
`PresentationEvent`, and applies the resulting actions to RITK viewer state.
The focused suite checks slice pixels, all three panels, panel-specific wheel
navigation, resize/minimize, DPI, focus-loss cancellation, close, and bounded
hidden capture. The test proves the host boundary and the existing DICOM
workflow; it does not add a DICOM parser to Métis. Browser handoff and GPU
upload remain migration work; the RITK-owned manifest now exercises the Métis
executable and Windows MSI path without moving DICOM behavior across the
presentation boundary.

Primary-button drags follow the selected RITK tool through the same event path.
For the Pan tool, the resulting image-space offset is applied while RITK
recomposes the three panels, so the next Métis framebuffer reflects the drag as
well as the updated viewer state. This is a state-and-pixels check; it does not
move DICOM parsing or geometry ownership into Métis.

Keyboard page navigation follows the same RITK-owned action path. Page Down
advances the active slice and recomposes the native Métis framebuffer; the
native session test verifies both the slice index and changed pixels.

Deferred viewer loads use a bounded Moirai task per primary or comparison
target. RITK assigns each request a generation and checks cooperative
cancellation before publication, so a superseded or closed request cannot
replace the current study. A failed replacement reports its error while the
previous decoded study remains displayed. The load-task tests use the real
synthetic DICOM fixtures and assert decoded shape, spacing, series identity,
supersession, and close cancellation.

Wheel input is now reduced and applied by RITK for every host. The native
Moirai `ModifierState` is translated into the format-neutral
`PresentationModifiers`; the current egui producer emits the same
`PresentationEvent::PointerWheel` shape. Ctrl/Command plus a vertical delta
uses the existing zoom policy, while an unmodified vertical delta steps the
active slice. Non-finite deltas and events outside a viewport are rejected or
ignored before viewer state changes. Métis carries the event snapshot only; it
does not parse DICOM, retain a decoded volume, choose a series, or apply a
clinical display transform.

The existing `dicom-window.png` below remains the egui/eframe baseline. The
`dicom-metis-native.png` image is the reviewed Métis content capture; it is
generated from the same synthetic study and is not relabeled as an OS-window
golden.

![Running native viewer with the synthetic DICOM study](images/dicom-window.png)

This egui/eframe capture runs on Windows at 125% display scale, producing a
1600 × 1000 root viewport. The viewer's current hanging protocol selects width
400 and center 60, unlike the software-grid oracle's width 510 and center 235.
Image placement preserves physical proportions in every layout. For this
fixture, displayed width/height is 2/3 for the depth slice, 1/3 for the row
slice, and 1/2 for the column slice: pixel count times sample spacing on each
axis. A quarter turn exchanges the two extents. The same placement determines
image bounds and cursor hit coordinates; tests inspect the actual egui image
shapes across layouts and rotations. Geometry that cannot be represented in
positive finite screen coordinates reports an explicit placement error.
The status bar's
cursor `[1, 1, 2]` maps to LPS `[12, 21, 31.5]` mm by the equation above, and
the decoded cursor value is 260. The capture demonstrates the existing viewer
baseline; it is not a Métis application screenshot.

Scalar readouts use “value” because this carrier does not establish physical
intensity units; PET SUV readouts retain their explicit SUV label.
Annotations use opaque backings for contrast. When the viewport cannot fit all
annotations without overlap, activate **Details** to read the complete metadata
in a scrollable popup. Press Escape or click outside the popup to close it.
With Details focused, press Enter or Space to open it. Arrow keys scroll by a
line, Page Up/Down by a page, and Home/End reach the content boundaries without
changing the study slice.

Secondary images use their own sampling distances. A fused image keeps the
primary output grid, then maps each primary voxel centre through the physical
affine

```text
P = origin + direction · diag(spacing) · [depth, row, column]
```

into the secondary grid before nearest-neighbour sampling. The two volumes
must carry the same `FrameOfReferenceUID`. When both identifiers are absent,
fusion is admitted only for exactly identical origin, direction and spacing;
different unknown grids are rejected. Non-parallel planes and a selected
secondary plane outside the secondary normal extent report an explicit error,
while secondary in-plane samples outside the field of view leave the primary
pixel unchanged. These rules prevent a normalized-coordinate blend from being
presented as registered anatomy.

The focused fusion suite contains manufactured translated, rotated and
anisotropic grids with known patient-space landmarks, plus incompatible-frame,
missing-frame, non-parallel and out-of-field rejection/retention cases. The
same affine transform is used by RT-STRUCT projection, so contour coordinates
and fused pixels share one physical convention.

The workflow writes a deterministic fused capture. It shifts the secondary
grid by one column spacing in LPS, gives both volumes the synthetic frame
identifier `2.25.20260905099`, and records the mapped landmarks in
`workflow.json`. The enlarged software buffer is the reviewed visual oracle;
the native window capture above remains the host-renderer demonstration.

![Patient-coordinate fusion capture](images/dicom-fusion.png)

| Primary voxel | Patient LPS (mm) | Secondary continuous voxel | Sampling result |
| --- | --- | --- | --- |
| `[1, 0, 0]` | `[12, 20, 30]` | `[1, 0, 1]` | Secondary column 1 |
| `[1, 0, 1]` | `[12, 20.5, 30]` | `[1, 0, 2]` | Secondary column 2 |
| `[1, 1, 2]` | `[12, 21, 31.5]` | `[1, 1, 3]` | Secondary column 3 |
| `[1, 1, 3]` | `[12, 21.5, 31.5]` | `[1, 1, 4]` | Primary retained (out of field) |

The capture is generated by the same `render_fused_slice` path used by the
viewer, rather than by a hand-assembled image. A changed capture must update
the reviewed `dicom-fusion.png` image only after the coordinate table and the
pixel tests agree.

## Keep cursor and measurements with transformed pixels

The viewer's **View → Orientation** commands and shortcuts apply the same
`ViewTransform` to the texture and every source-coordinate overlay:
**H** flips horizontally, **V** flips vertically, **R** rotates clockwise by
90°, **Shift+R** rotates counter-clockwise, and **O** restores the identity.
The linked cursor, pointer intensity, crosshair, label map, RT-STRUCT, RT-DOSE,
orientation labels, and measurement annotations all use source voxel
coordinates. Screen input first enters the displayed output rectangle and is
then mapped through the inverse transform, so a transformed click selects the
same voxel as an identity click at the corresponding source location.

The continuous mapping uses image-edge coordinates. For source size
`[width, height]`, the source domain is `[0,width] × [0,height]`; the output
dimensions swap for quarter turns. Pixel centres therefore map through the
same equations as the rendered image without a half-pixel offset. Annotation
lengths and ROI areas use the source plane spacing. Their checked constructors
reject non-positive, non-finite, or `f32`-unrepresentable spacing and reject a
non-finite derived result; the viewer reports the rejection and does not append
an invalid annotation.

The workflow also emits a deterministic transformed slice capture. It flips
the depth slice horizontally and rotates it clockwise, then enlarges the
result with nearest-neighbour sampling. The image is generated by the same
`apply_to_image` path used before texture upload and is compared byte-for-byte
with the reviewed manual image.

![Transformed depth slice pixel grid](images/dicom-orientation.png)

Temporal multiframe organization and default DICOM LINEAR/VOI semantics are
covered by the completed [RITK-SNAP-FRAMES-001](../../backlog.md#RITK-SNAP-FRAMES-001)
item. These workflows prepare the egui baseline for the Métis migration. The
browser handoff now has a compiled RITK adapter and manual workflow; GPU
upload, native packaging, and a browser runtime visual capture remain separate
acceptance items in [RITK-SNAP-METIS-001](../../backlog.md#RITK-SNAP-METIS-001).

The browser presentation seam is now explicit as well. `metis_web::CanvasFrame`
is implemented by RITK's `PresentationFrame`, and
`ritk_snap::presentation::WebCanvasPresenter` resolves a named canvas and
uploads that borrowed RGBA view through Métis and Moirai. The adapter carries
only dimensions and pixels; DICOM parsing, decoded volume state, geometry and
medical display policy remain in RITK. `start_web` continues to use the eframe
canvas for the existing shell, while `start_web_canvas` exercises the direct
single-slice browser workflow described above. Neither path moves DICOM
behavior into the GUI framework.
