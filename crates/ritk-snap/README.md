# ritk-snap

`ritk-snap` is the RITK medical-image viewer. The current desktop shell uses
egui/eframe, and the Windows Métis host is available for the migrated native
session described in [ADR 0026](../../docs/adr/0026-viewer-presentation-migration.md).

RITK owns DICOM opening, decoding, geometry, and medical display semantics.
The `presentation` module exposes a validated format-neutral RGBA frame for a
Métis host and translates native input into format-neutral events; the host
does not parse DICOM or retain viewer state.

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

The PNG is the final RITK source frame after host initialization; the visible
session remains open until the user closes it.

API reference: `cargo doc --locked -p ritk-snap --no-deps`.
