# ritk-snap

`ritk-snap` is the RITK medical-image viewer. The current desktop shell uses
egui/eframe; Métis is the planned replacement described in
[ADR 0026](../../docs/adr/0026-viewer-presentation-migration.md).

From a standalone RITK checkout, open a study directory with:

```console
cargo run --locked -p ritk-snap -- path/to/study
```

The [synthetic DICOM workflow](../../docs/manual/dicom-workflow.md) explains
the deterministic study used to check real file decoding, coordinates, and
rendered slice pixels. It includes reproducible captures and the current
verification limits.

API reference: `cargo doc --locked -p ritk-snap --no-deps`.
