# ritk-snap-eframe

`ritk-snap-eframe` packages the complete eframe compatibility shell for the
RITK medical image viewer. The default `ritk-snap` binary uses the Métis native
host on Windows; this binary keeps the existing eframe workflows available
while the host transition is completed.

Run it with an optional DICOM path:

```text
ritk-snap-eframe path/to/study
```

The `--capture PNG` option runs the bounded eframe capture workflow for a
startup study. DICOM decoding and viewer state remain in `ritk-snap`.
