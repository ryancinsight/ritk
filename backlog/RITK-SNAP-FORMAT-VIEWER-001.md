<a id="RITK-SNAP-FORMAT-VIEWER-001"></a>
## RITK-SNAP-FORMAT-VIEWER-001 — Exercise encoded studies in the viewer — todo
- outcome: Supported raw, color, multiframe, RLE, JPEG lossy/lossless, JPEG 2000 and JPEG-LS inputs have native visible-load value tests.
- acceptance: Load a tiny independent fixture for each declared transfer syntax through the actual native session and assert exact or codec-derived pixel values and slice identity.
- scope: crates/ritk-io/src/format/dicom/reader/tests/, crates/ritk-snap/src/dicom/loader/tests/, crates/ritk-snap/src/presentation/native_session/tests/
- verified: Default scalar and RGB multiframe tests assert displayed RGBA values. The optional `independent_lossless_mr_encodings` test compares file and byte loading of pydicom’s RLE, JPEG-LS and JPEG 2000 MR encodings with every reference voxel and the complete axial frame.
- verified: The grayscale JPEG-LS codec matches all 450 samples of pydicom's 8-bit and all 500 samples of its 16-bit near-lossless fixtures. The default codec test pins an independently decoded two-band scan at NEAR 0 and 2 in both directions. Interleaved RGB JPEG-LS remains unsupported.
- verified: Extensionless scalar studies cover packed native frames, native integer and IEEE floating-point pixels, retired big-endian words, byte-padded binary RLE, per-frame uncompressed/DEFLATE packing, and the exact 38-bit grayscale JPEG 2000 cursor and window. See the [scalar import regressions](../crates/ritk-snap/src/dicom/loader/tests/scalar_precision.rs) and [workflow limits](../docs/manual/dicom-workflow.md).
- verified: Independent JPEG 2000 and HTJ2K frames cover all five packet orders, header packing, coding overrides and multi-part tiles. File and byte import cover all three HTJ2K transfer syntaxes, signed coefficients and 38-bit source values. The multi-frame writer emits all three HT syntaxes.
- open: RT Dose grids require dose scaling, units and frame offsets. Full color/video, table-based VOI LUTs and SOP-specific presentation have separate acceptance requirements.
- next: Derive the declared syntax set from source and add missing loader-to-frame assertions. Reference: RadiAnt 2021.2; [feature comparison](../docs/manual/features.md#reading-and-annotations)
- basis: eb9a81dabb62777be2103fa16b8a2dcbab50f1e3
- needs: none
- priority: verification; milestone: M-viewer-reading
