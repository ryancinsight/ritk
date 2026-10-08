<a id="RITK-DICOM-OBJECT-PIXEL-PREFLIGHT-001"></a>

## RITK-DICOM-OBJECT-PIXEL-PREFLIGHT-001 — Validate DICOM pixels before writing — done
- outcome: Reject DICOM objects whose metadata cannot describe their encoded payload before opening the destination.
- acceptance: Validate rows, columns, samples, frames, BitsAllocated/BitsStored/HighBit/PixelRepresentation, pixel VR, checked length, and legal padding; absent NumberOfFrames means one, and rejection preserves an existing destination.
- scope: crates/ritk-io/src/format/dicom/writer_object.rs, writer modules, and DICOM writer tests
- next: Landed and verified. `writer/pixel_preflight.rs::preflight_native_pixel_data` requires Rows, Columns, SamplesPerPixel, BitsAllocated, BitsStored, HighBit, PixelRepresentation, and PhotometricInterpretation, derives frames with `None => 1`, and checks the pixel count with checked arithmetic plus the DICOM even-length padding rule (`pixel_bit_description_is_valid` is the shared predicate). Both image write paths call it before any destination is touched: `writer/series.rs::write_series_flat` preflights every slice at line 250 and only then calls `write_series_files` at line 261, and `writer/metadata.rs::write_dicom_series_with_metadata` does the same at lines 235/247. `writer_object.rs` stays a verbatim emitter (no preflight, no `write_series_files`) — `dicom_multiframe_rejects_declared_frame_count_mismatch` deliberately writes a declared-frame-count mismatch to exercise the reader, so guarding that path would break negative fixtures. Coverage: 19 tests in `writer/pixel_preflight_tests.rs`, plus destination preservation in `writer/tests/preflight.rs::series_preflight_rejects_later_slice_before_replacing_earlier_slice` and `...::metadata_preflight_rejects_invalid_pixel_descriptions_before_output_changes`. `cargo test -p ritk-io pixel_preflight` 19 passed / 0 failed.
- basis: 4b2d8b031c4ecdeec609295f8dbdff26c2cbeefb
- status: done
- needs: none
- priority: correctness
