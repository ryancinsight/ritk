use super::*;

fn layout_8bit(rows: usize, cols: usize, slope: f32, intercept: f32) -> PixelLayout {
    PixelLayout {
        rows,
        cols,
        samples_per_pixel: 1,
        bits_allocated: 8,
        pixel_representation: 0,
        rescale_slope: slope,
        rescale_intercept: intercept,
    }
}

/// Build a minimal single-component JPEG-LS 8-bit lossless frame.
fn build_jpeg_ls_frame(height: u16, width: u16, scan_data: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(29 + scan_data.len());
    frame.extend_from_slice(&[0xFF, 0xD8]);
    frame.extend_from_slice(&[0xFF, 0xF7, 0x00, 0x0B, 0x08]);
    frame.extend_from_slice(&height.to_be_bytes());
    frame.extend_from_slice(&width.to_be_bytes());
    frame.extend_from_slice(&[0x01, 0x01, 0x11, 0x00]);
    frame.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00]);
    frame.extend_from_slice(scan_data);
    frame.extend_from_slice(&[0xFF, 0xD9]);
    frame
}

#[test]
fn jpeg_ls_fragment_2x2_all_zero_decodes_correctly() {
    let frame = build_jpeg_ls_frame(2, 2, &[0xF8]);
    let layout = layout_8bit(2, 2, 1.0, 0.0);
    let result = decode_jpeg_ls_fragment(&frame, layout).unwrap();
    assert_eq!(result, vec![0.0f32, 0.0, 0.0, 0.0]);
}

#[test]
fn jpeg_ls_fragment_1x3_constant_value10_decodes_correctly() {
    // Derivation: first sample is a run interrupt with mapped error 19 at k=2;
    // subsequent equal samples are regular-mode zero errors at k=2 then k=1.
    let frame = build_jpeg_ls_frame(1, 3, &[0x07, 0x90]);
    let layout = layout_8bit(1, 3, 1.0, 0.0);
    let result = decode_jpeg_ls_fragment(&frame, layout).unwrap();
    assert_eq!(result, vec![10.0f32, 10.0, 10.0]);
}

#[test]
fn jpeg_ls_fragment_1x1_run_interrupt_with_modality_lut() {
    // Sample 2 with slope 2 and intercept -5 yields -1 after modality LUT.
    let frame = build_jpeg_ls_frame(1, 1, &[0x70]);
    let layout = layout_8bit(1, 1, 2.0, -5.0);
    let result = decode_jpeg_ls_fragment(&frame, layout).unwrap();
    assert_eq!(result, vec![-1.0f32]);
}
