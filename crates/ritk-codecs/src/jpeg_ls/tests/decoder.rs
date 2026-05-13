use super::*;

#[test]
fn jpeg_ls_marker_constants_correct() {
    assert_eq!(SOI, 0xFFD8);
    assert_eq!(SOF55, 0xFFF7);
    assert_eq!(SOS, 0xFFDA);
    assert_eq!(LSE, 0xFFF8);
    assert_eq!(EOI, 0xFFD9);
}

#[test]
fn decoder_new_initializes_defaults() {
    let decoder = JpegLsDecoder::new();
    assert_eq!(decoder.width, 0);
    assert_eq!(decoder.height, 0);
    assert_eq!(decoder.bits_per_sample, 8);
    assert_eq!(decoder.near, 0);
    assert_eq!(decoder.interleave_mode, 0);
    assert_eq!(decoder.point_transform, 0);
}

#[test]
fn decode_fragment_rejects_near_nonzero() {
    let decoder = one_component_decoder(100, 100, 8, 1, 0, 0);
    let result = decoder.decode_fragment(&[]);
    assert!(result.is_err());
    let msg = format!("{:?}", result.unwrap_err());
    assert!(msg.contains("NEAR"), "Expected 'NEAR' in error: {msg}");
}

#[test]
fn decode_fragment_rejects_zero_dimensions() {
    let decoder = one_component_decoder(0, 100, 8, 0, 0, 0);
    let result = decoder.decode_fragment(&[]);
    assert!(result.is_err());
}

#[test]
fn decode_fragment_rejects_nonzero_point_transform() {
    let decoder = one_component_decoder(100, 100, 8, 0, 0, 1);
    let result = decoder.decode_fragment(&[]);
    assert!(result.is_err());
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("point transform"),
        "Expected point-transform error, got: {msg}"
    );
}

fn one_component_decoder(
    width: usize,
    height: usize,
    bits_per_sample: u32,
    near: u32,
    interleave_mode: u8,
    point_transform: u8,
) -> JpegLsDecoder {
    JpegLsDecoder {
        width,
        height,
        bits_per_sample,
        components: vec![ComponentInfo {
            id: 1,
            mapping_table_selector: 0,
            context: [ContextState::default(); 365],
        }],
        near,
        interleave_mode,
        point_transform,
        restart_interval: 0,
        t1: 0,
        t2: 0,
        t3: 0,
    }
}
