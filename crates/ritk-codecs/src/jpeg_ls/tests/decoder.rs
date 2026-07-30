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
    assert_eq!(decoder.interleave_mode, InterleaveMode::None);
    assert_eq!(decoder.point_transform, 0);
}

#[test]
fn decode_fragment_near_lossless_bounded_error() {
    // NEAR=2 native encode → native decode must satisfy |s' − s| ≤ 2 for all
    // samples (ISO 14495-1 §A.4.4 analytical bound; tolerance is exact).
    let original: Vec<u16> = vec![
        10, 50, 100, 150, 200, 245, 30, 80, 130, 180, 220, 60, 110, 160, 210, 40,
    ];
    let stream = crate::jpeg_ls::encoder::encode_grayscale_jpeg_ls(&original, 4, 4, 8, 2)
        .expect("valid near-lossless fixture must encode");
    let layout = crate::PixelLayout {
        rows: 4,
        cols: 4,
        samples_per_pixel: 1,
        bits_allocated: 8,
        pixel_representation: crate::PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    };
    let decoded =
        decode_jpeg_ls_fragment(&stream, layout).expect("near-lossless decode must succeed");
    assert_eq!(decoded.len(), original.len());
    for (i, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
        let err = (f32::from(orig) - dec).abs();
        assert!(
            err <= 2.0,
            "sample[{i}]: |{dec} − {orig}| = {err} exceeds NEAR=2 bound"
        );
    }
}

#[test]
fn decode_fragment_rejects_zero_dimensions() {
    let decoder = one_component_decoder(0, 100, 8, 0, InterleaveMode::None, 0);
    let error = decoder
        .decode_fragment(&[])
        .expect_err("zero-width JPEG-LS fragments must fail before scan decode");
    assert!(
        format!("{error:#}").contains("invalid dimensions"),
        "unexpected zero-dimension error: {error:#}"
    );
}

#[test]
fn decode_fragment_rejects_nonzero_point_transform() {
    let decoder = one_component_decoder(100, 100, 8, 0, InterleaveMode::None, 1);
    let error = decoder
        .decode_fragment(&[])
        .expect_err("nonzero point transforms must fail before scan decode");
    let msg = format!("{error:#}");
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
    interleave_mode: InterleaveMode,
    point_transform: u8,
) -> JpegLsDecoder {
    JpegLsDecoder {
        width,
        height,
        bits_per_sample,
        components: vec![ComponentInfo {}],
        near,
        interleave_mode,
        point_transform,
        t1: 0,
        t2: 0,
        t3: 0,
    }
}
