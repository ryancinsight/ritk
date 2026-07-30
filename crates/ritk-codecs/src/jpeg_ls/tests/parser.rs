use super::*;

#[test]
fn parse_jpeg_ls_headers_rejects_missing_soi() {
    let mut decoder = JpegLsDecoder::new();
    let bad_data = [0x00u8, 0x00u8];
    let error =
        parse_jpeg_ls_headers(&mut decoder, &bad_data).expect_err("missing SOI must be rejected");
    assert!(
        error.to_string().contains("does not start with SOI"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn parse_jpeg_ls_headers_rejects_invalid_interleave_mode() {
    let data = [
        0xFF, 0xD8, // SOI
        0xFF, 0xDA, // SOS
        0x00, 0x08, // Ls
        0x01, // Ns
        0x01, 0x00, // component selector and mapping table
        0x00, 0x03, 0x00, // NEAR, invalid ILV, point transform
    ];
    let mut decoder = JpegLsDecoder::new();
    let error = parse_jpeg_ls_headers(&mut decoder, &data)
        .expect_err("invalid interleave mode must be rejected");
    assert!(
        error.to_string().contains("interleave mode 3"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn parse_jpeg_ls_headers_rejects_unsupported_mapping_table() {
    let data = [
        0xFF, 0xD8, // SOI
        0xFF, 0xDA, // SOS
        0x00, 0x08, // Ls
        0x01, // Ns
        0x01, 0x01, // component selector and unsupported mapping table
        0x00, 0x00, 0x00, // NEAR, ILV, point transform
        0x00, // scan byte
    ];
    let mut decoder = JpegLsDecoder::new();
    let error = parse_jpeg_ls_headers(&mut decoder, &data)
        .expect_err("unsupported mapping table must be rejected");
    assert!(
        error.to_string().contains("mapping table selector 1"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn parse_jpeg_ls_headers_rejects_restart_interval() {
    let data = [
        0xFF, 0xD8, // SOI
        0xFF, 0xDD, // DRI
        0x00, 0x04, 0x00, 0x01, // restart interval = 1
        0xFF, 0xDA, // SOS
        0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let mut decoder = JpegLsDecoder::new();
    let error = parse_jpeg_ls_headers(&mut decoder, &data)
        .expect_err("nonzero restart interval must be rejected");
    assert!(
        error.to_string().contains("restart interval 1"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn parse_jpeg_ls_headers_rejects_missing_sos() {
    let data = [0xFF, 0xD8];
    let mut decoder = JpegLsDecoder::new();
    let error =
        parse_jpeg_ls_headers(&mut decoder, &data).expect_err("missing SOS must be rejected");
    assert!(
        error.to_string().contains("SOS marker is missing"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn parse_jpeg_ls_headers_returns_bytes_after_sos_header() {
    let data: &[u8] = &[
        0xFF, 0xD8, // SOI
        0xFF, 0xDA, // SOS
        0x00, 0x08, // length
        0x01, // Ns
        0x01, 0x00, // component table
        0x00, 0x00, 0x00, // NEAR, ILV, Ah/Al
        0xAB, 0xCD, 0xEF, // scan data
    ];
    let mut decoder = JpegLsDecoder::new();
    let scan_data = parse_jpeg_ls_headers(&mut decoder, data).expect("scan data must be present");
    assert_eq!(scan_data, &[0xAB, 0xCD, 0xEF]);
}

#[test]
fn parse_jpeg_ls_headers_rejects_truncated_sos_segment() {
    let data = [
        0xFF, 0xD8, // SOI
        0xFF, 0xDA, // SOS
        0x00, 0x08, // claims an eight-byte segment
        0x01, 0x01, 0x00, // truncated before NEAR, ILV, and point transform
    ];
    let mut decoder = JpegLsDecoder::new();
    let error = parse_jpeg_ls_headers(&mut decoder, &data)
        .expect_err("truncated SOS segment must be rejected");
    assert!(
        error.to_string().contains("truncated JPEG-LS SOS segment"),
        "unexpected error: {error:#}"
    );
}

proptest::proptest! {
    #[test]
    fn arbitrary_marker_bytes_never_panic(
        data in proptest::collection::vec(proptest::num::u8::ANY, 0..=512),
    ) {
        let mut decoder = JpegLsDecoder::new();
        if let Ok(scan) = parse_jpeg_ls_headers(&mut decoder, &data) {
            proptest::prop_assert!(!scan.is_empty());
            proptest::prop_assert!(scan.len() <= data.len());
        }
    }
}
