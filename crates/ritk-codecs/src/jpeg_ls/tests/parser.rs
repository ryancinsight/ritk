use super::*;

#[test]
fn parse_jpeg_ls_headers_rejects_missing_soi() {
    let mut decoder = JpegLsDecoder::new();
    let bad_data = [0x00u8, 0x00u8];
    assert!(parse_jpeg_ls_headers(&mut decoder, &bad_data).is_err());
}

#[test]
fn find_scan_data_returns_none_without_sos() {
    let data = [0xFF, 0xD8];
    assert!(find_scan_data(&data).is_none());
}

#[test]
fn find_scan_data_returns_bytes_after_sos_header() {
    let data: &[u8] = &[
        0xFF, 0xD8, // SOI
        0xFF, 0xDA, // SOS
        0x00, 0x08, // length
        0x01, // Ns
        0x01, 0x00, // component table
        0x00, 0x00, 0x00, // NEAR, ILV, Ah/Al
        0xAB, 0xCD, 0xEF, // scan data
    ];
    let scan_data = find_scan_data(data).expect("scan data must be present");
    assert_eq!(scan_data, &[0xAB, 0xCD, 0xEF]);
}
