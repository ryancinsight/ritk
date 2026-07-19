use super::*;

#[test]
fn cursor_segment_body_round_trips_length() {
    let data: &[u8] = &[0xFF, 0x52, 0x00, 0x05, 0xAA, 0xBB, 0xCC];
    let mut cur = Cursor::new(data);
    let _m = cur.read_u16().unwrap(); // consume marker
    let body = cur.read_segment_body().unwrap();
    assert_eq!(body, &[0xAA, 0xBB, 0xCC]);
    assert_eq!(cur.pos(), 7);
}

#[test]
fn component_spec_precision_and_signed() {
    let c = ComponentSpec {
        ssiz: 0x87,
        xr_siz: 1,
        yr_siz: 1,
    };
    assert_eq!(c.precision(), 8);
    assert!(c.is_signed());
    let u = ComponentSpec {
        ssiz: 0x07,
        xr_siz: 1,
        yr_siz: 1,
    };
    assert_eq!(u.precision(), 8);
    assert!(!u.is_signed());
}
