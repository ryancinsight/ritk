use super::*;

fn siz_body(
    image_size: (u32, u32),
    image_origin: (u32, u32),
    tile_size: (u32, u32),
    tile_origin: (u32, u32),
    component: (u8, u8, u8),
) -> Vec<u8> {
    let mut body = Vec::with_capacity(39);
    body.extend_from_slice(&0u16.to_be_bytes());
    for value in [
        image_size.0,
        image_size.1,
        image_origin.0,
        image_origin.1,
        tile_size.0,
        tile_size.1,
        tile_origin.0,
        tile_origin.1,
    ] {
        body.extend_from_slice(&value.to_be_bytes());
    }
    body.extend_from_slice(&1u16.to_be_bytes());
    body.extend_from_slice(&[component.0, component.1, component.2]);
    body
}

fn sot_segment(isot: u16, psot: u32, tpsot: u8) -> [u8; 12] {
    let mut segment = [0u8; 12];
    segment[0..2].copy_from_slice(&marker::SOT.to_be_bytes());
    segment[2..4].copy_from_slice(&10u16.to_be_bytes());
    segment[4..6].copy_from_slice(&isot.to_be_bytes());
    segment[6..10].copy_from_slice(&psot.to_be_bytes());
    segment[10] = tpsot;
    segment
}

#[test]
fn cursor_segment_body_round_trips_length() {
    let data: &[u8] = &[0xFF, 0x52, 0x00, 0x05, 0xAA, 0xBB, 0xCC];
    let mut cur = Cursor::new(data);
    let _m = cur.read_u16().expect("infallible: validated precondition"); // consume marker
    let body = cur
        .read_segment_body()
        .expect("infallible: validated precondition");
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

#[test]
fn tile_bounds_intersect_the_image_area() {
    let siz = parse_siz(&siz_body((20, 16), (10, 6), (12, 8), (0, 0), (7, 1, 1)))
        .expect("T.800 B.3 geometry must parse");

    assert_eq!(siz.num_tiles().unwrap(), 4);
    assert_eq!(
        siz.tile_bounds(0).unwrap(),
        TileBounds {
            x0: 0,
            y0: 0,
            width: 2,
            height: 2,
        }
    );
    assert_eq!(
        siz.tile_bounds(3).unwrap(),
        TileBounds {
            x0: 2,
            y0: 2,
            width: 8,
            height: 8,
        }
    );
}

#[test]
fn siz_rejects_tile_origin_beyond_image_origin() {
    let err = parse_siz(&siz_body((20, 16), (10, 6), (12, 8), (11, 0), (7, 1, 1)))
        .expect_err("T.800 B-3 violation must fail");

    assert!(
        err.to_string().contains("must not exceed image origin"),
        "got: {err:#}"
    );
}

#[test]
fn siz_rejects_first_tile_that_misses_the_image() {
    let err = parse_siz(&siz_body((20, 16), (10, 6), (10, 6), (0, 0), (7, 1, 1)))
        .expect_err("T.800 B-4 violation must fail");

    assert!(
        err.to_string()
            .contains("does not intersect the image origin"),
        "got: {err:#}"
    );
}

#[test]
fn siz_rejects_zero_component_sampling() {
    let err = parse_siz(&siz_body((8, 8), (0, 0), (8, 8), (0, 0), (7, 0, 1)))
        .expect_err("zero XRsiz must fail");

    assert!(
        err.to_string().contains("must be in 1..=255"),
        "got: {err:#}"
    );
}

#[test]
fn siz_rejects_tile_grid_outside_sot_index_range() {
    let err = parse_siz(&siz_body((65_536, 1), (0, 0), (1, 1), (0, 0), (7, 1, 1)))
        .expect_err("65,536 tiles cannot be indexed by Isot");

    assert!(
        err.to_string().contains("65535-tile SOT index range"),
        "got: {err:#}"
    );
}

#[test]
fn tile_bounds_reject_out_of_range_sot_index() {
    let siz = parse_siz(&siz_body((8, 8), (0, 0), (8, 8), (0, 0), (7, 1, 1)))
        .expect("single-tile geometry must parse");

    let err = siz.tile_bounds(1).expect_err("tile index 1 must fail");
    assert!(err.to_string().contains("Isot=1"), "got: {err:#}");
    assert!(err.to_string().contains("outside 0..0"), "got: {err:#}");
}

#[test]
fn sot_rejects_reserved_or_too_short_fields() {
    for (segment, expected) in [
        (sot_segment(u16::MAX, 14, 0), "Isot=65535 is reserved"),
        (sot_segment(0, 13, 0), "Psot=13 is invalid"),
        (sot_segment(0, 14, u8::MAX), "TPsot=255 is reserved"),
    ] {
        let err = parse_sot(&segment, 0).expect_err("invalid SOT field must fail");
        assert!(err.to_string().contains(expected), "got: {err:#}");
    }
}
