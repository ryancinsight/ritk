#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;
use crate::PixelSignedness;

fn make_layout(rows: usize, cols: usize, bits: u16, signed: PixelSignedness) -> PixelLayout {
    PixelLayout {
        rows,
        cols,
        samples_per_pixel: 1,
        bits_allocated: if bits <= 8 { 8 } else { 16 },
        bits_stored: bits,
        pixel_representation: signed,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    }
}

#[test]
fn is_soc_accepts_valid_prefix() {
    assert!(is_soc(&[0xFF, 0x4F, 0x00]));
}

#[test]
fn is_soc_rejects_jpeg_baseline_prefix() {
    assert!(!is_soc(&[0xFF, 0xD8, 0xFF, 0xE0]));
}

#[test]
fn is_soc_rejects_empty_slice() {
    assert!(!is_soc(&[]));
}

#[test]
fn decode_j2k_fragment_rejects_non_soc() {
    let data = [0xFF_u8, 0xD8, 0x00];
    let err = decode_j2k_fragment(
        &data,
        make_layout(1, 1, 8, PixelSignedness::Unsigned),
        ImageReadBudget::DEFAULT,
    )
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("SOC") || format!("{err:#}").contains("0xFF4F"),
        "error must mention SOC; got: {err:#}"
    );
}

#[test]
fn unsigned_dc_restoration_covers_complete_32_bit_range() {
    assert_eq!(restore_component_sample(i32::MIN, 32, false), 0);
    assert_eq!(
        restore_component_sample(i32::MAX, 32, false),
        i64::from(u32::MAX)
    );
    assert_eq!(
        restore_component_sample(i32::MIN, 32, true),
        i64::from(i32::MIN)
    );
}
