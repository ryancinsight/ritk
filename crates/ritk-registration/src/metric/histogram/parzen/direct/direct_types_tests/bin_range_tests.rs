//! BinRange tests (ARCH-316-04).

use super::super::types::BinRange;

#[test]
fn bin_range_interior_value() {
    // Interior value: no clamping needed.
    let range = BinRange::new(10, 3, 32);
    assert_eq!(range.lo, 7);
    assert_eq!(range.hi, 13);
    assert_eq!(range.len(), 7);
    assert!(!range.is_empty());
}

#[test]
fn bin_range_near_lower_boundary() {
    // Value near 0: lo should clamp to 0.
    let range = BinRange::new(1, 3, 32);
    assert_eq!(range.lo, 0);
    assert_eq!(range.hi, 4);
    assert_eq!(range.len(), 5);
}

#[test]
fn bin_range_near_upper_boundary() {
    // Value near num_bins-1: hi should clamp to num_bins-1.
    let range = BinRange::new(30, 3, 32);
    assert_eq!(range.lo, 27);
    assert_eq!(range.hi, 31);
    assert_eq!(range.len(), 5);
}

#[test]
fn bin_range_primary_exceeds_num_bins() {
    // When primary > num_bins-1, both lo and hi clamp to the boundary.
    let range = BinRange::new(22, 3, 16);
    assert_eq!(range.lo, 15); // clamped from 19 to 15
    assert_eq!(range.hi, 15);
    assert_eq!(range.len(), 1);
}

#[test]
fn bin_range_primary_negative() {
    // When primary is negative, both lo and hi clamp to 0.
    let range = BinRange::new(-5, 3, 32);
    assert_eq!(range.lo, 0);
    assert_eq!(range.hi, 0);
    assert_eq!(range.len(), 1);
}

#[test]
fn bin_range_iter_produces_correct_indices() {
    let range = BinRange::new(5, 2, 32);
    let indices: Vec<usize> = range.iter().collect();
    assert_eq!(indices, vec![3, 4, 5, 6, 7]);
}
