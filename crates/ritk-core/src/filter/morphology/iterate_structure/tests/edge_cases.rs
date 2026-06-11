//! Edge case tests: singleton, empty, and 3D structures.

use super::*;

// ── Empty structure (all false) ─────────────────────────────────────────────

#[test]
fn empty_structure_iterations_2() {
    let s = struct_2d(&[&[0, 0, 0], &[0, 0, 0], &[0, 0, 0]]);
    let r = iterate_structure(s, 2);
    let expected = struct_2d(&[&[0, 0, 0, 0, 0]; 5]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5, 5]);
}

#[test]
fn empty_3d_structure_iterations_2() {
    let s = struct_3d(&[
        &[&[0u8, 0, 0], &[0, 0, 0], &[0, 0, 0]],
        &[&[0u8, 0, 0], &[0, 0, 0], &[0, 0, 0]],
        &[&[0u8, 0, 0], &[0, 0, 0], &[0, 0, 0]],
    ]);
    let r = iterate_structure(s, 2);
    let expected = struct_3d(&[&[&[0u8; 5]; 5]; 5]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5, 5, 5]);
}

// ── Singleton 1×1×1 structure (no growth) ───────────────────────────────────

#[test]
fn singleton_3d_iterations_3() {
    let s = struct_3d(&[&[&[1u8]]]);
    let r = iterate_structure(s, 3);
    let expected = struct_3d(&[&[&[1u8]]]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[1, 1, 1]);
}

#[test]
fn singleton_2d_iterations_5() {
    let s = struct_2d(&[&[1u8]]);
    let r = iterate_structure(s, 5);
    let expected = struct_2d(&[&[1u8]]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[1, 1]);
}

#[test]
fn singleton_3d_iterations_5() {
    let s = struct_3d(&[&[&[1u8]]]);
    let r = iterate_structure(s, 5);
    let expected = struct_3d(&[&[&[1u8]]]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[1, 1, 1]);
}
