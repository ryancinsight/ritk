//! Tests for `BoolStructure` construction, shape, indexing, center, and dilate.

use super::*;

#[test]
fn bool_structure_count_and_is_empty() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    assert_eq!(s.count(), 5);
    assert!(!s.is_empty());
    let e = struct_2d(&[&[0, 0], &[0, 0]]);
    assert_eq!(e.count(), 0);
    assert!(e.is_empty());
}

#[test]
fn bool_structure_center_default_origin() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    assert_eq!(s.center(), [1, 1]);
    let s4 = struct_2d(&[&[1, 1, 1, 1], &[1, 1, 1, 1]]);
    assert_eq!(s4.center(), [1, 2]);
}

#[test]
fn bool_structure_flat_to_multi_round_trip() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    for flat in 0..s.size() {
        let multi = s.flat_to_multi(flat);
        assert_eq!(s.multi_to_flat(&multi), flat, "flat = {}", flat);
    }
}

#[test]
fn bool_structure_from_shape_fn() {
    let s = BoolStructure::<2>::from_shape_fn([3, 3], |idx| idx[0] == idx[1]);
    let expected = struct_2d(&[&[1, 0, 0], &[0, 1, 0], &[0, 0, 1]]);
    assert_eq!(s, expected);
}

// ── Dilate smoke tests ─────────────────────────────────────────────────────

#[test]
fn dilate_1d_iteration_1() {
    // Dilation by a 1D structure that is True only at its center
    // ([F, T, F] of shape 3) is identity: the splat places the
    // structure's single true voxel exactly on the input pixel.
    let inp = BoolStructure::<1>::from_data([5], vec![true, false, false, false, false]);
    let ker = BoolStructure::<1>::from_data([3], vec![false, true, false]);
    let out = inp.dilate(&ker, 1);
    let expected = BoolStructure::<1>::from_data([5], vec![true, false, false, false, false]);
    assert_eq!(out, expected);
}

#[test]
fn dilate_1d_all_true_structure() {
    // Dilation by [T, T, T] of shape 3 (center 1) on input
    // [T, F, F, F, F]: out[0] = T (in[0]); out[1] = T (in[0]); rest F.
    // So result is [T, T, F, F, F]. scipy parity verified.
    let inp = BoolStructure::<1>::from_data([5], vec![true, false, false, false, false]);
    let ker = BoolStructure::<1>::from_data([3], vec![true, true, true]);
    let out = inp.dilate(&ker, 1);
    let expected = BoolStructure::<1>::from_data([5], vec![true, true, false, false, false]);
    assert_eq!(out, expected);
}

#[test]
fn dilate_1d_all_true_structure_centered_input() {
    // Same kernel, input with a single T at index 2: out is the
    // kernel-stamped view, so [F, T, T, T, F].
    let inp = BoolStructure::<1>::from_data([5], vec![false, false, true, false, false]);
    let ker = BoolStructure::<1>::from_data([3], vec![true, true, true]);
    let out = inp.dilate(&ker, 1);
    let expected = BoolStructure::<1>::from_data([5], vec![false, true, true, true, false]);
    assert_eq!(out, expected);
}

#[test]
fn dilate_2d_single_voxel_origin_at_center() {
    // Single True voxel at (0, 0) of 3x3 dilated by 3x3 cross once:
    // out is a 3x3 with (0, 0), (0, 1), (1, 0) = True
    let inp = struct_2d(&[&[1, 0, 0], &[0, 0, 0], &[0, 0, 0]]);
    let ker = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let out = inp.dilate(&ker, 1);
    let expected = struct_2d(&[&[1, 1, 0], &[1, 0, 0], &[0, 0, 0]]);
    assert_eq!(out, expected);
}

#[test]
fn dilate_2d_all_true_kernel_central_voxel() {
    // scipy v1.17.1 reference: 3x3 all-True kernel dilates 5x5 with a
    // single T at (0, 0) to a 2x2 block at the top-left.
    let inp = struct_2d(&[
        &[1, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0],
    ]);
    let ker = struct_2d(&[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]]);
    let out = inp.dilate(&ker, 1);
    let expected = struct_2d(&[
        &[1, 1, 0, 0, 0],
        &[1, 1, 0, 0, 0],
        &[0, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0],
    ]);
    assert_eq!(out, expected);
}

#[test]
fn dilate_2d_all_true_kernel_center_voxel() {
    // scipy v1.17.1 reference: 3x3 all-True kernel dilates 5x5 with a
    // single T at (2, 2) to a 3x3 block in the centre.
    let mut inp = struct_2d(&[&[0, 0, 0, 0, 0]; 5]);
    {
        let mut data = inp.as_mut_slice().to_vec();
        data[2 * 5 + 2] = true;
        inp = BoolStructure::<2>::from_data([5, 5], data);
    }
    let ker = struct_2d(&[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]]);
    let out = inp.dilate(&ker, 1);
    let expected = struct_2d(&[
        &[0, 0, 0, 0, 0],
        &[0, 1, 1, 1, 0],
        &[0, 1, 1, 1, 0],
        &[0, 1, 1, 1, 0],
        &[0, 0, 0, 0, 0],
    ]);
    assert_eq!(out, expected);
}
