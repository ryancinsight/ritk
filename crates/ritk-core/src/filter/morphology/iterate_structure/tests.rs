//! Tests for `iterate_structure` and `BoolStructure`.

use super::*;

#[test]
fn iterations_zero_returns_copy() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let r = iterate_structure(&s, 0);
    assert_eq!(r, s);
    assert_eq!(r.shape(), s.shape());
}

#[test]
fn iterations_one_returns_copy() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let r = iterate_structure(&s, 1);
    assert_eq!(r, s);
    assert_eq!(r.shape(), s.shape());
}

// ── 2-D cross, iterations 2/3/4/5 ───────────────────────────────────────────

#[test]
fn cross_2d_iterations_2_is_5x5_diamond() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let r = iterate_structure(&s, 2);
    let expected = struct_2d(&[
        &[0, 0, 1, 0, 0],
        &[0, 1, 1, 1, 0],
        &[1, 1, 1, 1, 1],
        &[0, 1, 1, 1, 0],
        &[0, 0, 1, 0, 0],
    ]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5, 5]);
}

#[test]
fn cross_2d_iterations_3_is_7x7_diamond() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let r = iterate_structure(&s, 3);
    let expected = struct_2d(&[
        &[0, 0, 0, 1, 0, 0, 0],
        &[0, 0, 1, 1, 1, 0, 0],
        &[0, 1, 1, 1, 1, 1, 0],
        &[1, 1, 1, 1, 1, 1, 1],
        &[0, 1, 1, 1, 1, 1, 0],
        &[0, 0, 1, 1, 1, 0, 0],
        &[0, 0, 0, 1, 0, 0, 0],
    ]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[7, 7]);
}

#[test]
fn cross_2d_iterations_4_is_9x9_diamond() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let r = iterate_structure(&s, 4);
    let expected = struct_2d(&[
        &[0, 0, 0, 0, 1, 0, 0, 0, 0],
        &[0, 0, 0, 1, 1, 1, 0, 0, 0],
        &[0, 0, 1, 1, 1, 1, 1, 0, 0],
        &[0, 1, 1, 1, 1, 1, 1, 1, 0],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[0, 1, 1, 1, 1, 1, 1, 1, 0],
        &[0, 0, 1, 1, 1, 1, 1, 0, 0],
        &[0, 0, 0, 1, 1, 1, 0, 0, 0],
        &[0, 0, 0, 0, 1, 0, 0, 0, 0],
    ]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[9, 9]);
}

#[test]
fn cross_2d_iterations_5_is_11x11_diamond() {
    // scipy v1.17.1 reference
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let r = iterate_structure(&s, 5);
    let expected = struct_2d(&[
        &[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        &[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        &[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        &[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        &[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        &[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        &[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        &[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[11, 11]);
}

// ── 3-D cross, iterations 2/3 ───────────────────────────────────────────────

#[test]
fn cross_3d_iterations_2_is_5x5x5_diamond() {
    let s = struct_3d(&[
        &[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]],
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
        &[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]],
    ]);
    let r = iterate_structure(&s, 2);
    let expected = struct_3d(&[
        &[
            &[0, 0, 1, 0, 0],
            &[0, 1, 1, 1, 0],
            &[1, 1, 1, 1, 1],
            &[0, 1, 1, 1, 0],
            &[0, 0, 1, 0, 0],
        ],
        &[
            &[0, 1, 1, 1, 0],
            &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1],
            &[0, 1, 1, 1, 0],
        ],
        &[
            &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1],
        ],
        &[
            &[0, 1, 1, 1, 0],
            &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1],
            &[1, 1, 1, 1, 1],
            &[0, 1, 1, 1, 0],
        ],
        &[
            &[0, 0, 1, 0, 0],
            &[0, 1, 1, 1, 0],
            &[1, 1, 1, 1, 1],
            &[0, 1, 1, 1, 0],
            &[0, 0, 1, 0, 0],
        ],
    ]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5, 5, 5]);
}

// ── 3-D cube, iterations 2/3/4 (cube stays cube) ───────────────────────────

#[test]
fn cube_3d_iterations_2_is_5x5x5_cube() {
    let s = struct_3d(&[
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
    ]);
    let r = iterate_structure(&s, 2);
    let expected = struct_3d(&[&[&[1u8; 5]; 5]; 5]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5, 5, 5]);
}

#[test]
fn cube_3d_iterations_3_is_7x7x7_cube() {
    let s = struct_3d(&[
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
    ]);
    let r = iterate_structure(&s, 3);
    let expected = struct_3d(&[&[&[1u8; 7]; 7]; 7]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[7, 7, 7]);
}

#[test]
fn cube_3d_iterations_4_is_9x9x9_cube() {
    let s = struct_3d(&[
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
    ]);
    let r = iterate_structure(&s, 4);
    let expected = struct_3d(&[&[&[1u8; 9]; 9]; 9]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[9, 9, 9]);
}

// ── Empty structure (all false) ─────────────────────────────────────────────

#[test]
fn empty_structure_iterations_2() {
    let s = struct_2d(&[&[0, 0, 0], &[0, 0, 0], &[0, 0, 0]]);
    let r = iterate_structure(&s, 2);
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
    let r = iterate_structure(&s, 2);
    let expected = struct_3d(&[&[&[0u8; 5]; 5]; 5]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5, 5, 5]);
}

// ── Singleton 1×1×1 structure (no growth) ───────────────────────────────────

#[test]
fn singleton_3d_iterations_3() {
    let s = struct_3d(&[&[&[1u8]]]);
    let r = iterate_structure(&s, 3);
    let expected = struct_3d(&[&[&[1u8]]]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[1, 1, 1]);
}

#[test]
fn singleton_2d_iterations_5() {
    let s = struct_2d(&[&[1u8]]);
    let r = iterate_structure(&s, 5);
    let expected = struct_2d(&[&[1u8]]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[1, 1]);
}

#[test]
fn singleton_3d_iterations_5() {
    let s = struct_3d(&[&[&[1u8]]]);
    let r = iterate_structure(&s, 5);
    let expected = struct_3d(&[&[&[1u8]]]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[1, 1, 1]);
}

// ── Asymmetric structures ───────────────────────────────────────────────────

#[test]
fn asymmetric_2d_iterations_2() {
    // 2×3 structure: only top-right 2×2 block is true
    let s = struct_2d(&[&[0, 1, 1], &[0, 1, 1]]);
    let r = iterate_structure(&s, 2);
    assert_eq!(r.shape(), &[3, 5]);
    // After stamping + 1 scipy dilation: the 2×2 block (rows 0-1,
    // columns 2-3 of the 3×5 canvas) spreads to columns 2-4 in rows
    // 0-1; row 2 remains empty (no true input there).
    let expected = struct_2d(&[&[0, 0, 1, 1, 1], &[0, 0, 1, 1, 1], &[0, 0, 0, 0, 0]]);
    assert_eq!(r, expected);
}

#[test]
fn l_shape_3d_iterations_2() {
    // 3D L-shape: True at (0, 0, 0) and (2, 2, 2) of 3×3×3.
    // scipy v1.17.1 reference: structure stays the same — the two voxels
    // are too far apart for 1 dilation to connect them.
    let s = struct_3d(&[
        &[&[1, 0, 0], &[0, 0, 0], &[0, 0, 0]],
        &[&[0, 0, 0], &[0, 0, 0], &[0, 0, 0]],
        &[&[0, 0, 0], &[0, 0, 0], &[0, 0, 1]],
    ]);
    let r = iterate_structure(&s, 2);
    let expected = struct_3d(&[&[&[1u8, 0, 0, 0, 0]; 5]; 5]);
    // Append the (4, 4, 4) True voxel
    let mut exp_data = expected.as_slice().to_vec();
    exp_data[4 * 25 + 4 * 5 + 4] = true;
    let expected = BoolStructure::<3>::from_data([5, 5, 5], exp_data);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5, 5, 5]);
}

// ── 1-D structures ──────────────────────────────────────────────────────────

#[test]
fn line_1d_iterations_2() {
    // 1D [F, T, F] structure: a single pixel. Iterating it 2 times
    // produces a structure of the same shape stamped at pos 1 in a
    // length-5 canvas, then dilated by [F, T, F] 1 time. Dilation by
    // [F, T, F] is identity (structure center is the only true voxel),
    // so the result is the single pixel at index 2.
    let s = BoolStructure::<1>::from_data([3], vec![false, true, false]);
    let r = iterate_structure(&s, 2);
    let expected = BoolStructure::<1>::from_data([5], vec![false, false, true, false, false]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5]);
}

#[test]
fn line_1d_iterations_3() {
    let s = BoolStructure::<1>::from_data([3], vec![false, true, false]);
    let r = iterate_structure(&s, 3);
    let expected =
        BoolStructure::<1>::from_data([7], vec![false, false, false, true, false, false, false]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[7]);
}

#[test]
fn line_1d_all_true_iterations_2() {
    // 1D [T, T, T] structure: full interval. Iterating grows it to
    // length 5 (1 + 1 * 2 on each side, ni=1).
    let s = BoolStructure::<1>::from_data([3], vec![true, true, true]);
    let r = iterate_structure(&s, 2);
    let expected = BoolStructure::<1>::from_data([5], vec![true, true, true, true, true]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5]);
}

#[test]
fn line_1d_all_true_iterations_4() {
    // 1D [T, T, T] structure iterations=4: length 9, all True.
    let s = BoolStructure::<1>::from_data([3], vec![true, true, true]);
    let r = iterate_structure(&s, 4);
    let expected = BoolStructure::<1>::from_data([9], vec![true; 9]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[9]);
}

#[test]
fn line_1d_two_pixels_iterations_2() {
    // 1D [T, F, T] structure: two pixels at indices 0 and 2, separated
    // by 1 voxel. After 1 dilation by [T, F, T], the two pixels merge
    // to a length-3 interval [T, T, T] in the centre of the length-5
    // output.
    let s = BoolStructure::<1>::from_data([3], vec![true, false, true]);
    let r = iterate_structure(&s, 2);
    let expected = BoolStructure::<1>::from_data([5], vec![true, false, true, false, true]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5]);
}

// ── 2x2 all-True ────────────────────────────────────────────────────────────

#[test]
fn two_by_two_all_true_iterations_3() {
    // 2x2 all-True structure iterations=3 → 4x4 all-True. Shape:
    // 2 + 2 * 1 = 4 per axis.
    let s = struct_2d(&[&[1, 1], &[1, 1]]);
    let r = iterate_structure(&s, 3);
    let expected = struct_2d(&[&[1, 1, 1, 1]; 4]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[4, 4]);
}

// ── iterate_structure_with_origin origin tracking ──────────────────────────

#[test]
fn origin_tracking_cross_center() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    for (it, expected_origin) in [
        (2usize, [2usize, 2usize]),
        (3, [3, 3]),
        (4, [4, 4]),
        (5, [5, 5]),
    ] {
        let (_r, o) = iterate_structure_with_origin(&s, it, [1, 1]);
        assert_eq!(o, expected_origin, "iterations = {}", it);
    }
}

#[test]
fn origin_tracking_zero_origin() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let (_r, o) = iterate_structure_with_origin(&s, 3, [0, 0]);
    assert_eq!(o, [0, 0]);
}

#[test]
fn origin_tracking_iterations_below_2() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let (_r, o) = iterate_structure_with_origin(&s, 1, [1, 1]);
    // scipy still scales: new_origin = iterations * origin = 1 * [1, 1]
    assert_eq!(o, [1, 1]);
}

#[test]
fn origin_tracking_3d_cross_center() {
    let s = struct_3d(&[
        &[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]],
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
        &[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]],
    ]);
    let (_r, o) = iterate_structure_with_origin(&s, 2, [1, 1, 1]);
    assert_eq!(o, [2, 2, 2]);
}

#[test]
fn origin_tracking_2d_zero_iter2() {
    // scipy reference: iterate_structure(cross_2d, 2, origin=(0, 0))
    // → returns (the 5x5 diamond, [0, 0])
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let (r, o) = iterate_structure_with_origin(&s, 2, [0, 0]);
    assert_eq!(o, [0, 0]);
    assert_eq!(r.shape(), &[5, 5]);
}

// ── BoolStructure helpers ──────────────────────────────────────────────────

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
    let inp = struct_2d(&[&[1, 0, 0, 0, 0]; 5]);
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

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Build a 2-D BoolStructure from a row-major vec of 0/1.
fn struct_2d<R: AsRef<[u8]>>(rows: &[R]) -> BoolStructure<2> {
    let shape = [rows.len(), rows[0].as_ref().len()];
    let data: Vec<bool> = rows
        .iter()
        .flat_map(|r| r.as_ref().iter())
        .map(|&v| v != 0)
        .collect();
    BoolStructure::from_data(shape, data)
}

/// Build a 3-D BoolStructure from a flat row-major vec of 0/1.
fn struct_3d<Y: AsRef<[u8]>, Z: AsRef<[Y]>>(zs: &[Z]) -> BoolStructure<3> {
    let nz = zs.len();
    let ny = zs[0].as_ref().len();
    let nx = zs[0].as_ref()[0].as_ref().len();
    let shape = [nz, ny, nx];
    let data: Vec<bool> = zs
        .iter()
        .flat_map(|z| z.as_ref().iter())
        .flat_map(|y| y.as_ref().iter())
        .map(|&v| v != 0)
        .collect();
    BoolStructure::from_data(shape, data)
}
