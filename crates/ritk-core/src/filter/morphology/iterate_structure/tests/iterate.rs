//! Tests for `iterate_structure` and `iterate_structure_with_origin`.

use super::*;

// ── iterations < 2 returns copy ────────────────────────────────────────────

#[test]
fn iterations_zero_returns_copy() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let r = iterate_structure(s.clone(), 0);
    assert_eq!(r, s);
    assert_eq!(r.shape(), s.shape());
}

#[test]
fn iterations_one_returns_copy() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let r = iterate_structure(s.clone(), 1);
    assert_eq!(r, s);
    assert_eq!(r.shape(), s.shape());
}

// ── 2-D cross, iterations 2/3/4/5 ───────────────────────────────────────────

#[test]
fn cross_2d_iterations_2_is_5x5_diamond() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let r = iterate_structure(s, 2);
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
    let r = iterate_structure(s, 3);
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
    let r = iterate_structure(s, 4);
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
    let r = iterate_structure(s, 5);
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

// ── 3-D cross, iterations 2 ───────────────────────────────────────────────

#[test]
fn cross_3d_iterations_2_is_5x5x5_diamond() {
    let s = struct_3d(&[
        &[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]],
        &[&[1, 1, 1], &[1, 1, 1], &[1, 1, 1]],
        &[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]],
    ]);
    let r = iterate_structure(s, 2);
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
    let r = iterate_structure(s, 2);
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
    let r = iterate_structure(s, 3);
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
    let r = iterate_structure(s, 4);
    let expected = struct_3d(&[&[&[1u8; 9]; 9]; 9]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[9, 9, 9]);
}

// ── Asymmetric structures ───────────────────────────────────────────────────

#[test]
fn asymmetric_2d_iterations_2() {
    // 2×3 structure: only top-right 2×2 block is true
    let s = struct_2d(&[&[0, 1, 1], &[0, 1, 1]]);
    let r = iterate_structure(s, 2);
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
    // After stamping at (1,1,1) in 5×5×5: True at (1,1,1) and (3,3,3).
    // Kernel offsets: (-1,-1,-1) and (1,1,1) (center=(1,1,1), odd size).
    // After 1 dilation: True at (0,0,0), (2,2,2), (4,4,4).
    let s = struct_3d(&[
        &[&[1, 0, 0], &[0, 0, 0], &[0, 0, 0]],
        &[&[0, 0, 0], &[0, 0, 0], &[0, 0, 0]],
        &[&[0, 0, 0], &[0, 0, 0], &[0, 0, 1]],
    ]);
    let r = iterate_structure(s, 2);
    assert_eq!(r.shape(), &[5, 5, 5]);
    // Three True voxels along the main diagonal.
    let mut exp_data = vec![false; 125];
    exp_data[0] = true; // (0, 0, 0)
    exp_data[2 * 25 + 2 * 5 + 2] = true; // (2, 2, 2)
    exp_data[4 * 25 + 4 * 5 + 4] = true; // (4, 4, 4)
    let expected = BoolStructure::<3>::from_data([5, 5, 5], exp_data);
    assert_eq!(r, expected);
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
    let r = iterate_structure(s, 2);
    let expected = BoolStructure::<1>::from_data([5], vec![false, false, true, false, false]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5]);
}

#[test]
fn line_1d_iterations_3() {
    let s = BoolStructure::<1>::from_data([3], vec![false, true, false]);
    let r = iterate_structure(s, 3);
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
    let r = iterate_structure(s, 2);
    let expected = BoolStructure::<1>::from_data([5], vec![true, true, true, true, true]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5]);
}

#[test]
fn line_1d_all_true_iterations_4() {
    // 1D [T, T, T] structure iterations=4: length 9, all True.
    let s = BoolStructure::<1>::from_data([3], vec![true, true, true]);
    let r = iterate_structure(s, 4);
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
    let r = iterate_structure(s, 2);
    let expected = BoolStructure::<1>::from_data([5], vec![true, false, true, false, true]);
    assert_eq!(r, expected);
    assert_eq!(r.shape(), &[5]);
}

// ── 2x2 all-True ────────────────────────────────────────────────────────────

#[test]
fn two_by_two_all_true_iterations_3() {
    // 2x2 all-True structure iterations=3 → 4x4 output.
    // Shape: 2 + 2 * 1 = 4 per axis. Stamp at (2, 2).
    // After stamping: True at (2,2),(2,3),(3,2),(3,3).
    // Kernel center=(1,1), even_offset=(1,1). Offsets: (-2,-2),(-1,-2),(-2,-1),(-1,-1).
    // 1st dilation: upper-left 3x3. 2nd dilation: upper-left 2x2 (even offset
    // makes dilation asymmetric — only stamps backward).
    let s = struct_2d(&[&[1, 1], &[1, 1]]);
    let r = iterate_structure(s, 3);
    assert_eq!(r.shape(), &[4, 4]);
    let expected = struct_2d(&[&[1, 1, 0, 0], &[1, 1, 0, 0], &[0, 0, 0, 0], &[0, 0, 0, 0]]);
    assert_eq!(r, expected);
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
        let (_r, o) = iterate_structure_with_origin(s.clone(), it, [1, 1]);
        assert_eq!(o, expected_origin, "iterations = {}", it);
    }
}

#[test]
fn origin_tracking_zero_origin() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let (_r, o) = iterate_structure_with_origin(s, 3, [0, 0]);
    assert_eq!(o, [0, 0]);
}

#[test]
fn origin_tracking_iterations_below_2() {
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let (_r, o) = iterate_structure_with_origin(s, 1, [1, 1]);
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
    let (_r, o) = iterate_structure_with_origin(s, 2, [1, 1, 1]);
    assert_eq!(o, [2, 2, 2]);
}

#[test]
fn origin_tracking_2d_zero_iter2() {
    // scipy reference: iterate_structure(cross_2d, 2, origin=(0, 0))
    // → returns (the 5x5 diamond, [0, 0])
    let s = struct_2d(&[&[0, 1, 0], &[1, 1, 1], &[0, 1, 0]]);
    let (r, o) = iterate_structure_with_origin(s, 2, [0, 0]);
    assert_eq!(o, [0, 0]);
    assert_eq!(r.shape(), &[5, 5]);
}
