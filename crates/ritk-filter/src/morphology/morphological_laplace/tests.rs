use super::*;

use ritk_core::image::Image;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn extract_vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data().to_vec()
}

/// Morphological Laplacian of a constant field is zero everywhere.
#[test]
fn constant_field_is_zero() {
    let vals = vec![5.0_f32; 27];
    let img = make_image(vals.clone(), [3, 3, 3]);
    let lap = MorphologicalLaplacian::new(1);
    let out = lap.apply(&img).expect("infallible: validated precondition");
    let got = extract_vals(&out);
    assert_eq!(got, vec![0.0_f32; 27]);
}

/// All-ones field: D=1, E=1, output = 1+1-2*1 = 0.
#[test]
fn all_ones_is_zero() {
    let vals = vec![1.0_f32; 27];
    let img = make_image(vals, [3, 3, 3]);
    let lap = MorphologicalLaplacian::new(1);
    let out = lap.apply(&img).expect("infallible: validated precondition");
    let got = extract_vals(&out);
    for &v in &got {
        assert_eq!(v, 0.0);
    }
}

/// Linear ramp 3x3x3 along x: with reflect padding the result is
/// [1, 0, -1] in each x-slice (verified against scipy v1.17.1).
#[test]
fn linear_ramp_3x3x3() {
    let mut vals = Vec::new();
    for _z in 0..3 {
        for _y in 0..3 {
            for x in 0..3 {
                vals.push(x as f32);
            }
        }
    }
    let img = make_image(vals, [3, 3, 3]);
    let lap = MorphologicalLaplacian::new(1);
    let out = lap.apply(&img).expect("infallible: validated precondition");
    let got = extract_vals(&out);
    let expected_left = 1.0_f32;
    let expected_mid = 0.0_f32;
    let expected_right = -1.0_f32;
    for chunk in got.chunks(3) {
        assert!((chunk[0] - expected_left).abs() < 1e-5, "got {:?}", chunk);
        assert!((chunk[1] - expected_mid).abs() < 1e-5, "got {:?}", chunk);
        assert!((chunk[2] - expected_right).abs() < 1e-5, "got {:?}", chunk);
    }
}

/// Single voxel in 5x5x5: 3x3x3 cube around voxel has -1 at center, +1 at neighbours, 0 elsewhere.
/// The structuring element is a cube (L∞ distance), so the cube includes all
/// voxels (z, y, x) with `|z-1| ≤ 1 && |y-1| ≤ 1 && |x-1| ≤ 1`.
#[test]
fn single_voxel_5x5x5_size_3() {
    let mut vals = vec![0.0_f32; 125];
    vals[31] = 1.0; // voxel (1,1,1): 1*25 + 1*5 + 1 = 31
    let img = make_image(vals, [5, 5, 5]);
    let lap = MorphologicalLaplacian::new(1);
    let out = lap.apply(&img).expect("infallible: validated precondition");
    let got = extract_vals(&out);

    for iz in 0..5 {
        for iy in 0..5 {
            for ix in 0..5 {
                let v = got[iz * 25 + iy * 5 + ix];
                let in_cube = (iz as i32 - 1).abs() <= 1
                    && (iy as i32 - 1).abs() <= 1
                    && (ix as i32 - 1).abs() <= 1;
                let expected = if (iz, iy, ix) == (1, 1, 1) {
                    -1.0
                } else if in_cube {
                    1.0
                } else {
                    0.0
                };
                assert!(
                    (v - expected).abs() < 1e-5,
                    "voxel ({iz},{iy},{ix}): got {v}, expected {expected}"
                );
            }
        }
    }
}

/// Single voxel in 5x5x5 with radius=2 (size=5): 5x5x5 cube around voxel has -1 at center, +1 at neighbours, 0 elsewhere.
#[test]
fn single_voxel_5x5x5_size_5() {
    let mut vals = vec![0.0_f32; 125];
    vals[2 * 25 + 2 * 5 + 2] = 1.0; // voxel (2,2,2)
    let img = make_image(vals, [5, 5, 5]);
    let lap = MorphologicalLaplacian::new(2);
    let out = lap.apply(&img).expect("infallible: validated precondition");
    let got = extract_vals(&out);

    for iz in 0..5 {
        for iy in 0..5 {
            for ix in 0..5 {
                let v = got[iz * 25 + iy * 5 + ix];
                let expected = if (iz, iy, ix) == (2, 2, 2) {
                    -1.0
                } else if (iz as i32 - 2).abs() <= 2
                    && (iy as i32 - 2).abs() <= 2
                    && (ix as i32 - 2).abs() <= 2
                    && (iz, iy, ix) != (2, 2, 2)
                {
                    1.0
                } else {
                    0.0
                };
                assert!(
                    (v - expected).abs() < 1e-5,
                    "voxel ({iz},{iy},{ix}): got {v}, expected {expected}"
                );
            }
        }
    }
}

/// Single voxel in 3x3x3: -1 at voxel, +1 at all 26 cube-neighbours.
/// This test verifies the cubic (size=3) case, equivalent to the 5x5x5 size=3
/// case but with no boundary issues.
#[test]
fn single_voxel_3x3x3() {
    let mut vals = vec![0.0_f32; 27];
    vals[13] = 1.0; // voxel (1,1,1): 1*9 + 1*3 + 1
    let img = make_image(vals, [3, 3, 3]);
    let lap = MorphologicalLaplacian::new(1);
    let out = lap.apply(&img).expect("infallible: validated precondition");
    let got = extract_vals(&out);
    let mut count_neg1 = 0;
    let mut count_pos1 = 0;
    let mut count_zero = 0;
    for &v in &got {
        if (v - (-1.0)).abs() < 1e-5 {
            count_neg1 += 1;
        } else if (v - 1.0).abs() < 1e-5 {
            count_pos1 += 1;
        } else if v.abs() < 1e-5 {
            count_zero += 1;
        } else {
            panic!("unexpected value {v}");
        }
    }
    assert_eq!(count_neg1, 1, "expected 1 voxel at -1, got {count_neg1}");
    assert_eq!(count_pos1, 26, "expected 26 voxels at +1, got {count_pos1}");
    assert_eq!(count_zero, 0, "expected 0 voxels at 0, got {count_zero}");
}

/// Degenerate axis: dim of length 1 still works (no panic). Verified against
/// scipy.ndimage.grey_dilation / grey_erosion on the same 1x3x3 array with
/// size=(1,3,3), mode='reflect' which gives D = 5 6 6 / 8 9 9 / 8 9 9 and
/// E = 1 1 2 / 1 1 2 / 4 4 5, hence L = D + E − 2f.
#[test]
fn degenerate_axis_size_1() {
    let vals = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let img = make_image(vals, [1, 3, 3]);
    let lap = MorphologicalLaplacian::new(1);
    let out = lap.apply(&img).expect("infallible: validated precondition");
    let got = extract_vals(&out);
    let expected = [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0];
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "voxel {i}: got {g}, expected {e}");
    }
}

/// Sanity check: applying the morphological Laplacian to a non-constant image
/// changes it (i.e., the implementation is not a no-op).
#[test]
fn operator_is_not_identity() {
    let mut vals = Vec::new();
    for i in 0..27 {
        vals.push(i as f32);
    }
    let img = make_image(vals.clone(), [3, 3, 3]);
    let lap = MorphologicalLaplacian::new(1);
    let out = lap.apply(&img).expect("infallible: validated precondition");
    let got = extract_vals(&out);
    assert_ne!(got, vals, "morphological_laplace should change the input");
}

/// Differential against scipy v1.17.1: 4x4x4 with two corner voxels.
/// scipy.ndimage.morphological_laplace of:
///   arr[0,0,0] = arr[3,3,3] = 1.0
/// gives:
///   z=0: -1  1  0  0     z=1:  1  1  0  0     z=2:  0  0  0  0     z=3:  0  0  0  0
///          1  1  0  0            1  1  0  0            0  0  0  0            0  0  0  0
///          0  0  0  0            0  0  0  0            0  0  1  1            0  0  1  1
///          0  0  0  0            0  0  0  0            0  0  1  1            0  0  1 -1
#[test]
fn differential_two_corner_voxels_4x4x4() {
    let mut vals = vec![0.0_f32; 64];
    vals[0] = 1.0; // (0,0,0): index 0
    vals[63] = 1.0; // (3,3,3): index 3*16 + 3*4 + 3 = 63
    let img = make_image(vals, [4, 4, 4]);
    let lap = MorphologicalLaplacian::new(1);
    let out = lap.apply(&img).expect("infallible: validated precondition");
    let got = extract_vals(&out);

    let z0 = [
        -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    for (i, (&g, &e)) in got[0..16].iter().zip(z0.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "z=0 voxel {i}: got {g}, expected {e}");
    }
    let z1 = [
        1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    for (i, (&g, &e)) in got[16..32].iter().zip(z1.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "z=1 voxel {i}: got {g}, expected {e}");
    }
    let z2 = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    ];
    for (i, (&g, &e)) in got[32..48].iter().zip(z2.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "z=2 voxel {i}: got {g}, expected {e}");
    }
    let z3 = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0,
    ];
    for (i, (&g, &e)) in got[48..64].iter().zip(z3.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "z=3 voxel {i}: got {g}, expected {e}");
    }
}
