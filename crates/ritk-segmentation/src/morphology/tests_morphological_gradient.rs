#![allow(clippy::needless_range_loop)]

use super::*;
use ritk_image::test_support::make_image;
use ritk_image::Image;
type Backend = burn_ndarray::NdArray<f32>;

fn make_mask(vals: Vec<f32>, shape: [usize; 3]) -> Image<Backend, 3> {
    make_image(vals, shape)
}

#[test]
fn test_gradient_all_zero_gives_all_zero() {
    let mask = make_mask(vec![0.0f32; 27], [3, 3, 3]);
    let result = MorphologicalGradient::new(1).apply(&mask);
    result.with_data_slice(|vals| {
        assert!(
            vals.iter().all(|&v| v < 0.5),
            "all-zero input must produce all-zero gradient"
        );
    });
}

/// BinaryErosion treats out-of-bounds neighbors as background (0.0).
/// Therefore, border voxels of an all-1 mask are eroded to 0, and the gradient
/// is 1 there (dilation=1, erosion=0).  Only interior voxels -- those whose
/// full r=1 neighborhood is within image bounds -- are invariant to this
/// boundary effect and must have gradient 0.
#[test]
fn test_gradient_all_one_interior_voxels_zero() {
    // 5x5x5 all-one mask; interior indices 1..=3 in each axis have a complete r=1 cube.
    let mask = make_mask(vec![1.0f32; 125], [5, 5, 5]);
    let result = MorphologicalGradient::new(1).apply(&mask);
    result.with_data_slice(|vals| {
        for iz in 1..=3usize {
            for iy in 1..=3usize {
                for ix in 1..=3usize {
                    let i = iz * 25 + iy * 5 + ix;
                    assert_eq!(
                        vals[i], 0.0,
                        "interior voxel ({},{},{}) must have zero gradient for all-one mask",
                        iz, iy, ix
                    );
                }
            }
        }
    });
}

#[test]
fn test_gradient_solid_ball_only_boundary_voxels_nonzero() {
    let shape = [7usize, 7, 7];
    let mut vals = vec![0.0f32; 343];
    for iz in 0..7usize {
        for iy in 0..7usize {
            for ix in 0..7usize {
                let d2 = ((iz as i32 - 3).pow(2) + (iy as i32 - 3).pow(2) + (ix as i32 - 3).pow(2))
                    as f32;
                if d2 <= 4.0 {
                    vals[iz * 49 + iy * 7 + ix] = 1.0;
                }
            }
        }
    }
    let mask = make_mask(vals.clone(), shape);
    let result = MorphologicalGradient::new(1).apply(&mask);
    result.with_data_slice(|out_vals| {
        let center_idx = 3 * 49 + 3 * 7 + 3;
        assert_eq!(
            out_vals[center_idx], 0.0,
            "center voxel must not be a boundary"
        );
        let boundary_count = out_vals.iter().filter(|&&v| v >= 0.5).count();
        assert!(
            boundary_count > 0,
            "gradient must detect at least one boundary voxel"
        );
    });
}

#[test]
fn test_gradient_output_shape_preserved() {
    let mask = make_mask(vec![0.0f32; 60], [3, 4, 5]);
    let result = MorphologicalGradient::new(1).apply(&mask);
    assert_eq!(result.shape(), [3, 4, 5]);
}

#[test]
fn test_gradient_values_binary() {
    let mut vals = vec![0.0f32; 125];
    for i in 50..75 {
        vals[i] = 1.0;
    }
    let mask = make_mask(vals, [5, 5, 5]);
    let result = MorphologicalGradient::new(1).apply(&mask);
    result.with_data_slice(|out_vals| {
        assert!(
            out_vals.iter().all(|&v| v == 0.0 || v == 1.0),
            "gradient output must be binary"
        );
    });
}
