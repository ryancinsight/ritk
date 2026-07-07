//! Differential coverage: each native binary-morphology wrapper must be
//! value-identical to the Burn filter it mirrors — both call the identical
//! substrate-agnostic core (shared harness in `native_support`).

use super::{binary_closing, binary_dilate, binary_erode, binary_opening};
use crate::morphology::{
    BinaryDilateFilter, BinaryErodeFilter, BinaryMorphologicalClosing, BinaryMorphologicalOpening,
};
use crate::native_support::assert_native_matches_burn;

mod erode {
    use super::*;

    fn check(vals: Vec<f32>, dims: [usize; 3], radius: usize) {
        assert_native_matches_burn(
            vals,
            dims,
            |img| {
                BinaryErodeFilter::new(radius)
                    .apply(img)
                    .expect("burn erode")
            },
            |img, backend| binary_erode(img, radius, Default::default(), backend),
        );
    }

    #[test]
    fn matches_burn_radius_zero_identity() {
        check(vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], [2, 2, 2], 0);
    }

    #[test]
    fn matches_burn_all_foreground_radius_one() {
        check(vec![1.0f32; 27], [3, 3, 3], 1);
    }

    #[test]
    fn matches_burn_scattered_foreground_radius_one() {
        let dims = [6usize, 5, 4];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        check(vals, dims, 1);
    }

    #[test]
    fn matches_burn_all_background() {
        check(vec![0.0f32; 8], [2, 2, 2], 1);
    }
}

mod dilate {
    use super::*;

    fn check(vals: Vec<f32>, dims: [usize; 3], radius: usize) {
        assert_native_matches_burn(
            vals,
            dims,
            |img| {
                BinaryDilateFilter::new(radius)
                    .apply(img)
                    .expect("burn dilate")
            },
            |img, backend| binary_dilate(img, radius, Default::default(), backend),
        );
    }

    #[test]
    fn matches_burn_radius_zero_identity() {
        check(vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], [2, 2, 2], 0);
    }

    #[test]
    fn matches_burn_single_seed_radius_one() {
        let dims = [3usize, 3, 3];
        let mut vals = vec![0.0f32; 27];
        vals[13] = 1.0; // centre
        check(vals, dims, 1);
    }

    #[test]
    fn matches_burn_scattered_foreground_radius_one() {
        let dims = [6usize, 5, 4];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
        check(vals, dims, 1);
    }

    #[test]
    fn matches_burn_all_background() {
        check(vec![0.0f32; 8], [2, 2, 2], 1);
    }
}

mod closing {
    use super::*;

    fn check(vals: Vec<f32>, dims: [usize; 3], radius: usize) {
        assert_native_matches_burn(
            vals,
            dims,
            |img| {
                BinaryMorphologicalClosing::new(radius)
                    .apply(img)
                    .expect("burn closing")
            },
            |img, backend| binary_closing(img, radius, Default::default(), backend),
        );
    }

    #[test]
    fn matches_burn_fills_small_gap_radius_one() {
        // Two foreground blocks separated by a one-voxel gap along x; closing at
        // r=1 bridges it. 1x1x5 line: fg at x=0,1 and x=3,4, gap at x=2.
        check(vec![1.0, 1.0, 0.0, 1.0, 1.0], [1, 1, 5], 1);
    }

    #[test]
    fn matches_burn_all_foreground_radius_one() {
        check(vec![1.0f32; 27], [3, 3, 3], 1);
    }

    #[test]
    fn matches_burn_scattered_foreground_radius_one() {
        let dims = [6usize, 5, 4];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| if i % 4 == 0 { 1.0 } else { 0.0 }).collect();
        check(vals, dims, 1);
    }

    #[test]
    fn matches_burn_all_background() {
        check(vec![0.0f32; 8], [2, 2, 2], 1);
    }
}

mod opening {
    use super::*;

    fn check(vals: Vec<f32>, dims: [usize; 3], radius: usize) {
        assert_native_matches_burn(
            vals,
            dims,
            |img| {
                BinaryMorphologicalOpening::new(radius)
                    .apply(img)
                    .expect("burn opening")
            },
            |img, backend| binary_opening(img, radius, Default::default(), backend),
        );
    }

    #[test]
    fn matches_burn_removes_thin_protrusion_radius_one() {
        // A 3x3x3 solid block plus one isolated foreground voxel that opening at
        // r=1 removes. Use a 3x3x3 all-fg cube (opening keeps the interior that
        // survives erosion then dilation).
        check(vec![1.0f32; 27], [3, 3, 3], 1);
    }

    #[test]
    fn matches_burn_isolated_voxel_removed_radius_one() {
        let dims = [3usize, 3, 3];
        let mut vals = vec![0.0f32; 27];
        vals[13] = 1.0; // lone centre voxel — erosion kills it, opening yields all-bg
        check(vals, dims, 1);
    }

    #[test]
    fn matches_burn_scattered_foreground_radius_one() {
        let dims = [6usize, 5, 4];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        check(vals, dims, 1);
    }

    #[test]
    fn matches_burn_all_background() {
        check(vec![0.0f32; 8], [2, 2, 2], 1);
    }
}
