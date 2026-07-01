//! Differential coverage: `binary_opening_coeus` must be value-identical to
//! the Burn-generic `BinaryMorphologicalOpening::apply` it mirrors — both
//! compose the identical `dilate ∘ erode` cores (shared harness in
//! `coeus_support`).

use super::binary_opening_coeus;
use crate::coeus_support::assert_coeus_matches_burn;
use crate::morphology::BinaryMorphologicalOpening;

fn check(vals: Vec<f32>, dims: [usize; 3], radius: usize) {
    assert_coeus_matches_burn(
        vals,
        dims,
        |img| {
            BinaryMorphologicalOpening::new(radius)
                .apply(img)
                .expect("burn opening")
        },
        |img, backend| binary_opening_coeus(img, radius, Default::default(), backend),
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
