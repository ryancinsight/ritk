//! Tests for grayscale_gradient
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_image::tensor::Tensor;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = coeus_core::SequentialBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, shape)
}

fn vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data_slice()
        .expect("invariant: contiguous host storage")
        .to_vec()
}

/// Constant image â†’ gradient = 0 everywhere.
///
/// # Proof
/// D_B(c)(x) = max_{b âˆˆ B} c = c.
/// E_B(c)(x) = min_{b âˆˆ B} c = c.
/// grad(x) = c âˆ’ c = 0.
#[test]
fn constant_image_zero_gradient() {
    let img = make_image(vec![5.0; 27], [3, 3, 3]);
    let out = GrayscaleMorphologicalGradientFilter::new(1)
        .apply(&img)
        .unwrap();
    for &v in vals(&out).iter() {
        assert_eq!(
            v, 0.0_f32,
            "constant image must yield zero gradient; got {v}"
        );
    }
}

/// radius = 0 â†’ gradient = 0 everywhere (degenerate SE = {0}).
///
/// # Proof
/// SE = {0}; D_B(f)(x) = f(x); E_B(f)(x) = f(x); grad = 0.
#[test]
fn radius_zero_always_zero() {
    let img = make_image(vec![0.0, 5.0, 10.0, 3.0, 8.0, 1.0], [1, 2, 3]);
    let out = GrayscaleMorphologicalGradientFilter::new(0)
        .apply(&img)
        .unwrap();
    for &v in vals(&out).iter() {
        assert_eq!(v, 0.0_f32, "radius=0 must yield zero gradient; got {v}");
    }
}

/// Output is non-negative everywhere.
///
/// # Proof
/// D_B(f)(x) â‰¥ f(x) â‰¥ E_B(f)(x) (extensivity / anti-extensivity of flat SE).
/// Therefore grad(x) = D_B(f)(x) âˆ’ E_B(f)(x) â‰¥ 0.
#[test]
fn output_nonnegative_everywhere() {
    // Arbitrary non-constant volume.
    let data: Vec<f32> = (0..27).map(|i| (i as f32) * 3.7 - 20.0).collect();
    let img = make_image(data, [3, 3, 3]);
    let out = GrayscaleMorphologicalGradientFilter::new(1)
        .apply(&img)
        .unwrap();
    for &v in vals(&out).iter() {
        assert!(v >= 0.0, "gradient must be non-negative; got {v}");
    }
}

/// Step-edge volume: gradient = 10 at boundary, 0 away from it.
///
/// # Analytical derivation
/// Volume 1Ã—3Ã—7 with values [0,0,0, 10,10,10,10] (step at column 3).
/// radius = 1, SE = [-1, 0, +1] along each axis; only the X axis matters here.
///
/// For boundary voxel at x=3 (value 10): neighbourhood = {x=2, x=3, x=4};
///   dilation = max(0, 10, 10) = 10; erosion = min(0, 10, 10) = 0; grad = 10.
/// For boundary voxel at x=2 (value 0): neighbourhood = {x=1, x=2, x=3};
///   dilation = max(0, 0, 10) = 10; erosion = min(0, 0, 10) = 0; grad = 10.
/// For interior voxel at x=0 (value 0): neighbourhood = {x=0, x=0, x=1} (clamped);
///   dilation = 0; erosion = 0; grad = 0.
/// For interior voxel at x=6 (value 10): neighbourhood = {x=5, x=6, x=6} (clamped);
///   dilation = 10; erosion = 10; grad = 0.
#[test]
fn step_edge_gradient_at_boundary() {
    let data = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
    let img = make_image(data, [1, 1, 7]);
    let out = GrayscaleMorphologicalGradientFilter::new(1)
        .apply(&img)
        .unwrap();
    let v = vals(&out);
    // Interior voxels far from the edge: gradient = 0
    assert_eq!(v[0], 0.0_f32, "voxel at x=0 (far left): gradient must be 0");
    assert_eq!(
        v[6], 0.0_f32,
        "voxel at x=6 (far right): gradient must be 0"
    );
    // Boundary voxels at x=2 and x=3
    assert_eq!(
        v[2], 10.0_f32,
        "boundary voxel x=2: dilation=10, erosion=0 â†’ gradient=10"
    );
    assert_eq!(
        v[3], 10.0_f32,
        "boundary voxel x=3: dilation=10, erosion=0 â†’ gradient=10"
    );
}

/// Spatial metadata is preserved.
#[test]
fn spatial_metadata_preserved() {
    let sp = Spacing::new([2.5, 3.5, 4.5]);
    let t = Tensor::<f32, B>::from_slice([1usize, 1, 3], &[1.0_f32, 2.0, 3.0]);
    let img = Image::new(t, Point::new([0.0, 0.0, 0.0]), sp, Direction::identity())
        .expect("invariant: fixture tensor has the declared rank");
    let out = GrayscaleMorphologicalGradientFilter::new(1)
        .apply(&img)
        .unwrap();
    assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
}

/// Single bright voxel in a dark volume: gradient is non-zero only in
/// the neighbourhood of the bright voxel.
///
/// # Derivation
/// Volume 1Ã—5Ã—5, all voxels = 0 except center (2,2) = 100.
/// With radius = 1:
///   - The bright voxel and its 8 2D-neighbours (plus 1 above/below in z=0) have
///     dilation = 100 (center in neighbourhood) and erosion = 0 (0 in neighbourhood).
///     gradient = 100.
///   - Corner/edge voxels â‰¥ 2 away from the center: both dilation and erosion
///     are 0 (the bright voxel is not in the 3Ã—3Ã—3 neighbourhood); gradient = 0.
#[test]
fn single_bright_voxel_gradient_ring() {
    let mut data = vec![0.0_f32; 25]; // 1Ã—5Ã—5
    data[12] = 100.0; // center voxel at (0, 2, 2)
    let img = make_image(data, [1, 5, 5]);
    let out = GrayscaleMorphologicalGradientFilter::new(1)
        .apply(&img)
        .unwrap();
    let v = vals(&out);
    // Corners of the 5Ã—5 image (flat z=0): indices 0, 4, 20, 24 are â‰¥ 2 away
    // from center (2,2) in the x dimension â€” no contact with the bright voxel.
    assert_eq!(
        v[0], 0.0_f32,
        "corner (0,0): far from bright voxel, gradient must be 0"
    );
    assert_eq!(
        v[4], 0.0_f32,
        "corner (0,4): far from bright voxel, gradient must be 0"
    );
    assert_eq!(
        v[20], 0.0_f32,
        "corner (4,0): far from bright voxel, gradient must be 0"
    );
    assert_eq!(
        v[24], 0.0_f32,
        "corner (4,4): far from bright voxel, gradient must be 0"
    );
    // Center voxel itself: in its own neighbourhood â†’ dilation=100, erosion=0; grad=100.
    assert_eq!(
        v[12], 100.0_f32,
        "center bright voxel: gradient must be 100"
    );
}
