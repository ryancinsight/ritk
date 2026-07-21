//! Tests for threshold_level_set.
//! Extracted to keep the 500-line structural limit.

use super::*;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::test_support::make_image_with;

type B = coeus_core::SequentialBackend;

fn make_image(dims: [usize; 3], val: f32) -> Image<f32, B, 3> {
    let n: usize = dims.iter().product();
    let tensor = Tensor::<f32, B>::from_slice(dims, &vec![val; n]);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

fn make_image_with_metadata(
    data: Vec<f32>,
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<f32, B, 3> {
    make_image_with(
        data,
        dims,
        Some(Point::new(origin)),
        Some(Spacing::new(spacing)),
        None,
    )
}

fn sphere_phi(dims: [usize; 3], center: [f64; 3], radius: f64) -> Image<f32, B, 3> {
    let n: usize = dims.iter().product();
    let [nz, ny, nx] = dims;
    let mut data = vec![0.0_f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dist = ((iz as f64 - center[0]).powi(2)
                    + (iy as f64 - center[1]).powi(2)
                    + (ix as f64 - center[2]).powi(2))
                .sqrt();
                data[iz * ny * nx + iy * nx + ix] = (dist - radius) as f32;
            }
        }
    }
    let device = Default::default();
    let tensor = Tensor::<f32, B>::from_slice_on(dims, &data, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

fn make_bimodal_image(
    dims: [usize; 3],
    center: [f64; 3],
    radius: f64,
    inside_val: f32,
    outside_val: f32,
) -> Image<f32, B, 3> {
    let n: usize = dims.iter().product();
    let [nz, ny, nx] = dims;
    let mut data = vec![outside_val; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dist = ((iz as f64 - center[0]).powi(2)
                    + (iy as f64 - center[1]).powi(2)
                    + (ix as f64 - center[2]).powi(2))
                .sqrt();
                if dist <= radius {
                    data[iz * ny * nx + iy * nx + ix] = inside_val;
                }
            }
        }
    }
    let device = Default::default();
    let tensor = Tensor::<f32, B>::from_slice_on(dims, &data, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

fn get_values(image: &Image<f32, B, 3>) -> Vec<f32> {
    image.data().to_vec()
}

fn count_foreground(image: &Image<f32, B, 3>) -> usize {
    get_values(image).iter().filter(|&&v| v == 1.0).count()
}

fn count_phi_inside(phi: &Image<f32, B, 3>) -> usize {
    get_values(phi).iter().filter(|&&v| v < 0.0).count()
}

#[test]
fn test_threshold_expands_within_range() {
    let dims = [15, 15, 15];
    let center = [7.0, 7.0, 7.0];
    let image = make_bimodal_image(dims, center, 4.0, 150.0, 0.0);
    let phi = sphere_phi(dims, center, 3.0);
    let initial_inside = count_phi_inside(&phi);

    let mut ls = ThresholdLevelSet::new(100.0, 200.0);
    ls.propagation_weight = 1.0;
    ls.curvature_weight = 0.1;
    ls.max_iterations = 300;

    let result = ls.apply(&image, &phi).expect("infallible: validated precondition");
    let final_fg = count_foreground(&result);

    assert!(
        final_fg > initial_inside,
        "Contour must expand within threshold range: final {} > initial {}",
        final_fg,
        initial_inside
    );
}

#[test]
fn test_threshold_contracts_outside_range() {
    let dims = [11, 11, 11];
    let center = [5.0, 5.0, 5.0];
    // Both intensities (50.0 and 300.0) are OUTSIDE threshold range [100, 250].
    let image = make_bimodal_image(dims, center, 3.0, 50.0, 300.0);
    let phi = sphere_phi(dims, center, 3.0);
    let initial_inside = count_phi_inside(&phi);

    let ls = ThresholdLevelSet::new(100.0, 250.0);
    let result = ls.apply(&image, &phi).expect("infallible: validated precondition");
    let final_fg = count_foreground(&result);

    assert!(
        final_fg < initial_inside,
        "Contour must contract outside threshold range: final {} < initial {}",
        final_fg,
        initial_inside
    );
}

#[test]
fn test_output_is_binary() {
    let dims = [8, 8, 8];
    let image = make_image(dims, 128.0);
    let phi = sphere_phi(dims, [4.0, 4.0, 4.0], 2.0);

    let result = ThresholdLevelSet::new(100.0, 200.0)
        .apply(&image, &phi)
        .expect("infallible: validated precondition");
    let vals = get_values(&result);

    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v == 0.0 || v == 1.0,
            "Output voxel {} has non-binary value {}",
            i,
            v
        );
    }
}

#[test]
fn test_metadata_preserved() {
    let dims = [6, 6, 6];
    let origin = [2.0, 3.0, 4.0];
    let spacing = [0.5, 0.5, 1.0];
    let image = make_image_with_metadata(vec![128.0; dims.iter().product()], dims, origin, spacing);
    let phi = sphere_phi(dims, [3.0, 3.0, 3.0], 2.0);

    let result = ThresholdLevelSet::new(100.0, 200.0)
        .apply(&image, &phi)
        .expect("infallible: validated precondition");

    assert_eq!(
        result.origin(),
        &Point::new(origin),
        "Origin must be preserved"
    );
    assert_eq!(
        result.spacing(),
        &Spacing::new(spacing),
        "Spacing must be preserved"
    );
    assert_eq!(
        result.direction(),
        image.direction(),
        "Direction must be preserved"
    );
    assert_eq!(result.shape(), dims, "Shape must be preserved");
}

#[test]
fn test_shape_mismatch_returns_error() {
    let image = make_image([5, 5, 5], 128.0);
    let phi = sphere_phi([3, 3, 3], [1.0, 1.0, 1.0], 1.0);
    let result = ThresholdLevelSet::new(100.0, 200.0).apply(&image, &phi);
    assert!(result.is_err(), "Shape mismatch must return Err");
}

#[test]
fn test_default_matches_new() {
    let d = ThresholdLevelSet::default();
    let n = ThresholdLevelSet::new(0.0, 255.0);

    assert_eq!(d.lower_threshold, n.lower_threshold);
    assert_eq!(d.upper_threshold, n.upper_threshold);
    assert_eq!(d.propagation_weight, n.propagation_weight);
    assert_eq!(d.curvature_weight, n.curvature_weight);
    assert_eq!(d.dt, n.dt);
    assert_eq!(d.max_iterations, n.max_iterations);
    assert_eq!(d.tolerance, n.tolerance);
}

#[test]
fn test_uniform_in_range_expands() {
    let dims = [11, 11, 11];
    let center = [5.0, 5.0, 5.0];
    let image = make_image(dims, 128.0);
    let phi = sphere_phi(dims, center, 3.0);
    let initial_inside = count_phi_inside(&phi);

    let mut ls = ThresholdLevelSet::new(100.0, 200.0);
    ls.propagation_weight = 1.0;
    ls.curvature_weight = 0.1;
    ls.max_iterations = 200;

    let result = ls.apply(&image, &phi).expect("infallible: validated precondition");
    let final_fg = count_foreground(&result);

    assert!(
        final_fg > initial_inside,
        "Uniform in-range image must cause expansion: final {} > initial {}",
        final_fg,
        initial_inside
    );
}
