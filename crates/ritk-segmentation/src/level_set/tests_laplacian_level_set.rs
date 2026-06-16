//! Tests for laplacian level set segmentation.
//! Extracted to keep the 500-line structural limit.

use super::*;
use ritk_core::spatial::{Direction, Point, Spacing};

type B = burn_ndarray::NdArray<f32>;

fn make_image(dims: [usize; 3], val: f32) -> Image<B, 3> {
    let n: usize = dims.iter().product();
    let device = Default::default();
    let tensor =
        Tensor::<B, 3>::from_data(TensorData::new(vec![val; n], Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn make_image_with_metadata(
    data: Vec<f32>,
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<B, 3> {
    make_image_with(
        data, dims,
        Some(Point::new(origin)),
        Some(Spacing::new(spacing)),
        None,
    )
}),
        Some(ritk_spatial::Spacing::new(spacing)),
        None,
    )
}

fn sphere_phi(dims: [usize; 3], center: [f64; 3], radius: f64) -> Image<B, 3> {
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
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn count_foreground(image: &Image<B, 3>) -> usize {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .filter(|&&v| v == 1.0)
        .count()
}

fn count_phi_inside(phi: &Image<B, 3>) -> usize {
    image_values(phi).iter().filter(|&&v| v < 0.0).count()
}

fn image_values(image: &Image<B, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// All output voxels must be exactly 0.0 or 1.0.
#[test]
fn test_output_is_binary() {
    let dims = [8, 8, 8];
    let image = make_image(dims, 128.0);
    let phi = sphere_phi(dims, [4.0, 4.0, 4.0], 2.0);
    let result = LaplacianLevelSet::new().apply(&image, &phi).unwrap();
    let vals = image_values(&result);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v == 0.0 || v == 1.0,
            "Output voxel {} has non-binary value {}",
            i,
            v
        );
    }
}

/// Spatial metadata must be copied unchanged from image to output.
#[test]
fn test_metadata_preserved() {
    let dims = [6, 6, 6];
    let origin = [2.0, 3.0, 4.0];
    let spacing = [0.5, 0.5, 1.0];
    let image = make_image_with_metadata(dims, 128.0, origin, spacing);
    let phi = sphere_phi(dims, [3.0, 3.0, 3.0], 2.0);
    let result = LaplacianLevelSet::new().apply(&image, &phi).unwrap();
    assert_eq!(result.origin(), &Point::new(origin));
    assert_eq!(result.spacing(), &Spacing::new(spacing));
    assert_eq!(result.direction(), image.direction());
    assert_eq!(result.shape(), dims);
}

/// Mismatched image and phi shapes must return an Err.
#[test]
fn test_shape_mismatch_returns_error() {
    let image = make_image([5, 5, 5], 128.0);
    let phi = sphere_phi([3, 3, 3], [1.0, 1.0, 1.0], 1.0);
    let result = LaplacianLevelSet::new().apply(&image, &phi);
    assert!(result.is_err());
}

/// Default::default() must produce fields identical to new().
#[test]
fn test_default_matches_new() {
    let d = LaplacianLevelSet::default();
    let n = LaplacianLevelSet::new();
    assert_eq!(d.propagation_weight, n.propagation_weight);
    assert_eq!(d.curvature_weight, n.curvature_weight);
    assert_eq!(d.sigma, n.sigma);
    assert_eq!(d.dt, n.dt);
    assert_eq!(d.max_iterations, n.max_iterations);
    assert_eq!(d.tolerance, n.tolerance);
}

/// Contour must expand when the image has a bright local maximum (negative Laplacian).
///
/// Analytical invariant for a 3-D Gaussian blob I(r) = exp(-r^2/(2*sigma^2)),
/// sigma = 3.0, at the centre r = 0:
///
///   L = -3/sigma^2 = -3/9 = -0.333
///   F = L/(1+|L|) = -0.25
///   kappa = 2/R = 2/2 = 1.0  (sphere SDF, R=2)
///   speed = w_p*F + w_c*kappa = 1.0*(-0.25) + 0.1*1.0 = -0.15 < 0  => expansion
///
/// Note: a uniform image (all same value) has zero Laplacian everywhere and
/// therefore zero propagation speed; only a spatially varying bright region
/// (negative Laplacian at maxima) drives expansion in this model.
#[test]
fn test_laplacian_expands_in_bright_region() {
    let dims = [15, 15, 15];
    let center = [7.0, 7.0, 7.0];
    let [nz, ny, nx] = dims;
    let n: usize = nz * ny * nx;
    // Gaussian blob: I(r) = exp(-r^2 / (2*3^2)).
    // Discrete Laplacian at centre: L approx -3/9 = -0.333, F approx -0.25.
    let sigma_img: f64 = 3.0;
    let mut img_data = vec![0.0_f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let r2 = (iz as f64 - center[0]).powi(2)
                    + (iy as f64 - center[1]).powi(2)
                    + (ix as f64 - center[2]).powi(2);
                img_data[iz * ny * nx + iy * nx + ix] =
                    f64::exp(-r2 / (2.0 * sigma_img * sigma_img)) as f32;
            }
        }
    }
    let device = Default::default();
    let img_tensor =
        Tensor::<B, 3>::from_data(TensorData::new(img_data, Shape::new(dims)), &device);
    let image = Image::new(
        img_tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    let phi = sphere_phi(dims, center, 2.0);
    let initial_inside = count_phi_inside(&phi);
    let mut ls = LaplacianLevelSet::new();
    ls.propagation_weight = 1.0;
    ls.curvature_weight = 0.1;
    ls.sigma = GaussianSigma::new_unchecked(0.5);
    ls.max_iterations = 200;
    let result = ls.apply(&image, &phi).unwrap();
    let final_fg = count_foreground(&result);
    assert!(
        final_fg > initial_inside,
        "Laplacian level set must expand in bright region: final {} > initial {}",
        final_fg,
        initial_inside
    );
}

/// With max_iterations = 0 the PDE loop never executes.
/// Output must equal threshold of the initial phi.
#[test]
fn test_zero_iterations_returns_initial_phi_thresholded() {
    let dims = [7, 7, 7];
    let center = [3.0, 3.0, 3.0];
    let image = make_image(dims, 1.0);
    let phi = sphere_phi(dims, center, 2.0);
    let expected_inside = count_phi_inside(&phi);
    let mut ls = LaplacianLevelSet::new();
    ls.max_iterations = 0;
    let result = ls.apply(&image, &phi).unwrap();
    let actual_fg = count_foreground(&result);
    assert_eq!(
        actual_fg, expected_inside,
        "Zero iterations: fg count must equal initial phi threshold: {} == {}",
        actual_fg, expected_inside
    );
}
