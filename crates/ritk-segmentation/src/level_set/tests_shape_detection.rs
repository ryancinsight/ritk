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
    dims: [usize; 3],
    val: f32,
    origin: [f64; 3],
    spacing: [f64; 3],
) -> Image<B, 3> {
    let n: usize = dims.iter().product();
    let device = Default::default();
    let tensor =
        Tensor::<B, 3>::from_data(TensorData::new(vec![val; n], Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new(origin),
        Spacing::new(spacing),
        Direction::identity(),
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

#[test]
fn test_shape_detection_expands_in_homogeneous_region() {
    let dims = [15, 15, 15];
    let center = [7.0, 7.0, 7.0];
    let image = make_image(dims, 40.0);
    let phi = sphere_phi(dims, center, 3.0);
    let initial_inside = count_phi_inside(&phi);

    let mut ls = ShapeDetectionSegmentation::new();
    ls.propagation_weight = 1.0;
    ls.curvature_weight = 0.1;
    ls.advection_weight = 0.0;
    ls.edge_k = 1.0;
    ls.max_iterations = 200;

    let result = ls.apply(&image, &phi).unwrap();
    let final_fg = count_foreground(&result);

    assert!(
        final_fg > initial_inside,
        "Shape detection must expand in homogeneous regions: final {} > initial {}",
        final_fg,
        initial_inside
    );
}

#[test]
fn test_shape_detection_stops_at_edges() {
    let dims = [17, 17, 17];
    let center = [8.0, 8.0, 8.0];
    let image = make_image(dims, 0.0);
    let mut img_data = image_values(&image);
    let [nz, ny, nx] = dims;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dist = ((iz as f64 - center[0]).powi(2)
                    + (iy as f64 - center[1]).powi(2)
                    + (ix as f64 - center[2]).powi(2))
                .sqrt();
                if (dist - 5.0).abs() < 0.5 {
                    img_data[iz * ny * nx + iy * nx + ix] = 255.0;
                }
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

    let mut ls = ShapeDetectionSegmentation::new();
    ls.propagation_weight = 1.0;
    ls.curvature_weight = 0.2;
    ls.advection_weight = 0.5;
    ls.edge_k = 5.0;
    ls.sigma = GaussianSigma::new_unchecked(1.0);
    ls.max_iterations = 200;

    let result = ls.apply(&image, &phi).unwrap();
    let final_fg = count_foreground(&result);

    assert!(
        final_fg >= initial_inside,
        "Edge stopping must prevent collapse past the initial contour"
    );
}

#[test]
fn test_output_is_binary() {
    let dims = [8, 8, 8];
    let image = make_image(dims, 128.0);
    let phi = sphere_phi(dims, [4.0, 4.0, 4.0], 2.0);

    let result = ShapeDetectionSegmentation::new()
        .apply(&image, &phi)
        .unwrap();
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

#[test]
fn test_metadata_preserved() {
    let dims = [6, 6, 6];
    let origin = [2.0, 3.0, 4.0];
    let spacing = [0.5, 0.5, 1.0];
    let image = make_image_with_metadata(dims, 128.0, origin, spacing);
    let phi = sphere_phi(dims, [3.0, 3.0, 3.0], 2.0);

    let result = ShapeDetectionSegmentation::new()
        .apply(&image, &phi)
        .unwrap();

    assert_eq!(result.origin(), &Point::new(origin));
    assert_eq!(result.spacing(), &Spacing::new(spacing));
    assert_eq!(result.direction(), image.direction());
    assert_eq!(result.shape(), dims);
}

#[test]
fn test_shape_mismatch_returns_error() {
    let image = make_image([5, 5, 5], 128.0);
    let phi = sphere_phi([3, 3, 3], [1.0, 1.0, 1.0], 1.0);
    let result = ShapeDetectionSegmentation::new().apply(&image, &phi);
    assert!(result.is_err());
}

#[test]
fn test_default_matches_new() {
    let d = ShapeDetectionSegmentation::default();
    let n = ShapeDetectionSegmentation::new();

    assert_eq!(d.curvature_weight, n.curvature_weight);
    assert_eq!(d.propagation_weight, n.propagation_weight);
    assert_eq!(d.advection_weight, n.advection_weight);
    assert_eq!(d.edge_k, n.edge_k);
    assert_eq!(d.sigma, n.sigma);
    assert_eq!(d.dt, n.dt);
    assert_eq!(d.max_iterations, n.max_iterations);
    assert_eq!(d.tolerance, n.tolerance);
}
