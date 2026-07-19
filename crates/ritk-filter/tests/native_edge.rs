//! Native-image contracts for Canny and differential edge filters.

use coeus_core::SequentialBackend;
use ritk_filter::{
    CannyEdgeDetector, GaussianSigma, GradientImageFilter, GradientMagnitudeFilter,
    GradientRecursiveGaussianImageFilter, LaplacianFilter, LaplacianOfGaussianFilter, SobelFilter,
};
use ritk_image::{ColorVolume, Image};
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

fn image(values: Vec<f32>, shape: [usize; 3], spacing: [f64; 3]) -> Image<f32, B, 3> {
    Image::from_flat_on(
        values,
        shape,
        Point::origin(),
        Spacing::new(spacing),
        Direction::identity(),
        &B::default(),
    )
    .expect("invariant: valid native test image")
}

fn values(image: &Image<f32, B, 3>) -> &[f32] {
    image
        .data_slice()
        .expect("invariant: contiguous native test image")
}

fn components(volume: &ColorVolume<f32, B, 3>) -> &[f32] {
    volume.data().as_slice()
}

#[test]
fn canny_constant_field_has_no_interior_edges() {
    let shape = [24, 24, 24];
    let input = image(vec![100.0; shape.iter().product()], shape, [1.0, 1.0, 1.0]);
    let output = CannyEdgeDetector::new(GaussianSigma::new_unchecked(1.0), 0.1, 0.2)
        .apply_native(&input, &B::default())
        .expect("native Canny succeeds");

    let margin = 6;
    let [depth, rows, columns] = shape;
    let edge_count = (margin..depth - margin)
        .flat_map(|z| {
            (margin..rows - margin)
                .flat_map(move |y| (margin..columns - margin).map(move |x| (z, y, x)))
        })
        .filter(|&(z, y, x)| values(&output)[z * rows * columns + y * columns + x] > 0.5)
        .count();
    assert_eq!(
        edge_count, 0,
        "constant field must have no interior Canny edges"
    );
}

#[test]
fn canny_step_edges_concentrate_at_the_step_plane() {
    let [depth, rows, columns] = [16usize, 16, 32];
    let step = columns / 2;
    let input = image(
        (0..depth * rows * columns)
            .map(|flat| if flat % columns < step { 0.0 } else { 100.0 })
            .collect(),
        [depth, rows, columns],
        [1.0, 1.0, 1.0],
    );
    let output = CannyEdgeDetector::new(GaussianSigma::new_unchecked(1.0), 2.0, 10.0)
        .apply_native(&input, &B::default())
        .expect("native Canny succeeds");

    let margin = 8;
    let mut near = 0usize;
    let mut far = 0usize;
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..columns {
                if values(&output)[z * rows * columns + y * columns + x] > 0.5 {
                    if x >= step - margin && x <= step + margin {
                        near += 1;
                    } else {
                        far += 1;
                    }
                }
            }
        }
    }
    assert!(near > 0, "step must produce an edge response");
    assert!(
        near > far,
        "step edges must concentrate near the step: {near} <= {far}"
    );
}

#[test]
#[should_panic(expected = "low_threshold")]
fn canny_rejects_inverted_thresholds() {
    let _ = CannyEdgeDetector::new(GaussianSigma::new_unchecked(1.0), 5.0, 2.0);
}

#[test]
fn canny_two_dimensional_step_retains_the_edge_column() {
    let rows = 20usize;
    let columns = 20usize;
    let mut input_values = vec![0.0; rows * columns];
    for row in 0..rows {
        for column in 10..columns {
            input_values[row * columns + column] = 1.0;
        }
    }
    let input = image(input_values, [1, rows, columns], [1.0, 1.0, 1.0]);
    let output = CannyEdgeDetector::new(GaussianSigma::new_unchecked(1.0), 0.05, 0.15)
        .apply_native(&input, &B::default())
        .expect("native Canny succeeds");
    let edge_rows = (0..rows)
        .filter(|&row| {
            values(&output)[row * columns + 9] > 0.5 || values(&output)[row * columns + 10] > 0.5
        })
        .count();

    assert!(
        edge_rows >= 15,
        "expected at least 15 step-edge rows, got {edge_rows}"
    );
}

#[test]
fn canny_nonmaximum_suppression_thins_a_linear_ramp() {
    let rows = 20usize;
    let columns = 20usize;
    let input = image(
        (0..rows * columns)
            .map(|index| (index % columns) as f32 / columns as f32)
            .collect(),
        [1, rows, columns],
        [1.0, 1.0, 1.0],
    );
    let output = CannyEdgeDetector::new(GaussianSigma::new_unchecked(0.5), 0.03, 0.06)
        .apply_native(&input, &B::default())
        .expect("native Canny succeeds");
    let edge_count = values(&output).iter().filter(|&&value| value > 0.5).count();

    assert!(
        edge_count < (rows * columns * 30) / 100,
        "nonmaximum suppression left too many edges: {edge_count}"
    );
}

#[test]
fn gradient_of_x_ramp_is_unit_along_x() {
    let [depth, rows, columns] = [3usize, 3, 4];
    let input = image(
        (0..depth * rows * columns)
            .map(|flat| (flat % columns) as f32)
            .collect(),
        [depth, rows, columns],
        [1.0, 1.0, 1.0],
    );
    let gradient = GradientImageFilter::new(true)
        .apply(&input, &B::default())
        .expect("native gradient succeeds");
    let result = components(&gradient);
    assert_eq!(result.len(), depth * rows * columns * 3);

    for z in 0..depth {
        for y in 0..rows {
            for x in 1..columns - 1 {
                let voxel = z * rows * columns + y * columns + x;
                assert!(
                    (result[3 * voxel] - 1.0).abs() < 1e-6,
                    "dx at {voxel}: {}",
                    result[3 * voxel]
                );
            }
        }
    }
    for voxel in 0..depth * rows * columns {
        assert!(
            result[3 * voxel + 1].abs() < 1e-6,
            "dy at {voxel}: {}",
            result[3 * voxel + 1]
        );
        assert!(
            result[3 * voxel + 2].abs() < 1e-6,
            "dz at {voxel}: {}",
            result[3 * voxel + 2]
        );
    }
}

#[test]
fn recursive_gradient_of_x_ramp_is_unit_along_x() {
    let [depth, rows, columns] = [6usize, 6, 12];
    let input = image(
        (0..depth * rows * columns)
            .map(|flat| (flat % columns) as f32)
            .collect(),
        [depth, rows, columns],
        [1.0, 1.0, 1.0],
    );
    let gradient = GradientRecursiveGaussianImageFilter::new(1.0)
        .apply(&input, &B::default())
        .expect("native recursive gradient succeeds");
    let result = components(&gradient);

    for z in 1..depth - 1 {
        for y in 1..rows - 1 {
            for x in 4..columns - 4 {
                let voxel = z * rows * columns + y * columns + x;
                assert!(
                    (result[3 * voxel] - 1.0).abs() < 5e-3,
                    "dx at {voxel}: {}",
                    result[3 * voxel]
                );
                assert!(
                    result[3 * voxel + 1].abs() < 5e-3,
                    "dy at {voxel}: {}",
                    result[3 * voxel + 1]
                );
                assert!(
                    result[3 * voxel + 2].abs() < 5e-3,
                    "dz at {voxel}: {}",
                    result[3 * voxel + 2]
                );
            }
        }
    }
}

#[test]
fn gradient_magnitude_of_constant_field_is_zero() {
    let shape = [4, 4, 4];
    let input = image(vec![3.5; shape.iter().product()], shape, [1.0, 1.0, 1.0]);
    let output = GradientMagnitudeFilter::unit()
        .apply_native(&input)
        .expect("native gradient magnitude succeeds");
    assert!(values(&output).iter().all(|&value| value == 0.0));
}

#[test]
fn laplacian_of_linear_field_is_zero_interior() {
    let [depth, rows, columns] = [5usize, 5, 5];
    let input = image(
        (0..depth * rows * columns)
            .map(|flat| {
                let z = flat / (rows * columns);
                let y = (flat / columns) % rows;
                let x = flat % columns;
                2.0 * z as f32 - 3.0 * y as f32 + 1.5 * x as f32 + 4.0
            })
            .collect(),
        [depth, rows, columns],
        [1.0, 1.0, 1.0],
    );
    let output = LaplacianFilter::unit()
        .apply_native(&input)
        .expect("native Laplacian succeeds");
    for z in 1..depth - 1 {
        for y in 1..rows - 1 {
            for x in 1..columns - 1 {
                let value = values(&output)[z * rows * columns + y * columns + x];
                assert!(
                    value.abs() < 1e-4,
                    "interior Laplacian at ({z}, {y}, {x}): {value}"
                );
            }
        }
    }
}

#[test]
fn sobel_of_constant_field_is_zero() {
    let shape = [4, 4, 4];
    let input = image(vec![-7.0; shape.iter().product()], shape, [1.0, 1.0, 1.0]);
    let output = SobelFilter::unit()
        .apply_native(&input)
        .expect("native Sobel succeeds");
    assert!(values(&output).iter().all(|&value| value == 0.0));
}

#[test]
fn log_of_constant_field_is_zero() {
    let shape = [16, 16, 16];
    let input = image(vec![42.0; shape.iter().product()], shape, [1.0, 1.0, 1.0]);
    let output = LaplacianOfGaussianFilter::new(GaussianSigma::new_unchecked(1.5))
        .apply_native(&input, &B::default())
        .expect("native LoG succeeds");
    for (index, &value) in values(&output).iter().enumerate() {
        assert!(
            value.abs() < 0.1,
            "LoG constant response at {index}: {value}"
        );
    }
}

#[test]
fn log_of_bright_gaussian_blob_is_negative_at_its_centre() {
    let [depth, rows, columns] = [32usize, 32, 32];
    let centre = [depth as f64 / 2.0, rows as f64 / 2.0, columns as f64 / 2.0];
    let sigma = 3.0_f64;
    let input = image(
        (0..depth * rows * columns)
            .map(|flat| {
                let z = flat / (rows * columns);
                let y = (flat / columns) % rows;
                let x = flat % columns;
                let radius_squared = (z as f64 - centre[0]).powi(2)
                    + (y as f64 - centre[1]).powi(2)
                    + (x as f64 - centre[2]).powi(2);
                (100.0 * (-radius_squared / (2.0 * sigma * sigma)).exp()) as f32
            })
            .collect(),
        [depth, rows, columns],
        [1.0, 1.0, 1.0],
    );
    let output = LaplacianOfGaussianFilter::new(GaussianSigma::new_unchecked(sigma))
        .apply_native(&input, &B::default())
        .expect("native LoG succeeds");
    let centre_value =
        values(&output)[(depth / 2) * rows * columns + (rows / 2) * columns + columns / 2];

    assert!(
        centre_value < -0.1,
        "LoG centre response must be negative: {centre_value}"
    );
}

#[test]
fn log_of_linear_field_is_zero_interior() {
    let [depth, rows, columns] = [16usize, 16, 16];
    let input = image(
        (0..depth * rows * columns)
            .map(|flat| {
                let z = flat / (rows * columns);
                let y = (flat / columns) % rows;
                let x = flat % columns;
                (x + y + z) as f32
            })
            .collect(),
        [depth, rows, columns],
        [1.0, 1.0, 1.0],
    );
    let output = LaplacianOfGaussianFilter::new(GaussianSigma::new_unchecked(1.5))
        .apply_native(&input, &B::default())
        .expect("native LoG succeeds");

    let margin = 5;
    for z in margin..depth - margin {
        for y in margin..rows - margin {
            for x in margin..columns - margin {
                let value = values(&output)[z * rows * columns + y * columns + x];
                assert!(
                    value.abs() < 0.5,
                    "LoG linear response at ({z}, {y}, {x}): {value}"
                );
            }
        }
    }
}
