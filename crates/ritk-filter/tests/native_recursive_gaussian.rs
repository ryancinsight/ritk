//! Native-image contracts for recursive Gaussian filtering.

use coeus_core::SequentialBackend;
use ritk_filter::recursive_gaussian::{
    gradient_magnitude_recursive_gaussian, laplacian_recursive_gaussian,
    recursive_gaussian_directional,
};
use ritk_filter::{DerivativeOrder, RecursiveGaussianFilter};
use ritk_image::Image;
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

/// A constant signal is the Deriche recurrence's steady state under its
/// unit-DC-gain coefficients, so smoothing preserves every voxel.
#[test]
fn smoothing_preserves_constant_and_metadata() {
    let shape = [16, 16, 16];
    let constant = 42.0_f32;
    let input = image(
        vec![constant; shape.iter().product()],
        shape,
        [1.0, 1.0, 1.0],
    );
    let output = RecursiveGaussianFilter::new(2.0)
        .apply_native(&input, &B::default())
        .expect("native recursive Gaussian smoothing succeeds");

    for (index, &value) in values(&output).iter().enumerate() {
        assert!(
            (value - constant).abs() < 1e-3,
            "constant image smoothing: voxel {index} = {value}, expected {constant}"
        );
    }
    assert_eq!(output.origin(), input.origin());
    assert_eq!(output.spacing(), input.spacing());
    assert_eq!(output.direction(), input.direction());
}

/// Gaussian smoothing has unit DC gain, so global intensity is conserved up to
/// the deterministic replicated-boundary approximation error.
#[test]
fn smoothing_preserves_sum() {
    let shape = [20, 20, 20];
    let input_values: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|index| (index % 17) as f32)
        .collect();
    let input_sum: f64 = input_values.iter().map(|&value| value as f64).sum();
    let input = image(input_values, shape, [1.0, 1.0, 1.0]);
    let output = RecursiveGaussianFilter::new(1.5)
        .apply_native(&input, &B::default())
        .expect("native recursive Gaussian smoothing succeeds");
    let output_sum: f64 = values(&output).iter().map(|&value| value as f64).sum();
    let relative_error = (output_sum - input_sum).abs() / input_sum.abs().max(1e-12);

    assert!(
        relative_error < 0.05,
        "sum not preserved: input={input_sum}, output={output_sum}, relative error={relative_error}"
    );
}

/// The first derivative of a linear ramp is spatially constant away from the
/// IIR boundary transient.
#[test]
fn first_derivative_of_linear_ramp_is_constant_interior() {
    let length = 64usize;
    let input = image(
        (0..length).map(|index| index as f32).collect(),
        [1, 1, length],
        [1.0, 1.0, 1.0],
    );
    let output = RecursiveGaussianFilter::new(3.0)
        .with_derivative_order(DerivativeOrder::First)
        .apply_native(&input, &B::default())
        .expect("native first derivative succeeds");

    let margin = 12;
    let interior = &values(&output)[margin..length - margin];
    let mean: f64 = interior.iter().map(|&value| value as f64).sum::<f64>() / interior.len() as f64;
    for (offset, &value) in interior.iter().enumerate() {
        let deviation = (value as f64 - mean).abs();
        assert!(
            deviation < mean.abs() * 0.15 + 0.1,
            "first derivative is not constant at {}: {value} versus mean {mean}",
            offset + margin
        );
    }
    assert!(
        mean.abs() > 0.01,
        "first derivative mean must be nonzero: {mean}"
    );
}

/// The second derivative of a quadratic is spatially constant away from the
/// IIR boundary transient.
#[test]
fn second_derivative_of_quadratic_is_constant_interior() {
    let length = 64usize;
    let input = image(
        (0..length)
            .map(|index| {
                let value = index as f32;
                value * value
            })
            .collect(),
        [1, 1, length],
        [1.0, 1.0, 1.0],
    );
    let output = RecursiveGaussianFilter::new(3.0)
        .with_derivative_order(DerivativeOrder::Second)
        .apply_native(&input, &B::default())
        .expect("native second derivative succeeds");

    let margin = 15;
    let interior = &values(&output)[margin..length - margin];
    let mean: f64 = interior.iter().map(|&value| value as f64).sum::<f64>() / interior.len() as f64;
    for (offset, &value) in interior.iter().enumerate() {
        let deviation = (value as f64 - mean).abs();
        assert!(
            deviation < mean.abs() * 0.25 + 0.5,
            "second derivative is not constant at {}: {value} versus mean {mean}",
            offset + margin
        );
    }
    assert!(
        mean.abs() > 0.5,
        "second derivative mean must be substantially nonzero: {mean}"
    );
}

/// For `f(x) = x²`, the physical Laplacian is exactly two regardless of voxel
/// spacing. The tolerance bounds the finite IIR approximation after its boundary
/// transient decays.
#[test]
fn laplacian_of_physical_quadratic_is_two() {
    let length = 160usize;
    let margin = 48;
    for &spacing_x in &[1.0_f64, 2.0, 0.5] {
        let input = image(
            (0..length)
                .map(|index| (index as f64 * spacing_x).powi(2) as f32)
                .collect(),
            [1, 1, length],
            [1.0, 1.0, spacing_x],
        );
        let output = laplacian_recursive_gaussian(&input, 3.0, &B::default())
            .expect("native Laplacian recursive Gaussian succeeds");
        for (offset, &value) in values(&output)[margin..length - margin].iter().enumerate() {
            assert!(
                (value as f64 - 2.0).abs() < 0.02,
                "physical Laplacian must be two at spacing {spacing_x}, x={}: {value}",
                offset + margin
            );
        }
    }
}

/// For `f(x) = a·x`, the physical gradient magnitude is `|a|` regardless of
/// voxel spacing once the boundary transient decays.
#[test]
fn gradient_magnitude_of_physical_ramp_is_slope() {
    let length = 160usize;
    let margin = 48;
    let slope = 3.0_f64;
    for &spacing_x in &[1.0_f64, 2.0, 0.5] {
        let input = image(
            (0..length)
                .map(|index| (slope * index as f64 * spacing_x) as f32)
                .collect(),
            [1, 1, length],
            [1.0, 1.0, spacing_x],
        );
        let output = gradient_magnitude_recursive_gaussian(&input, 3.0, &B::default())
            .expect("native gradient magnitude recursive Gaussian succeeds");
        for (offset, &value) in values(&output)[margin..length - margin].iter().enumerate() {
            assert!(
                (value as f64 - slope).abs() < 0.02,
                "physical gradient magnitude must be {slope} at spacing {spacing_x}, x={}: {value}",
                offset + margin
            );
        }
    }
}

#[test]
fn directional_first_derivative_of_ramp_is_signed_slope() {
    let length = 96usize;
    let slope = 3.0_f64;
    let input = image(
        (0..length)
            .map(|index| (slope * index as f64) as f32)
            .collect(),
        [1, 1, length],
        [1.0, 1.0, 1.0],
    );
    let output =
        recursive_gaussian_directional(&input, 3.0, DerivativeOrder::First, 2, &B::default())
            .expect("native directional recursive Gaussian succeeds");

    let margin = 24;
    for (offset, &value) in values(&output)[margin..length - margin].iter().enumerate() {
        assert!(
            (value as f64 - slope).abs() < 0.02,
            "directional derivative must be {slope} at x={}: {value}",
            offset + margin
        );
    }
}

#[test]
fn subpixel_sigma_is_identity() {
    let shape = [8, 8, 8];
    let input_values: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|index| (index % 13) as f32)
        .collect();
    let input = image(input_values.clone(), shape, [1.0, 1.0, 1.0]);
    let output = RecursiveGaussianFilter::new(0.1)
        .apply_native(&input, &B::default())
        .expect("native subpixel recursive Gaussian succeeds");

    for (index, (&expected, &actual)) in input_values.iter().zip(values(&output)).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "subpixel sigma changed voxel {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn derivative_orders_annihilate_constant_field() {
    let shape = [10, 10, 10];
    for (order, constant) in [
        (DerivativeOrder::First, 9.0_f32),
        (DerivativeOrder::Second, -4.0),
    ] {
        let input = image(
            vec![constant; shape.iter().product()],
            shape,
            [1.0, 1.0, 1.0],
        );
        let output = RecursiveGaussianFilter::new(1.5)
            .with_derivative_order(order)
            .apply_native(&input, &B::default())
            .expect("native recursive Gaussian derivative succeeds");
        for (index, &value) in values(&output).iter().enumerate() {
            assert!(
                value.abs() < 1e-3,
                "{order:?} of a constant must vanish at voxel {index}: {value}"
            );
        }
    }
}
