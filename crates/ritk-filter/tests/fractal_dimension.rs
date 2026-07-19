//! Native-image contracts for stochastic fractal dimension.

use coeus_core::SequentialBackend;
use ritk_filter::StochasticFractalDimensionFilter;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

fn image(values: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    Image::from_flat_on(
        values,
        shape,
        Point::origin(),
        Spacing::uniform(1.0),
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

#[test]
fn textured_image_has_a_finite_varying_fractal_dimension_field() {
    let [depth, rows, columns] = [8usize; 3];
    let input = (0..depth * rows * columns)
        .map(|flat| {
            let z = flat / (rows * columns);
            let y = (flat / columns) % rows;
            let x = flat % columns;
            ((x as f32 * 0.7).sin() + (y as f32 * 1.1).cos() + (z as f32 * 0.5).sin()) * 30.0 + 50.0
        })
        .collect();
    let output = StochasticFractalDimensionFilter::default()
        .apply(&image(input, [depth, rows, columns]), &B::default())
        .expect("textured image is readable on the native backend");
    assert!(values(&output).iter().all(|value| value.is_finite()));

    let (minimum, maximum) = values(&output).iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), &value| (minimum.min(value), maximum.max(value)),
    );
    assert_ne!(minimum, maximum, "field range must not collapse to zero");
}

#[test]
fn power_of_two_intensity_scaling_preserves_finite_dimension_values() {
    let shape = [7usize, 9, 6];
    let input = (0..shape.iter().product())
        .map(|index| ((index as f32 * 0.37).sin() * (index as f32 * 0.11).cos()) * 20.0 + 40.0)
        .collect::<Vec<_>>();
    // Multiplication by two is exact for these finite f32 values, so every
    // nonzero pairwise difference changes by one common logarithmic offset.
    // The least-squares slope must therefore remain bitwise identical.
    let scaled = input.iter().map(|&value| 2.0 * value).collect();
    let filter = StochasticFractalDimensionFilter::default();
    let baseline = filter
        .apply(&image(input, shape), &B::default())
        .expect("baseline image is readable");
    let rescaled = filter
        .apply(&image(scaled, shape), &B::default())
        .expect("rescaled image is readable");

    for (index, (&baseline, &rescaled)) in
        values(&baseline).iter().zip(values(&rescaled)).enumerate()
    {
        if baseline.is_finite() && rescaled.is_finite() {
            assert_eq!(baseline, rescaled, "voxel {index}");
        }
    }
}

#[test]
fn output_preserves_shape_and_physical_geometry() {
    let shape = [5usize; 3];
    let input = Image::from_flat_on(
        (0..shape.iter().product())
            .map(|index| (index as f32 * 1.3).sin())
            .collect(),
        shape,
        Point::new([3.0, -2.0, 7.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &B::default(),
    )
    .expect("invariant: valid native test image");
    let output = StochasticFractalDimensionFilter::new([1; 3])
        .apply(&input, &B::default())
        .expect("native image is readable");

    assert_eq!(output.shape(), shape);
    assert_eq!(output.origin(), input.origin());
    assert_eq!(output.spacing(), input.spacing());
    assert_eq!(output.direction(), input.direction());
}
