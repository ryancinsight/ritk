//! Tests for intensity projection filters.
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

// ── Helper ────────────────────────────────────────────────────────────────────

fn make_volume(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(data, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .expect("test: f32 backend required")
}

// ── 1. max_projection_z_shape ─────────────────────────────────────────────────

/// MaxIP along Z of a [4,3,2] volume must produce shape [1,3,2].
///
/// Invariant: collapsed axis → size 1; other axes unchanged.
#[test]
fn max_projection_z_shape() {
    let img = make_volume(vec![0.0_f32; 4 * 3 * 2], [4, 3, 2]);
    let filter = MaxIntensityProjectionFilter::new(ProjectionAxis::Z);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 3, 2], "MaxIP-Z shape must be [1, 3, 2]");
}

// ── 2. max_projection_z_values ────────────────────────────────────────────────

/// MaxIP along Z of a [3,2,2] image with known layer values.
///
/// Layer Z=0: [1,2,3,4], Z=1: [5,6,7,8], Z=2: [2,3,1,9].
/// For each (y,x) output pixel, the maximum across Z layers is:
///   (y=0,x=0): max(1,5,2)=5
///   (y=0,x=1): max(2,6,3)=6
///   (y=1,x=0): max(3,7,1)=7
///   (y=1,x=1): max(4,8,9)=9
#[test]
fn max_projection_z_values() {
    // Row-major [Z,Y,X]: Z=0→[1,2,3,4], Z=1→[5,6,7,8], Z=2→[2,3,1,9]
    let data: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8., 2., 3., 1., 9.];
    let img = make_volume(data, [3, 2, 2]);
    let filter = MaxIntensityProjectionFilter::new(ProjectionAxis::Z);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 2, 2]);
    let vals = extract_vals(&out);
    let expected = [5.0_f32, 6.0, 7.0, 9.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "MaxIP-Z pixel {i}: got {got}, expected {exp}"
        );
    }
}

// ── 3. min_projection_y_shape ─────────────────────────────────────────────────

/// MinIP along Y of a [3,5,4] volume must produce shape [3,1,4].
#[test]
fn min_projection_y_shape() {
    let img = make_volume(vec![0.0_f32; 3 * 5 * 4], [3, 5, 4]);
    let filter = MinIntensityProjectionFilter::new(ProjectionAxis::Y);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [3, 1, 4], "MinIP-Y shape must be [3, 1, 4]");
}

// ── 4. mean_projection_x_shape ────────────────────────────────────────────────

/// MeanIP along X of a [3,4,6] volume must produce shape [3,4,1].
#[test]
fn mean_projection_x_shape() {
    let img = make_volume(vec![0.0_f32; 3 * 4 * 6], [3, 4, 6]);
    let filter = MeanIntensityProjectionFilter::new(ProjectionAxis::X);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [3, 4, 1], "MeanIP-X shape must be [3, 4, 1]");
}

// ── 5. mean_projection_x_values ───────────────────────────────────────────────

/// MeanIP along X of a constant-1 image must return all-1 output.
///
/// Proof: mean(1, 1, …, 1) = 1 for any n ≥ 1.
#[test]
fn mean_projection_x_values() {
    let img = make_volume(vec![1.0_f32; 4 * 3 * 2], [4, 3, 2]);
    let filter = MeanIntensityProjectionFilter::new(ProjectionAxis::X);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [4, 3, 1]);
    let vals = extract_vals(&out);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 1.0_f32).abs() < 1e-6,
            "MeanIP-X constant-1 image: pixel {i} = {v}, expected 1.0"
        );
    }
}

// ── 6. sum_projection_z_values ────────────────────────────────────────────────

/// SumIP along Z of a [2,2,2] image with Z=0 all 1.0 and Z=1 all 2.0 must
/// produce shape [1,2,2] with every pixel = 3.0.
///
/// Proof: sum(1.0, 2.0) = 3.0 at each (y,x).
#[test]
fn sum_projection_z_values() {
    // Z=0: [1,1,1,1], Z=1: [2,2,2,2]
    let data: Vec<f32> = vec![1., 1., 1., 1., 2., 2., 2., 2.];
    let img = make_volume(data, [2, 2, 2]);
    let filter = SumIntensityProjectionFilter::new(ProjectionAxis::Z);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 2, 2]);
    let vals = extract_vals(&out);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 3.0_f32).abs() < 1e-6,
            "SumIP-Z pixel {i}: got {v}, expected 3.0"
        );
    }
}

// ── 7. stddev_projection_z_values ─────────────────────────────────────────────

/// StdDevIP along Z of a [2,1,1] image with values [0.0, 1.0] must produce
/// shape [1,1,1] with value 0.5.
///
/// Derivation (population std-dev):
///   μ = (0 + 1) / 2 = 0.5
///   σ = sqrt(((0 − 0.5)² + (1 − 0.5)²) / 2) = sqrt(0.25) = 0.5
#[test]
fn stddev_projection_z_values() {
    let img = make_volume(vec![0.0_f32, 1.0_f32], [2, 1, 1]);
    let filter = StdDevIntensityProjectionFilter::new(ProjectionAxis::Z);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 1]);
    let vals = extract_vals(&out);
    assert_eq!(vals.len(), 1);
    assert!(
        (vals[0] - 0.5_f32).abs() < 1e-5,
        "StdDevIP-Z of [0,1] must equal 0.5 (population std-dev), got {}",
        vals[0]
    );
}
