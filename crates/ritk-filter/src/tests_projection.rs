//! Tests for intensity projection filters.
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

type B = coeus_core::SequentialBackend;

// â”€â”€ Helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn make_volume(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    let tensor = Tensor::<f32, B>::from_slice(shape, &data);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

fn extract_vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data().to_vec()
}

// â”€â”€ 1. max_projection_z_shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// MaxIP along Z of a [4,3,2] volume must produce shape [1,3,2].
///
/// Invariant: collapsed axis â†’ size 1; other axes unchanged.
#[test]
fn max_projection_z_shape() {
    let img = make_volume(vec![0.0_f32; 4 * 3 * 2], [4, 3, 2]);
    let filter = MaxIntensityProjectionFilter::new(ProjectionAxis::Z);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 3, 2], "MaxIP-Z shape must be [1, 3, 2]");
}

// â”€â”€ 2. max_projection_z_values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    // Row-major [Z,Y,X]: Z=0â†’[1,2,3,4], Z=1â†’[5,6,7,8], Z=2â†’[2,3,1,9]
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

// â”€â”€ 3. min_projection_y_shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// MinIP along Y of a [3,5,4] volume must produce shape [3,1,4].
#[test]
fn min_projection_y_shape() {
    let img = make_volume(vec![0.0_f32; 3 * 5 * 4], [3, 5, 4]);
    let filter = MinIntensityProjectionFilter::new(ProjectionAxis::Y);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [3, 1, 4], "MinIP-Y shape must be [3, 1, 4]");
}

// â”€â”€ 4. mean_projection_x_shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// MeanIP along X of a [3,4,6] volume must produce shape [3,4,1].
#[test]
fn mean_projection_x_shape() {
    let img = make_volume(vec![0.0_f32; 3 * 4 * 6], [3, 4, 6]);
    let filter = MeanIntensityProjectionFilter::new(ProjectionAxis::X);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [3, 4, 1], "MeanIP-X shape must be [3, 4, 1]");
}

// â”€â”€ 5. mean_projection_x_values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// MeanIP along X of a constant-1 image must return all-1 output.
///
/// Proof: mean(1, 1, â€¦, 1) = 1 for any n â‰¥ 1.
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

// â”€â”€ 6. sum_projection_z_values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ 7. stddev_projection_z_values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// StdDevIP along Z of a [2,1,1] image with values [0.0, 1.0] must produce
/// shape [1,1,1] with the sample standard deviation.
///
/// Derivation (sample / Nâˆ’1 std-dev, matching ITK):
///   Î¼ = (0 + 1) / 2 = 0.5
///   Ïƒ = sqrt(((0 âˆ’ 0.5)Â² + (1 âˆ’ 0.5)Â²) / (2 âˆ’ 1)) = sqrt(0.5) â‰ˆ 0.70710678
#[test]
fn stddev_projection_z_values() {
    let img = make_volume(vec![0.0_f32, 1.0_f32], [2, 1, 1]);
    let filter = StdDevIntensityProjectionFilter::new(ProjectionAxis::Z);
    let out = filter.apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 1]);
    let vals = extract_vals(&out);
    assert_eq!(vals.len(), 1);
    assert!(
        (vals[0] - 0.5_f32.sqrt()).abs() < 1e-5,
        "StdDevIP-Z of [0,1] must equal sqrt(0.5) (sample std-dev), got {}",
        vals[0]
    );
}

/// Median projection along X: per row, the median of the row values. For an even
/// count, ITK's `nth_element` at size/2 takes the upper-middle element.
#[test]
fn median_projection_x_values() {
    // 1Ã—2Ã—4: row0 = [1,2,3,4] (median@2 = 3), row1 = [10,5,5,5] (sorted 5,5,5,10 â†’ @2 = 5)
    let img = make_volume(vec![1.0, 2.0, 3.0, 4.0, 10.0, 5.0, 5.0, 5.0], [1, 2, 4]);
    let out = MedianIntensityProjectionFilter::new(ProjectionAxis::X)
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 2, 1]);
    assert_eq!(extract_vals(&out), vec![3.0, 5.0]);
}

/// Binary projection: foreground if any voxel along the axis equals foreground.
#[test]
fn binary_projection_x_any_foreground() {
    // 1Ã—2Ã—3: row0 = [0,1,0] (has fg 1 â†’ 1), row1 = [0,0,0] (no fg â†’ 0)
    let img = make_volume(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1, 2, 3]);
    let out = BinaryProjectionFilter::new(ProjectionAxis::X, 1.0, 0.0)
        .apply(&img)
        .unwrap();
    assert_eq!(extract_vals(&out), vec![1.0, 0.0]);
}

/// Binary-threshold projection: foreground if any voxel along the axis >= threshold.
#[test]
fn binary_threshold_projection_x_any_ge() {
    // 1Ã—2Ã—3: row0 = [1,2,3] (max 3 â‰¥ 3 â†’ 1), row1 = [1,1,2] (max 2 < 3 â†’ 0)
    let img = make_volume(vec![1.0, 2.0, 3.0, 1.0, 1.0, 2.0], [1, 2, 3]);
    let out = BinaryThresholdProjectionFilter::new(ProjectionAxis::X, 3.0, 1.0, 0.0)
        .apply(&img)
        .unwrap();
    assert_eq!(extract_vals(&out), vec![1.0, 0.0]);
}

// â”€â”€ T-3: even-axis-length median â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Median along X of a [1, 1, 4] image [1.0, 2.0, 3.0, 4.0].
///
/// With n = 4 (even), `median_at_half` uses k = n/2 = 2.
/// `select_nth_unstable_by(2, â€¦)` on a 4-element sequence returns the
/// element that ranks at index 2 in sorted order: sorted = [1,2,3,4], so
/// the result is 3.0 (upper-middle, matching ITKâ€™s nth_element at size/2).
#[test]
fn median_projection_x_even_axis_length() {
    let img = make_volume(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 1, 4]);
    let out = MedianIntensityProjectionFilter::new(ProjectionAxis::X)
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 1]);
    let vals = extract_vals(&out);
    assert_eq!(vals.len(), 1);
    assert_eq!(
        vals[0], 3.0_f32,
        "even-length (n=4) median at n/2=2 must be 3.0, got {}",
        vals[0]
    );
}
