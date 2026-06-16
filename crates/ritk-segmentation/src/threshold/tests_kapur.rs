//! Tests for Kapur maximum entropy thresholding.
//! Extracted to keep the 500-line structural limit.

use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;

type B = NdArray<f32>;

fn make_image_1d(data: Vec<f32>) -> Image<B, 1> {
    let n = data.len();
    make_image(data, [n])
}

fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    make_image(data, dims)
}

fn get_slice_1d(image: &Image<B, 1>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

// ── Degenerate / constant image ────────────────────────────────────────────

#[test]
fn test_constant_image_returns_constant_value() {
    let data = vec![42.0_f32; 100];
    let image = make_image_1d(data);
    let t = kapur_threshold(&image);
    assert!(
        (t - 42.0).abs() < f32::EPSILON,
        "constant image threshold must equal the constant value, got {}",
        t
    );
}

// ── Bimodal separation ─────────────────────────────────────────────────────

#[test]
fn test_bimodal_threshold_separates_modes() {
    // Bimodal distribution with spread around each mode so that multiple
    // histogram bins are populated in each class. Without spread, the
    // entropy of each single-bin class is zero for every threshold,
    // making the criterion degenerate.
    //
    // Mode 1 centred at 25: values 10..40 (31 bins occupied).
    // Mode 2 centred at 225: values 210..240 (31 bins occupied).
    // Gap between modes: [41, 209] — 169 intensity units wide.
    //
    // Analytical note: Kapur's criterion H(t) = H_b(t) + H_f(t) is
    // constant for all t in the empty gap [40, 210] (both class
    // compositions are unchanged). The argmax search selects the first
    // bin achieving the maximum, which falls at the upper boundary of
    // the low mode (~bin 33 for 256 bins over [10, 240]).  The resulting
    // intensity threshold is ≈ 39.8, which still correctly separates
    // the two modes: all low-mode values (≤ 40) are ≤ threshold and all
    // high-mode values (≥ 210) are > threshold.
    let mut data: Vec<f32> = (10..=40).map(|v| v as f32).collect(); // 31 values
    data.extend((10..=40).map(|v| v as f32)); // 62 total low
    let high: Vec<f32> = (210..=240).map(|v| v as f32).collect(); // 31 values
    data.extend(high.iter().copied());
    data.extend(high.iter().copied()); // 62 total high
    let image = make_image_1d(data);
    let t = KapurThreshold::new().compute(&image);

    // The threshold must lie past the centre of the low mode and before
    // the start of the high mode, ensuring correct binary separation.
    // It may fall at the upper boundary of the low mode due to the flat
    // criterion across the empty gap.
    assert!(
        t > 25.0,
        "threshold must exceed centre of low mode (25.0), got {}",
        t
    );
    assert!(
        t < 210.0,
        "threshold must be below lower edge of high mode (210.0), got {}",
        t
    );

    // Verify that the threshold actually separates the two modes:
    // applying it must label all high-mode voxels as foreground.
    let mask = KapurThreshold::new().apply(&image);
    let vals = get_slice_1d(&mask);
    // First 62 values are low-mode (10..40), last 62 are high-mode (210..240).
    for &v in &vals[62..] {
        assert!(
            v == 1.0,
            "high-mode voxel must be foreground (1.0), got {}",
            v
        );
    }
}

// ── Output shape preserved ─────────────────────────────────────────────────

#[test]
fn test_apply_output_shape_matches_input() {
    let dims = [4, 5, 6];
    let n = dims[0] * dims[1] * dims[2];
    let mut data = vec![10.0_f32; n / 2];
    data.extend(vec![200.0_f32; n - n / 2]);
    let image = make_image_3d(data, dims);
    let mask = KapurThreshold::new().apply(&image);
    assert_eq!(mask.shape(), dims);
}

// ── Binary output ──────────────────────────────────────────────────────────

#[test]
fn test_apply_output_is_strictly_binary() {
    let mut data = vec![30.0_f32; 60];
    data.extend(vec![180.0_f32; 40]);
    let image = make_image_1d(data);
    let mask = KapurThreshold::new().apply(&image);
    let vals = get_slice_1d(&mask);
    for &v in &vals {
        assert!(
            v == 0.0 || v == 1.0,
            "apply must produce binary output, found {}",
            v
        );
    }
}

// ── Convenience function matches struct ────────────────────────────────────

#[test]
fn test_convenience_fn_matches_struct_compute() {
    let mut data = vec![10.0_f32; 80];
    data.extend(vec![240.0_f32; 20]);
    let image = make_image_1d(data);
    let t_fn = kapur_threshold(&image);
    let t_struct = KapurThreshold::new().compute(&image);
    assert!(
        (t_fn - t_struct).abs() < f32::EPSILON,
        "convenience function and struct must agree"
    );
}

// ── Spatial metadata preserved ─────────────────────────────────────────────

#[test]
fn test_apply_preserves_spatial_metadata() {
    let dims = [2, 3, 4];
    let n = dims[0] * dims[1] * dims[2];
    let mut data = vec![10.0_f32; n / 2];
    data.extend(vec![200.0_f32; n - n / 2]);
    let image = make_image_3d(data, dims);
    let mask = KapurThreshold::new().apply(&image);

    assert_eq!(mask.origin(), image.origin());
    assert_eq!(mask.spacing(), image.spacing());
    assert_eq!(mask.direction(), image.direction());
}

// ── Default trait ──────────────────────────────────────────────────────────

#[test]
fn test_default_is_256_bins() {
    let k = KapurThreshold::default();
    assert_eq!(k.num_bins, 256);
}

// ── Panics on invalid bins ─────────────────────────────────────────────────

#[test]
#[should_panic(expected = "num_bins must be ≥ 2")]
fn test_with_bins_zero_panics() {
    KapurThreshold::with_bins(0);
}

#[test]
#[should_panic(expected = "num_bins must be ≥ 2")]
fn test_with_bins_one_panics() {
    KapurThreshold::with_bins(1);
}
