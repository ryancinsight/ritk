//! Tests for n4 bias field correction.
//! Extracted to keep the 500-line structural limit.

use super::*;
use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_image::test_support as ts;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing, VolumeDims};

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn extract_vals(img: Image<f32, B, 3>) -> Vec<f32> {
    img.data_slice()
        .expect("invariant: contiguous host storage")
        .to_vec()
}

#[test]
fn native_n4_preserves_geometry_and_matches_values_provider() {
    let dimensions = [4, 4, 4];
    let values: Vec<f32> = (0..64).map(|index| 20.0 + index as f32 * 0.25).collect();
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let image = NativeImage::from_flat_on(
        values.clone(),
        dimensions,
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let config = N4Config {
        num_fitting_levels: 1,
        num_iterations: 1,
        ..Default::default()
    };

    let output = N4BiasFieldCorrectionFilter::new(config.clone())
        .apply_native(&image, &SequentialBackend)
        .expect("native N4 succeeds");

    assert_eq!(output.shape(), dimensions);
    assert_eq!(*output.origin(), origin);
    assert_eq!(*output.spacing(), spacing);
    assert_eq!(*output.direction(), direction);
    assert_eq!(
        output.data_slice().expect("contiguous output"),
        apply_n4_bias_correction_values(&values, dimensions, &config)
            .expect("values provider succeeds")
    );
}

/// Coefficient of variation (Ïƒ/Î¼) for a subset of voxels identified by indices.
fn within_class_cov(vals: &[f32], indices: &[usize]) -> f64 {
    assert!(!indices.is_empty(), "within_class_cov: empty class");
    let n = indices.len() as f64;
    let mean: f64 = indices.iter().map(|&i| vals[i] as f64).sum::<f64>() / n;
    let var: f64 = indices
        .iter()
        .map(|&i| ((vals[i] as f64) - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt() / mean.abs().max(1e-10)
}

fn rms_diff(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let ss: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ((ai - bi) as f64).powi(2))
        .sum();
    (ss / a.len() as f64).sqrt()
}

/// N4 stability on a two-class image.
#[test]
fn two_class_n4_stability_discrete_histogram() {
    let nz = 16usize;
    let ny = 16usize;
    let nx = 16usize;
    let n = nz * ny * nx;

    let mut class_a: Vec<usize> = Vec::new();
    let mut class_b: Vec<usize> = Vec::new();

    let vals: Vec<f32> = (0..n)
        .map(|vi| {
            let ix = vi % nx;
            let iy = (vi / nx) % ny;
            let bias = 1.0_f32 + 0.25 * (ix as f32 / (nx - 1) as f32 - 0.5);
            let true_intensity = if iy < ny / 2 { 100.0_f32 } else { 40.0_f32 };
            if iy < ny / 2 {
                class_a.push(vi);
            } else {
                class_b.push(vi);
            }
            true_intensity * bias
        })
        .collect();

    let cov_a_before = within_class_cov(&vals, &class_a);
    let cov_b_before = within_class_cov(&vals, &class_b);

    let image = make_image(vals, [nz, ny, nx]);
    let config = N4Config {
        num_fitting_levels: 1,
        num_iterations: 5,
        convergence_threshold: 1e-4,
        num_histogram_bins: 200,
        bias_field_fwhm: 0.15,
        bspline_mesh: VolumeDims::new([1, 1, 1]),
        noise_estimate: 0.07,
        shrink_factor: 1,
    };

    let out = extract_vals(
        N4BiasFieldCorrectionFilter::new(config)
            .apply(&image)
            .expect("N4 two-class stability apply failed"),
    );

    let cov_a_after = within_class_cov(&out, &class_a);
    let cov_b_after = within_class_cov(&out, &class_b);

    assert!(
        cov_a_after <= cov_a_before * 1.01,
        "Class A CoV increased: before={cov_a_before:.4} after={cov_a_after:.4}"
    );
    assert!(
        cov_b_after <= cov_b_before * 1.01,
        "Class B CoV increased: before={cov_b_before:.4} after={cov_b_after:.4}"
    );
    for &v in &out {
        assert!(v > 0.0 && v.is_finite(), "non-positive/nan output: {v}");
    }
}

/// histogram_sharpen reduces within-mode spread for a continuous bimodal distribution.
#[test]
fn histogram_sharpen_continuous_bimodal_reduces_spread() {
    let n_per_mode = 200usize;
    let mode_a_lo = 4.45_f32;
    let mode_a_hi = 4.55_f32;
    let mode_b_lo = 3.38_f32;
    let mode_b_hi = 3.48_f32;

    let mut w = Vec::with_capacity(2 * n_per_mode);
    for i in 0..n_per_mode {
        let t = i as f32 / (n_per_mode - 1) as f32;
        w.push(mode_a_lo + t * (mode_a_hi - mode_a_lo));
    }
    for i in 0..n_per_mode {
        let t = i as f32 / (n_per_mode - 1) as f32;
        w.push(mode_b_lo + t * (mode_b_hi - mode_b_lo));
    }

    let mut scratch = HistogramSharpenScratch::new(200, w.len());
    histogram_sharpen(&w, 200, 0.087, 0.01, &mut scratch).expect("histogram_sharpen failed");
    let w_sharp = scratch.w_sharp;
    assert_eq!(w_sharp.len(), w.len());

    let centre_a = 0.5 * (mode_a_lo + mode_a_hi);
    let var_a_before: f64 = w
        .iter()
        .take(n_per_mode)
        .map(|&v| ((v - centre_a) as f64).powi(2))
        .sum::<f64>()
        / n_per_mode as f64;
    let mean_a_after: f64 = w_sharp
        .iter()
        .take(n_per_mode)
        .map(|&v| v as f64)
        .sum::<f64>()
        / n_per_mode as f64;
    let var_a_after: f64 = w_sharp
        .iter()
        .take(n_per_mode)
        .map(|&v| ((v as f64) - mean_a_after).powi(2))
        .sum::<f64>()
        / n_per_mode as f64;
    assert!(var_a_after < var_a_before, "histogram_sharpen did not reduce Mode-A variance: before={var_a_before:.6} after={var_a_after:.6}");

    let centre_b = 0.5 * (mode_b_lo + mode_b_hi);
    let var_b_before: f64 = w
        .iter()
        .skip(n_per_mode)
        .map(|&v| ((v - centre_b) as f64).powi(2))
        .sum::<f64>()
        / n_per_mode as f64;
    let mean_b_after: f64 = w_sharp
        .iter()
        .skip(n_per_mode)
        .map(|&v| v as f64)
        .sum::<f64>()
        / n_per_mode as f64;
    let var_b_after: f64 = w_sharp
        .iter()
        .skip(n_per_mode)
        .map(|&v| ((v as f64) - mean_b_after).powi(2))
        .sum::<f64>()
        / n_per_mode as f64;
    assert!(var_b_after < var_b_before, "histogram_sharpen did not reduce Mode-B variance: before={var_b_before:.6} after={var_b_after:.6}");
}

/// Constant image: no crash; all output values within 100.0 Â± 5.0.
#[test]
fn constant_image_stable() {
    let dims = [8usize, 8, 8];
    let n = 8 * 8 * 8;
    let image = make_image(vec![100.0f32; n], dims);

    let config = N4Config {
        num_fitting_levels: 1,
        num_iterations: 5,
        convergence_threshold: 0.001,
        num_histogram_bins: 50,
        bias_field_fwhm: 0.15,
        bspline_mesh: VolumeDims::new([1, 1, 1]),
        noise_estimate: 0.01,
        shrink_factor: 1,
    };

    let out = extract_vals(
        N4BiasFieldCorrectionFilter::new(config)
            .apply(&image)
            .expect("N4 constant failed"),
    );
    for &v in &out {
        assert!(
            (v - 100.0).abs() < 5.0,
            "constant image: expected ~100.0, got {v:.4}"
        );
    }
}

/// All corrected output values are strictly positive.
#[test]
fn output_all_positive() {
    let nz = 8;
    let ny = 8;
    let nx = 8;
    let n = nz * ny * nx;

    let vals: Vec<f32> = (0..n)
        .map(|vi| {
            let iz = vi / (ny * nx);
            let t = iz as f32 / (nz - 1) as f32;
            50.0_f32 * (1.0 + 0.3 * t * t)
        })
        .collect();

    let image = make_image(vals, [nz, ny, nx]);
    let config = N4Config {
        num_fitting_levels: 1,
        num_iterations: 5,
        convergence_threshold: 0.001,
        num_histogram_bins: 50,
        bias_field_fwhm: 0.15,
        bspline_mesh: VolumeDims::new([1, 1, 1]),
        noise_estimate: 0.01,
        shrink_factor: 1,
    };

    let out = extract_vals(
        N4BiasFieldCorrectionFilter::new(config)
            .apply(&image)
            .expect("N4 positive failed"),
    );
    for &v in &out {
        assert!(v > 0.0, "non-positive output value: {v}");
    }
}

/// next_pow2 correctness at boundary values.
#[test]
fn next_pow2_boundaries() {
    assert_eq!(next_pow2(0), 1);
    assert_eq!(next_pow2(1), 1);
    assert_eq!(next_pow2(2), 2);
    assert_eq!(next_pow2(3), 4);
    assert_eq!(next_pow2(128), 128);
    assert_eq!(next_pow2(129), 256);
    assert_eq!(next_pow2(200), 256);
}

/// Gaussian kernel is L1-normalised and symmetric.
#[test]
fn gaussian_kernel_normalised_and_symmetric() {
    for &sigma in &[0.5_f64, 1.0, 2.5, 5.0] {
        let k = crate::gaussian_kernel(sigma, None);
        let sum: f64 = k.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "sigma={sigma}: kernel sum = {sum:.15}"
        );
        let len = k.len();
        for i in 0..len / 2 {
            assert!(
                (k[i] - k[len - 1 - i]).abs() < 1e-15,
                "sigma={sigma}: asymmetry at i={i}"
            );
        }
    }
}

/// rms_diff is zero for identical slices and positive for different ones.
#[test]
fn rms_diff_identity_and_positive() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(rms_diff(&a, &a), 0.0);
    let b = vec![2.0f32, 3.0, 4.0, 5.0];
    let d = rms_diff(&a, &b);
    assert!((d - 1.0).abs() < 1e-6, "expected rms=1.0, got {d}");
}

#[test]
fn n4_value_helper_rejects_shape_length_mismatch() {
    let err = apply_n4_bias_correction_values(&[1.0, 2.0, 3.0], [1, 2, 2], &N4Config::default())
        .unwrap_err();

    assert_eq!(
        err.to_string(),
        "N4 input value count 3 does not match voxel count 4 for dims [1, 2, 2]"
    );
}

fn dft_real(data: &[f64], n: usize) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); n];
    dft_real_into(data, n, &mut out);
    out
}

fn idft_real(freq: &[(f64, f64)], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    idft_real_into(freq, n, &mut out);
    out
}

/// DFT round-trip: IDFT(DFT(x)) â‰ˆ x for a short real sequence.
#[test]
fn dft_round_trip() {
    let data = vec![1.0f64, 2.0, 0.5, 0.0, 1.5, 0.0, 0.0, 0.0];
    let n = data.len();
    let freq = dft_real(&data, n);
    let recovered = idft_real(&freq, n);
    for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (orig - rec).abs() < 1e-9,
            "index {i}: orig={orig:.10} rec={rec:.10}"
        );
    }
}

/// histogram_sharpen returns the input unchanged for a constant signal.
#[test]
fn histogram_sharpen_passthrough_for_constant_input() {
    let w = vec![2.71f32; 64];
    let mut scratch = HistogramSharpenScratch::new(100, w.len());
    histogram_sharpen(&w, 100, 0.01, 0.01, &mut scratch).unwrap();
    let out = scratch.w_sharp;
    for (&o, &i) in out.iter().zip(w.iter()) {
        assert_eq!(o, i);
    }
}
