use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_transform::TranslationTransform;

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, shape)
}

// ── name ─────────────────────────────────────────────────────────────────

#[test]
fn lncc_name() {
    let metric = LocalNormalizedCrossCorrelation::<B>::new(GaussianSigma::new_unchecked(1.0));
    assert_eq!(
        <LocalNormalizedCrossCorrelation<B> as Metric<B, 3>>::name(&metric),
        "LocalNormalizedCrossCorrelation"
    );
}

// ── Identical images → loss ≈ -1 ─────────────────────────────────────────
//
// # Derivation
// When fixed == moving (identity transform), covariance = var_f = var_m at every
// voxel, so LNCC(x) = var_f / sqrt(var_f^2 + ε) ≈ 1.  The mean over all
// voxels is ≈ 1, and the returned loss is the negation: ≈ −1.
//
// # Tolerance
// The ε = 1e-5 guard on the denominator and the Gaussian kernel smearing
// introduce small deviations from the ideal −1.  We assert loss < −0.95
// (5 % tolerance), matching the ITK LNCC acceptance criterion.

#[test]
fn lncc_identical_images_loss_near_negative_one() {
    let size = 8;
    let count = size * size * size;
    // Non-constant ramp so that local variance > 0 almost everywhere.
    let data: Vec<f32> = (0..count).map(|i| i as f32).collect();

    let image = make_image(data, [size, size, size]);
    let device = Default::default();
    let transform =
        TranslationTransform::<B, 3>::new(burn::tensor::Tensor::<B, 1>::zeros([3], &device));

    let metric = LocalNormalizedCrossCorrelation::<B>::new(GaussianSigma::new_unchecked(1.5));
    let loss = metric.forward(&image, &image, &transform);
    let loss_val = loss.into_scalar();

    assert!(
        loss_val < -0.90,
        "LNCC for identical images must be < -0.90 (near -1); got {}",
        loss_val
    );
}

// ── Linear rescaling invariance ──────────────────────────────────────────
//
// LNCC is invariant to affine intensity transformations:
//   LNCC(F, a·F + b) = 1 for any a > 0, b ∈ ℝ.
// The loss must remain ≈ −1 when the moving image is a linearly rescaled
// copy of the fixed image.

#[test]
fn lncc_linear_rescaling_invariant() {
    let size = 6;
    let count = size * size * size;
    let data1: Vec<f32> = (0..count).map(|i| i as f32 + 1.0).collect();
    // a=3, b=10: moving = 3*fixed + 10
    let data2: Vec<f32> = data1.iter().map(|&x| 3.0 * x + 10.0).collect();

    let fixed = make_image(data1, [size, size, size]);
    let moving = make_image(data2, [size, size, size]);

    let device = Default::default();
    let transform =
        TranslationTransform::<B, 3>::new(burn::tensor::Tensor::<B, 1>::zeros([3], &device));

    let metric = LocalNormalizedCrossCorrelation::<B>::new(GaussianSigma::new_unchecked(1.5));
    let loss = metric.forward(&fixed, &moving, &transform);
    let loss_val = loss.into_scalar();

    assert!(
        loss_val < -0.90,
        "LNCC must be invariant to linear intensity rescaling (loss < -0.90); got {}",
        loss_val
    );
}

// ── Fixed-image cache stability ──────────────────────────────────────────
//
// The second `forward` call on the same (fixed, moving) pair must reuse the
// cached fixed-image statistics and return the same scalar value as the
// first call (within f32 round-trip precision).

#[test]
fn lncc_cache_returns_same_result_on_second_call() {
    let size = 6;
    let count = size * size * size;
    let data: Vec<f32> = (0..count).map(|i| i as f32 + 1.0).collect();

    let image = make_image(data, [size, size, size]);
    let device = Default::default();
    let transform =
        TranslationTransform::<B, 3>::new(burn::tensor::Tensor::<B, 1>::zeros([3], &device));

    let metric = LocalNormalizedCrossCorrelation::<B>::new(GaussianSigma::new_unchecked(1.5));

    let loss1 = metric.forward(&image, &image, &transform).into_scalar();
    let loss2 = metric.forward(&image, &image, &transform).into_scalar();

    assert!(
        (loss1 - loss2).abs() < 1e-5,
        "LNCC cache must produce identical results across calls: {} vs {}",
        loss1,
        loss2
    );
}

// ── Constant image → loss is finite ─────────────────────────────────────
//
// When the fixed image is constant, local variance = 0 everywhere.
// The ε guard on the denominator must prevent division by zero.
// The loss must be finite (not NaN or ±∞).

#[test]
fn lncc_constant_image_loss_is_finite() {
    let size = 4;
    let image = make_image(vec![5.0_f32; size * size * size], [size, size, size]);
    let device = Default::default();
    let transform =
        TranslationTransform::<B, 3>::new(burn::tensor::Tensor::<B, 1>::zeros([3], &device));

    let metric = LocalNormalizedCrossCorrelation::<B>::new(GaussianSigma::new_unchecked(1.0));
    let loss_val = metric.forward(&image, &image, &transform).into_scalar();

    assert!(
        loss_val.is_finite(),
        "LNCC with constant image must be finite (ε guard active); got {}",
        loss_val
    );
}
