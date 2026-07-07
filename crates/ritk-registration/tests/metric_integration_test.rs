//! Comprehensive integration tests for all registration metrics with known-answer verification.
//!
//! Covers Mattes MI, NMI, CR, MI with separate ranges, MI with stochastic sampling,
//! and MI monotonicity under rotation — all with absolute-value checks and relative
//! ordering assertions against synthetic images.

use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_registration::metric::{
    CorrelationDirection, CorrelationRatio, Metric, MutualInformation, MutualInformationVariant,
    NormalizationMethod,
};
use ritk_statistics::IntensityRange;
use ritk_transform::{RigidTransform, TranslationTransform};

type B = NdArray<f32>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_test_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let tensor = Tensor::from_data(TensorData::new(data, Shape::new(shape)), &device);
    let spacing = Spacing::new([1.0, 1.0, 1.0]);
    let origin = Point::new([0.0, 0.0, 0.0]);
    let direction = Direction::identity();
    Image::new(tensor, origin, spacing, direction)
}

fn create_identity_transform() -> TranslationTransform<B, 3> {
    let device = Default::default();
    TranslationTransform::new(Tensor::zeros([3], &device))
}

fn create_small_rotation_transform(angle_rad: f32) -> RigidTransform<B, 3> {
    let device = Default::default();
    let rotation = Tensor::<B, 1>::from_data(TensorData::from([angle_rad, 0.0, 0.0]), &device);
    let translation = Tensor::<B, 1>::zeros([3], &device);
    let center = Tensor::<B, 1>::zeros([3], &device);
    RigidTransform::new(translation, rotation, center)
}

/// 3-D Gaussian blob with the given amplitude.
fn create_gaussian_blob(size: usize, amplitude: f32) -> Vec<f32> {
    let center = size as f32 / 2.0;
    let sigma = size as f32 / 6.0;
    (0..size.pow(3))
        .map(|i| {
            let z = (i / (size * size)) as f32 - center;
            let y = ((i % (size * size)) / size) as f32 - center;
            let x = (i % size) as f32 - center;
            amplitude * (-(x * x + y * y + z * z) / (2.0 * sigma * sigma)).exp()
        })
        .collect()
}

// ===========================================================================
// 1. Mattes MI — perfect alignment
// ===========================================================================

#[test]
fn test_mattes_mi_perfect_alignment() {
    let size = 10;
    let count = size * size * size;
    let data: Vec<f32> = (0..count).map(|i| (i as f32) / (count as f32)).collect();
    let image = create_test_image(data, [size, size, size]);
    let transform = create_identity_transform();

    let device = Default::default();
    let mattes = MutualInformation::<B>::new_mattes(32, 0.0, 1.0, &device);

    let loss = mattes.forward(&image, &image, &transform);
    let loss_val = loss.into_scalar();

    // For identical images MI = H(X), which for 1000 distinct values is high.
    // Negative MI as loss should be well below -0.5.
    assert!(
        loss_val < -0.5,
        "Mattes MI loss for identical images should be < -0.5, got {}",
        loss_val
    );
}

// ===========================================================================
// 2. Mattes MI — monotonicity under translation
// ===========================================================================

#[test]
fn test_mattes_mi_monotonicity() {
    // Use a Gaussian blob — a linear ramp shifted by a few voxels still has
    // near-perfect linear correlation, making MI differences noise-level (~0.005).
    // A Gaussian blob has actual spatial structure where translation meaningfully
    // changes the joint distribution, producing clear monotonic MI degradation.
    let size = 20;
    let blob = create_gaussian_blob(size, 1.0);
    let image = create_test_image(blob, [size, size, size]);
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let mattes = MutualInformation::<B>::new_mattes(32, 0.0, 1.0, &device);

    // Identity
    let t0 = TranslationTransform::new(Tensor::zeros([3], &device));
    let loss0 = mattes.forward(&image, &image, &t0).into_scalar();

    // Moderate translation (3 voxels along x)
    let t3 = TranslationTransform::new(Tensor::from_data(
        TensorData::from([3.0f32, 0.0, 0.0]),
        &device,
    ));
    let loss3 = mattes.forward(&image, &image, &t3).into_scalar();

    // Larger translation (6 voxels along x)
    let t6 = TranslationTransform::new(Tensor::from_data(
        TensorData::from([6.0f32, 0.0, 0.0]),
        &device,
    ));
    let loss6 = mattes.forward(&image, &image, &t6).into_scalar();

    // More negative loss = higher MI.  Monotonic decrease with misalignment.
    assert!(
        loss0 < loss3,
        "MI at identity ({}) should be higher (more negative loss) than at 3-voxel shift ({})",
        loss0,
        loss3
    );
    assert!(
        loss3 < loss6,
        "MI at 3-voxel shift ({}) should be higher than at 6-voxel shift ({})",
        loss3,
        loss6
    );
}

// ===========================================================================
// 3. NMI — perfect alignment ≈ 2.0
// ===========================================================================

#[test]
fn test_nmi_perfect_alignment() {
    let size = 10;
    let count = size * size * size;
    let data: Vec<f32> = (0..count).map(|i| (i as f32) / (count as f32)).collect();
    let image = create_test_image(data, [size, size, size]);
    let transform = create_identity_transform();

    let device = Default::default();
    let nmi = MutualInformation::<B>::new(
        MutualInformationVariant::Normalized(NormalizationMethod::JointEntropy),
        32,
        0.0,
        1.0,
        0.03,
        &device,
    );

    let loss = nmi.forward(&image, &image, &transform);
    let loss_val = loss.into_scalar();

    // NMI = (H(X)+H(Y))/H(X,Y).  For identical images NMI → 2, loss → -2.
    // Discretization in a small test image relaxes the threshold to -1.3
    // (consistent with existing normalized_mi_test).
    assert!(
        loss_val < -1.3,
        "NMI loss for identical images should be close to -2.0, got {}",
        loss_val
    );
}

// ===========================================================================
// 4. NMI — shifted images have lower NMI
// ===========================================================================

#[test]
fn test_nmi_shift_sensitivity() {
    let size = 10;
    let count = size * size * size;
    let data1: Vec<f32> = (0..count).map(|i| (i as f32) / (count as f32)).collect();
    let data2: Vec<f32> = (0..count)
        .map(|i| ((i + 5) % count) as f32 / (count as f32))
        .collect();

    let fixed = create_test_image(data1, [size, size, size]);
    let moving = create_test_image(data2, [size, size, size]);
    let transform = create_identity_transform();

    let device = Default::default();
    let nmi = MutualInformation::<B>::new(
        MutualInformationVariant::Normalized(NormalizationMethod::JointEntropy),
        32,
        0.0,
        1.0,
        0.03,
        &device,
    );

    let loss_perfect = nmi.forward(&fixed, &fixed, &transform).into_scalar();
    let loss_shifted = nmi.forward(&fixed, &moving, &transform).into_scalar();

    assert!(
        loss_perfect < loss_shifted,
        "NMI for identical images ({}) should be higher (more negative loss) than for shifted ({})",
        loss_perfect,
        loss_shifted
    );
}

// ===========================================================================
// 5. CR — perfect alignment ≈ 1.0
// ===========================================================================

#[test]
fn test_cr_perfect_alignment() {
    let size = 10;
    let count = size * size * size;
    let data: Vec<f32> = (0..count).map(|i| (i as f32) / (count as f32)).collect();
    let image = create_test_image(data, [size, size, size]);
    let transform = create_identity_transform();

    let device = Default::default();
    let cr = CorrelationRatio::<B>::new(
        32,
        IntensityRange::new_unchecked(0.0_f32, 1.0_f32),
        0.03,
        CorrelationDirection::MovingGivenFixed,
        &device,
    );

    let loss = cr.forward(&image, &image, &transform);
    let loss_val = loss.into_scalar();

    // CR ≈ 1.0 for identical images → loss ≈ -1.0
    assert!(
        loss_val < -0.9,
        "CR loss for identical images should be close to -1.0, got {}",
        loss_val
    );
}

// ===========================================================================
// 6. CR — direction symmetry
// ===========================================================================

#[test]
fn test_cr_direction_symmetry() {
    let size = 10;
    let count = size * size * size;
    let data: Vec<f32> = (0..count).map(|i| (i as f32) / (count as f32)).collect();
    let image = create_test_image(data, [size, size, size]);
    let transform = create_identity_transform();

    let device = Default::default();
    let cr_mgf = CorrelationRatio::<B>::new(
        32,
        IntensityRange::new_unchecked(0.0_f32, 1.0_f32),
        0.03,
        CorrelationDirection::MovingGivenFixed,
        &device,
    );
    let cr_fgm = CorrelationRatio::<B>::new(
        32,
        IntensityRange::new_unchecked(0.0_f32, 1.0_f32),
        0.03,
        CorrelationDirection::FixedGivenMoving,
        &device,
    );

    let loss_mgf = cr_mgf.forward(&image, &image, &transform).into_scalar();
    let loss_fgm = cr_fgm.forward(&image, &image, &transform).into_scalar();

    // Both directions should yield CR near 1.0 (loss near -1.0) for identical images.
    assert!(
        loss_mgf < -0.9,
        "CR(MovingGivenFixed) loss for identical images should be close to -1.0, got {}",
        loss_mgf
    );
    assert!(
        loss_fgm < -0.9,
        "CR(FixedGivenMoving) loss for identical images should be close to -1.0, got {}",
        loss_fgm
    );
    // Both should agree (within a tolerance) since X = Y.
    assert!(
        (loss_mgf - loss_fgm).abs() < 0.2,
        "CR directions should agree for identical images: MGF={}, FGM={}",
        loss_mgf,
        loss_fgm
    );
}

// ===========================================================================
// 7. MI with separate intensity ranges
// ===========================================================================

#[test]
fn test_mi_separate_intensity_ranges() {
    let size = 10;
    let count = size * size * size;

    // Fixed image in [0, 1]
    let fixed_data: Vec<f32> = (0..count).map(|i| (i as f32) / (count as f32)).collect();
    // Moving image in [0, 10] — same monotonic pattern but different range
    let moving_data: Vec<f32> = (0..count)
        .map(|i| ((i as f32) / (count as f32)) * 10.0)
        .collect();

    let fixed = create_test_image(fixed_data, [size, size, size]);
    let moving = create_test_image(moving_data, [size, size, size]);
    let transform = create_identity_transform();

    let device = Default::default();

    // With separate ranges, the metric can allocate full bin resolution to each image.
    let mi_sep = MutualInformation::<B>::new_with_separate_ranges(
        MutualInformationVariant::Mattes,
        32,
        0.0,
        1.0, // fixed range
        0.0,
        10.0, // moving range
        &device,
    );

    let loss = mi_sep.forward(&fixed, &moving, &transform);
    let loss_val = loss.into_scalar();

    // The two images are deterministically related (Y = 10·X), so MI should be high.
    // Loss = -MI should be well below -0.5.
    assert!(
        loss_val < -0.5,
        "MI with separate ranges for Y=10X should be high (loss < -0.5), got {}",
        loss_val
    );
}

// ===========================================================================
// 8. MI with stochastic sampling
// ===========================================================================

#[test]
fn test_mi_stochastic_sampling() {
    let size = 10;
    let count = size * size * size;
    let data: Vec<f32> = (0..count).map(|i| (i as f32) / (count as f32)).collect();
    let image = create_test_image(data, [size, size, size]);
    let transform = create_identity_transform();

    let device = Default::default();

    let mi_full = MutualInformation::<B>::new_mattes(32, 0.0, 1.0, &device);
    let mi_sampled = MutualInformation::<B>::new_mattes(32, 0.0, 1.0, &device).with_sampling(0.20);

    let loss_full = mi_full.forward(&image, &image, &transform).into_scalar();
    let loss_sampled = mi_sampled.forward(&image, &image, &transform).into_scalar();

    // 20% sampled MI should approximate full MI.  Allow generous tolerance because
    // stochastic subsampling introduces variance, especially in a small image.
    assert!(
        (loss_full - loss_sampled).abs() < 1.5,
        "20% sampled MI ({}) should approximate full MI ({}), diff={}",
        loss_sampled,
        loss_full,
        (loss_full - loss_sampled).abs()
    );

    // Both should still detect high mutual information for identical images.
    assert!(
        loss_sampled < -0.3,
        "Sampled MI loss for identical images should still be significantly negative, got {}",
        loss_sampled
    );
}

// ===========================================================================
// 9. MI monotonicity with rotation
// ===========================================================================

#[test]
fn test_mi_monotonicity_with_rotation() {
    let size = 20;
    let blob = create_gaussian_blob(size, 1.0);
    let image = create_test_image(blob, [size, size, size]);

    let device = Default::default();
    let mattes = MutualInformation::<B>::new_mattes(32, 0.0, 1.0, &device);

    // Small rotation (5°)
    let small_angle = 5.0_f32.to_radians();
    let t_small = create_small_rotation_transform(small_angle);
    let loss_small = mattes.forward(&image, &image, &t_small).into_scalar();

    // Large rotation (30°)
    let large_angle = 30.0_f32.to_radians();
    let t_large = create_small_rotation_transform(large_angle);
    let loss_large = mattes.forward(&image, &image, &t_large).into_scalar();

    // Identity (no rotation)
    let loss_identity = mattes
        .forward(&image, &image, &create_identity_transform())
        .into_scalar();

    // More negative loss = higher MI.  Ordering: identity < small < large
    assert!(
        loss_identity < loss_small,
        "MI at identity ({}) should be higher than at 5° rotation ({})",
        loss_identity,
        loss_small
    );
    assert!(
        loss_small < loss_large,
        "MI at 5° rotation ({}) should be higher than at 30° rotation ({})",
        loss_small,
        loss_large
    );
}
