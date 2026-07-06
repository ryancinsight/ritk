//! Unit tests for global MI registration: config, intensity, matrix, and convergence.

mod extended;
mod integration;

use super::config::{GlobalMiConfig, GlobalMiTransformType};
use super::registration::GlobalMiRegistration;
use super::transforms::{
    compute_image_center, estimate_intensity_range, rigid_matrix_to_homogeneous,
    translation_matrix_to_homogeneous,
};
use crate::optimizer::RegularStepGdConfig;
use crate::optimizer::{HistoryPolicy, PopulationEval};
use ritk_image::burn::backend::Autodiff;
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_filter::GaussianSigma;
use ritk_image::tensor::{Backend, Shape, Tensor, TensorData};
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::{RigidTransform, TranslationTransform};

pub(super) type TestBackend = Autodiff<NdArray<f32>>;

/// Create a 3D Gaussian blob image: I(x,y,z) = 255·exp(−||pos−center||²/(2σ²)).
pub(super) fn make_gaussian_blob(
    shape: [usize; 3],
    center: [f32; 3],
    sigma: f32,
    device: &<TestBackend as Backend>::Device,
) -> Image<TestBackend, 3> {
    let n = shape[0] * shape[1] * shape[2];
    let mut data = vec![0.0f32; n];
    let sigma_sq = sigma * sigma;
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let dz = z as f32 - center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                data[z * shape[1] * shape[2] + y * shape[2] + x] =
                    255.0 * (-dist_sq / (2.0 * sigma_sq)).exp();
            }
        }
    }
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(shape)), device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

/// Create a 3D ellipsoid: I = 255·max(0, 1 − ((x−cx)/a)² − ((y−cy)/b)² − ((z−cz)/c)²).
pub(super) fn make_ellipsoid(
    shape: [usize; 3],
    center: [f32; 3],
    scales: [f32; 3],
    device: &<TestBackend as Backend>::Device,
) -> Image<TestBackend, 3> {
    let n = shape[0] * shape[1] * shape[2];
    let mut data = vec![0.0f32; n];
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let dx = (x as f32 - center[0]) / scales[0];
                let dy = (y as f32 - center[1]) / scales[1];
                let dz = (z as f32 - center[2]) / scales[2];
                let val = 1.0 - dx * dx - dy * dy - dz * dz;
                data[z * shape[1] * shape[2] + y * shape[2] + x] = 255.0 * val.max(0.0);
            }
        }
    }
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(shape)), device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

// ── Configuration Tests ───────────────────────────────────────────────────────

#[test]
fn config_default_validates() {
    assert!(GlobalMiConfig::default().validate().is_ok());
}

#[test]
fn config_rigid_default_validates() {
    let cfg = GlobalMiConfig::rigid_default();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.transform_type, GlobalMiTransformType::Rigid);
}

#[test]
fn config_affine_default_validates() {
    let cfg = GlobalMiConfig::affine_default();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.transform_type, GlobalMiTransformType::Affine);
}

#[test]
fn config_translation_default_validates() {
    let cfg = GlobalMiConfig::translation_default();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.transform_type, GlobalMiTransformType::Translation);
}

#[test]
fn config_rejects_zero_levels() {
    let mut cfg = GlobalMiConfig::default();
    cfg.num_levels = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_mismatched_shrink_factors() {
    let mut cfg = GlobalMiConfig::default();
    cfg.shrink_factors = vec![4, 2];
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_too_few_mi_bins() {
    let mut cfg = GlobalMiConfig::default();
    cfg.num_mi_bins = 3;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_rejects_zero_sampling_percentage() {
    let mut cfg = GlobalMiConfig::default();
    cfg.sampling_percentage = 0.0;
    assert!(cfg.validate().is_err());
}

// ── Intensity Range Tests ─────────────────────────────────────────────────────

#[test]
fn intensity_range_estimation_gaussian() {
    let device = Default::default();
    let image = make_gaussian_blob([16, 16, 16], [8.0, 8.0, 8.0], 3.0, &device);
    let (min_val, max_val) = estimate_intensity_range(&image);
    assert!(max_val > 250.0, "max should be near 255, got {max_val}");
    assert!(min_val < 1.0, "min should be near 0, got {min_val}");
}

#[test]
fn intensity_range_adds_margin() {
    let device = Default::default();
    let image = make_gaussian_blob([16, 16, 16], [8.0, 8.0, 8.0], 3.0, &device);
    let (min_val, max_val) = estimate_intensity_range(&image);
    assert!(
        min_val < 0.0,
        "min with margin should be < 0, got {min_val}"
    );
    assert!(
        max_val > 255.0,
        "max with margin should be > 255, got {max_val}"
    );
}

// ── Matrix Extraction Tests ───────────────────────────────────────────────────

#[test]
fn rigid_identity_matrix_is_identity_homogeneous() {
    let device = Default::default();
    let center =
        Tensor::<TestBackend, 1>::from_data(TensorData::from([16.0f32, 16.0, 16.0]), &device);
    let transform = RigidTransform::<TestBackend, 3>::identity(Some(center), &device);
    let matrix = rigid_matrix_to_homogeneous(&transform);
    assert!((matrix.0[0] - 1.0).abs() < 1e-3, "R[0,0]={}", matrix.0[0]);
    assert!((matrix.0[5] - 1.0).abs() < 1e-3, "R[1,1]={}", matrix.0[5]);
    assert!((matrix.0[10] - 1.0).abs() < 1e-3, "R[2,2]={}", matrix.0[10]);
    assert!((matrix.0[15] - 1.0).abs() < 1e-6, "R[3,3]={}", matrix.0[15]);
}

#[test]
fn translation_identity_matrix_is_identity_homogeneous() {
    let device = Default::default();
    let transform =
        TranslationTransform::<TestBackend, 3>::new(Tensor::<TestBackend, 1>::zeros([3], &device));
    let matrix = translation_matrix_to_homogeneous(&transform);
    let identity: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    for (i, (&got, &expected)) in matrix.0.iter().zip(identity.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-9,
            "Matrix[{i}]: expected {expected}, got {got}"
        );
    }
}

// ── Center Computation Test ───────────────────────────────────────────────────

#[test]
fn image_center_computation_is_correct() {
    let device = Default::default();
    let image = make_gaussian_blob([32, 32, 32], [16.0, 16.0, 16.0], 3.0, &device);
    let center = compute_image_center(&image);
    assert!((center[0] - 16.0).abs() < 0.1, "Center X={}", center[0]);
    assert!((center[1] - 16.0).abs() < 0.1, "Center Y={}", center[1]);
    assert!((center[2] - 16.0).abs() < 0.1, "Center Z={}", center[2]);
}

// ── CmaMiConfig Preset Tests ─────────────────────────────────────────────────

#[test]
fn cma_mi_brain_rigid_default_has_expected_fields() {
    use super::cma_mi::CmaMiConfig;
    use super::cma_mi::InitStrategy;
    let cfg = CmaMiConfig::brain_rigid_default();
    assert_eq!(cfg.coarse_shrink, 8);
    assert!((cfg.sampling_percentage - 0.30).abs() < 1e-6);
    assert_eq!(cfg.init_strategy, InitStrategy::Manual);
    assert_eq!(cfg.ipop_restarts, 0);
    assert!(cfg.shrink_per_axis.is_none());
    assert!((cfg.cma_config.sigma0 - 0.7).abs() < 1e-9);
    assert_eq!(cfg.cma_config.max_generations, 200);
}

#[test]
fn cma_mi_fast_exploratory_uses_coarser_pyramid() {
    use super::cma_mi::CmaMiConfig;
    let fast = CmaMiConfig::fast_exploratory();
    let brain = CmaMiConfig::brain_rigid_default();

    // Fast exploratory must be coarser
    assert!(
        fast.coarse_shrink >= brain.coarse_shrink,
        "fast_exploratory shrink ({}) should be >= brain_rigid_default shrink ({})",
        fast.coarse_shrink,
        brain.coarse_shrink,
    );

    // And use fewer generations
    assert!(
        fast.cma_config.max_generations <= brain.cma_config.max_generations,
        "fast_exploratory max_gen ({}) should be <= brain_rigid_default ({})",
        fast.cma_config.max_generations,
        brain.cma_config.max_generations,
    );

    // Wider search range
    assert!(
        fast.translation_range_mm >= brain.translation_range_mm,
        "fast_exploratory translation_range ({} mm) should be >= brain ({}mm)",
        fast.translation_range_mm,
        brain.translation_range_mm,
    );
}

#[test]
fn cma_mi_brain_rigid_default_uses_nmi() {
    use super::cma_mi::CmaMiConfig;
    use crate::metric::{MutualInformationVariant, NormalizationMethod};

    let cfg = CmaMiConfig::brain_rigid_default();

    // NMI (AverageEntropy) is immune to OOB zero-pad artefact during rotation.
    assert_eq!(
        cfg.mi_variant,
        MutualInformationVariant::Normalized(NormalizationMethod::AverageEntropy),
        "brain_rigid_default should use NMI(AverageEntropy)"
    );

    // No cascade schedule — uses single-level path.
    assert!(
        cfg.pyramid_schedule.is_empty(),
        "brain_rigid_default should not use pyramid_schedule"
    );
}

#[test]
fn cma_mi_default_uses_mattes_no_schedule() {
    use super::cma_mi::CmaMiConfig;
    use crate::metric::MutualInformationVariant;

    let cfg = CmaMiConfig::default();
    assert_eq!(cfg.mi_variant, MutualInformationVariant::Mattes);
    assert!(cfg.pyramid_schedule.is_empty());
}

#[test]
fn cma_mi_multiscale_has_three_levels() {
    use super::cma_mi::CmaMiConfig;
    use super::cma_mi::InitStrategy;

    let cfg = CmaMiConfig::brain_rigid_multiscale();
    assert_eq!(
        cfg.pyramid_schedule.len(),
        3,
        "brain_rigid_multiscale must have 3 levels"
    );

    // Levels must go coarse → medium → fine (shrink decreasing).
    let shrinks: Vec<usize> = cfg.pyramid_schedule.iter().map(|l| l.shrink).collect();
    assert!(
        shrinks[0] > shrinks[1] && shrinks[1] > shrinks[2],
        "Pyramid levels should be in decreasing shrink order: {:?}",
        shrinks
    );

    // σ₀ should decrease level by level (narrowing the search).
    let sigmas: Vec<f64> = cfg.pyramid_schedule.iter().map(|l| l.cma_sigma0).collect();
    assert!(
        sigmas[0] > sigmas[1] && sigmas[1] > sigmas[2],
        "Pyramid level sigma0 should decrease coarse→fine: {:?}",
        sigmas
    );

    // NMI should be used across all levels.
    use crate::metric::{MutualInformationVariant, NormalizationMethod};
    assert_eq!(
        cfg.mi_variant,
        MutualInformationVariant::Normalized(NormalizationMethod::AverageEntropy),
    );

    // CoM init must be disabled for CT↔MRI.
    assert_eq!(cfg.init_strategy, InitStrategy::Manual);
}

#[test]
fn cma_mi_level_config_new_sets_defaults() {
    use super::cma_mi::CmaMiLevelConfig;

    let level = CmaMiLevelConfig::new(8, GaussianSigma::new_unchecked(4.0), 0.5, 100);
    assert_eq!(level.shrink, 8);
    assert!((level.sigma_mm.get() - 4.0).abs() < 1e-9);
    assert!((level.cma_sigma0 - 0.5).abs() < 1e-9);
    assert_eq!(level.max_generations, 100);
    assert_eq!(level.lambda, 0, "lambda should default to 0 (auto)");
    assert_eq!(level.ipop_restarts, 0);
    assert!(level.shrink_per_axis.is_none());
}

#[test]
fn cma_mi_thin_slab_ct_uses_anisotropic_shrink() {
    use super::cma_mi::CmaMiConfig;

    let cfg = CmaMiConfig::thin_slab_ct_default();
    let [sz, sy, sx] = cfg
        .shrink_per_axis
        .expect("thin_slab should have shrink_per_axis");

    // z-axis must not be downsampled (sz = 1)
    assert_eq!(sz, 1, "thin_slab_ct shrink z must be 1 (preserve z-slices)");

    // xy must be downsampled more aggressively
    assert!(sy > 1, "thin_slab_ct shrink y must be > 1");
    assert!(sx > 1, "thin_slab_ct shrink x must be > 1");
}

// ── Sprint 290: Brain Masking Tests ───────────────────────────────────────

/// Create a 3-D binary mask image (1.0 inside a box, 0.0 outside).
pub(super) fn make_box_mask(
    shape: [usize; 3],
    // box corners in voxel space: [z_lo..z_hi, y_lo..y_hi, x_lo..x_hi]
    z_range: std::ops::Range<usize>,
    y_range: std::ops::Range<usize>,
    x_range: std::ops::Range<usize>,
    device: &<TestBackend as Backend>::Device,
) -> Image<TestBackend, 3> {
    let n = shape[0] * shape[1] * shape[2];
    let mut data = vec![0.0f32; n];
    for z in z_range {
        for y in y_range.clone() {
            for x in x_range.clone() {
                data[z * shape[1] * shape[2] + y * shape[2] + x] = 1.0;
            }
        }
    }
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(shape)), device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

#[test]
fn cma_mi_register_rigid_with_mask_accepts_full_foreground_mask() {
    // Smoke test: register_rigid_with_mask with an all-ones mask must not panic
    // and must return the same type as register_rigid.
    use super::cma_mi::{CmaMiConfig, CmaMiRegistration};

    let device = Default::default();
    let shape = [8, 8, 8];
    let fixed = make_gaussian_blob(shape, [4.0, 4.0, 4.0], 2.0, &device);
    let moving = make_gaussian_blob(shape, [4.0, 4.0, 4.0], 2.0, &device);

    // All-ones mask — all voxels are foreground.
    let mask = make_box_mask(shape, 0..8, 0..8, 0..8, &device);

    let config = CmaMiConfig {
        cma_config: crate::optimizer::CmaEsConfig {
            sigma0: 0.3,
            lambda: 0,
            max_generations: 2, // minimal — just checking it runs
            sigma_tol: 1e-8,
            ftol: f64::NEG_INFINITY,
            seed: 42,
            parallel_population: PopulationEval::Sequential,
            record_history: HistoryPolicy::Discard,
        },
        coarse_shrink: 4,
        coarse_sigma_mm: GaussianSigma::new_unchecked(2.0),
        sampling_percentage: 0.50,
        ..CmaMiConfig::default()
    };

    // Must not panic; transform type check only.
    let (transform_no_mask, _) =
        CmaMiRegistration::register_rigid(&fixed, &moving, [0.0; 3], None, &config);
    let (transform_with_mask, _) = CmaMiRegistration::register_rigid_with_mask(
        &fixed,
        &moving,
        [0.0; 3],
        None,
        &config,
        Some(&mask),
    );

    // Both should produce finite translation vectors.
    let t_no_mask = transform_no_mask
        .translation()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let t_with_mask = transform_with_mask
        .translation()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    for v in t_no_mask.iter().chain(t_with_mask.iter()) {
        assert!(
            v.is_finite(),
            "translation component must be finite, got {v}"
        );
    }
}
