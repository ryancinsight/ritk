//! Unit tests for global MI registration: config, intensity, matrix, and convergence.

mod integration;

use super::config::{GlobalMiConfig, GlobalMiTransformType};
use super::registration::GlobalMiRegistration;
use super::transforms::{
    compute_image_center, estimate_intensity_range, rigid_matrix_to_homogeneous,
    translation_matrix_to_homogeneous,
};
use crate::optimizer::RegularStepGdConfig;
use burn::backend::Autodiff;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_core::transform::{RigidTransform, TranslationTransform};

pub(super) type TestBackend = Autodiff<NdArray<f32>>;

/// Create a 3D Gaussian blob image: I(x,y,z) = 255·exp(−||pos−center||²/(2σ²)).
pub(super) fn make_gaussian_blob(
    shape: [usize; 3],
    center: [f32; 3],
    sigma: f32,
    device: &<TestBackend as burn::tensor::backend::Backend>::Device,
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
    device: &<TestBackend as burn::tensor::backend::Backend>::Device,
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

    assert!((matrix[0] - 1.0).abs() < 1e-3, "R[0,0]={}", matrix[0]);
    assert!((matrix[5] - 1.0).abs() < 1e-3, "R[1,1]={}", matrix[5]);
    assert!((matrix[10] - 1.0).abs() < 1e-3, "R[2,2]={}", matrix[10]);
    assert!((matrix[15] - 1.0).abs() < 1e-6, "R[3,3]={}", matrix[15]);
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
    for (i, (&got, &expected)) in matrix.iter().zip(identity.iter()).enumerate() {
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

// ── Convergence History Test ──────────────────────────────────────────────────

#[test]
fn convergence_history_tracks_per_level() {
    let device = Default::default();
    let fixed = make_gaussian_blob([16, 16, 16], [8.0, 8.0, 8.0], 3.0, &device);
    let moving = make_gaussian_blob([16, 16, 16], [8.0, 8.0, 8.0], 3.0, &device);

    let initial_transform =
        TranslationTransform::<TestBackend, 3>::new(Tensor::<TestBackend, 1>::zeros([3], &device));

    let config = GlobalMiConfig {
        num_levels: 3,
        shrink_factors: vec![4, 2, 1],
        smoothing_sigmas: vec![4.0, 2.0, 0.0],
        num_mi_bins: 16,
        sampling_percentage: 0.30,
        rsgd_configs: vec![
            RegularStepGdConfig {
                initial_step_length: 1.0,
                relaxation_factor: 0.5,
                minimum_step_length: 1e-4,
                maximum_step_length: 5.0,
                gradient_tolerance: 1e-4,
                maximum_iterations: 30,
            },
            RegularStepGdConfig {
                initial_step_length: 0.5,
                relaxation_factor: 0.5,
                minimum_step_length: 1e-5,
                maximum_step_length: 2.0,
                gradient_tolerance: 1e-5,
                maximum_iterations: 30,
            },
            RegularStepGdConfig {
                initial_step_length: 0.2,
                relaxation_factor: 0.5,
                minimum_step_length: 1e-6,
                maximum_step_length: 1.0,
                gradient_tolerance: 1e-6,
                maximum_iterations: 30,
            },
        ],
        transform_type: GlobalMiTransformType::Translation,
        center: None,
    };

    let (_, result) = GlobalMiRegistration::register_translation_full(
        &fixed,
        &moving,
        initial_transform,
        &config,
    );

    assert_eq!(result.convergence_history.len(), 3);
    assert_eq!(result.iterations_per_level.len(), 3);
    for (level, &iters) in result.iterations_per_level.iter().enumerate() {
        assert!(
            iters > 0,
            "Level {} must have positive iterations",
            level + 1
        );
    }
}
