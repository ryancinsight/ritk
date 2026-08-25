//! NODDI Watson-dispersion model verification against synthetic oracles.
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::*;
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::{Point, Vector};

fn multi_shell_scheme(b0_count: usize, dir_count: usize, b_values: &[f64]) -> GradientScheme {
    let mut entries: Vec<GradientDirection> = Vec::new();
    let b0_weighting = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let zero_dir = Vector::new([0.0, 0.0, 0.0]);
    for _ in 0..b0_count {
        entries.push(GradientDirection::new(b0_weighting, zero_dir).expect("b0 entry"));
    }

    let phi_golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for i in 0..dir_count {
        let z = 1.0 - (2.0 * i as f64 + 1.0) / dir_count as f64;
        let radius = (1.0 - z * z).sqrt();
        let phi = phi_golden * i as f64;
        let g = Vector::new([radius * phi.cos(), radius * phi.sin(), z]);

        for &b in b_values {
            let weighting =
                DiffusionWeighting::from_seconds_per_square_millimeter(b).expect("finite b");
            entries.push(GradientDirection::new(weighting, g).expect("weighted direction"));
        }
    }

    GradientScheme::new(entries, GradientFrame::ImageAxis).expect("valid scheme")
}

/// Generate noiseless NODDI signals with Watson dispersion.
fn noddi_signals(
    scheme: &GradientScheme,
    s0: f64,
    f_intra: f64,
    f_iso: f64,
    odi: f64,
    direction: [f64; 3],
) -> Vec<f64> {
    let kappa = odi_to_kappa(odi);
    let quad = quadrature_sphere();
    scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return s0;
            }
            let g = entry.direction().to_array();
            let a_ic = watson_stick(b, g, direction, kappa, quad);
            let a_ec = (-b * D_EXTRA).exp();
            let a_iso = (-b * D_ISO).exp();
            s0 * ((1.0 - f_iso) * (f_intra * a_ic + (1.0 - f_intra) * a_ec) + f_iso * a_iso)
        })
        .collect()
}

/// 3-shell × 30 dirs = richer kurtosis signal for NODDI compartment separation.
fn default_scheme() -> GradientScheme {
    multi_shell_scheme(4, 30, &[500.0, 1500.0, 3000.0])
}

fn default_config() -> NoddiConfig {
    NoddiConfig::default()
}

fn z_dir() -> [f64; 3] {
    [0.0, 0.0, 1.0]
}

// ── Dispersion edge cases ─────────────────────────────────────────────────────

#[test]
fn perfectly_aligned_stick_recovers_near_zero_odi() {
    let scheme = default_scheme();
    let s0 = 1000.0;
    let signals = noddi_signals(&scheme, s0, 0.7, 0.0, 0.03, z_dir());

    let fit = estimate_noddi(&scheme, &signals, &default_config())
        .expect("aligned stick NODDI fit should succeed");

    assert!(fit.converged());
    assert!(
        (fit.ndi() - 0.7).abs() < 0.15,
        "NDI mismatch: {:.3}",
        fit.ndi()
    );
    assert!(
        fit.odi() < 0.15,
        "ODI should be near 0 for aligned sticks, got {}",
        fit.odi()
    );
    assert!(fit.f_iso() < 0.1, "f_iso should be near 0");
}

#[test]
fn moderately_dispersed_stick_recovers_positive_odi() {
    let scheme = default_scheme();
    let s0 = 1000.0;
    // ODI = 0.3 → κ = 1/tan(π·0.3/2) ≈ 1/tan(0.471) ≈ 1/0.509 ≈ 1.96
    let signals = noddi_signals(&scheme, s0, 0.6, 0.0, 0.3, z_dir());

    let fit = estimate_noddi(&scheme, &signals, &default_config())
        .expect("moderately dispersed NODDI fit should succeed");

    assert!(fit.converged());
    assert!(
        fit.odi() > 0.1,
        "ODI should be positive for dispersed sticks, got {}",
        fit.odi()
    );
    // ODI recovery tolerance: quadrature and LM convergence limit precision.
    assert!(
        (fit.odi() - 0.3).abs() < 0.2,
        "ODI recovery: {:.3} vs 0.3",
        fit.odi()
    );
}

#[test]
fn highly_dispersed_stick_approaches_isotropic() {
    let scheme = default_scheme();
    let s0 = 1000.0;
    // ODI = 0.85 → near-isotropic dispersion.
    let signals = noddi_signals(&scheme, s0, 0.5, 0.0, 0.85, z_dir());

    let fit = estimate_noddi(&scheme, &signals, &default_config())
        .expect("highly dispersed NODDI fit should succeed");

    assert!(fit.converged());
    assert!(
        fit.odi() > 0.5,
        "ODI should be high for near-isotropic dispersion, got {}",
        fit.odi()
    );
}

// ── Compartment recovery ──────────────────────────────────────────────────────

#[test]
fn pure_csf_is_identified_correctly() {
    let scheme = default_scheme();
    let s0 = 1000.0;
    let signals = noddi_signals(&scheme, s0, 0.0, 1.0, 0.03, z_dir());

    let fit = estimate_noddi(&scheme, &signals, &default_config())
        .expect("pure CSF NODDI fit should succeed");

    assert!(fit.converged());
    assert!(
        fit.f_iso() > 0.7,
        "f_iso should be high for pure CSF, got {}",
        fit.f_iso()
    );
}

#[test]
fn pure_ball_recovers_zero_intra_and_low_csf() {
    let scheme = default_scheme();
    let s0 = 1000.0;
    let signals = noddi_signals(&scheme, s0, 0.0, 0.0, 0.03, z_dir());

    let fit = estimate_noddi(&scheme, &signals, &default_config())
        .expect("pure ball NODDI fit should succeed");

    assert!(fit.converged());
    assert!(fit.ndi() < 0.2, "NDI should be near 0, got {}", fit.ndi());
    assert!(
        fit.f_iso() < 0.2,
        "f_iso should be near 0, got {}",
        fit.f_iso()
    );
}

#[test]
fn direction_is_recovered_for_x_fibre() {
    let scheme = default_scheme();
    let s0 = 1000.0;
    let dir = [1.0_f64, 0.0, 0.0];
    let signals = noddi_signals(&scheme, s0, 0.7, 0.0, 0.05, dir);

    let fit = estimate_noddi(&scheme, &signals, &default_config())
        .expect("x-fibre NODDI fit should succeed");

    let fit_dir = fit.principal_direction();
    let abs_x = fit_dir[0].abs();
    assert!(abs_x > 0.96, "direction should be along x, got |x|={abs_x}");
}

#[test]
fn signal_prediction_round_trips() {
    let scheme = default_scheme();
    let s0 = 1000.0;
    let signals = noddi_signals(&scheme, s0, 0.6, 0.0, 0.15, z_dir());

    let fit =
        estimate_noddi(&scheme, &signals, &default_config()).expect("NODDI fit should succeed");

    for (idx, entry) in scheme.directions().iter().enumerate() {
        let b = entry.weighting().seconds_per_square_millimeter();
        let predicted = fit.predict_signal(entry.direction().to_array(), b);
        let rel_err = (predicted - signals[idx]).abs() / signals[idx].max(1.0);
        assert!(
            rel_err < 0.03,
            "signal[{idx}]: predicted={predicted:.2} true={:.2} rel_err={rel_err:.3e}",
            signals[idx]
        );
    }
}

// ── Compartment fraction sum ──────────────────────────────────────────────────

#[test]
fn compartment_fractions_sum_to_one() {
    let scheme = default_scheme();
    let s0 = 1000.0;
    let signals = noddi_signals(&scheme, s0, 0.5, 0.1, 0.1, z_dir());

    let fit =
        estimate_noddi(&scheme, &signals, &default_config()).expect("NODDI fit should succeed");

    let stick = fit.f_intra() * (1.0 - fit.f_iso());
    let ball = (1.0 - fit.f_intra()) * (1.0 - fit.f_iso());
    let csf = fit.f_iso();
    let total = stick + ball + csf;
    assert!(
        (total - 1.0).abs() < 1e-12,
        "compartments must sum to 1.0, got {total}"
    );
}

// ── Error cases ───────────────────────────────────────────────────────────────

#[test]
fn rejects_signal_length_mismatch() {
    let scheme = default_scheme();
    let signals = vec![1.0; scheme.len() - 1];
    let err = estimate_noddi(&scheme, &signals, &default_config()).unwrap_err();
    assert!(matches!(err, NoddiError::SignalLengthMismatch { .. }));
}

#[test]
fn rejects_non_finite_signal() {
    let scheme = default_scheme();
    let mut signals = vec![1.0; scheme.len()];
    signals[5] = f64::NAN;
    let err = estimate_noddi(&scheme, &signals, &default_config()).unwrap_err();
    assert!(matches!(err, NoddiError::NonFiniteSignal { .. }));
}

#[test]
fn rejects_all_dwi_scheme() {
    let mut entries = Vec::new();
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let z = Vector::new([0.0, 0.0, 0.0]);
    for _ in 0..4 {
        entries.push(GradientDirection::new(b0, z).unwrap());
    }
    let scheme = GradientScheme::new(entries, GradientFrame::ImageAxis).unwrap();
    let signals = vec![1.0; scheme.len()];
    let err = estimate_noddi(&scheme, &signals, &default_config()).unwrap_err();
    assert!(matches!(err, NoddiError::NoDwiDirections));
}

// ── ODF convergence to ball-stick ─────────────────────────────────────────────

#[test]
fn zero_odi_matches_single_stick_signal() {
    // At ODI ≈ 0 (κ → ∞), the Watson distribution collapses to a delta at μ,
    // so A_ic should match the single-stick signal exp(−b·d_‖·(g·μ)²).
    let quad = quadrature_sphere();
    let dir = [0.0, 0.0, 1.0];
    let g = [1.0_f64 / 2.0_f64.sqrt(), 0.0, 1.0 / 2.0_f64.sqrt()];
    let b = 1500.0;

    let watson = watson_stick(b, g, dir, odi_to_kappa(0.01), quad);
    let single = (-b * D_PARALLEL * dot3(g, dir).powi(2)).exp();

    assert!(
        (watson - single).abs() < 0.02,
        "zero-ODI Watson stick ({watson:.6}) must ≈ single stick ({single:.6})"
    );
}

// ── ADR 0036 verification condition 7: gradient reorientation ────────────

fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (s, c) = angle.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

#[test]
fn reorient_gradients_recovers_original_direction() {
    let scheme = default_scheme();
    let original_dir = z_dir();

    let angle = std::f64::consts::FRAC_PI_4;
    let rotation = rotation_y(angle);
    let scheme_reoriented = scheme.reorient(rotation).expect("valid rotation");

    let s0 = 1000.0;
    let signals = noddi_signals(&scheme_reoriented, s0, 0.7, 0.0, 0.1, original_dir);

    let fit_correct = estimate_noddi(&scheme_reoriented, &signals, &default_config())
        .expect("NODDI fit with reoriented scheme");
    let abs_z = fit_correct.principal_direction()[2].abs();
    assert!(
        abs_z > 0.96,
        "with reorientation |z| should be near 1, got {abs_z}"
    );

    let fit_wrong = estimate_noddi(&scheme, &signals, &default_config())
        .expect("NODDI fit without reorientation");
    let abs_z_wrong = fit_wrong.principal_direction()[2].abs();
    assert!(
        abs_z_wrong < 0.85,
        "without reorientation |z| should deviate, got {abs_z_wrong}"
    );
}

// ── NoddiVolume tests ────────────────────────────────────────────────────

/// Build a minimal 2×2×2 volume with z-aligned NODDI directions.
fn two_by_two_z_volume() -> Result<NoddiVolume, NoddiError> {
    let scheme = default_scheme();
    let s0 = 1000.0;
    let signals = noddi_signals(&scheme, s0, 0.7, 0.0, 0.03, z_dir());
    let fit = estimate_noddi(&scheme, &signals, &default_config())?;
    let dir = fit.principal_direction();

    let n_voxels = 8; // 2×2×2
    let mut flat = Vec::with_capacity(n_voxels * 3);
    for _ in 0..n_voxels {
        flat.extend_from_slice(&dir);
    }
    NoddiVolume::new(
        flat.into_boxed_slice(),
        [2, 2, 2],
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        GradientFrame::ImageAxis,
    )
}

#[test]
fn volume_construction_validates_inputs() {
    let dirs: Box<[f64]> = vec![0.0; 8 * 3].into_boxed_slice();

    // Zero shape.
    assert!(
        NoddiVolume::new(
            dirs.clone(),
            [0, 2, 2],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            GradientFrame::ImageAxis,
        )
        .is_err()
    );

    // Mismatched direction count.
    assert!(
        NoddiVolume::new(
            vec![0.0; 4 * 3].into_boxed_slice(),
            [2, 2, 2],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            GradientFrame::ImageAxis,
        )
        .is_err()
    );

    // Negative spacing.
    assert!(
        NoddiVolume::new(
            dirs.clone(),
            [2, 2, 2],
            [-1.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            GradientFrame::ImageAxis,
        )
        .is_err()
    );
}

#[test]
fn direction_at_voxel_centre_recovers_z_axis() -> Result<(), NoddiError> {
    let volume = two_by_two_z_volume()?;

    let dir = volume
        .direction_at(&Point::new([0.0, 0.0, 0.0]))
        .expect("in bounds");
    let abs_z = dir.to_array()[2].abs();
    assert!(
        abs_z > 0.98,
        "NODDI volume direction should be near z, got |z|={abs_z}"
    );
    Ok(())
}

#[test]
fn direction_at_outside_volume_returns_none() {
    let volume = two_by_two_z_volume().expect("valid volume");

    assert!(
        volume
            .direction_at(&Point::new([-10.0, 0.0, 0.0]))
            .is_none()
    );
    assert!(volume.direction_at(&Point::new([0.0, 0.0, 10.0])).is_none());
    assert!(
        volume
            .direction_at(&Point::new([f64::NAN, 0.0, 0.0]))
            .is_none()
    );
}

#[test]
fn shape_is_ordered_x_fastest_not_like_image_shape() {
    // Pins the axis-order contract. A non-cubic volume is required: on a cubic
    // grid a transposed shape still indexes in bounds and the error is silent,
    // which is exactly the failure this guards.
    //
    // shape [nx, ny, nz] = [4, 2, 1] with storage z-slowest means offset
    // = z*(ny*nx) + y*nx + x, so the voxel at x = 3 is flat index 3, and the
    // voxel at y = 1 is flat index 4.
    // Bound to SHAPE so the buffer and the declared shape cannot disagree.
    const SHAPE: [usize; 3] = [4, 2, 1];
    let mut flat = vec![0.0_f64; SHAPE.iter().product::<usize>() * 3];
    flat[3 * 3] = 1.0; // x = 3, y = 0: direction +x
    flat[4 * 3 + 1] = 1.0; // x = 0, y = 1: direction +y

    let volume = NoddiVolume::new(
        flat.into_boxed_slice(),
        SHAPE,
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        GradientFrame::ImageAxis,
    )
    .expect("valid volume");

    // Physical (3, 0, 0) is x = 3: the +x voxel.
    let at_x = volume
        .direction_at(&ritk_spatial::Point::new([3.0, 0.0, 0.0]))
        .expect("x = 3 is inside a volume 4 wide in x");
    assert!(
        at_x.to_array()[0].abs() > 0.999,
        "expected the +x voxel at x = 3, got {at_x:?} -- shape read slowest-first \
         would have placed it out of bounds"
    );

    // Physical (0, 1, 0) is y = 1: the +y voxel.
    let at_y = volume
        .direction_at(&ritk_spatial::Point::new([0.0, 1.0, 0.0]))
        .expect("y = 1 is inside a volume 2 deep in y");
    assert!(
        at_y.to_array()[1].abs() > 0.999,
        "expected the +y voxel at y = 1"
    );

    // The far corner of a transposed reading: x = 0, y = 0, z = 3 must be
    // outside, because nz is 1.
    assert!(
        volume
            .direction_at(&ritk_spatial::Point::new([0.0, 0.0, 3.0]))
            .is_none(),
        "z = 3 is outside a volume 1 deep in z; a transposed shape would admit it"
    );
}
