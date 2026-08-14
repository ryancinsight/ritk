#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;
use ritk_diffusion_scheme::{GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::{Point, Vector};

fn weighting(value: f64) -> DiffusionWeighting {
    DiffusionWeighting::from_seconds_per_square_millimeter(value).expect("finite weighting")
}

fn scheme(direction_count: usize) -> GradientScheme {
    let mut entries = vec![
        GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0])).expect("valid b0"),
    ];
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for index in 0..direction_count {
        let z = 1.0 - 2.0 * (index as f64 + 0.5) / direction_count as f64;
        let radius = (1.0 - z * z).sqrt();
        let phi = golden_angle * index as f64;
        entries.push(
            GradientDirection::new(
                weighting(3_000.0),
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .expect("unit Fibonacci direction"),
        );
    }
    GradientScheme::new(entries, GradientFrame::Lps).expect("valid scheme")
}

/// Signal of a single axially symmetric tensor aligned with `axis`.
fn tensor_signal(scheme: &GradientScheme, axis: [f64; 3], b0: f64, ad: f64, rd: f64) -> Vec<f64> {
    scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return b0;
            }
            let direction = entry.direction().to_array();
            let projection = direction
                .iter()
                .zip(axis)
                .map(|(left, right)| left * right)
                .sum::<f64>();
            let apparent = rd + (ad - rd) * projection.powi(2);
            b0 * (-b * apparent).exp()
        })
        .collect()
}

/// Multi-fibre signal: weighted sum of two single-fibre signals.
fn two_fibre_signal(
    scheme: &GradientScheme,
    axis_a: [f64; 3],
    axis_b: [f64; 3],
    fraction_a: f64,
) -> Vec<f64> {
    let sig_a = tensor_signal(scheme, axis_a, fraction_a, 0.0017, 0.0003);
    let sig_b = tensor_signal(scheme, axis_b, 1.0 - fraction_a, 0.0017, 0.0003);
    sig_a.iter().zip(sig_b.iter()).map(|(a, b)| a + b).collect()
}

// ── Response function tests ───────────────────────────────────────────────

#[test]
fn tensor_response_r0_is_unity() -> Result<(), CsdError> {
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    assert!((response.harmonics()[0] - 1.0).abs() < 1e-12);
    Ok(())
}

#[test]
fn tensor_response_r2_is_nonzero_for_anisotropic_tensor() -> Result<(), CsdError> {
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    // r_2 can be negative at high b-values because the perpendicular signal
    // (larger, P_2 < 0) dominates the parallel signal (smaller, P_2 > 0).
    assert!(response.harmonics()[1].abs() > 1e-12);
    Ok(())
}

#[test]
fn tensor_response_reconstruction_is_non_negative() -> Result<(), CsdError> {
    // The response function is a signal profile and must be non-negative
    // everywhere.  Reconstruct it from the Legendre coefficients at a
    // dense set of angles and assert non-negativity.
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let harmonics = response.harmonics();
    const N_THETA: usize = 200;
    for i in 0..N_THETA {
        let theta = std::f64::consts::PI * (i as f64 + 0.5) / N_THETA as f64;
        let cos_theta = theta.cos();
        let mut reconstruction = 0.0;
        for (index, &r_l) in harmonics.iter().enumerate() {
            let degree = index * 2;
            reconstruction += r_l * legendre_p(degree, cos_theta);
        }
        assert!(
            reconstruction >= -1e-10,
            "response reconstruction negative ({reconstruction}) at θ = {theta}"
        );
    }
    Ok(())
}

#[test]
fn isotropic_tensor_response_all_zeros_beyond_r0() -> Result<(), CsdError> {
    // Isotropic tensor (ad == rd).  Numerical quadrature with 512 samples
    // leaves sub-1e-4 residual; 1e-4 is the verified tolerance.
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0017, 4)?;
    assert!((response.harmonics()[0] - 1.0).abs() < 1e-12);
    for &r in &response.harmonics()[1..] {
        assert!(r.abs() < 2e-5, "isotropic r_l must be near zero, got {r}");
    }
    Ok(())
}

#[test]
fn response_new_rejects_non_unity_r0() {
    let err = ResponseFunction::new(vec![0.5, 0.3, 0.1]).unwrap_err();
    assert!(matches!(err, CsdError::InvalidR0 { value: 0.5 }));
}

// ── CSD fODF estimation tests ────────────────────────────────────────────

#[test]
fn single_fibre_recovers_peak_on_axis() -> Result<(), CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [1.0, 0.0, 0.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    assert!(fod.nnls_converged());
    let x = fod.evaluate_at_direction([1.0, 0.0, 0.0])?;
    let y = fod.evaluate_at_direction([0.0, 1.0, 0.0])?;
    assert!(x > y, "fODF({x}) on axis must exceed fODF({y}) off axis");
    Ok(())
}

#[test]
fn single_fibre_antipodal_symmetry() -> Result<(), CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [0.0, 0.0, 1.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    let pos = fod.evaluate_at_direction([0.0, 0.0, 1.0])?;
    let neg = fod.evaluate_at_direction([0.0, 0.0, -1.0])?;
    assert!((pos - neg).abs() < 1e-12);
    Ok(())
}

#[test]
fn all_coefficients_are_non_negative() -> Result<(), CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [1.0, 0.0, 0.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    for (i, &c) in fod.coefficients().iter().enumerate() {
        assert!(c >= 0.0, "coefficient {i} is negative: {c}");
    }
    Ok(())
}

#[test]
fn two_fibre_crossing_has_two_peaks() -> Result<(), CsdError> {
    let scheme = scheme(120);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let axis_a = [1.0_f64, 0.0, 0.0];
    let axis_b = [0.0_f64, 1.0, 0.0];
    let signals = two_fibre_signal(&scheme, axis_a, axis_b, 0.5);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    let fod_x = fod.evaluate_at_direction(axis_a)?;
    let fod_y = fod.evaluate_at_direction(axis_b)?;
    let fod_z = fod.evaluate_at_direction([0.0, 0.0, 1.0])?;
    assert!(
        fod_x > fod_z,
        "crossing peak x {fod_x} must exceed z {fod_z}"
    );
    assert!(
        fod_y > fod_z,
        "crossing peak y {fod_y} must exceed z {fod_z}"
    );
    Ok(())
}

#[test]
fn constant_signal_fits_isotropic_fod() -> Result<(), CsdError> {
    // When all signals are equal, the normalised signal is ~1 everywhere.
    // The only degree that can represent a constant is l=0, so c_0 should
    // absorb all weight and higher-degree coefficients be negligible.
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = vec![1.0; scheme.len()];
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;
    assert!(fod.coefficients()[0] > 0.0);
    for (i, &c) in fod.coefficients().iter().enumerate().skip(1) {
        assert!(c.abs() < 1e-10, "coefficient {i} should be zero, got {c}");
    }
    Ok(())
}

#[test]
fn response_degree_too_low_errors() -> Result<(), CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 4)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [1.0, 0.0, 0.0], 1.0, 0.0017, 0.0003);
    let err = estimate_fod(&scheme, &signals, &response, &config).unwrap_err();
    assert!(matches!(err, CsdError::ResponseDegreeTooLow { .. }));
    Ok(())
}

#[test]
fn response_higher_degree_is_accepted() -> Result<(), CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(4, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [1.0, 0.0, 0.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;
    assert!(fod.nnls_converged());
    Ok(())
}

#[test]
fn signal_length_mismatch_errors() {
    let scheme = scheme(30);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8).unwrap();
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default()).unwrap();
    let err = estimate_fod(&scheme, &[1.0; 5], &response, &config).unwrap_err();
    assert!(matches!(err, CsdError::SignalLengthMismatch { .. }));
}

#[test]
fn grid_is_flat_and_finite() -> Result<(), CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [1.0, 0.0, 0.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;
    let grid = fod.evaluate_on_grid(8, 16)?;
    assert_eq!(grid.shape(), [8, 16]);
    assert_eq!(grid.values().len(), 128);
    assert!(grid.values().iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn default_config_is_valid() {
    let config = CsdConfig::default();
    assert_eq!(config.l_max(), 8);
    assert!(config.nnls_config().max_iterations > 0);
}

// ── Peak extraction tests ────────────────────────────────────────────────

#[test]
fn single_fibre_peak_agrees_with_evaluation() -> Result<(), CsdError> {
    // Use a z-axis fibre — the pole is away from the antipodal-symmetric
    // ringing region and the peak finder reliably recovers it.
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [0.0, 0.0, 1.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    let peaks = fod.find_peaks(50, 100, 0.1)?;
    assert!(!peaks.is_empty(), "must find at least one peak");
    // The strongest peak direction should approximate (0,0,±1).
    let abs_z = peaks[0].direction[2].abs();
    assert!(abs_z > 0.8, "strongest peak z={abs_z} should be near ±1");
    Ok(())
}

#[test]
fn two_fibre_fod_has_two_peaks() -> Result<(), CsdError> {
    let scheme = scheme(120);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let axis_a = [1.0_f64, 0.0, 0.0];
    let axis_b = [0.0_f64, 1.0, 0.0];
    let signals = two_fibre_signal(&scheme, axis_a, axis_b, 0.5);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    let peaks = fod.find_peaks(50, 100, 0.1)?;
    assert!(
        peaks.len() >= 2,
        "expected at least 2 peaks, got {}",
        peaks.len()
    );
    // The first two peaks should align with the two fibre axes.
    let dots: Vec<f64> = peaks
        .iter()
        .take(2)
        .map(|peak| {
            let dx = peak.direction[0].abs();
            let dy = peak.direction[1].abs();
            f64::max(dx, dy)
        })
        .collect();
    for &dot in &dots {
        assert!(dot > 0.85, "peak should align with x or y axis, got {dot}");
    }
    Ok(())
}

#[test]
fn relative_threshold_discards_weak_peaks() -> Result<(), CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [1.0, 0.0, 0.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    let all_peaks = fod.find_peaks(50, 100, 0.0)?;
    let strict_peaks = fod.find_peaks(50, 100, 0.99)?;
    // Stricter threshold should not return more peaks.
    assert!(
        strict_peaks.len() <= all_peaks.len(),
        "stricter threshold gave {} peaks, relaxed gave {}",
        strict_peaks.len(),
        all_peaks.len()
    );
    Ok(())
}

#[test]
fn peak_amplitudes_are_sorted_descending() -> Result<(), CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [1.0, 0.0, 0.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    let peaks = fod.find_peaks(50, 100, 0.0)?;
    for window in peaks.windows(2) {
        assert!(window[0].amplitude >= window[1].amplitude);
    }
    Ok(())
}

// === ADR 0036 verification condition 7: gradient reorientation ===

fn rotation_y(angle: f64) -> [[f64; 3]; 3] {
    let (s, c) = angle.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

#[test]
fn reorient_gradients_recovers_original_peak_skip_reorientation_gives_wrong_peak()
-> Result<(), CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let original_peak = [0.0, 0.0, 1.0];

    let angle = std::f64::consts::FRAC_PI_4;
    let rotation = rotation_y(angle);
    let scheme_reoriented = scheme.reorient(rotation).expect("valid rotation");

    let signals = tensor_signal(&scheme_reoriented, original_peak, 1.0, 0.0017, 0.0003);

    // With reorientation: fit with reoriented scheme recovers original peak.
    let fod_correct = estimate_fod(&scheme_reoriented, &signals, &response, &config)?;
    let peaks_correct = fod_correct.find_peaks(50, 100, 0.1)?;
    assert!(
        !peaks_correct.is_empty(),
        "must find peak with reorientation"
    );
    let abs_z = peaks_correct[0].direction[2].abs();
    assert!(
        abs_z > 0.8,
        "with reorientation peak z={abs_z} should be near +-1"
    );

    // Without reorientation: fit with original scheme gives wrong peak.
    let fod_wrong = estimate_fod(&scheme, &signals, &response, &config)?;
    let peaks_wrong = fod_wrong.find_peaks(50, 100, 0.1)?;
    assert!(
        !peaks_wrong.is_empty(),
        "must find peak without reorientation"
    );
    // Peak should be rotated away from z-axis: PEV = R^T.[0,0,1] = [-r2,0,r2].
    let abs_z_wrong = peaks_wrong[0].direction[2].abs();
    assert!(
        abs_z_wrong < 0.9,
        "without reorientation peak z={abs_z_wrong} must deviate from +-1"
    );
    let r2 = std::f64::consts::SQRT_2 / 2.0;
    let rotated_dir = [-r2, 0.0, r2];
    let dot = (peaks_wrong[0].direction[0] * rotated_dir[0]
        + peaks_wrong[0].direction[1] * rotated_dir[1]
        + peaks_wrong[0].direction[2] * rotated_dir[2])
        .abs();
    assert!(
        dot > 0.9,
        "without reorientation peak must align with [-r2,0,r2]; dot={dot:.4}"
    );

    Ok(())
}

// ── FodVolume tests ───────────────────────────────────────────────────────

/// Build a minimal 2×2×2 volume with the z-aligned single-fibre fODF
/// replicated at every voxel.
fn two_by_two_z_fibre_volume() -> Result<FodVolume, CsdError> {
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [0.0, 0.0, 1.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    let nc = fod.coefficients().len();
    let n_voxels = 8; // 2×2×2
    let mut flat = Vec::with_capacity(n_voxels * nc);
    for _ in 0..n_voxels {
        flat.extend_from_slice(fod.coefficients());
    }
    let basis = RealSphericalHarmonicBasis::new(8).expect("valid basis");
    FodVolume::new(
        flat.into_boxed_slice(),
        [2, 2, 2],
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        basis,
        GradientFrame::Lps,
    )
}

#[test]
fn volume_construction_rejects_every_invalid_input() {
    let basis = RealSphericalHarmonicBasis::new(8).expect("valid basis");
    let nc = basis.num_coefficients();

    // Zero dimension.
    let err = FodVolume::new(
        vec![0.0; 8 * nc].into_boxed_slice(),
        [0, 2, 2],
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        basis.clone(),
        GradientFrame::Lps,
    )
    .unwrap_err();
    assert!(matches!(err, CsdError::VolumeShapeEmpty { .. }));

    // Coefficient count mismatch.
    let err = FodVolume::new(
        vec![0.0; 4 * nc].into_boxed_slice(),
        [2, 2, 2],
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        basis.clone(),
        GradientFrame::Lps,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        CsdError::VolumeCoefficientCountMismatch { .. }
    ));

    // Invalid spacing.
    for bad_spacing in [[-1.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, f64::NAN]] {
        let err = FodVolume::new(
            vec![0.0; 8 * nc].into_boxed_slice(),
            [2, 2, 2],
            bad_spacing,
            [0.0, 0.0, 0.0],
            basis.clone(),
            GradientFrame::Lps,
        )
        .unwrap_err();
        assert!(matches!(err, CsdError::VolumeSpacingInvalid { .. }));
    }

    // Invalid origin.
    let err = FodVolume::new(
        vec![0.0; 8 * nc].into_boxed_slice(),
        [2, 2, 2],
        [2.0, 2.0, 2.0],
        [0.0, f64::INFINITY, 0.0],
        basis.clone(),
        GradientFrame::Lps,
    )
    .unwrap_err();
    assert!(matches!(err, CsdError::VolumeOriginInvalid { .. }));
}

#[test]
fn interpolation_at_voxel_centre_recovers_exact_coefficients() -> Result<(), CsdError> {
    let volume = two_by_two_z_fibre_volume()?;

    let centre = Point::new([0.0, 0.0, 0.0]);
    let interp = volume
        .interpolate_coefficients_at(&centre)
        .expect("in bounds");
    let nc = volume.coefficient_count();

    for (c, value) in interp.iter().enumerate().take(nc) {
        let delta = (*value
            - volume
                .coefficients
                .as_ref()
                .get(c)
                .copied()
                .unwrap_or(f64::NAN))
        .abs();
        assert!(delta < 1e-12, "coefficient {c} delta {delta} too large");
    }
    Ok(())
}

#[test]
fn interpolation_at_midpoint_averages_two_voxels() -> Result<(), CsdError> {
    let volume = two_by_two_z_fibre_volume()?;
    let nc = volume.coefficient_count();

    let mid = Point::new([1.0, 1.0, 1.0]);
    let interp = volume.interpolate_coefficients_at(&mid).expect("in bounds");

    for (c, value) in interp.iter().enumerate().take(nc) {
        let delta = (*value
            - volume
                .coefficients
                .as_ref()
                .get(c)
                .copied()
                .unwrap_or(f64::NAN))
        .abs();
        assert!(
            delta < 1e-12,
            "coefficient {c} at midpoint delta {delta} too large"
        );
    }
    Ok(())
}

#[test]
fn interpolation_outside_volume_returns_none() {
    let volume = two_by_two_z_fibre_volume().expect("valid volume");

    for bad_point in [
        Point::new([-1.1, 1.0, 1.0]),
        Point::new([3.1, 1.0, 1.0]),
        Point::new([1.0, -1.1, 1.0]),
        Point::new([1.0, 1.0, 3.1]),
        Point::new([f64::NAN, 1.0, 1.0]),
    ] {
        assert!(
            volume.interpolate_coefficients_at(&bad_point).is_none(),
            "point {bad_point:?} should be out of bounds"
        );
    }
}

#[test]
fn direction_at_recovers_z_axis_in_homogeneous_volume() -> Result<(), CsdError> {
    let volume = two_by_two_z_fibre_volume()?;

    let dir = volume
        .direction_at(&Point::new([1.0, 1.0, 1.0]), 50, 100, 0.1)
        .expect("direction inside volume");
    let abs_z = dir.to_array()[2].abs();
    assert!(abs_z > 0.8, "interpolated peak z={abs_z} must be near ±1");
    Ok(())
}

#[test]
fn shape_is_ordered_x_fastest_not_like_image_shape() -> Result<(), CsdError> {
    // Pins the axis-order contract, as `noddi::tests` does for its volume. A
    // non-cubic grid is required: on a cubic one a transposed shape still
    // indexes in bounds and the error is silent, which is the failure this
    // guards against.
    //
    // shape [nx, ny, nz] = [4, 2, 1], storage z-slowest, so the voxel at x = 3
    // is flat index 3. A real fODF is placed there and nowhere else: reading
    // the shape slowest-first would make x = 3 out of bounds and z = 3 in
    // bounds, reversing both answers below.
    let scheme = scheme(60);
    let response = ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)?;
    let config = CsdConfig::new(8, weighting(50.0), NnlsConfig::default())?;
    let signals = tensor_signal(&scheme, [0.0, 0.0, 1.0], 1.0, 0.0017, 0.0003);
    let fod = estimate_fod(&scheme, &signals, &response, &config)?;

    let nc = fod.coefficients().len();
    // Bound to SHAPE so the buffer and the declared shape cannot disagree.
    const SHAPE: [usize; 3] = [4, 2, 1];
    let mut flat = vec![0.0_f64; SHAPE.iter().product::<usize>() * nc];
    flat[3 * nc..4 * nc].copy_from_slice(fod.coefficients());

    let basis = RealSphericalHarmonicBasis::new(8).expect("valid basis");
    let volume = FodVolume::new(
        flat.into_boxed_slice(),
        SHAPE,
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        basis,
        GradientFrame::Lps,
    )?;

    assert_eq!(
        volume.shape(),
        SHAPE,
        "the shape is reported in the order it was given"
    );

    let peak = volume
        .direction_at(&ritk_spatial::Point::new([3.0, 0.0, 0.0]), 50, 100, 0.1)
        .expect("x = 3 holds the fODF and is inside a volume 4 wide in x");
    assert!(
        peak.to_array()[2].abs() > 0.9,
        "the z-aligned fibre must be recovered at x = 3, got {peak:?}"
    );

    assert!(
        volume
            .direction_at(&ritk_spatial::Point::new([0.0, 0.0, 3.0]), 50, 100, 0.1)
            .is_none(),
        "z = 3 is outside a volume 1 deep in z; a transposed shape would admit it"
    );
    Ok(())
}
