use super::*;
use ritk_diffusion_scheme::{GradientDirection, GradientFrame};
use ritk_spatial::Vector;

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
                weighting(1_000.0),
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .expect("unit Fibonacci direction"),
        );
    }
    GradientScheme::new(entries, GradientFrame::Lps).expect("valid scheme")
}

fn tensor_signal(scheme: &GradientScheme, axis: [f64; 3]) -> Vec<f64> {
    const PARALLEL_DIFFUSIVITY: f64 = 0.0017;
    const PERPENDICULAR_DIFFUSIVITY: f64 = 0.0003;
    scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return 1.0;
            }
            let direction = entry.direction().to_array();
            let projection = direction
                .iter()
                .zip(axis)
                .map(|(left, right)| left * right)
                .sum::<f64>();
            let apparent = PERPENDICULAR_DIFFUSIVITY
                + (PARALLEL_DIFFUSIVITY - PERPENDICULAR_DIFFUSIVITY) * projection.powi(2);
            (-b * apparent).exp()
        })
        .collect()
}

#[test]
fn funk_radon_legendre_factors_match_closed_forms() {
    assert_eq!(legendre_at_zero(0), 1.0);
    assert_eq!(legendre_at_zero(2), -0.5);
    assert_eq!(legendre_at_zero(4), 0.375);
    assert_eq!(legendre_at_zero(6), -0.3125);
}

#[test]
fn isotropic_signal_produces_constant_antipodal_odf() -> Result<(), OdfError> {
    let scheme = scheme(30);
    let signals = std::iter::once(1.0)
        .chain(std::iter::repeat_n(0.5, 30))
        .collect::<Vec<_>>();
    let odf = estimate_odf(&scheme, &signals, OdfConfig::default())?;
    let x = odf.evaluate_at_direction([1.0, 0.0, 0.0])?;
    let negative_x = odf.evaluate_at_direction([-1.0, 0.0, 0.0])?;
    let z = odf.evaluate_at_direction([0.0, 0.0, 1.0])?;
    assert!((x - negative_x).abs() < 1.0e-12);
    assert!(
        (x - z).abs() < 2.0e-3,
        "isotropic ODF differs by {}",
        (x - z).abs()
    );
    Ok(())
}

#[test]
fn tensor_phantom_odf_peaks_on_analytical_axis() -> Result<(), OdfError> {
    let scheme = scheme(60);
    let odf = estimate_odf(
        &scheme,
        &tensor_signal(&scheme, [1.0, 0.0, 0.0]),
        OdfConfig::new(6, 0.002, weighting(50.0), weighting(0.0))?,
    )?;
    let x = odf.evaluate_at_direction([1.0, 0.0, 0.0])?;
    let y = odf.evaluate_at_direction([0.0, 1.0, 0.0])?;
    let z = odf.evaluate_at_direction([0.0, 0.0, 1.0])?;
    assert!(x > y, "x-axis ODF {x} must exceed y-axis ODF {y}");
    assert!(x > z, "x-axis ODF {x} must exceed z-axis ODF {z}");
    assert_eq!(odf.coefficients().len(), 28);
    assert!(odf.normalized_signal_residual().is_finite());
    assert!(odf.normalized_signal_residual() >= 0.0);

    const POLAR_INTERVALS: usize = 180;
    const AZIMUTH_SAMPLES: usize = 360;
    let mut peak = ([0.0; 3], f64::NEG_INFINITY);
    for polar_index in 0..=POLAR_INTERVALS {
        let theta = std::f64::consts::PI * polar_index as f64 / POLAR_INTERVALS as f64;
        for azimuth_index in 0..AZIMUTH_SAMPLES {
            let phi = std::f64::consts::TAU * azimuth_index as f64 / AZIMUTH_SAMPLES as f64;
            let direction = [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ];
            let value = odf.evaluate_at_direction(direction)?;
            if value > peak.1 {
                peak = (direction, value);
            }
        }
    }
    let angular_error = peak.0[0].abs().clamp(-1.0, 1.0).acos();
    let grid_bound = std::f64::consts::PI / POLAR_INTERVALS as f64;
    assert!(
        angular_error <= grid_bound,
        "ODF peak error {} degrees exceeds the one-degree scan bound",
        angular_error.to_degrees()
    );
    Ok(())
}

#[test]
fn invalid_configuration_signals_and_grid_are_typed_errors() {
    assert!(matches!(
        OdfConfig::new(3, 0.0, weighting(50.0), weighting(0.0)),
        Err(OdfError::Basis(_))
    ));
    assert!(matches!(
        OdfConfig::new(4, f64::NAN, weighting(50.0), weighting(0.0)),
        Err(OdfError::InvalidRegularization { .. })
    ));
    let scheme = scheme(30);
    let mut signals = vec![1.0; 31];
    signals[7] = f64::INFINITY;
    assert!(matches!(
        estimate_odf(&scheme, &signals, OdfConfig::default()),
        Err(OdfError::NonFiniteSignal { index: 7, .. })
    ));
    let odf =
        estimate_odf(&scheme, &vec![1.0; 31], OdfConfig::default()).expect("valid constant signal");
    assert!(matches!(
        odf.evaluate_on_grid(0, 12),
        Err(OdfError::InvalidGrid { .. })
    ));

    let mut extreme_signals = vec![f64::MAX; 31];
    extreme_signals[0] = f64::MIN_POSITIVE;
    assert!(matches!(
        estimate_odf(&scheme, &extreme_signals, OdfConfig::default()),
        Err(OdfError::NonFiniteNormalizedSignal { index: 1, .. })
    ));
}

#[test]
fn acquisition_partitions_report_exact_typed_errors() {
    let complete = scheme(30);
    assert!(matches!(
        estimate_odf(&complete, &[1.0; 5], OdfConfig::default()),
        Err(OdfError::SignalLengthMismatch {
            signal_count: 5,
            acquisition_count: 31,
        })
    ));

    let weighted_only = GradientScheme::new(
        vec![
            GradientDirection::new(weighting(1_000.0), Vector::new([1.0, 0.0, 0.0]))
                .expect("valid weighted entry"),
        ],
        GradientFrame::Lps,
    )
    .expect("nonempty weighted scheme");
    assert!(matches!(
        estimate_odf(&weighted_only, &[1.0], OdfConfig::default()),
        Err(OdfError::NoB0Volumes)
    ));

    let baseline_only = GradientScheme::new(
        vec![
            GradientDirection::new(weighting(0.0), Vector::new([0.0, 0.0, 0.0]))
                .expect("valid baseline entry"),
        ],
        GradientFrame::Lps,
    )
    .expect("nonempty baseline scheme");
    assert!(matches!(
        estimate_odf(&baseline_only, &[1.0], OdfConfig::default()),
        Err(OdfError::NoDwiDirections)
    ));

    let underdetermined = scheme(6);
    assert!(matches!(
        estimate_odf(&underdetermined, &[1.0; 7], OdfConfig::default()),
        Err(OdfError::Underdetermined {
            direction_count: 6,
            coefficient_count: 15,
        })
    ));

    let mut invalid_baseline = [1.0; 31];
    invalid_baseline[0] = 0.0;
    assert!(matches!(
        estimate_odf(&complete, &invalid_baseline, OdfConfig::default()),
        Err(OdfError::InvalidBaseline { value: 0.0 })
    ));
}

#[test]
fn mixed_shells_fail_and_frame_is_preserved() -> Result<(), OdfError> {
    let mut scheme = scheme(30);
    let mut pairs = scheme
        .directions()
        .iter()
        .map(|entry| {
            (
                entry.weighting().seconds_per_square_millimeter(),
                entry.direction(),
            )
        })
        .collect::<Vec<_>>();
    pairs[2].0 = 1_100.0;
    scheme = GradientScheme::from_seconds_per_square_millimeter(pairs, GradientFrame::ImageAxis)
        .expect("valid mixed-shell scheme");
    let signals = vec![1.0; scheme.len()];
    assert!(matches!(
        estimate_odf(&scheme, &signals, OdfConfig::default()),
        Err(OdfError::MixedShells { index: 2, .. })
    ));

    pairs = scheme
        .directions()
        .iter()
        .map(|entry| {
            (
                if entry.weighting().is_unweighted() {
                    0.0
                } else {
                    1_000.0
                },
                entry.direction(),
            )
        })
        .collect();
    let image_axis =
        GradientScheme::from_seconds_per_square_millimeter(pairs, GradientFrame::ImageAxis)
            .expect("single-shell image-axis scheme");
    let odf = estimate_odf(&image_axis, &signals, OdfConfig::default())?;
    assert_eq!(odf.frame(), GradientFrame::ImageAxis);

    let mut tolerated_pairs = image_axis
        .directions()
        .iter()
        .map(|entry| {
            (
                entry.weighting().seconds_per_square_millimeter(),
                entry.direction(),
            )
        })
        .collect::<Vec<_>>();
    tolerated_pairs[2].0 = 1_000.5;
    let tolerated_scheme = GradientScheme::from_seconds_per_square_millimeter(
        tolerated_pairs,
        GradientFrame::ImageAxis,
    )
    .expect("scheme inside the configured shell tolerance");
    let tolerance = OdfConfig::new(4, 0.006, weighting(50.0), weighting(1.0))?;
    let tolerated = estimate_odf(&tolerated_scheme, &signals, tolerance)?;
    assert_eq!(tolerated.frame(), GradientFrame::ImageAxis);
    Ok(())
}

#[test]
fn spherical_grid_is_flat_and_finite() -> Result<(), OdfError> {
    let scheme = scheme(30);
    let odf = estimate_odf(&scheme, &vec![1.0; 31], OdfConfig::default())?;
    let grid = odf.evaluate_on_grid(8, 16)?;
    assert_eq!(grid.shape(), [8, 16]);
    assert_eq!(grid.values().len(), 128);
    assert!(grid.values().iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn evaluation_rejects_finite_coefficient_overflow() {
    let basis = RealSphericalHarmonicBasis::new(6).expect("valid even basis");
    let mut coefficients = vec![0.0; basis.iter_lm().count()];
    let degree_six_zonal = basis
        .iter_lm()
        .position(|(_, degree, order)| degree == 6 && order == 0)
        .expect("degree-six zonal coefficient exists");
    coefficients[degree_six_zonal] = f64::MAX;
    let odf = OdField {
        coefficients: coefficients.into_boxed_slice(),
        basis,
        baseline_signal: 1.0,
        normalized_signal_residual: 0.0,
        frame: GradientFrame::Lps,
    };

    assert!(matches!(
        odf.evaluate(0.0, 0.0),
        Err(OdfError::NonFiniteEvaluation { .. })
    ));
    assert!(matches!(
        odf.evaluate_at_direction([0.0, 0.0, 1.0]),
        Err(OdfError::NonFiniteEvaluation { .. })
    ));

    let grid_basis = RealSphericalHarmonicBasis::new(6).expect("valid even basis");
    let grid_coefficients = grid_basis
        .iter_lm()
        .map(|(_, degree, order)| {
            f64::MAX.copysign(
                real_spherical_harmonic(degree, order, std::f64::consts::FRAC_PI_2, 0.0)
                    .expect("finite analytical basis value"),
            )
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let grid_odf = OdField {
        coefficients: grid_coefficients,
        basis: grid_basis,
        baseline_signal: 1.0,
        normalized_signal_residual: 0.0,
        frame: GradientFrame::Lps,
    };
    assert!(matches!(
        grid_odf.evaluate_on_grid(1, 1),
        Err(OdfError::NonFiniteEvaluation { .. })
    ));
}
