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
