use super::*;

/// Tolerance for a closed-form comparison at order-one magnitudes.
const TOLERANCE: f64 = 1.0e-12;

/// Canonical shapes, at physiological magnitudes in mm²/s.
///
/// These four cover the corners of the shape space every invariant here is
/// meant to discriminate.
const PROLATE: [f64; 3] = [1.7e-3, 0.25e-3, 0.25e-3];
const OBLATE: [f64; 3] = [1.2e-3, 1.2e-3, 0.2e-3];
const ISOTROPIC: [f64; 3] = [0.8e-3, 0.8e-3, 0.8e-3];
const ORTHOTROPIC: [f64; 3] = [1.5e-3, 1.0e-3, 0.5e-3];

// ── Diffusivities ────────────────────────────────────────────────────────

#[test]
fn mean_diffusivity_is_one_third_of_the_trace() {
    let trace: f64 = ORTHOTROPIC.iter().sum();
    assert!((mean_diffusivity(ORTHOTROPIC) - trace / 3.0).abs() < TOLERANCE);
}

#[test]
fn axial_and_radial_split_the_eigenvalues_about_the_principal_axis() {
    assert!((axial_diffusivity(PROLATE) - 1.7e-3).abs() < TOLERANCE);
    assert!((radial_diffusivity(PROLATE) - 0.25e-3).abs() < TOLERANCE);
    // The three diffusivities are not independent: AD + 2·RD = 3·MD.
    let reconstructed = axial_diffusivity(ORTHOTROPIC) + 2.0 * radial_diffusivity(ORTHOTROPIC);
    assert!((reconstructed - 3.0 * mean_diffusivity(ORTHOTROPIC)).abs() < TOLERANCE);
}

// ── Fractional anisotropy ────────────────────────────────────────────────

#[test]
fn fractional_anisotropy_is_zero_for_an_isotropic_tensor() {
    assert!(fractional_anisotropy(ISOTROPIC).abs() < TOLERANCE);
}

/// The maximally anisotropic tensor `(λ, 0, 0)` reaches exactly one.
///
/// This is the upper end of the scale FA is normalised against, so it is a
/// closed-form value rather than an approximate one.
#[test]
fn fractional_anisotropy_is_one_for_a_line_tensor() {
    assert!((fractional_anisotropy([1.7e-3, 0.0, 0.0]) - 1.0).abs() < TOLERANCE);
}

/// FA is scale invariant: it measures shape, so multiplying every eigenvalue
/// leaves it unchanged. This is the property that lets an FA map be compared
/// across acquisitions with different overall diffusivity.
#[test]
fn fractional_anisotropy_is_invariant_under_uniform_scaling() {
    let scaled = ORTHOTROPIC.map(|value| value * 7.3);
    assert!((fractional_anisotropy(ORTHOTROPIC) - fractional_anisotropy(scaled)).abs() < TOLERANCE);
}

/// A closed-form reference value rather than a self-consistency check.
///
/// For `(λ, λ, 0)` — a perfect disc — the mean is `2λ/3`, the deviations are
/// `(λ/3, λ/3, −2λ/3)`, so `Σ(λᵢ−λ̄)² = 2λ²/3` and `Σλᵢ² = 2λ²`. Hence
/// `FA = √(3/2 · (2/3)/2) = 1/√2`.
#[test]
fn fractional_anisotropy_of_a_disc_is_one_over_root_two() {
    let expected = 1.0 / std::f64::consts::SQRT_2;
    assert!((fractional_anisotropy([1.0e-3, 1.0e-3, 0.0]) - expected).abs() < TOLERANCE);
}

#[test]
fn fractional_anisotropy_of_a_null_tensor_is_zero_rather_than_nan() {
    assert_eq!(fractional_anisotropy([0.0; 3]), 0.0);
}

// ── Relative anisotropy ──────────────────────────────────────────────────

#[test]
fn relative_anisotropy_is_zero_for_an_isotropic_tensor() {
    assert!(relative_anisotropy(ISOTROPIC).abs() < TOLERANCE);
}

/// The line tensor `(λ, 0, 0)` gives the upper bound `√2`.
///
/// Mean is `λ/3`, deviations `(2λ/3, −λ/3, −λ/3)`, so the deviatoric norm is
/// `λ√(6)/3` and `RA = λ√6/3 / (√3 · λ/3) = √2`.
#[test]
fn relative_anisotropy_of_a_line_tensor_is_root_two() {
    let value = relative_anisotropy([1.7e-3, 0.0, 0.0]);
    assert!(
        (value - std::f64::consts::SQRT_2).abs() < TOLERANCE,
        "got {value}"
    );
}

#[test]
fn relative_anisotropy_is_invariant_under_uniform_scaling() {
    let scaled = ORTHOTROPIC.map(|value| value * 0.31);
    assert!((relative_anisotropy(ORTHOTROPIC) - relative_anisotropy(scaled)).abs() < TOLERANCE);
}

#[test]
fn relative_anisotropy_of_a_null_tensor_is_zero_rather_than_nan() {
    assert_eq!(relative_anisotropy([0.0; 3]), 0.0);
}

// ── Westin measures ──────────────────────────────────────────────────────

/// The three measures partition the tensor's shape, so they sum to one for any
/// nonnegative triple. This is the property that makes them readable as
/// fractions, and it is not true of FA and RA.
#[test]
fn westin_measures_sum_to_one() {
    for eigenvalues in [PROLATE, OBLATE, ISOTROPIC, ORTHOTROPIC, [1.0e-3, 0.0, 0.0]] {
        let (linear, planar, spherical) = westin_measures(eigenvalues);
        assert!(
            (linear + planar + spherical - 1.0).abs() < TOLERANCE,
            "{eigenvalues:?} gave ({linear}, {planar}, {spherical})"
        );
    }
}

#[test]
fn a_line_tensor_is_purely_linear() {
    let (linear, planar, spherical) = westin_measures([1.0e-3, 0.0, 0.0]);
    assert!((linear - 1.0).abs() < TOLERANCE);
    assert!(planar.abs() < TOLERANCE);
    assert!(spherical.abs() < TOLERANCE);
}

#[test]
fn a_disc_tensor_is_purely_planar() {
    let (linear, planar, spherical) = westin_measures([1.0e-3, 1.0e-3, 0.0]);
    assert!(linear.abs() < TOLERANCE);
    assert!((planar - 1.0).abs() < TOLERANCE);
    assert!(spherical.abs() < TOLERANCE);
}

#[test]
fn a_sphere_tensor_is_purely_spherical() {
    let (linear, planar, spherical) = westin_measures(ISOTROPIC);
    assert!(linear.abs() < TOLERANCE);
    assert!(planar.abs() < TOLERANCE);
    assert!((spherical - 1.0).abs() < TOLERANCE);
}

/// The discrimination FA cannot make.
///
/// A cigar and a pancake can carry identical FA while having opposite shape.
/// This pair is constructed to make that exact: for a prolate `(a, b, b)` the
/// definition reduces to `FA = |a−b| / √(a² + 2b²)`, and for an oblate
/// `(c, c, d)` to `FA = |c−d| / √(2c² + d²)`. Substituting `(1, ½, ½)` gives
/// `½/√(3/2) = 1/√6`, and `(1, 1, ⅖)` gives `⅗/√(54/25) = 1/√6` — the same
/// value, from two tensors whose shapes are as different as shapes get.
///
/// FA therefore cannot tell them apart, and the Westin measures must. That is
/// what makes these measures information rather than a restatement, and it is
/// the concrete reason a high-FA voxel is not automatically a coherent fibre:
/// it may be a plane of crossing bundles.
#[test]
fn westin_measures_separate_shapes_that_share_a_fractional_anisotropy() {
    let prolate = [1.0e-3, 0.5e-3, 0.5e-3];
    let oblate = [1.0e-3, 1.0e-3, 0.4e-3];

    let shared = 1.0 / 6.0_f64.sqrt();
    assert!(
        (fractional_anisotropy(prolate) - shared).abs() < TOLERANCE,
        "prolate FA must be 1/√6, got {}",
        fractional_anisotropy(prolate)
    );
    assert!(
        (fractional_anisotropy(oblate) - shared).abs() < TOLERANCE,
        "oblate FA must be 1/√6, got {}",
        fractional_anisotropy(oblate)
    );

    let (prolate_linear, prolate_planar, _) = westin_measures(prolate);
    let (oblate_linear, oblate_planar, _) = westin_measures(oblate);

    assert!(
        prolate_linear > prolate_planar,
        "the prolate tensor must read as linear: cl {prolate_linear}, cp {prolate_planar}"
    );
    assert!(
        oblate_planar > oblate_linear,
        "the oblate tensor must read as planar: cl {oblate_linear}, cp {oblate_planar}"
    );
    // Mode makes the same separation on a single signed axis.
    assert!(mode(prolate) > 0.9, "got {}", mode(prolate));
    assert!(mode(oblate) < -0.9, "got {}", mode(oblate));
}

#[test]
fn westin_measures_of_a_null_tensor_are_the_spherical_limit() {
    assert_eq!(westin_measures([0.0; 3]), (0.0, 0.0, 1.0));
}

// ── Mode ─────────────────────────────────────────────────────────────────

/// A prolate tensor sits at the `+1` end of the mode scale.
#[test]
fn mode_of_a_prolate_tensor_is_one() {
    assert!(
        (mode([1.7e-3, 0.25e-3, 0.25e-3]) - 1.0).abs() < 1.0e-12,
        "got {}",
        mode([1.7e-3, 0.25e-3, 0.25e-3])
    );
}

/// An oblate tensor sits at the `−1` end.
#[test]
fn mode_of_an_oblate_tensor_is_minus_one() {
    assert!(
        (mode([1.2e-3, 1.2e-3, 0.2e-3]) + 1.0).abs() < 1.0e-12,
        "got {}",
        mode([1.2e-3, 1.2e-3, 0.2e-3])
    );
}

/// An arithmetic sequence of eigenvalues is the orthotropic midpoint: the
/// deviations are `(+a, 0, −a)`, so the deviatoric determinant vanishes.
#[test]
fn mode_of_an_orthotropic_tensor_is_zero() {
    assert!(
        mode(ORTHOTROPIC).abs() < TOLERANCE,
        "got {}",
        mode(ORTHOTROPIC)
    );
}

#[test]
fn mode_is_invariant_under_uniform_scaling() {
    let scaled = PROLATE.map(|value| value * 4.7);
    assert!((mode(PROLATE) - mode(scaled)).abs() < 1.0e-12);
}

/// Mode is a shape descriptor, so shifting every eigenvalue by a constant —
/// which changes MD but not the deviatoric part — must leave it unchanged.
#[test]
fn mode_is_invariant_under_an_isotropic_shift() {
    let shifted = ORTHOTROPIC.map(|value| value + 0.5e-3);
    assert!((mode(ORTHOTROPIC) - mode(shifted)).abs() < 1.0e-12);
}

#[test]
fn mode_stays_within_its_bounds_across_the_shape_space() {
    // Sweep the middle eigenvalue from the prolate to the oblate limit.
    for step in 0..=100 {
        let fraction = f64::from(step) / 100.0;
        let middle = 0.2e-3 + fraction * (1.0e-3 - 0.2e-3);
        let value = mode([1.0e-3, middle, 0.2e-3]);
        assert!(
            (-1.0..=1.0).contains(&value),
            "mode must stay in [-1, 1]; middle eigenvalue {middle:.3e} gave {value}"
        );
    }
}

#[test]
fn mode_of_an_isotropic_tensor_is_zero_rather_than_nan() {
    assert_eq!(mode(ISOTROPIC), 0.0);
}

// ── Norm and colour ──────────────────────────────────────────────────────

#[test]
fn tensor_norm_is_the_root_sum_of_squared_eigenvalues() {
    let expected: f64 = ORTHOTROPIC
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!((tensor_norm(ORTHOTROPIC) - expected).abs() < TOLERANCE);
}

/// Colour scales by FA, so an isotropic voxel is black whatever direction its
/// arbitrary principal eigenvector happens to point.
#[test]
fn isotropic_tensors_colour_black() {
    assert_eq!(
        colour_by_orientation(ISOTROPIC, [0.577, 0.577, 0.577]),
        [0.0; 3]
    );
}

/// A fully anisotropic tensor along an axis saturates that channel alone.
#[test]
fn a_line_tensor_along_an_axis_saturates_one_channel() {
    let colour = colour_by_orientation([1.0e-3, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert!((colour[2] - 1.0).abs() < TOLERANCE, "got {colour:?}");
    assert_eq!(colour[0], 0.0);
    assert_eq!(colour[1], 0.0);
}

/// The sign of an eigenvector carries no information, so the two representatives
/// of one orientation must colour identically.
#[test]
fn colour_is_invariant_under_eigenvector_sign() {
    let direction = [0.3, -0.5, 0.812];
    let flipped = direction.map(|component: f64| -component);
    assert_eq!(
        colour_by_orientation(PROLATE, direction),
        colour_by_orientation(PROLATE, flipped)
    );
}

#[test]
fn colour_components_stay_within_the_unit_range() {
    for direction in [[1.0, 1.0, 1.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]] {
        for channel in colour_by_orientation([1.0e-3, 0.0, 0.0], direction) {
            assert!(
                (0.0..=1.0).contains(&channel),
                "channel {channel} out of range for {direction:?}"
            );
        }
    }
}
