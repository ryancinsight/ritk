//! Strain-window filter tests.
//!
//! The oracles are exact wherever the arithmetic permits it. Linear
//! interpolation of a linear ramp reproduces that ramp, and nearest-value
//! extrapolation copies a stored value, so both cases are checked against the
//! value they must produce rather than against a smoothness heuristic.

use super::{strain_window_filter, StrainWindowParams};
use crate::DisplacementField;

/// Interpolating a linear ramp is exact in real arithmetic. In floating point
/// it costs three roundings (`z_hi - z_lo`, the scaled difference, the sum), so
/// the error is bounded by a few ULP of the interpolated magnitude. The ramp
/// values here are below 1.0, whose ULP is ~2.2e-16, giving ~1e-15.
const INTERPOLATION_TOLERANCE: f64 = 1e-15;

/// Axial slope of the synthetic ramp, in voxel/voxel. Well inside the default
/// 0.1 bound, so an unfiltered ramp must survive untouched.
const RAMP_STRAIN: f64 = 0.02;

/// A single axial line of `n` blocks at lateral position `(y, x)`, carrying a
/// ramp of slope [`RAMP_STRAIN`] and a fully correlated peak everywhere.
fn ramp_line(n: usize, y: usize, x: usize) -> DisplacementField {
    DisplacementField {
        centres: (0..n).map(|z| [z, y, x]).collect(),
        displacements: (0..n).map(|z| [RAMP_STRAIN * z as f64, 0.0, 0.0]).collect(),
        peak_similarities: vec![0.99; n],
    }
}

fn concat(a: DisplacementField, b: DisplacementField) -> DisplacementField {
    DisplacementField {
        centres: [a.centres, b.centres].concat(),
        displacements: [a.displacements, b.displacements].concat(),
        peak_similarities: [a.peak_similarities, b.peak_similarities].concat(),
    }
}

#[test]
fn plausible_field_is_returned_bit_for_bit() {
    let field = ramp_line(7, 0, 0);
    let report = strain_window_filter(&field, 1, StrainWindowParams::default()).unwrap();

    assert_eq!(
        report.field, field,
        "a field whose strain is everywhere inside the bound must not be altered"
    );
    assert_eq!(report.replaced, 0);
    assert_eq!(
        report.iterations, 1,
        "convergence is detected on the first pass, not after exhausting the budget"
    );
    assert!(report.unrecoverable.is_empty());
}

#[test]
fn peak_hop_is_replaced_by_the_ramp_it_interrupted() {
    let mut field = ramp_line(7, 0, 0);
    // A hop of ~5 voxels: the scale a wavelength-lobe jump produces, and far
    // outside anything the 0.02 ramp could reach.
    field.displacements[3][0] = 5.0;

    let report = strain_window_filter(&field, 1, StrainWindowParams::default()).unwrap();

    // The hop makes the gradient implausible on both sides of block 3 and on
    // the outward side of blocks 2 and 4, so all three are substituted. Donors
    // are blocks 1 and 5, and interpolating a linear ramp reproduces the ramp
    // exactly — the filter restores the signal, it does not merely smooth it.
    assert_eq!(report.replaced, 3);
    assert!(report.unrecoverable.is_empty());
    for z in 0..7 {
        let expected = RAMP_STRAIN * z as f64;
        assert!(
            (report.field.displacements[z][0] - expected).abs() <= INTERPOLATION_TOLERANCE,
            "block {z}: got {}, expected the underlying ramp value {expected}",
            report.field.displacements[z][0]
        );
    }
}

#[test]
fn boundary_outlier_takes_the_nearest_reliable_value() {
    let mut field = ramp_line(7, 0, 0);
    field.displacements[0][0] = 5.0;

    let report = strain_window_filter(&field, 1, StrainWindowParams::default()).unwrap();

    // Blocks 0 and 1 are implausible and have no reliable block below them, so
    // both take the nearest reliable value above — block 2 — unchanged. This is
    // ITK's extrapolate-by-nearest behaviour, and being a copy it is exact.
    let nearest = RAMP_STRAIN * 2.0;
    assert_eq!(report.field.displacements[0][0], nearest);
    assert_eq!(report.field.displacements[1][0], nearest);
    assert_eq!(report.replaced, 2);
}

#[test]
fn constant_block_is_replaced_on_similarity_alone() {
    let mut field = ramp_line(7, 0, 0);
    // A zero-variance window: track_volume records a non-finite similarity and
    // a zero displacement. Zero happens to sit near the ramp here, so the
    // strain bound alone would not catch it — only the similarity does.
    field.peak_similarities[3] = f64::NAN;
    field.displacements[3][0] = 0.06;

    let report = strain_window_filter(&field, 1, StrainWindowParams::default()).unwrap();

    assert_eq!(report.replaced, 1, "the constant block must be substituted");
    assert!(report.unrecoverable.is_empty());
    let expected = RAMP_STRAIN * 3.0;
    assert!((report.field.displacements[3][0] - expected).abs() <= INTERPOLATION_TOLERANCE);
}

#[test]
fn a_non_finite_displacement_is_implausible_not_incomparable() {
    let mut field = ramp_line(7, 0, 0);
    // A NaN displacement has no ordering against the bound at all. Comparing it
    // with a plain  would report "not greater", quietly admitting it; the
    // gradient test must reject what it cannot order.
    field.displacements[3][0] = f64::NAN;

    let report = strain_window_filter(&field, 1, StrainWindowParams::default()).unwrap();

    assert!(report.unrecoverable.is_empty());
    for z in 0..7 {
        let value = report.field.displacements[z][0];
        assert!(value.is_finite(), "block {z} is still {value}");
        let expected = RAMP_STRAIN * z as f64;
        assert!(
            (value - expected).abs() <= INTERPOLATION_TOLERANCE,
            "block {z}: got {value}, expected the underlying ramp value {expected}"
        );
    }
}

#[test]
fn a_line_without_any_reliable_block_is_reported_not_invented() {
    let mut field = ramp_line(5, 0, 0);
    for i in 0..5 {
        field.peak_similarities[i] = f64::NAN;
    }

    let report = strain_window_filter(&field, 1, StrainWindowParams::default()).unwrap();

    assert_eq!(
        report.field.displacements, field.displacements,
        "with no reliable donor there is nothing to interpolate from, so the \
         displacements must be left as measured rather than fabricated"
    );
    assert_eq!(report.replaced, 0);
    assert_eq!(report.unrecoverable, vec![0, 1, 2, 3, 4]);
}

#[test]
fn filtering_is_confined_to_the_affected_axial_line() {
    let mut corrupted = ramp_line(7, 0, 0);
    corrupted.displacements[3][0] = 5.0;
    let clean = ramp_line(7, 0, 1);
    let field = concat(corrupted, clean.clone());

    let report = strain_window_filter(&field, 1, StrainWindowParams::default()).unwrap();

    for z in 0..7 {
        assert_eq!(
            report.field.displacements[7 + z],
            clean.displacements[z],
            "block {z} of the neighbouring line shares no axial line with the \
             outlier and must be untouched"
        );
    }
    assert_eq!(report.replaced, 3);
}

#[test]
fn a_larger_bound_admits_the_same_field_unchanged() {
    let mut field = ramp_line(7, 0, 0);
    field.displacements[3][0] = 5.0;

    // The bound is a plausibility limit, not a denoiser: raised above the hop
    // it produces, nothing is rejected. This pins that the filter acts on the
    // stated criterion rather than on outlier-ness in general.
    let params = StrainWindowParams {
        max_abs_strain: 10.0,
        max_iterations: 3,
    };
    let report = strain_window_filter(&field, 1, params).unwrap();

    assert_eq!(report.field, field);
    assert_eq!(report.replaced, 0);
}

#[test]
fn the_axial_stride_scales_the_strain_estimate() {
    let mut field = ramp_line(7, 0, 0);
    field.displacements[3][0] = 0.5;

    // At stride 1 the jump to the neighbour below implies a gradient of
    // (0.5 - 0.04)/1 = 0.46, outside the bound. At stride 8 the same
    // displacements imply 0.0575, inside it: the blocks are further apart, so
    // the same jump is a gentler gradient and no longer implausible.
    let strict = strain_window_filter(&field, 1, StrainWindowParams::default()).unwrap();
    assert!(strict.replaced > 0);

    let lenient = strain_window_filter(&field, 8, StrainWindowParams::default()).unwrap();
    assert_eq!(lenient.replaced, 0);
    assert_eq!(lenient.field, field);
}

#[test]
fn an_empty_field_is_accepted() {
    let field = DisplacementField {
        centres: Vec::new(),
        displacements: Vec::new(),
        peak_similarities: Vec::new(),
    };
    let report = strain_window_filter(&field, 1, StrainWindowParams::default()).unwrap();
    assert_eq!(report.replaced, 0);
    assert!(report.unrecoverable.is_empty());
}

#[test]
fn non_positive_or_non_finite_bounds_are_rejected() {
    let field = ramp_line(3, 0, 0);
    for bound in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let params = StrainWindowParams {
            max_abs_strain: bound,
            max_iterations: 3,
        };
        // Infinity would accept everything and zero would reject everything;
        // neither is a usable plausibility bound.
        assert!(
            strain_window_filter(&field, 1, params).is_err(),
            "max_abs_strain {bound} must be rejected"
        );
    }
}

#[test]
fn zero_axial_stride_is_rejected() {
    let field = ramp_line(3, 0, 0);
    assert!(strain_window_filter(&field, 0, StrainWindowParams::default()).is_err());
}
