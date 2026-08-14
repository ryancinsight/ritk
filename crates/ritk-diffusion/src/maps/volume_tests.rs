//! Tests for spatial lookup over a fitted tensor field.

use super::*;
use crate::test_support::{dti_signal, scheme};

const ANISOTROPIC: [f64; 6] = [1.7e-3, 3.0e-4, 3.0e-4, 0.0, 0.0, 0.0];
const ISOTROPIC: [f64; 6] = [8.0e-4, 8.0e-4, 8.0e-4, 0.0, 0.0, 0.0];

/// Fit a volume whose voxels carry the given tensors, in `[depth, row, column]`
/// order, with masking off so every listed voxel is fitted.
fn volume(tensors: &[[f64; 6]], shape: [usize; 3], floor: f64) -> DtiVolume {
    let scheme = scheme(30);
    let per_voxel: Vec<Vec<f64>> = tensors
        .iter()
        .map(|elements| dti_signal(&scheme, *elements, 1000.0))
        .collect();
    let volumes: Vec<Vec<f64>> = (0..scheme.len())
        .map(|acquisition| per_voxel.iter().map(|v| v[acquisition]).collect())
        .collect();
    let borrowed: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();

    let config = DiffusionMapsConfig {
        background_fraction: 0.0,
        ..DiffusionMapsConfig::default()
    };
    let maps = fit_diffusion_maps(&scheme, &borrowed, &config).expect("well-formed series");
    DtiVolume::new(maps, shape, floor).expect("shape matches the fitted voxels")
}

/// A one-voxel-per-index line volume along the depth axis.
fn line(tensors: &[[f64; 6]], floor: f64) -> DtiVolume {
    volume(tensors, [tensors.len(), 1, 1], floor)
}

fn at(volume: &DtiVolume, index: [f64; 3]) -> Option<Vector<3>> {
    volume.direction_at(&Point::new(index))
}

#[test]
fn a_fitted_voxel_reports_its_unit_orientation() {
    let volume = line(&[ANISOTROPIC], 0.2);
    let direction = at(&volume, [0.0, 0.0, 0.0]).expect("anisotropic voxel has a direction");

    let [x, y, z] = direction.to_array();
    assert!(x.abs() > 0.999, "prolate along x, got {x}");
    assert!(y.abs() < 0.01 && z.abs() < 0.01);

    let norm = (x * x + y * y + z * z).sqrt();
    assert!(
        (norm - 1.0).abs() < 1.0e-12,
        "direction must be unit, got {norm}"
    );
}

#[test]
fn below_the_anisotropy_floor_there_is_no_direction() {
    // Tracking through near-isotropic tissue follows an eigenvector the data
    // does not distinguish from any other, so the streamline must stop.
    let volume = line(&[ISOTROPIC], 0.2);
    assert!(at(&volume, [0.0, 0.0, 0.0]).is_none());

    // The same voxel is trackable once the floor admits it, which separates
    // "the fit failed" from "the policy excluded it".
    let permissive = line(&[ISOTROPIC], 0.0);
    assert!(at(&permissive, [0.0, 0.0, 0.0]).is_some());
}

#[test]
fn outside_the_grid_there_is_no_direction() {
    let volume = line(&[ANISOTROPIC, ANISOTROPIC], 0.2);
    assert!(at(&volume, [-1.0, 0.0, 0.0]).is_none(), "before the grid");
    assert!(
        at(&volume, [2.0, 0.0, 0.0]).is_none(),
        "past the last voxel"
    );
    assert!(
        at(&volume, [0.0, 1.0, 0.0]).is_none(),
        "outside a unit axis"
    );
    assert!(
        at(&volume, [f64::NAN, 0.0, 0.0]).is_none(),
        "not a coordinate"
    );
    assert!(
        at(&volume, [1.0, 0.0, 0.0]).is_some(),
        "the last voxel is inside"
    );
}

#[test]
fn lookup_is_nearest_neighbour() {
    // A step lands between voxels far more often than on one, so the rounding
    // boundary is the common case rather than an edge case.
    let volume = line(&[ANISOTROPIC, ANISOTROPIC], 0.2);
    assert!(
        at(&volume, [0.49, 0.0, 0.0]).is_some(),
        "rounds down into voxel 0"
    );
    assert!(
        at(&volume, [1.49, 0.0, 0.0]).is_some(),
        "rounds down into voxel 1"
    );
    assert!(
        at(&volume, [1.51, 0.0, 0.0]).is_none(),
        "rounds up past the last voxel rather than clamping to it"
    );
}

#[test]
fn indices_are_ordered_depth_row_column() {
    // The order must match `Image::shape()` and the layout the maps were fitted
    // from. Transposing it would sample a different voxel and silently track
    // through the wrong tissue.
    //
    // Voxel 1 of a [1, 2, 1] volume is row 1, so it is reached at index
    // [0, 1, 0] and not at [1, 0, 0].
    let volume = volume(&[ISOTROPIC, ANISOTROPIC], [1, 2, 1], 0.2);
    assert!(
        at(&volume, [0.0, 1.0, 0.0]).is_some(),
        "the anisotropic voxel is at row 1"
    );
    assert!(
        at(&volume, [1.0, 0.0, 0.0]).is_none(),
        "depth 1 is outside a volume one slice deep"
    );
}

#[test]
fn an_unfitted_voxel_reports_no_direction() {
    // Masked-out voxels store exact zeros. Returning that as a direction would
    // send a streamline off along a meaningless axis instead of stopping it.
    let scheme = scheme(30);
    let bright = dti_signal(&scheme, ANISOTROPIC, 1000.0);
    let dim = dti_signal(&scheme, ANISOTROPIC, 10.0);
    let volumes: Vec<Vec<f64>> = (0..scheme.len())
        .map(|acquisition| vec![bright[acquisition], dim[acquisition]])
        .collect();
    let borrowed: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();

    let maps = fit_diffusion_maps(&scheme, &borrowed, &DiffusionMapsConfig::default())
        .expect("well-formed series");
    assert_eq!(maps.mask(), [true, false], "the dim voxel is background");

    let volume = DtiVolume::new(maps, [2, 1, 1], 0.2).expect("shape matches");
    assert!(at(&volume, [0.0, 0.0, 0.0]).is_some());
    assert!(at(&volume, [1.0, 0.0, 0.0]).is_none());
}

#[test]
fn a_shape_disagreeing_with_the_fitted_voxels_is_rejected() {
    let scheme = scheme(30);
    let signals = dti_signal(&scheme, ANISOTROPIC, 1000.0);
    let volumes: Vec<Vec<f64>> = signals.iter().map(|value| vec![*value]).collect();
    let borrowed: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();
    let maps = fit_diffusion_maps(&scheme, &borrowed, &DiffusionMapsConfig::default())
        .expect("well-formed series");

    let error = DtiVolume::new(maps, [2, 2, 2], 0.2).expect_err("one voxel is not eight");
    assert!(matches!(
        error,
        DiffusionMapsError::VolumeLengthMismatch {
            length: 1,
            expected: 8,
            ..
        }
    ));
}

#[test]
fn an_anisotropy_floor_outside_zero_to_one_is_rejected() {
    // FA is a fraction by construction, so a floor above 1 would silently make
    // every voxel untrackable.
    let scheme = scheme(30);
    let signals = dti_signal(&scheme, ANISOTROPIC, 1000.0);
    let volumes: Vec<Vec<f64>> = signals.iter().map(|value| vec![*value]).collect();
    let borrowed: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();
    let maps = fit_diffusion_maps(&scheme, &borrowed, &DiffusionMapsConfig::default())
        .expect("well-formed series");

    let error = DtiVolume::new(maps, [1, 1, 1], 1.5).expect_err("not a fraction");
    assert!(matches!(
        error,
        DiffusionMapsError::InvalidConfiguration {
            parameter: "anisotropy_floor",
            ..
        }
    ));
}

// ── Interpolation ─────────────────────────────────────────────────────────────

/// Eigenvalues of a strongly prolate voxel; only their ordering and anisotropy
/// matter to a direction lookup.
const PROLATE: [f64; 3] = [1.7e-3, 3.0e-4, 3.0e-4];

/// Build a volume from stored orientations directly, bypassing the fit.
fn oriented(directions: &[[f64; 3]], shape: [usize; 3], floor: f64) -> DtiVolume {
    let maps = DiffusionMaps::from_parts(
        vec![PROLATE; directions.len()],
        directions.to_vec(),
        vec![true; directions.len()],
    );
    DtiVolume::new(maps, shape, floor).expect("shape matches the supplied voxels")
}

#[test]
fn opposite_stored_signs_do_not_cancel() {
    // The defect this guards against. An eigenvector has no sign, so two
    // neighbours may legitimately store v and -v for the same fibre. Averaging
    // the vectors gives exactly zero and the streamline stops mid-bundle;
    // averaging the outer product cannot, because (-v)(-v)ᵀ = v vᵀ.
    let volume = oriented(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [2, 1, 1], 0.2);

    let direction = at(&volume, [0.5, 0.0, 0.0]).expect("a fibre runs through the midpoint");
    let [x, y, z] = direction.to_array();
    assert!(
        x.abs() > 0.999,
        "the midpoint of a straight bundle must follow it, got {direction:?}"
    );
    assert!(y.abs() < 1.0e-9 && z.abs() < 1.0e-9);
}

#[test]
fn interpolation_lies_between_the_contributing_orientations() {
    // Two neighbours 40 degrees apart: the midpoint must land between them, not
    // snap to either. This is what distinguishes interpolation from a
    // nearest-neighbour lookup that merely happens to be smooth.
    let angle = 40.0_f64.to_radians();
    let volume = oriented(
        &[[1.0, 0.0, 0.0], [angle.cos(), angle.sin(), 0.0]],
        [2, 1, 1],
        0.2,
    );

    let midpoint = at(&volume, [0.5, 0.0, 0.0]).expect("both neighbours contribute");
    let [x, y, _] = midpoint.to_array();
    let recovered = y.atan2(x).to_degrees().abs();
    assert!(
        (recovered - 20.0).abs() < 1.0e-6,
        "the midpoint of 0 and 40 degrees is 20, got {recovered}"
    );
}

#[test]
fn nearest_mode_is_piecewise_constant() {
    // The comparison baseline: the same query points under Nearest must snap to
    // one voxel's orientation, which is the discontinuity interpolation removes.
    let angle = 40.0_f64.to_radians();
    let volume = oriented(
        &[[1.0, 0.0, 0.0], [angle.cos(), angle.sin(), 0.0]],
        [2, 1, 1],
        0.2,
    )
    .with_interpolation(DirectionInterpolation::Nearest);

    let [x, y, _] = at(&volume, [0.4, 0.0, 0.0])
        .expect("inside voxel 0")
        .to_array();
    assert!(
        y.atan2(x).abs() < 1.0e-9,
        "voxel 0 is exactly on the x axis"
    );

    let [x, y, _] = at(&volume, [0.6, 0.0, 0.0])
        .expect("inside voxel 1")
        .to_array();
    assert!(
        (y.atan2(x).to_degrees().abs() - 40.0).abs() < 1.0e-9,
        "voxel 1 is exactly 40 degrees"
    );
}

#[test]
fn interpolation_does_not_extend_the_trackable_region() {
    // Where a streamline may go is decided by the voxel it is in, not by its
    // neighbours. If interpolation could bridge a masked voxel, switching
    // interpolation would silently change where tracking stops.
    let maps = DiffusionMaps::from_parts(
        vec![PROLATE; 2],
        vec![[1.0, 0.0, 0.0]; 2],
        vec![true, false],
    );
    let volume = DtiVolume::new(maps, [2, 1, 1], 0.2).expect("shape matches");

    assert!(at(&volume, [0.0, 0.0, 0.0]).is_some(), "the fitted voxel");
    assert!(
        at(&volume, [1.0, 0.0, 0.0]).is_none(),
        "a masked voxel stays untrackable however its neighbour is oriented"
    );
}

#[test]
fn contradictory_orientations_fall_back_to_the_voxel_rather_than_averaging() {
    // Perpendicular neighbours at equal weight sum to a dyadic with two equal
    // leading eigenvalues, so no axis is preferred -- which is what a fibre
    // crossing looks like to a single tensor. Interpolation reports nothing
    // there, and the lookup uses the voxel's own orientation.
    //
    // The alternative failure this pins down is averaging: the mean of x and y
    // points at 45 degrees, an orientation neither voxel holds and no fibre
    // follows.
    let volume = oriented(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [2, 1, 1], 0.2);

    // 0.5 rounds away from zero, so the nearest voxel is 1, which holds y.
    let [x, y, _] = at(&volume, [0.5, 0.0, 0.0])
        .expect("the voxel is trackable even where interpolation is ambiguous")
        .to_array();
    assert!(
        y.abs() > 0.999 && x.abs() < 1.0e-9,
        "expected voxel 1's own orientation, got ({x}, {y}) -- a diagonal here          would mean the orientations were averaged"
    );

    // Off-centre one contribution dominates, so interpolation resolves again
    // and bends the direction toward the other voxel.
    let [x, y, _] = at(&volume, [0.2, 0.0, 0.0])
        .expect("voxel 0 dominates")
        .to_array();
    let tilt = y.atan2(x).to_degrees().abs();
    assert!(
        (0.0..45.0).contains(&tilt),
        "0.2 lies nearer voxel 0, so the direction tilts part way toward voxel 1, got {tilt} degrees"
    );
}

#[test]
fn a_uniform_bundle_interpolates_to_itself() {
    // Interpolation must be exact where there is nothing to interpolate --
    // otherwise it would bend a straight bundle.
    let volume = oriented(&[[1.0, 0.0, 0.0]; 4], [4, 1, 1], 0.2);
    for position in [0.0, 0.5, 1.25, 2.75, 3.0] {
        let [x, y, z] = at(&volume, [position, 0.0, 0.0])
            .unwrap_or_else(|| panic!("uniform bundle at {position}"))
            .to_array();
        assert!(x.abs() > 0.999 && y.abs() < 1.0e-9 && z.abs() < 1.0e-9);
    }
}
