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
