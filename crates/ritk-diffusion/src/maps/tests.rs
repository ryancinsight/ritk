//! Tests for whole-volume tensor fitting.
//!
//! Signals are generated from a known tensor through the forward model, so
//! every recovered map has an exact expected value rather than a plausible one.

use super::*;
use crate::test_support::{dti_signal, scheme, schemes_with_references};

/// An anisotropic tensor with distinct eigenvalues, in mm²/s.
///
/// Prolate and axis-aligned, so its eigenvalues are the diagonal itself and
/// every derived measure has a closed form: AD = 1.7e-3, RD = 3.0e-4,
/// MD = 7.667e-4.
const ANISOTROPIC: [f64; 6] = [1.7e-3, 3.0e-4, 3.0e-4, 0.0, 0.0, 0.0];

/// An isotropic tensor: equal eigenvalues, so FA is exactly zero.
const ISOTROPIC: [f64; 6] = [8.0e-4, 8.0e-4, 8.0e-4, 0.0, 0.0, 0.0];

/// Signals for `tensors`, one voxel each, transposed into per-volume slices.
///
/// A series stores volume-major (all voxels of volume 0, then volume 1), while
/// the forward model produces voxel-major, so the transpose is what turns
/// generated signals into something shaped like a real acquisition.
fn series(scheme: &GradientScheme, tensors: &[([f64; 6], f64)]) -> Vec<Vec<f64>> {
    let per_voxel: Vec<Vec<f64>> = tensors
        .iter()
        .map(|(elements, s0)| dti_signal(scheme, *elements, *s0))
        .collect();
    (0..scheme.len())
        .map(|volume| per_voxel.iter().map(|voxel| voxel[volume]).collect())
        .collect()
}

fn borrow(volumes: &[Vec<f64>]) -> Vec<&[f64]> {
    volumes.iter().map(Vec::as_slice).collect()
}

fn fit(
    scheme: &GradientScheme,
    volumes: &[Vec<f64>],
    config: &DiffusionMapsConfig,
) -> DiffusionMaps {
    fit_diffusion_maps(scheme, &borrow(volumes), config).expect("well-formed series")
}

/// Masking off, so every test voxel is fitted regardless of its S₀.
fn unmasked() -> DiffusionMapsConfig {
    DiffusionMapsConfig {
        background_fraction: 0.0,
        ..DiffusionMapsConfig::default()
    }
}

#[test]
fn scalar_maps_match_the_generating_tensor() {
    // The eigenvalues are the diagonal of an axis-aligned tensor, so each map
    // is checked against arithmetic on the input rather than on the output.
    let scheme = scheme(30);
    let volumes = series(&scheme, &[(ANISOTROPIC, 1000.0)]);
    let maps = fit(&scheme, &volumes, &unmasked());

    let [l1, l2, l3] = [1.7e-3, 3.0e-4, 3.0e-4];
    let tolerance = 1.0e-9;

    assert!((maps.axial_diffusivity()[0] - l1).abs() < tolerance);
    assert!((maps.radial_diffusivity()[0] - (l2 + l3) / 2.0).abs() < tolerance);
    assert!((maps.mean_diffusivity()[0] - (l1 + l2 + l3) / 3.0).abs() < tolerance);

    // FA closed form for a prolate tensor with two equal eigenvalues.
    let mean = (l1 + l2 + l3) / 3.0;
    let deviation = (l1 - mean).powi(2) + (l2 - mean).powi(2) + (l3 - mean).powi(2);
    let magnitude = l1 * l1 + l2 * l2 + l3 * l3;
    let expected_fa = (1.5 * deviation / magnitude).sqrt();
    assert!(
        (maps.fractional_anisotropy()[0] - expected_fa).abs() < tolerance,
        "FA {} should equal the closed form {expected_fa}",
        maps.fractional_anisotropy()[0]
    );
}

#[test]
fn isotropic_diffusion_has_zero_anisotropy() {
    // Equal eigenvalues make the deviation term vanish exactly, so this is a
    // value check and not a threshold check.
    let scheme = scheme(30);
    let volumes = series(&scheme, &[(ISOTROPIC, 1000.0)]);
    let maps = fit(&scheme, &volumes, &unmasked());

    assert!(maps.fractional_anisotropy()[0] < 1.0e-6);
    assert!((maps.mean_diffusivity()[0] - 8.0e-4).abs() < 1.0e-9);
}

#[test]
fn principal_eigenvector_follows_the_dominant_axis() {
    // ANISOTROPIC is prolate along x, so the recovered orientation must be ±x.
    // Absolute components because an eigenvector carries no sign.
    let scheme = scheme(30);
    let volumes = series(&scheme, &[(ANISOTROPIC, 1000.0)]);
    let maps = fit(&scheme, &volumes, &unmasked());

    let [x, y, z] = maps.principal_eigenvector()[0];
    assert!(x.abs() > 0.999, "principal axis should be x, got {x}");
    assert!(y.abs() < 0.01 && z.abs() < 0.01);
}

#[test]
fn eigenvalues_are_returned_in_descending_order() {
    // Every derived map indexes eigenvalues positionally -- AD reads [0], RD
    // averages [1] and [2] -- so the ordering is part of the contract.
    let scheme = scheme(30);
    let volumes = series(&scheme, &[(ANISOTROPIC, 1000.0)]);
    let maps = fit(&scheme, &volumes, &unmasked());

    let [l1, l2, l3] = maps.eigenvalues()[0];
    assert!(
        l1 >= l2 && l2 >= l3,
        "expected descending, got {l1} {l2} {l3}"
    );
}

#[test]
fn background_voxels_are_masked_out_and_read_zero() {
    // Two voxels with the same tensor but S₀ differing by 100x: the dim one
    // falls below 12% of the bright one's reference and must not be fitted.
    let scheme = scheme(30);
    let volumes = series(&scheme, &[(ANISOTROPIC, 1000.0), (ANISOTROPIC, 10.0)]);
    let maps = fit(&scheme, &volumes, &DiffusionMapsConfig::default());

    assert_eq!(maps.mask(), [true, false]);
    assert_eq!(maps.fitted_count(), 1);
    assert_eq!(maps.fractional_anisotropy()[1], 0.0);
    assert_eq!(maps.principal_eigenvector()[1], [0.0, 0.0, 0.0]);
    assert!(maps.fractional_anisotropy()[0] > 0.5);
}

#[test]
fn disabling_the_mask_fits_every_voxel() {
    // The same dim voxel is a real measurement once masking is off, which
    // separates "excluded by policy" from "could not be fitted".
    let scheme = scheme(30);
    let volumes = series(&scheme, &[(ANISOTROPIC, 1000.0), (ANISOTROPIC, 10.0)]);
    let maps = fit(&scheme, &volumes, &unmasked());

    assert_eq!(maps.mask(), [true, true]);
    let fa = maps.fractional_anisotropy();
    assert!(
        (fa[0] - fa[1]).abs() < 1.0e-9,
        "the same tensor at any S0 must give the same FA"
    );
}

#[test]
fn a_supraphysical_fit_is_rejected() {
    // Diffusivity an order above free water cannot be a measurement. The voxel
    // fits successfully -- it is the physical bound that excludes it, which is
    // the distinction the rejection exists to make.
    let scheme = scheme(30);
    let fast = [3.0e-2, 3.0e-2, 3.0e-2, 0.0, 0.0, 0.0];
    let volumes = series(&scheme, &[(fast, 1000.0)]);

    let maps = fit(&scheme, &volumes, &unmasked());
    assert_eq!(maps.mask(), [false], "above the free-water ceiling");

    let permissive = DiffusionMapsConfig {
        diffusivity_ceiling: 1.0,
        ..unmasked()
    };
    let maps = fit(&scheme, &volumes, &permissive);
    assert_eq!(maps.mask(), [true], "admitted once the ceiling allows it");
}

#[test]
fn a_collapsed_rank_one_fit_is_rejected() {
    // The defect this guards: a tensor with two near-zero eigenvalues is
    // positive-definite, so a sign check accepts it, and its FA approaches 1.
    // It must be excluded on the eigenvalue floor instead.
    let scheme = scheme(30);
    let collapsed = [1.7e-3, 1.0e-9, 1.0e-9, 0.0, 0.0, 0.0];
    let volumes = series(&scheme, &[(collapsed, 1000.0)]);

    let maps = fit(&scheme, &volumes, &unmasked());
    assert_eq!(
        maps.mask(),
        [false],
        "below the restricted-diffusivity floor"
    );

    // Confirm it would otherwise have produced the impossible anisotropy.
    let permissive = DiffusionMapsConfig {
        diffusivity_floor: 0.0,
        ..unmasked()
    };
    let maps = fit(&scheme, &volumes, &permissive);
    assert!(
        maps.fractional_anisotropy()[0] > 0.99,
        "the rejected fit is exactly the FA-approaching-1 case"
    );
}

#[test]
fn every_reference_volume_contributes_to_the_mask() {
    // With four references, a voxel bright in one and dark in three averages
    // below the floor. Picking the first reference instead of averaging would
    // admit it, so this fails if the averaging regresses to a single volume.
    let scheme = schemes_with_references(30, 4);
    let mut volumes = series(&scheme, &[(ANISOTROPIC, 1000.0), (ANISOTROPIC, 1000.0)]);

    // Voxel 1: bright in reference 0, near-zero in references 1..4.
    for reference in &mut volumes[1..4] {
        reference[1] = 1.0;
    }

    let maps = fit(&scheme, &volumes, &DiffusionMapsConfig::default());
    assert_eq!(
        maps.mask(),
        [true, false],
        "a voxel bright in only one of four references is background"
    );
}

#[test]
fn f32_input_is_accepted_without_conversion_by_the_caller() {
    // Image data arrives as f32; requiring callers to widen it first would put
    // an allocation of the whole volume at every call site.
    let scheme = scheme(30);
    let volumes = series(&scheme, &[(ANISOTROPIC, 1000.0)]);
    let narrowed: Vec<Vec<f32>> = volumes
        .iter()
        .map(|volume| volume.iter().map(|value| *value as f32).collect())
        .collect();
    let borrowed: Vec<&[f32]> = narrowed.iter().map(Vec::as_slice).collect();

    let maps = fit_diffusion_maps(&scheme, &borrowed, &unmasked()).expect("f32 series");
    assert_eq!(maps.mask(), [true]);
    assert!((maps.mean_diffusivity()[0] - 7.666_666e-4).abs() < 1.0e-8);
}

#[test]
fn a_series_disagreeing_with_its_scheme_is_rejected() {
    let scheme = scheme(30);
    let volumes = series(&scheme, &[(ANISOTROPIC, 1000.0)]);
    let mut short = borrow(&volumes);
    short.pop();

    let error = fit_diffusion_maps(&scheme, &short, &unmasked()).expect_err("count mismatch");
    assert!(matches!(
        error,
        DiffusionMapsError::VolumeCountMismatch { .. }
    ));
}

#[test]
fn volumes_of_differing_length_are_rejected() {
    let scheme = scheme(30);
    let mut volumes = series(&scheme, &[(ANISOTROPIC, 1000.0), (ANISOTROPIC, 1000.0)]);
    volumes[3].pop();

    let error =
        fit_diffusion_maps(&scheme, &borrow(&volumes), &unmasked()).expect_err("ragged series");
    assert!(matches!(
        error,
        DiffusionMapsError::VolumeLengthMismatch {
            index: 3,
            length: 1,
            expected: 2
        }
    ));
}

#[test]
fn an_empty_series_is_rejected() {
    let scheme = scheme(30);
    let error = fit_diffusion_maps::<f64>(&scheme, &[], &unmasked()).expect_err("no volumes");
    assert!(matches!(error, DiffusionMapsError::NoVolumes));
}

#[test]
fn a_nonsensical_bound_is_rejected_before_fitting() {
    let scheme = scheme(30);
    let volumes = series(&scheme, &[(ANISOTROPIC, 1000.0)]);
    let config = DiffusionMapsConfig {
        background_fraction: f64::NAN,
        ..DiffusionMapsConfig::default()
    };

    let error = fit_diffusion_maps(&scheme, &borrow(&volumes), &config).expect_err("NaN fraction");
    assert!(matches!(
        error,
        DiffusionMapsError::InvalidConfiguration {
            parameter: "background_fraction",
            ..
        }
    ));
}
