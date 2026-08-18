//! SRAD oracles.
//!
//! The tests that matter here assert the property that *distinguishes* SRAD
//! from the gradient-keyed diffusion already in this module: because the
//! instantaneous coefficient of variation is a ratio, SRAD's behaviour is
//! scale-free under the multiplicative noise ultrasound actually has.

use super::*;
use crate::diffusion::perona_malik::{
    AnisotropicDiffusionFilter, DiffusionConfig, ExponentialConductance,
};
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn image_from(values: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(values, dims)
}

fn values_of(image: &Image<f32, B, 3>) -> Vec<f32> {
    extract_vec_infallible(image).0
}

/// Deterministic multiplicative speckle: a Rayleigh-like modulation about 1.
fn speckle(seed: u64) -> f64 {
    let mixed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let unit = ((mixed >> 33) as f64) / ((1_u64 << 31) as f64);
    // Mean ~1, spread comparable to fully developed speckle.
    0.45 + 1.1 * unit
}

fn variance(values: &[f32]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    values
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n
}

/// A constant image has zero gradient and zero Laplacian everywhere, so the
/// divergence term is identically zero and the image must be returned bit-for-bit.
///
/// This is the strongest available exactness check: it pins that the stencil,
/// the boundary clamping and the update are all balanced, since any asymmetry
/// would leak a non-zero divergence at the edges.
#[test]
fn constant_image_is_unchanged() {
    let dims = [3, 8, 8];
    let image = image_from(vec![7.5_f32; 3 * 8 * 8], dims);
    let filtered = SpeckleReducingDiffusionFilter::new(SradConfig::default())
        .apply(&image)
        .expect("apply");
    for (got, want) in values_of(&filtered).iter().zip(values_of(&image).iter()) {
        assert_eq!(got, want, "constant image must be untouched");
    }
}

/// Speckle in a homogeneous region must be suppressed while a step edge
/// survives. Smoothing everything, or preserving everything, both fail.
#[test]
fn suppresses_speckle_and_keeps_the_step_edge() {
    let (nz, ny, nx) = (1_usize, 24_usize, 24_usize);
    let dims = [nz, ny, nx];
    // Left half 40, right half 120, each multiplied by speckle.
    let mut values = vec![0.0_f32; ny * nx];
    for y in 0..ny {
        for x in 0..nx {
            let base = if x < nx / 2 { 40.0 } else { 120.0 };
            values[y * nx + x] = (base * speckle((y * nx + x) as u64)) as f32;
        }
    }
    let image = image_from(values.clone(), dims);
    let filtered = SpeckleReducingDiffusionFilter::new(SradConfig::default())
        .apply(&image)
        .expect("apply");
    let out = values_of(&filtered);

    // Interior columns of each half, away from the seam, so the edge itself is
    // excluded from the homogeneity measurement.
    let region = |src: &[f32], lo: usize, hi: usize| -> Vec<f32> {
        let mut v = Vec::new();
        for y in 4..ny - 4 {
            for x in lo..hi {
                v.push(src[y * nx + x]);
            }
        }
        v
    };
    let before_left = variance(&region(&values, 2, 8));
    let after_left = variance(&region(&out, 2, 8));
    assert!(
        after_left < 0.5 * before_left,
        "speckle variance must fall in the dim region: {before_left} -> {after_left}"
    );

    // Edge contrast: mean of the two halves must stay well separated.
    let mean = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
    let contrast = mean(&region(&out, nx / 2 + 2, nx / 2 + 8)) - mean(&region(&out, 2, 8));
    assert!(
        contrast > 60.0,
        "the 80-unit step must survive diffusion, got {contrast}"
    );
}

/// The defining SRAD property, and the reason it exists alongside Perona–Malik.
///
/// Speckle is *multiplicative*, so the meaningful statement is about scaling.
/// Both `|∇I|/I` and `∇²I/I` are invariant under `I → k·I`, so the instantaneous
/// coefficient of variation `q`, and hence the diffusion coefficient `c`, are
/// unchanged; the update `I + Δt·div(c∇I)` is then exactly homogeneous of
/// degree one. SRAD is therefore **scale-equivariant**:
///
/// ```text
/// SRAD(k·I) = k·SRAD(I)
/// ```
///
/// This is an exact algebraic identity, not a statistical tendency, which makes
/// it a far sharper oracle than comparing smoothing strength between regions.
/// Perona–Malik keys its conductance on the *absolute* gradient against a fixed
/// threshold, so it is not equivariant — brightening an image changes which
/// edges it preserves. That contrast is the reason this filter exists, and the
/// second assertion pins it.
#[test]
fn is_scale_equivariant_where_gradient_keyed_diffusion_is_not() {
    let (ny, nx) = (16_usize, 16_usize);
    let dims = [1, ny, nx];
    const K: f32 = 10.0;

    let base: Vec<f32> = (0..ny * nx)
        .map(|i| (20.0 * speckle(i as u64)) as f32)
        .collect();
    let scaled: Vec<f32> = base.iter().map(|&v| v * K).collect();

    let config = SradConfig {
        num_iterations: 5,
        ..SradConfig::default()
    };
    let filter = SpeckleReducingDiffusionFilter::new(config);
    let from_base = values_of(&filter.apply(&image_from(base.clone(), dims)).expect("srad"));
    let from_scaled = values_of(
        &filter
            .apply(&image_from(scaled.clone(), dims))
            .expect("srad"),
    );

    // Tolerance: f32 carries ~1.2e-7 relative precision and the update
    // accumulates over 5 iterations of six-neighbour sums, so a few hundred
    // roundings; 1e-4 relative is far above that and far below any real
    // asymmetry, which would show up as a percent-level divergence.
    let mut worst = 0.0_f64;
    for (&b, &sc) in from_base.iter().zip(from_scaled.iter()) {
        let expected = f64::from(b) * f64::from(K);
        let got = f64::from(sc);
        let relative = (got - expected).abs() / expected.abs().max(1.0e-6);
        worst = worst.max(relative);
    }
    assert!(
        worst < 1.0e-4,
        "SRAD must satisfy SRAD(k*I) = k*SRAD(I); worst relative deviation {worst:e}"
    );

    // Perona-Malik on the same inputs must visibly fail the identity.
    let pm = AnisotropicDiffusionFilter::<ExponentialConductance>::new(DiffusionConfig {
        num_iterations: 5,
        ..DiffusionConfig::default()
    });
    let pm_base = values_of(&pm.apply(&image_from(base, dims)).expect("pm"));
    let pm_scaled = values_of(&pm.apply(&image_from(scaled, dims)).expect("pm"));
    let mut pm_worst = 0.0_f64;
    for (&b, &sc) in pm_base.iter().zip(pm_scaled.iter()) {
        let expected = f64::from(b) * f64::from(K);
        let relative = (f64::from(sc) - expected).abs() / expected.abs().max(1.0e-6);
        pm_worst = pm_worst.max(relative);
    }
    assert!(
        pm_worst > 100.0 * worst,
        "gradient-keyed diffusion should not be scale-equivariant:          SRAD {worst:e} vs Perona-Malik {pm_worst:e}"
    );
}

/// Zero iterations is the identity, and the filter must not silently do work
/// the caller did not ask for.
#[test]
fn zero_iterations_is_the_identity() {
    let dims = [1, 4, 4];
    let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let image = image_from(values.clone(), dims);
    let config = SradConfig {
        num_iterations: 0,
        ..SradConfig::default()
    };
    let filtered = SpeckleReducingDiffusionFilter::new(config)
        .apply(&image)
        .expect("apply");
    assert_eq!(values_of(&filtered), values);
}

#[test]
fn rejects_invalid_configuration() {
    let bad = [
        SradConfig {
            time_step: 0.0,
            ..SradConfig::default()
        },
        SradConfig {
            time_step: -1.0,
            ..SradConfig::default()
        },
        SradConfig {
            q0: 0.0,
            ..SradConfig::default()
        },
        SradConfig {
            rho: -0.1,
            ..SradConfig::default()
        },
        SradConfig {
            q0: f64::NAN,
            ..SradConfig::default()
        },
    ];
    for config in bad {
        assert!(config.validate().is_err(), "must reject {config:?}");
    }
    assert!(SradConfig::default().validate().is_ok());
}
