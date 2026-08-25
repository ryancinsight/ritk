//! Does the coarse-to-fine pyramid close the aperture gap?
//!
//! `deformable_recovery_test.rs` established that single-resolution Thirion
//! Demons recovers a known translation to better than 0.08 voxel at every
//! structural scale, but recovers only part of a non-rigid deformation's
//! amplitude, and that the shortfall grows with the image's structural
//! wavelength — roughly 30% missing at periods 13/15/17. The cause is the
//! rank-1 aperture structure of the Thirion force: it constrains only the
//! gradient-parallel component per voxel, so when the gradient direction is
//! near-constant across the smoother's support the perpendicular components
//! stay under-determined. A sigma sweep ruled out the regulariser.
//!
//! Multi-resolution Demons is the textbook remedy: at a coarse level the same
//! physical wavelength spans fewer voxels, so the gradient direction
//! decorrelates within the smoother's support and the aperture constraint is
//! better conditioned. Whether that actually recovers the missing amplitude on
//! this implementation was never measured — the multi-resolution path had no
//! ground-truth recovery test at all.
//!
//! # What this measures
//!
//! Both registrations run on the *same* fixed/moving pair, at the same grid,
//! wavelength, smoothing width and iteration budget, so the amplitude ratios
//! are directly comparable and the comparison does not depend on reproducing
//! an earlier run's conditions.
//!
//! Ground truth is exact by construction: `moving[p] = I(p)` and
//! `fixed[p] = I(p + u(p))`, both sampled analytically from one continuous
//! field, so `D = u` is the unique correct displacement under the crate's
//! `warped(p) = moving(p + D(p))` convention.
//!
//! The reported statistic is the best-fit amplitude ratio
//! `alpha = d·u / |u|^2`: 1.0 means the deformation's amplitude is fully
//! recovered, and a value below 1 is the fraction that was not.
//!
//! # Measured answer
//!
//! At periods 13/15/17 on a 20³ grid, sigma 0.75, 240 full-resolution
//! iterations:
//!
//! | Arm | alpha | RMS error |
//! | --- | --- | --- |
//! | Single resolution | 0.7177 | 0.4336 voxel |
//! | 2-level pyramid | 0.7704 | 0.3911 voxel |
//!
//! **The pyramid helps but does not close the gap.** It recovers 0.053 more of
//! the amplitude, which is about 19% of the 0.282 that single resolution
//! leaves on the table — a real improvement, and well short of the remedy the
//! textbook framing implies. The residual shortfall is intrinsic first-order
//! Demons behaviour at this structural scale, not a convergence failure: the
//! single-resolution arm here reproduces the 0.712 that
//! `deformable_recovery_test.rs` recorded independently, so both measurements
//! are describing the same effect.
//!
//! Anyone reaching for the pyramid to fix long-wavelength recovery should
//! expect roughly this much of the gap, not all of it.

use ritk_filter::GaussianSigma;
use ritk_registration::demons::{DemonsVariant, MultiResDemonsConfig, MultiResDemonsRegistration};
use ritk_registration::{DemonsConfig, ThirionDemonsRegistration};

/// Volume extent `[nz, ny, nx]`.
///
/// Matches `deformable_recovery_test.rs` so the single-resolution arm here is
/// the same measurement that file already characterises.
const DIMS: [usize; 3] = [20, 20, 20];

/// Isotropic unit voxel spacing, so displacement voxel units are physical units.
const SPACING: [f64; 3] = [1.0, 1.0, 1.0];

/// Voxels excluded from the statistic at each face.
///
/// Same three boundary effects as the single-resolution file — one-sided
/// gradients, the smoother's replicate boundary over its `3σ` support, and
/// clamped warp samples — plus the pyramid's own downsample/upsample edges,
/// which the existing margin already covers at these factors.
const MARGIN: usize = 5;

/// Sinusoid periods (voxels) along `[z, y, x]`, at the **long** structural
/// scale where the aperture problem dominates.
///
/// This is the row of the recorded single-resolution table with the largest
/// shortfall (alpha ~ 0.71). Testing the remedy anywhere shorter would measure
/// a gap that is barely there.
const PERIODS: [f64; 3] = [13.0, 15.0, 17.0];

/// Per-axis sinusoid amplitude (intensity units).
const AMPLITUDE: f64 = 40.0;

/// Mean intensity; with three ±40 sinusoids the range is `[8, 248]`.
const MEAN_INTENSITY: f64 = 128.0;

/// Diffusion smoothing width, in voxels.
const SIGMA_DIFFUSION: f64 = 0.75;

/// Iteration budget at full resolution.
///
/// The pyramid divides this by each level's shrink factor, so the two arms are
/// not given identical total work — that is the point. Multi-resolution is
/// claimed to reach a *better* answer for *less* fine-level iteration, so
/// matching the full-resolution budget is the honest comparison.
const ITERATIONS: usize = 240;

/// Pyramid depth. Factors are `[2, 1]`, so the coarse level is 10³.
///
/// Three levels would make the coarsest 5³, at which the 5-voxel margin leaves
/// no interior at all and the level contributes nothing but noise.
const LEVELS: usize = 2;

/// Least amplitude gain the pyramid must show over one resolution.
///
/// The measured gain is 0.053. Both arms are deterministic — no RNG, fixed
/// inputs, fixed iteration counts — so run-to-run variation is floating-point
/// only, and this threshold keeps a 2.6x margin against the measurement while
/// still failing if the pyramid regresses to parity. It is a floor on the
/// claim, not a restatement of the number.
const MIN_PYRAMID_GAIN: f64 = 0.02;

/// Applied deformation amplitude, in voxels.
const DEFORMATION_AMPLITUDE: f64 = 1.0;

/// Applied deformation wavelength, in voxels.
const DEFORMATION_WAVELENGTH: f64 = 20.0;

/// Analytic intensity at a continuous voxel coordinate.
///
/// A sum of sinusoids, one per axis, so each gradient component is independent
/// of the other two axes and the gradient vanishes only at isolated points.
fn intensity(coord: [f64; 3]) -> f32 {
    let structure: f64 = coord
        .iter()
        .zip(PERIODS.iter())
        .map(|(&c, &period)| (std::f64::consts::TAU * c / period).sin())
        .sum();
    (MEAN_INTENSITY + AMPLITUDE * structure) as f32
}

/// Sample the analytic field displaced by `displacement`.
fn sample_volume(displacement: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<f32> {
    let [nz, ny, nx] = DIMS;
    let mut volume = Vec::with_capacity(nz * ny * nx);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let base = [iz as f64, iy as f64, ix as f64];
                let offset = displacement(base);
                volume.push(intensity([
                    base[0] + offset[0],
                    base[1] + offset[1],
                    base[2] + offset[2],
                ]));
            }
        }
    }
    volume
}

/// Divergence-free smooth deformation, so no voxel is created or destroyed.
///
/// Each component depends only on the axes it is perpendicular to, which makes
/// the divergence identically zero and keeps the deformation a pure shear.
fn smooth_deformation(coord: [f64; 3]) -> [f64; 3] {
    let k = std::f64::consts::TAU / DEFORMATION_WAVELENGTH;
    [
        DEFORMATION_AMPLITUDE * (k * coord[1]).sin(),
        DEFORMATION_AMPLITUDE * (k * coord[2]).sin(),
        DEFORMATION_AMPLITUDE * (k * coord[0]).sin(),
    ]
}

/// Best-fit amplitude ratio and RMS error of a recovered field.
struct Recovery {
    /// `alpha = d·u / |u|^2`. 1.0 is full recovery.
    amplitude_ratio: f64,
    /// RMS magnitude of the error vector, in voxels.
    rms: f64,
}

fn recovery(disp: [&[f32]; 3], truth: impl Fn([f64; 3]) -> [f64; 3]) -> Recovery {
    let [nz, ny, nx] = DIMS;
    let (mut sq_error, mut sq_truth, mut dot) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut count = 0_usize;

    for iz in MARGIN..nz - MARGIN {
        for iy in MARGIN..ny - MARGIN {
            for ix in MARGIN..nx - MARGIN {
                let flat = (iz * ny + iy) * nx + ix;
                let expected = truth([iz as f64, iy as f64, ix as f64]);
                for axis in 0..3 {
                    let recovered = f64::from(disp[axis][flat]);
                    let error = recovered - expected[axis];
                    sq_error += error * error;
                    sq_truth += expected[axis] * expected[axis];
                    dot += recovered * expected[axis];
                }
                count += 1;
            }
        }
    }

    Recovery {
        amplitude_ratio: dot / sq_truth,
        rms: (sq_error / count as f64).sqrt(),
    }
}

fn base_config() -> DemonsConfig {
    DemonsConfig {
        max_iterations: ITERATIONS,
        sigma_diffusion: Some(
            GaussianSigma::new(SIGMA_DIFFUSION).expect("SIGMA_DIFFUSION is positive"),
        ),
        sigma_fluid: None,
        max_step_length: 2.0,
    }
}

/// The pyramid must not recover *less* amplitude than one resolution does.
///
/// This is the whole claim for running it: at long structural scale the
/// single-resolution force is aperture-limited, and a coarse level is supposed
/// to condition that constraint better. Both arms run on the same pair at the
/// same grid, wavelength, sigma and full-resolution iteration budget, so the
/// ratio between them is a direct measurement rather than a comparison across
/// two runs' conditions.
///
/// The assertion is a floor on the gain rather than the measured value itself,
/// so it fails if the pyramid regresses to parity without being brittle to the
/// exact figure. The measured pair is reported in the failure message either
/// way, and the module docs record what this run produced.
#[test]
fn multi_resolution_recovers_at_least_as_much_amplitude_as_one_resolution() {
    let moving = sample_volume(|_| [0.0; 3]);
    let fixed = sample_volume(smooth_deformation);

    let single = ThirionDemonsRegistration::new(base_config())
        .register(&fixed, &moving, DIMS, SPACING)
        .expect("matched image lengths and dims");
    let single = recovery(
        [&single.disp_z, &single.disp_y, &single.disp_x],
        smooth_deformation,
    );

    let pyramid = MultiResDemonsRegistration::new(MultiResDemonsConfig {
        base_config: base_config(),
        levels: LEVELS,
        variant: DemonsVariant::Classic,
        n_squarings: 6,
    })
    .register(
        &fixed,
        &moving,
        DIMS,
        [SPACING[0] as f32, SPACING[1] as f32, SPACING[2] as f32],
    )
    .expect("matched image lengths and dims");
    let pyramid = recovery(
        [&pyramid.disp_z, &pyramid.disp_y, &pyramid.disp_x],
        smooth_deformation,
    );

    assert!(
        pyramid.amplitude_ratio > single.amplitude_ratio + MIN_PYRAMID_GAIN,
        "the {LEVELS}-level pyramid recovered less amplitude than one resolution at periods \
         {PERIODS:?}: alpha {:.4} against {:.4} (RMS {:.4} against {:.4} voxel). The pyramid \
         exists to condition the aperture constraint at this scale; recovering less means it \
         is not doing that here.",
        pyramid.amplitude_ratio,
        single.amplitude_ratio,
        pyramid.rms,
        single.rms
    );

    // Both arms must actually register something. A pyramid that returned a
    // near-zero field would satisfy the comparison above only if the
    // single-resolution arm had also failed, but this pins each independently.
    for (label, measured) in [("single-resolution", &single), ("pyramid", &pyramid)] {
        assert!(
            measured.amplitude_ratio > 0.5,
            "{label} recovered only {:.4} of the deformation's amplitude, which is not a \
             registration result — the aperture shortfall this file characterises is ~0.7, \
             not ~0.5",
            measured.amplitude_ratio
        );
    }
}
