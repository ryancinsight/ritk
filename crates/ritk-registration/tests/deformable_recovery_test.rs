//! Ground-truth recovery tests for the deformable registration family.
//!
//! Every other deformable test in this crate asserts that the *metric* improves
//! (MSE decreases, the displacement field is finite, the mean shift has the
//! right sign). None of those can distinguish a registration that recovers the
//! applied deformation from one that recovers a fraction of it, or the wrong
//! axis, or a blurred approximation. These tests close that gap: a known
//! analytic deformation is applied to a synthetic image and the registration is
//! required to reproduce that deformation field to a derived accuracy.
//!
//! # Construction
//!
//! `I` is a smooth analytic intensity defined on continuous voxel coordinates.
//! With `u` the ground-truth displacement field the images are
//!
//! ```text
//! moving[p] = I(p)
//! fixed[p]  = I(p + u(p))
//! ```
//!
//! both sampled from the analytic `I`, so no resampling error is baked into the
//! inputs. Under the crate's forward-warp convention
//! `warped(p) = moving(p + D(p))` (see `deformable_field_ops::warp_image`), the
//! field that carries `moving` onto `fixed` is exactly `D = u`:
//! `warp(moving, u)[p] = moving(p + u(p)) = I(p + u(p)) = fixed[p]`.
//! `u` is therefore the unique correct answer, not an approximation of one.
//!
//! # Accuracy budget
//!
//! The tolerance is the sum of two terms, both fixed before any registration is
//! run; neither is fitted to an observed result.
//!
//! **1. Trilinear reconstruction of the moving image (`INTERPOLATION_BIAS`).**
//! `fixed` is analytic but `moving` is known only at grid nodes and is resampled
//! by trilinear interpolation, so the registration drives the intensity residual
//! to zero against the *interpolant*, not against `I`, and the displacement it
//! settles on is offset by the reconstruction error divided by the gradient. For
//! a smooth `f` on unit grid spacing the linear-interpolation error at fractional
//! cell position `s` is `−½·f''·s(1−s)·h²`; averaging `s(1−s)` over the cell
//! gives `1/6`, so the mean magnitude is `|f''|·h²/12` per axis. See
//! [`INTERPOLATION_BIAS`] for the arithmetic on this image.
//!
//! **2. Sub-voxel accuracy target (`SUB_VOXEL_LIMIT`).** Displacements differing
//! by less than the grid spacing `h` are only weakly separated by data sampled at
//! spacing `h`; `h/2` is the conventional ambiguity limit. The tests require
//! `h/4 = 0.25` voxel — twice as strict as that limit — as the accuracy the
//! algorithm must reach on a smooth, well-textured, fully converged problem.
//!
//! # Why the image's structural scale is part of the specification
//!
//! The Thirion force `diff·∇F/(|∇F|² + …)` is a rank-1 projection onto the
//! fixed-image gradient: at any single voxel it constrains only the component of
//! `u − d` parallel to `∇F`, and the perpendicular components are supplied
//! entirely by the Gaussian field smoother coupling neighbouring voxels. If the
//! gradient direction is near-constant across a smoothing neighbourhood, the
//! perpendicular components stay under-determined and a spatially varying
//! deformation is systematically under-recovered — the aperture problem, an
//! intrinsic property of first-order Demons rather than a defect.
//!
//! This was measured directly (20³ grid, `Λ = 20` deformation, 240 iterations,
//! recovered-vs-true amplitude ratio `α = d·u/|u|²`):
//!
//! ```text
//!   periods (voxels)   σ = 0.75            σ = 1.00
//!   [ 5,  6,  7]       α 0.879  rms 0.171  α 0.841  rms 0.204
//!   [ 7,  8,  9]       α 0.856  rms 0.205  α 0.821  rms 0.238
//!   [ 9, 11, 13]       α 0.798  rms 0.295  α 0.767  rms 0.322
//!   [13, 15, 17]       α 0.712  rms 0.450  α 0.692  rms 0.464
//! ```
//!
//! `α` tracks the structural period and is nearly independent of `σ` (at
//! `[13,15,17]` it stays within 0.69–0.72 across `σ ∈ [0.4, 1.0]`), which rules
//! out the diffusion regulariser's spectral attenuation as the cause and
//! confirms the aperture explanation. [`PERIODS`] is therefore chosen so the
//! gradient direction decorrelates within the smoother's support, making the
//! measurement one of registration accuracy rather than of the aperture problem.
//! A constant translation is unaffected — a single constant field satisfies every
//! voxel's rank-1 constraint simultaneously — which is why translation recovers
//! to within 0.08 voxel at every period tested.

use ritk_filter::GaussianSigma;
use ritk_registration::{DemonsConfig, DiffeomorphicDemonsRegistration, ThirionDemonsRegistration};

// ── Synthetic problem definition ─────────────────────────────────────────────

/// Volume extent `[nz, ny, nx]`.
///
/// 20³ = 8000 voxels leaves a 10³ interior after the boundary margin — enough
/// samples for the RMS statistic to be stable — while keeping the debug-profile
/// (`opt-level = 0`) runtime of a 240-iteration Demons run near 0.3 s, far
/// inside the workspace's 30 s per-test budget.
const DIMS: [usize; 3] = [20, 20, 20];

/// Isotropic unit voxel spacing, so displacement voxel units equal physical units.
const SPACING: [f64; 3] = [1.0, 1.0, 1.0];

/// Grid spacing in voxels, the unit the accuracy budget is expressed in.
const GRID_SPACING: f64 = 1.0;

/// Voxels excluded from the accuracy statistic at each face.
///
/// Three sources of boundary error must be excluded: the one-sided finite
/// differences `compute_gradient` uses on the outermost plane, the replicate
/// boundary condition of the Gaussian field smoother (support `3σ = 2.25`
/// voxels, so 3), and warp samples that fall outside the grid and are clamped
/// (up to `max|u| = 2` voxels). `3 + 2 = 5` covers all three.
const MARGIN: usize = 5;

/// Sinusoid periods (voxels) along `[z, y, x]`.
///
/// Two competing constraints fix this choice, both stated in the module docs:
/// the period must be **short** enough that the gradient direction decorrelates
/// within the smoother's `3σ ≈ 2.25`-voxel support (otherwise the aperture
/// problem, not the registration, dominates the measurement), and **long**
/// enough that no applied displacement reaches half a period, which would admit
/// an aliased second solution. The largest per-axis displacement is 1.5 voxels
/// against a shortest half-period of 3.5, leaving better than a 2× margin.
/// The three values are pairwise co-prime so the pattern does not repeat over
/// the volume.
const PERIODS: [f64; 3] = [7.0, 8.0, 9.0];

/// Per-axis sinusoid amplitude (intensity units).
const AMPLITUDE: f64 = 40.0;

/// Mean intensity; with three ±40 sinusoids the range is `[8, 248]`.
const MEAN_INTENSITY: f64 = 128.0;

/// RMS gradient magnitude of `I`, in intensity units per voxel.
///
/// Per axis the peak is `A·2π/L = (35.904, 31.416, 27.925)`, and a sinusoid's
/// RMS is `1/√2` of its peak, giving `(25.387, 22.214, 19.747)`; the magnitude
/// is their root-sum-square, `√1527.9 = 39.089`.
const RMS_GRADIENT: f64 = 39.089;

/// Mean trilinear reconstruction error of `I`, in intensity units.
///
/// Summing the three axis curvature peaks `A·(2π/L)² = (32.219, 24.674, 19.496)`
/// gives `76.389`; a sinusoid's RMS is `1/√2` of its peak and the mean cell sag
/// is `|f''|·h²/12`, so the mean error is `76.389/(√2·12) = 4.501`.
const RECONSTRUCTION_ERROR: f64 = 4.501;

/// Displacement error (voxels) contributed by trilinear reconstruction of
/// `moving`: `4.501 / 39.089 = 0.1152`. Accuracy budget, term 1.
const INTERPOLATION_BIAS: f64 = RECONSTRUCTION_ERROR / RMS_GRADIENT;

/// Sub-voxel accuracy target, `h/4`. Accuracy budget, term 2.
const SUB_VOXEL_LIMIT: f64 = GRID_SPACING / 4.0;

/// Total permitted RMS error of the recovered displacement field, in voxels:
/// `0.1152 + 0.25 = 0.3652`.
const RECOVERY_TOLERANCE: f64 = INTERPOLATION_BIAS + SUB_VOXEL_LIMIT;

/// Diffusion smoothing width used by every test, in voxels.
const SIGMA_DIFFUSION: f64 = 0.75;

/// Iteration count used by every test.
///
/// Demons is run to its fixed point rather than stopped early, so the tests
/// measure the algorithm's converged accuracy and not its convergence rate:
/// raising this to 400 changes the recovered field's RMS error by less than
/// 1e-3 voxel.
const ITERATIONS: usize = 240;

/// Analytic intensity at a continuous voxel coordinate.
///
/// `I(z,y,x) = 128 + 40·[sin(2πz/7) + sin(2πy/8) + sin(2πx/9)]`
///
/// A sum (not a product) of sinusoids keeps every gradient component independent
/// of the other two axes, so the planes on which one component's gradient
/// vanishes are not shared between components and the gradient never vanishes
/// entirely except at isolated points.
fn intensity(coord: [f64; 3]) -> f32 {
    let structure: f64 = coord
        .iter()
        .zip(PERIODS.iter())
        .map(|(c, period)| (std::f64::consts::TAU * c / period).sin())
        .sum();
    (MEAN_INTENSITY + AMPLITUDE * structure) as f32
}

/// Sample the analytic image over `DIMS` at `index + displacement(index)`.
///
/// `displacement` returning zero yields the moving image; returning the
/// ground-truth field `u` yields the fixed image.
fn sample_volume(displacement: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<f32> {
    let [nz, ny, nx] = DIMS;
    (0..nz * ny * nx)
        .map(|index| {
            let base = [
                (index / (ny * nx)) as f64,
                ((index / nx) % ny) as f64,
                (index % nx) as f64,
            ];
            let offset = displacement(base);
            intensity([
                base[0] + offset[0],
                base[1] + offset[1],
                base[2] + offset[2],
            ])
        })
        .collect()
}

/// Accuracy of a recovered displacement field against its ground truth.
#[derive(Debug)]
struct Recovery {
    /// RMS of the error vector magnitude `|d − u|` over the interior region.
    rms: f64,
    /// Per-component mean of `d − u` — the systematic bias, which exposes a
    /// wrong sign or a permuted axis that an unsigned magnitude statistic hides.
    mean_error: [f64; 3],
    /// Per-component mean of the recovered field itself.
    mean_recovered: [f64; 3],
    /// RMS of `|u|`: the error a registration returning a zero field would score.
    baseline_rms: f64,
    /// `d·u / |u|²`, the best-fit uniform scale of the recovered field. 1.0 means
    /// the deformation's amplitude is fully recovered; a value below 1 is the
    /// under-recovery signature described in the module docs.
    amplitude_ratio: f64,
}

/// Compare a recovered displacement field against the analytic ground truth
/// over the interior region.
fn recovery(disp: [&[f32]; 3], truth: impl Fn([f64; 3]) -> [f64; 3]) -> Recovery {
    let [nz, ny, nx] = DIMS;
    let mut sq_error = 0.0_f64;
    let mut sq_truth = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut sum_error = [0.0_f64; 3];
    let mut sum_recovered = [0.0_f64; 3];
    let mut count = 0_usize;

    for iz in MARGIN..nz - MARGIN {
        for iy in MARGIN..ny - MARGIN {
            for ix in MARGIN..nx - MARGIN {
                let flat = (iz * ny + iy) * nx + ix;
                let expected = truth([iz as f64, iy as f64, ix as f64]);
                let recovered = [
                    f64::from(disp[0][flat]),
                    f64::from(disp[1][flat]),
                    f64::from(disp[2][flat]),
                ];
                for (((sum_e, sum_r), &r), &e) in sum_error
                    .iter_mut()
                    .zip(sum_recovered.iter_mut())
                    .zip(recovered.iter())
                    .zip(expected.iter())
                {
                    let error = r - e;
                    sq_error += error * error;
                    sq_truth += e * e;
                    dot += r * e;
                    *sum_e += error;
                    *sum_r += r;
                }
                count += 1;
            }
        }
    }

    let n = count as f64;
    Recovery {
        rms: (sq_error / n).sqrt(),
        mean_error: sum_error.map(|s| s / n),
        mean_recovered: sum_recovered.map(|s| s / n),
        baseline_rms: (sq_truth / n).sqrt(),
        amplitude_ratio: dot / sq_truth,
    }
}

/// Demons configuration shared by every recovery test.
///
/// `sigma_fluid` is disabled so diffusion is the only regularisation in play.
/// `max_step_length = 2.0` exceeds the largest applied displacement (1.95
/// voxels), so the force clamp never truncates the correct answer.
fn recovery_config() -> DemonsConfig {
    DemonsConfig {
        max_iterations: ITERATIONS,
        sigma_diffusion: Some(
            GaussianSigma::new(SIGMA_DIFFUSION).expect("SIGMA_DIFFUSION is positive"),
        ),
        sigma_fluid: None,
        max_step_length: 2.0,
    }
}

// ── Ground-truth deformations ────────────────────────────────────────────────

/// Constant translation `(dz, dy, dx) = (−1.0, 0.75, 1.5)`, magnitude 1.95 voxels.
///
/// The three components are distinct and not all of one sign, so an axis
/// permutation or a sign inversion in the force, warp, or accumulation path
/// produces a component-wise mismatch that the per-component bias assertion
/// catches. None is an integer multiple of another, so a partially applied field
/// cannot coincidentally match.
const TRANSLATION: [f64; 3] = [-1.0, 0.75, 1.5];

/// Amplitude of the non-rigid deformation, in voxels.
const NONRIGID_AMPLITUDE: f64 = 1.0;

/// Wavelength of the non-rigid deformation, in voxels. Equal to the volume
/// extent, so exactly one period spans the grid.
const NONRIGID_WAVELENGTH: f64 = 20.0;

/// Divergence-free smooth deformation of amplitude 1 voxel and wavelength 20.
///
/// Each component varies along a *different* axis than the one it displaces, so
/// `∂u_z/∂z = ∂u_y/∂y = ∂u_x/∂x = 0` and `div u ≡ 0` exactly: the deformation is
/// volume-preserving and its Jacobian determinant stays near 1, so no folding
/// occurs and the deformation is invertible over the whole grid.
fn smooth_deformation(coord: [f64; 3]) -> [f64; 3] {
    let [z, y, x] = coord;
    let wave = |c: f64| (std::f64::consts::TAU * c / NONRIGID_WAVELENGTH).sin();
    [
        NONRIGID_AMPLITUDE * wave(x),
        NONRIGID_AMPLITUDE * wave(z),
        NONRIGID_AMPLITUDE * wave(y),
    ]
}

/// Assert that a recovered field reproduces `TRANSLATION`.
///
/// Shared by the Thirion and diffeomorphic tests, which differ only in the
/// algorithm that produced the field.
fn assert_translation_recovered(algorithm: &str, measured: &Recovery, final_mse: f64) {
    assert!(
        measured.rms < RECOVERY_TOLERANCE,
        "{algorithm} failed to recover the translation {TRANSLATION:?}: RMS error {:.4} voxel \
         exceeds the budget {RECOVERY_TOLERANCE:.4} (interpolation {INTERPOLATION_BIAS:.4} \
         + sub-voxel target {SUB_VOXEL_LIMIT:.4}); zero-field baseline is {:.4}, \
         recovered mean {:?}, amplitude ratio {:.4}, final MSE {final_mse:.4}",
        measured.rms,
        measured.baseline_rms,
        measured.mean_recovered,
        measured.amplitude_ratio
    );
    // The only systematic term for a spatially constant ground truth is the
    // trilinear reconstruction bias, so that term alone bounds the per-axis mean.
    for (axis, bias) in measured.mean_error.iter().enumerate() {
        assert!(
            bias.abs() < INTERPOLATION_BIAS,
            "{algorithm} axis {axis}: mean recovered displacement {:.4} differs from the \
             applied {:.4} by {bias:.4} voxel, exceeding the systematic-bias budget \
             {INTERPOLATION_BIAS:.4}",
            measured.mean_recovered[axis],
            TRANSLATION[axis]
        );
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// Classic Thirion Demons must recover a known constant translation.
///
/// This is the weakest deformation the algorithm can be asked to find and the
/// one case free of the aperture limitation described in the module docs, so it
/// is the case that must work before any non-rigid claim is credible.
///
/// A registration returning a zero field scores `|t| = 1.953`, so clearing the
/// 0.3652-voxel budget requires recovering 81% of the applied displacement; the
/// per-axis bias bound of 0.1152 voxel is the tighter assertion and requires
/// essentially full recovery on every axis independently.
#[test]
fn thirion_demons_recovers_a_known_translation() {
    let moving = sample_volume(|_| [0.0; 3]);
    let fixed = sample_volume(|_| TRANSLATION);

    let result = ThirionDemonsRegistration::new(recovery_config())
        .register(&fixed, &moving, DIMS, SPACING)
        .expect("matched image lengths and dims");

    let measured = recovery([&result.disp_z, &result.disp_y, &result.disp_x], |_| {
        TRANSLATION
    });
    assert_translation_recovered("Thirion Demons", &measured, result.final_mse);
}

/// Diffeomorphic Demons must recover the same known translation through its
/// velocity-field parameterisation.
///
/// The recovered `disp_*` is `exp(v)` computed by scaling and squaring. For a
/// spatially constant velocity field the flow for unit time is translation by
/// exactly `v`, and composing `2ⁿ` copies of the constant field `v/2ⁿ` is exact
/// under trilinear interpolation (interpolating a constant field returns the
/// constant), so the ground truth for the exponentiated field is the same
/// `TRANSLATION` and the same budget applies. A defect in the exponential map —
/// a wrong squaring count, a composition that adds instead of composes — shows
/// up here as an amplitude error the assertion catches.
#[test]
fn diffeomorphic_demons_recovers_a_known_translation() {
    let moving = sample_volume(|_| [0.0; 3]);
    let fixed = sample_volume(|_| TRANSLATION);

    let result = DiffeomorphicDemonsRegistration::new(recovery_config())
        .register(&fixed, &moving, DIMS, SPACING)
        .expect("matched image lengths and dims");

    let measured = recovery([&result.disp_z, &result.disp_y, &result.disp_x], |_| {
        TRANSLATION
    });
    assert_translation_recovered("Diffeomorphic Demons", &measured, result.final_mse);
}

/// Thirion Demons must recover a known non-rigid deformation.
///
/// The deformation is spatially varying, so unlike the translation case no
/// single constant field can satisfy it: passing requires the registration to
/// reproduce the field's spatial structure, not merely its mean. A registration
/// that collapses the deformation to its spatial mean — the classic failure mode
/// of an over-regularised deformable method, and the zero field here, since the
/// deformation integrates to zero over the interior — scores the baseline 1.225
/// and fails by a factor of 3.4.
///
/// The amplitude-ratio assertion is the sharper of the two: it fails if the
/// recovered field points the right way but is systematically scaled down, which
/// an RMS bound alone partly tolerates. Its 0.75 floor sits below the 0.856
/// measured here and above the 0.712 the same algorithm produces once the
/// image's structural scale stops breaking the aperture problem (module docs),
/// so it discriminates a real regression from measurement noise.
#[test]
fn thirion_demons_recovers_a_known_smooth_deformation() {
    let moving = sample_volume(|_| [0.0; 3]);
    let fixed = sample_volume(smooth_deformation);

    let result = ThirionDemonsRegistration::new(recovery_config())
        .register(&fixed, &moving, DIMS, SPACING)
        .expect("matched image lengths and dims");

    let measured = recovery(
        [&result.disp_z, &result.disp_y, &result.disp_x],
        smooth_deformation,
    );

    assert!(
        measured.rms < RECOVERY_TOLERANCE,
        "Thirion Demons failed to recover the smooth deformation: RMS error {:.4} voxel \
         exceeds the budget {RECOVERY_TOLERANCE:.4} (interpolation {INTERPOLATION_BIAS:.4} \
         + sub-voxel target {SUB_VOXEL_LIMIT:.4}); zero-field baseline is {:.4}, \
         amplitude ratio {:.4}, final MSE {:.4}",
        measured.rms,
        measured.baseline_rms,
        measured.amplitude_ratio,
        result.final_mse
    );
    assert!(
        measured.amplitude_ratio > 0.75,
        "Thirion Demons under-recovered the deformation's amplitude: best-fit scale {:.4} \
         is below the 0.75 floor (RMS error {:.4}, final MSE {:.4})",
        measured.amplitude_ratio,
        measured.rms,
        result.final_mse
    );
}
