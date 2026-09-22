//! Shared forward-Euler evolution engine for the gradient-magnitude
//! level-set solvers.
//!
//! [`LaplacianLevelSet`](super::LaplacianLevelSet),
//! [`ThresholdLevelSet`](super::ThresholdLevelSet),
//! [`ShapeDetectionSegmentation`](super::ShapeDetectionSegmentation), and
//! [`GeodesicActiveContourSegmentation`](super::GeodesicActiveContourSegmentation)
//! each evolved `phi` through their own copy of the same loop: curvature,
//! φ-gradient, an optional upwind-advection pre-pass, the slice-parallel
//! update, the double-buffer swap, and a convergence test. Eight copies of
//! that scaffolding had to be kept in step with one another; the solvers
//! differed only in how a point's increment `dphi` is computed and which
//! convergence metric closes the iteration.
//!
//! This module owns the scaffolding once. A solver supplies:
//!
//! - a **pre-step** (`FnMut(&[f64], &mut Scratch)`) for per-iteration fields
//!   derived from `phi` — the edge-stopping solvers write their upwind
//!   advection term into [`Scratch::extra`]; Laplacian and Threshold pass a
//!   no-op;
//! - a **step** (`Fn(usize, f64, &Scratch) -> f64`) returning the increment
//!   `dphi` at one index given `|∇phi|` and the scratch view — the solvers'
//!   physics lives exactly here;
//! - a **convergence** marker type — the zero-sized [`MaxAbsRate`] or
//!   [`RootMeanSquare`], selected at the call site and monomorphized away,
//!   so no metric branch survives in the element loop.
//!
//! Everything is generic and monomorphized: the step and pre-step closures
//! inline into the slice-parallel loop, the markers carry no state, and the
//! per-iteration scratch is allocated once for the whole evolution
//! (the SEG-01 rationale preserved from the geodesic solver: re-allocating
//! the gradient buffers per PDE iteration cost `4 × N × 8` bytes of heap
//! traffic on every sweep).

use super::ops::{compute_curvature_into, compute_field_gradient_into, evolve_slices_with_metric};

/// Validate that the image and level-set field share one shape, returning it.
///
/// The identical check opened every `apply`/`apply_native` pair in this
/// module; the error text is unchanged from those copies so existing
/// callers matching on the message see the same string.
pub(crate) fn checked_dims(
    image_dims: [usize; 3],
    phi_dims: [usize; 3],
) -> anyhow::Result<[usize; 3]> {
    if image_dims != phi_dims {
        anyhow::bail!(
            "image shape {:?} and initial_phi shape {:?} must match",
            image_dims,
            phi_dims
        );
    }
    Ok(image_dims)
}

/// Threshold `phi` to the binary segmentation mask.
///
/// Polarity shared by all four engine-driven solvers: `1.0` where
/// `phi < 0` (inside the contour), `0.0` elsewhere. (Chan–Vese uses the
/// opposite polarity and does not run through this engine.)
pub(crate) fn binary_mask(phi: &[f64]) -> Vec<f32> {
    phi.iter()
        .map(|&v| if v < 0.0 { 1.0_f32 } else { 0.0_f32 })
        .collect()
}

/// Per-iteration scratch shared by the pre-step, the step, and the engine.
///
/// `kappa` and the φ-gradient buffers are refreshed by the engine each
/// iteration; `extra` is solver-owned payload — the edge-stopping solvers
/// keep their upwind advection term there so the step closure reads it
/// without a second borrow channel.
pub(crate) struct Scratch {
    /// Mean curvature `κ = div(∇φ / |∇φ|)` of the current field.
    pub(crate) kappa: Vec<f64>,
    /// Auxiliary per-iteration field (upwind `∇g·∇φ` for edge-stopping
    /// solvers; unused by Laplacian and Threshold).
    pub(crate) extra: Vec<f64>,
    grad_z: Vec<f64>,
    grad_y: Vec<f64>,
    grad_x: Vec<f64>,
}

/// How the per-slice increments reduce to a convergence verdict.
///
/// Implementations are zero-sized markers selected with turbofish at the
/// call site (`evolve_to_convergence::<MaxAbsRate, _, _>`), so the metric
/// is monomorphized into the loop rather than branched on per element.
pub(crate) trait Convergence: Copy + Send + Sync + 'static {
    /// The fresh per-slice accumulator value.
    fn initial() -> f64;
    /// Fold one element's increment into its slice accumulator.
    fn accumulate(local: &mut f64, dphi: f64, dt: f64);
    /// Decide the iteration from the per-slice accumulators.
    fn converged(accum: &[f64], n: usize, tolerance: f64) -> bool;
}

/// `max |Δφ| / dt < tolerance` over all slices — the criterion used by
/// Laplacian, Threshold, and Shape Detection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaxAbsRate;

impl Convergence for MaxAbsRate {
    #[inline]
    fn initial() -> f64 {
        0.0
    }

    #[inline]
    fn accumulate(local: &mut f64, dphi: f64, dt: f64) {
        let rate = dphi.abs() / dt;
        if rate > *local {
            *local = rate;
        }
    }

    #[inline]
    fn converged(accum: &[f64], _n: usize, tolerance: f64) -> bool {
        accum.iter().copied().fold(0.0_f64, f64::max) < tolerance
    }
}

/// `sqrt(Σ Δφ² / N) < tolerance` — ITK's
/// `FiniteDifferenceImageFilter::GetRMSChange()` criterion, used by
/// Geodesic Active Contour.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RootMeanSquare;

impl Convergence for RootMeanSquare {
    #[inline]
    fn initial() -> f64 {
        0.0
    }

    #[inline]
    fn accumulate(local: &mut f64, dphi: f64, _dt: f64) {
        *local += dphi * dphi;
    }

    #[inline]
    fn converged(accum: &[f64], n: usize, tolerance: f64) -> bool {
        let sum_sq: f64 = accum.iter().sum();
        (sum_sq / n as f64).sqrt() < tolerance
    }
}

/// Evolve `phi` under `step` until `M` reports convergence or
/// `max_iterations` elapses; returns the converged field.
///
/// The iteration is exactly the sequence the per-solver copies performed:
/// refresh `κ` and `∇φ`, run the pre-step, update every point in parallel
/// into the double buffer, swap, test convergence.
///
/// `step` receives `(idx, |∇phi|, &scratch)` and returns the increment
/// `dphi` for that point; the engine applies `phi[idx] + dphi`. The
/// arithmetic order inside `step` belongs to the solver, so each caller
/// reproduces its original update bit-for-bit.
pub(crate) fn evolve_to_convergence<M, Pre, Step>(
    mut phi: Vec<f64>,
    dims: [usize; 3],
    dt: f64,
    max_iterations: usize,
    tolerance: f64,
    mut pre: Pre,
    step: Step,
) -> Vec<f64>
where
    M: Convergence,
    Pre: FnMut(&[f64], &mut Scratch),
    Step: Fn(usize, f64, &Scratch) -> f64 + Send + Sync,
{
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    debug_assert_eq!(phi.len(), n, "phi length mismatch: {} vs {}", phi.len(), n);
    let slice_len = ny * nx;

    let mut scratch = Scratch {
        kappa: vec![0.0_f64; n],
        extra: vec![0.0_f64; n],
        grad_z: vec![0.0_f64; n],
        grad_y: vec![0.0_f64; n],
        grad_x: vec![0.0_f64; n],
    };
    let mut phi_new = phi.clone();
    let mut accum = vec![0.0_f64; nz];

    for _iter in 0..max_iterations {
        compute_curvature_into(&phi, dims, &mut scratch.kappa);
        compute_field_gradient_into(
            &phi,
            dims,
            &mut scratch.grad_z,
            &mut scratch.grad_y,
            &mut scratch.grad_x,
        );
        pre(&phi, &mut scratch);

        let phi_ref = &phi;
        let scratch_ref = &scratch;
        evolve_slices_with_metric(&mut phi_new, &mut accum, slice_len, |iz, slice| {
            let base = iz * slice_len;
            let mut local = M::initial();
            for (i, out) in slice.iter_mut().enumerate() {
                let idx = base + i;
                let gz = scratch_ref.grad_z[idx];
                let gy = scratch_ref.grad_y[idx];
                let gx = scratch_ref.grad_x[idx];
                let grad_phi_mag = (gz * gz + gy * gy + gx * gx).sqrt();

                let dphi = step(idx, grad_phi_mag, scratch_ref);
                *out = phi_ref[idx] + dphi;
                M::accumulate(&mut local, dphi, dt);
            }
            local
        });

        std::mem::swap(&mut phi, &mut phi_new);

        if M::converged(&accum, n, tolerance) {
            break;
        }
    }

    phi
}
