//! Differentiable 1-D linear image sampling on Coeus autograd `Var`s.
//!
//! Interpolates a signal at continuous coordinates such that the gradient of
//! the sampled values flows back to the **coordinate** leaf. This is the step
//! that makes a registration loss a function of the transform parameters: the
//! transform produces continuous coordinates, sampling reads the moving image
//! at them, and the coordinate gradient drives the optimizer.
//!
//! # Mechanism (why the coordinate gradient is correct)
//!
//! For a coordinate `x`, linear interpolation is
//! `out = signal[i0]·(1 − f) + signal[i1]·f` where `i0 = clamp(⌊x⌋)`,
//! `i1 = clamp(⌊x⌋+1)`, and `f = x − ⌊x⌋`. The two corner *indices* are
//! piecewise-constant in `x` (Coeus `gather` treats its index as
//! non-differentiable — gradient flows only through the gathered values), so
//! the coordinate gradient flows entirely through the fractional weight:
//!
//! ```text
//! ∂out/∂x = signal[i0]·∂(1−f)/∂x + signal[i1]·∂f/∂x
//!         = −signal[i0] + signal[i1]   (since ∂f/∂x = 1)
//!         = signal[i1] − signal[i0]
//! ```
//!
//! i.e. the local finite slope of the signal — exactly the analytical gradient
//! the tests assert (for a linear ramp `signal[i] = a + b·i`, this is the
//! constant slope `b`).
//!
//! The weight `f` is built as `sub(coords, ⌊coords⌋)`, keeping the coordinate
//! on the autograd tape (its derivative is 1); `⌊coords⌋` and the clamped
//! integer corner indices are materialized as constant (`requires_grad = false`)
//! `Var`s. Reading `coords`' host values to compute those constants does not
//! detach `coords` from the tape — the differentiable path `coords → f → out`
//! is unbroken (migration gate #3).

use coeus_autograd::{add, gather, mul, sub, Var};
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Differentiable linear interpolation of a 1-D `signal` (shape `[L]`) at
/// continuous `coords` (shape `[N]`), returning sampled intensities (shape
/// `[N]`). Out-of-range coordinates clamp to the signal edges (indices are
/// clamped independently, matching the RITK trilinear reference); the weight
/// derives from the unclamped floor, so the mapping is continuous.
///
/// The result's autograd graph links back to `coords` (through the fractional
/// weights) and to `signal` (through `gather`, if `signal` requires grad).
///
/// # Panics
///
/// Panics if `signal` or `coords` is not 1-D, if `signal` is empty, or if
/// either tensor is non-contiguous (host readback of coordinate values and
/// signal length requires contiguity) — all caller invariants, not
/// input-data errors.
pub fn sample_linear_1d_coeus<T, B>(signal: &Var<T, B>, coords: &Var<T, B>) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();

    let signal_shape = signal.tensor.shape();
    assert_eq!(signal_shape.len(), 1, "sample_linear_1d_coeus: signal must be 1-D");
    let len = signal_shape[0];
    assert!(len > 0, "sample_linear_1d_coeus: signal must be non-empty");
    let max_index = len - 1;

    let coords_shape = coords.tensor.shape();
    assert_eq!(coords_shape.len(), 1, "sample_linear_1d_coeus: coords must be 1-D");
    let n = coords_shape[0];

    // Host-read the coordinate values to build the (non-differentiable) floor
    // and clamped integer corner indices. This constructs constants; it does
    // not detach `coords` from the tape used below.
    let coord_vals = coords.tensor.as_slice();

    let mut floor_vals = Vec::with_capacity(n);
    let mut idx0_vals = Vec::with_capacity(n);
    let mut idx1_vals = Vec::with_capacity(n);
    for &c in coord_vals {
        let cf = c.to_f64();
        let floor = cf.floor();
        floor_vals.push(T::from_f64(floor));
        // Independent clamp of each corner to [0, len-1].
        let i0 = clamp_index(floor, max_index);
        let i1 = clamp_index(floor + 1.0, max_index);
        idx0_vals.push(T::from_f64(i0 as f64));
        idx1_vals.push(T::from_f64(i1 as f64));
    }

    let floor_const = Var::new(Tensor::from_slice_on([n], &floor_vals, &backend), false);
    let idx0 = Var::new(Tensor::from_slice_on([n], &idx0_vals, &backend), false);
    let idx1 = Var::new(Tensor::from_slice_on([n], &idx1_vals, &backend), false);
    let ones = Var::new(Tensor::full_on([n], T::one(), &backend), false);

    // Fractional weight f = coords − ⌊coords⌋ (tape: ∂f/∂coords = 1); w0 = 1 − f.
    let w1 = sub(coords, &floor_const);
    let w0 = sub(&ones, &w1);

    // Corner values via differentiable gather (gradient flows to `signal`).
    let v0 = gather(signal, 0, &idx0);
    let v1 = gather(signal, 0, &idx1);

    // out = v0·w0 + v1·w1
    add(&mul(&v0, &w0), &mul(&v1, &w1))
}

/// Clamp a floored coordinate to a valid `[0, max_index]` signal index.
#[inline]
fn clamp_index(floor: f64, max_index: usize) -> usize {
    if floor <= 0.0 {
        0
    } else if floor >= max_index as f64 {
        max_index
    } else {
        floor as usize
    }
}

#[cfg(test)]
#[path = "tests_sampling.rs"]
mod tests;
