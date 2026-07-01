//! Minimal gradient-descent parameter update for Coeus autograd `Var`s.
//!
//! Coeus provides a fused low-level `sgd_step` over raw device buffers, but no
//! `Var`-level parameter-update helper. A tape-based autograd optimizer works
//! by rebuilding the computation graph each iteration from fresh parameter
//! leaves, so the update produces a *new* `requires_grad` leaf rather than
//! mutating the spent one in place. [`sgd_step_var`] is that step — the
//! smallest piece a gradient-descent registration loop needs on top of the
//! differentiable metric.
//!
//! The update is intentionally off-tape (it reads the parameter and its
//! gradient and produces a detached new leaf); migration gate #3 forbids host
//! extraction only on the *differentiable* path, and an optimizer step is by
//! definition not on it.

use coeus_autograd::Var;
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// One vanilla gradient-descent step: returns a fresh `requires_grad` leaf
/// equal to `param − lr · grad`, preserving `param`'s shape.
///
/// Call after `.backward()` on a loss that depends on `param`. The returned
/// `Var` is a new leaf (the input's autograd tape is spent after backward), so
/// a descent loop feeds it into the next forward pass.
///
/// # Panics
///
/// Panics if `param` has no accumulated gradient (i.e. it was not
/// `requires_grad`, or `.backward()` has not been called), or if the parameter
/// tensor is non-contiguous — caller invariants.
#[must_use]
pub fn sgd_step_var<T, B>(param: &Var<T, B>, lr: T) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let grad = param
        .grad()
        .expect("invariant: sgd_step_var requires an accumulated gradient (call after backward on a requires_grad Var)");
    let p = param.tensor.as_slice();
    let g = grad.as_slice();
    debug_assert_eq!(p.len(), g.len(), "parameter and gradient length mismatch");

    let updated: Vec<T> = p.iter().zip(g.iter()).map(|(&pv, &gv)| pv - lr * gv).collect();
    let shape = param.tensor.shape().to_vec();
    Var::new(Tensor::from_slice_on(shape, &updated, &B::default()), true)
}

#[cfg(test)]
#[path = "tests_optim.rs"]
mod tests;
