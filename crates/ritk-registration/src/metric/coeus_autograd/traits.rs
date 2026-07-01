//! Coeus-native registration `Transform` seam (ADR 0001).
//!
//! Parallel to `ritk_core::transform::Transform` but over Coeus autograd
//! `Var`s: a differentiable map of a `[N, 3]` batch of points to a `[N, 3]`
//! batch, with trainable parameters on the autograd tape so a metric's gradient
//! reaches them. `[N, 3]` is the canonical coordinate convention at this seam
//! (per-axis splitting is an internal sampler detail — see [`super::metric`]).
//!
//! Introduced now (not deferred) because it already has two real implementors
//! ([`super::transform::Translation`], [`super::transform::Affine`]); the
//! analogous Coeus `Metric` trait is deferred until a second metric type exists
//! (ADR 0001, Consequences).

use coeus_autograd::Var;
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_ops::BackendOps;

/// A differentiable coordinate transform on Coeus autograd `Var`s.
///
/// `transform_points` maps `[N, 3]` points (rows ordered `(z, y, x)`) to
/// `[N, 3]` transformed points; the result's autograd graph links back to the
/// implementor's parameters, so `.backward()` on a downstream loss accumulates
/// their gradients.
pub trait CoeusTransform<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    /// Transform a `[N, 3]` batch of points into a `[N, 3]` batch.
    fn transform_points(&self, points: &Var<T, B>) -> Var<T, B>;
}

#[cfg(test)]
#[path = "tests_traits.rs"]
mod tests;
