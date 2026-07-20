//! Coeus-native registration seams (ADR 0001): [`Transform`] (coordinate
//! transform) and [`Metric`] (intensity-loss reduction).
//!
//! Both are parallel to the coeus-bound `ritk_core`/`ritk_registration` traits
//! but over Coeus autograd `Var`s, with trainable state on the tape so a loss
//! gradient reaches it. Each is introduced once it has ≥2 real implementors
//! (seam-first, not speculative): `Transform` has
//! [`super::transform::Translation`]/[`super::transform::Affine`];
//! `Metric` has [`super::mse::Mse`]/[`super::ncc::Ncc`].

use coeus_autograd::Var;
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float, Scalar};
use coeus_ops::BackendOps;

/// A differentiable coordinate transform on Coeus autograd `Var`s.
///
/// `transform_points` maps `[N, 3]` points (rows ordered `(z, y, x)`) to
/// `[N, 3]` transformed points; the result's autograd graph links back to the
/// implementor's parameters, so `.backward()` on a downstream loss accumulates
/// their gradients.
pub trait Transform<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    /// Transform a `[N, 3]` batch of points into a `[N, 3]` batch.
    fn transform_points(&self, points: &Var<T, B>) -> Var<T, B>;
}

/// A differentiable intensity-similarity reduction on Coeus autograd `Var`s.
///
/// `reduce` maps a `[N]` sampled-intensity vector and the `[N]` fixed-intensity
/// vector to a scalar (`[1]`) minimization loss, keeping the tape intact so the
/// gradient flows back to `sampled` (hence, upstream, to the transform
/// parameters). This is the minimal role interface distinguishing metric types
/// (MSE vs NCC vs future MI); the shared transform-and-sample step is owned by
/// [`super::metric::evaluate`], not by implementors. `T: Float` because
/// correlation/normalization metrics need `sqrt`.
pub trait Metric<T, B>
where
    T: Float,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    /// Reduce `(sampled, fixed)` intensity vectors (`[N]`) to a scalar loss.
    fn reduce(&self, sampled: &Var<T, B>, fixed: &Var<T, B>) -> Var<T, B>;
}

#[cfg(test)]
#[path = "tests_traits.rs"]
mod tests;
