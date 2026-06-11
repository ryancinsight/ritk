//! Dimension dispatch for regularizers.
//!
//! Routes regularizer `compute_loss` calls to the correct dimension-specific
//! implementation based on `const D: usize`. Only D ∈ {4, 5} is supported
//! (representing 2D and 3D displacement fields respectively). Because the
//! dispatch is a simple `match` over a `const` parameter, the compiler
//! monomorphizes each branch and dead-code eliminates unreachable arms —
//! achieving zero-cost dispatch without requiring a `where` bound on callers.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::trait_::utils;

/// Shared implementation for bending-energy and curvature: Laplacian squared, averaged, weighted.
#[inline]
fn dispatch_laplacian_squared<B: Backend, const D: usize>(
    displacement: Tensor<B, D>,
    weight: f64,
) -> Tensor<B, 1> {
    match D {
        4 => {
            let shape = displacement.shape();
            let [b, c, h, w] = [shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]];
            let d4: Tensor<B, 4> = displacement.reshape([b, c, h, w]);
            utils::spatial_laplacian_2d(d4)
                .powf_scalar(2.0)
                .mean()
                .mul_scalar(weight)
        }
        5 => {
            let shape = displacement.shape();
            let [b, c, d, h, w] = [
                shape.dims[0],
                shape.dims[1],
                shape.dims[2],
                shape.dims[3],
                shape.dims[4],
            ];
            let d5: Tensor<B, 5> = displacement.reshape([b, c, d, h, w]);
            utils::spatial_laplacian_3d(d5)
                .powf_scalar(2.0)
                .mean()
                .mul_scalar(weight)
        }
        _ => panic!("Laplacian-squared regularizer only supports D ∈ {{4, 5}}, got D = {D}"),
    }
}

/// Dispatch bending energy loss to dimension-specific Laplacian.
#[inline]
pub fn dispatch_bending_energy<B: Backend, const D: usize>(
    displacement: Tensor<B, D>,
    weight: f64,
) -> Tensor<B, 1> {
    dispatch_laplacian_squared::<B, D>(displacement, weight)
}

/// Dispatch curvature loss to dimension-specific Laplacian.
#[inline]
pub fn dispatch_curvature<B: Backend, const D: usize>(
    displacement: Tensor<B, D>,
    weight: f64,
) -> Tensor<B, 1> {
    dispatch_laplacian_squared::<B, D>(displacement, weight)
}

/// Dispatch diffusion loss to dimension-specific gradient computation.
#[inline]
pub fn dispatch_diffusion<B: Backend, const D: usize>(
    displacement: Tensor<B, D>,
    weight: f64,
) -> Tensor<B, 1> {
    match D {
        4 => {
            let shape = displacement.shape();
            let [b, c, h, w] = [shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]];
            let d4: Tensor<B, 4> = displacement.reshape([b, c, h, w]);
            let (grad_h, grad_w) = utils::spatial_gradient_2d(d4);
            (grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0))
                .mean()
                .mul_scalar(weight)
        }
        5 => {
            let shape = displacement.shape();
            let [b, c, d, h, w] = [
                shape.dims[0],
                shape.dims[1],
                shape.dims[2],
                shape.dims[3],
                shape.dims[4],
            ];
            let d5: Tensor<B, 5> = displacement.reshape([b, c, d, h, w]);
            let (grad_d, grad_h, grad_w) = utils::spatial_gradient_3d(d5);
            (grad_d.powf_scalar(2.0) + grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0))
                .mean()
                .mul_scalar(weight)
        }
        _ => panic!("DiffusionRegularizer only supports D ∈ {{4, 5}}, got D = {D}"),
    }
}

/// Dispatch elastic loss to dimension-specific gradient computation.
#[inline]
pub fn dispatch_elastic<B: Backend, const D: usize>(
    displacement: Tensor<B, D>,
    alpha: f64,
    beta: f64,
) -> Tensor<B, 1> {
    match D {
        4 => {
            // NOTE: Each gradient tensor is consumed twice: once by powf_scalar (membrane)
            // and once by narrow (divergence). The clones are mandatory until burn adds
            // a borrow-based slice/narrow API (slice_ref/narrow_ref). See Sprint 338.
            let shape = displacement.shape();
            let [b, c, h, w] = [shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]];
            let d4: Tensor<B, 4> = displacement.reshape([b, c, h, w]);
            let (grad_h, grad_w) = utils::spatial_gradient_2d(d4);
            let membrane = grad_h.clone().powf_scalar(2.0) + grad_w.clone().powf_scalar(2.0);
            let div_u = grad_h.narrow(1, 0, 1) + grad_w.narrow(1, 1, 1);
            let volume_term = div_u.powf_scalar(2.0);
            membrane.mean() * alpha + volume_term.mean() * beta
        }
        5 => {
            // NOTE: Each gradient tensor is consumed twice: once by powf_scalar (membrane)
            // and once by narrow (divergence). The clones are mandatory until burn adds
            // a borrow-based slice/narrow API (slice_ref/narrow_ref). See Sprint 338.
            let shape = displacement.shape();
            let [b, c, d, h, w] = [
                shape.dims[0],
                shape.dims[1],
                shape.dims[2],
                shape.dims[3],
                shape.dims[4],
            ];
            let d5: Tensor<B, 5> = displacement.reshape([b, c, d, h, w]);
            let (grad_d, grad_h, grad_w) = utils::spatial_gradient_3d(d5);
            let membrane = grad_d.clone().powf_scalar(2.0)
                + grad_h.clone().powf_scalar(2.0)
                + grad_w.clone().powf_scalar(2.0);
            let div_u = grad_d.narrow(1, 0, 1) + grad_h.narrow(1, 1, 1) + grad_w.narrow(1, 2, 1);
            let volume_term = div_u.powf_scalar(2.0);
            membrane.mean() * alpha + volume_term.mean() * beta
        }
        _ => panic!("ElasticRegularizer only supports D ∈ {{4, 5}}, got D = {D}"),
    }
}

/// Dispatch total variation loss to dimension-specific gradient computation.
#[inline]
pub fn dispatch_total_variation<B: Backend, const D: usize>(
    displacement: Tensor<B, D>,
    weight: f64,
) -> Tensor<B, 1> {
    match D {
        4 => {
            let shape = displacement.shape();
            let [b, c, h, w] = [shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]];
            let d4: Tensor<B, 4> = displacement.reshape([b, c, h, w]);
            let (grad_h, grad_w) = utils::spatial_gradient_2d(d4);
            let grad_mag = (grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0)).sqrt();
            grad_mag.mean().mul_scalar(weight)
        }
        5 => {
            let shape = displacement.shape();
            let [b, c, d, h, w] = [
                shape.dims[0],
                shape.dims[1],
                shape.dims[2],
                shape.dims[3],
                shape.dims[4],
            ];
            let d5: Tensor<B, 5> = displacement.reshape([b, c, d, h, w]);
            let (grad_d, grad_h, grad_w) = utils::spatial_gradient_3d(d5);
            let grad_mag =
                (grad_d.powf_scalar(2.0) + grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0))
                    .sqrt();
            grad_mag.mean().mul_scalar(weight)
        }
        _ => panic!("TotalVariationRegularizer only supports D ∈ {{4, 5}}, got D = {D}"),
    }
}

#[cfg(test)]
#[path = "tests_dispatch.rs"]
mod tests;
