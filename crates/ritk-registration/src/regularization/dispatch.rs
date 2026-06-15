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
            spatial_laplacian_planar(d4)
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
            spatial_laplacian_volumetric(d5)
                .powf_scalar(2.0)
                .mean()
                .mul_scalar(weight)
        }
        // D is `const`; only D=4 and D=5 ever instantiate this function. The
        // `_` arm is statically unreachable in monomorphized code, but Rust's
        // stable exhaustiveness checker cannot prove that across const
        // generics, so the arm is required for type-checking.
        _ => unreachable!("Laplacian-squared regularizer: D ∈ {{4, 5}}, got D = {D}"),
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
            let (grad_h, grad_w) = spatial_gradient_planar(d4);
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
            let (grad_d, grad_h, grad_w) = spatial_gradient_volumetric(d5);
            (grad_d.powf_scalar(2.0) + grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0))
                .mean()
                .mul_scalar(weight)
        }
        // See `dispatch_laplacian_squared` for the const-generic `_` arm rationale.
        _ => unreachable!("DiffusionRegularizer: D ∈ {{4, 5}}, got D = {D}"),
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
            let (grad_h, grad_w) = spatial_gradient_planar(d4);
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
            let (grad_d, grad_h, grad_w) = spatial_gradient_volumetric(d5);
            let membrane = grad_d.clone().powf_scalar(2.0)
                + grad_h.clone().powf_scalar(2.0)
                + grad_w.clone().powf_scalar(2.0);
            let div_u = grad_d.narrow(1, 0, 1) + grad_h.narrow(1, 1, 1) + grad_w.narrow(1, 2, 1);
            let volume_term = div_u.powf_scalar(2.0);
            membrane.mean() * alpha + volume_term.mean() * beta
        }
        // See `dispatch_laplacian_squared` for the const-generic `_` arm rationale.
        _ => unreachable!("ElasticRegularizer: D ∈ {{4, 5}}, got D = {D}"),
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
            let (grad_h, grad_w) = spatial_gradient_planar(d4);
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
            let (grad_d, grad_h, grad_w) = spatial_gradient_volumetric(d5);
            let grad_mag =
                (grad_d.powf_scalar(2.0) + grad_h.powf_scalar(2.0) + grad_w.powf_scalar(2.0))
                    .sqrt();
            grad_mag.mean().mul_scalar(weight)
        }
        // See `dispatch_laplacian_squared` for the const-generic `_` arm rationale.
        _ => unreachable!("TotalVariationRegularizer: D ∈ {{4, 5}}, got D = {D}"),
    }
}

// ── Private spatial-operator helpers ─────────────────────────────────────────

/// Spatial gradients via forward finite differences on a 2-D field `[B, C, H, W]`.
fn spatial_gradient_planar<B: Backend>(field: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [b, c, h, w] = field.dims();
    let device = field.device();

    // Gradient in H direction (vertical)
    let top = field.clone().slice([0..b, 0..c, 1..h, 0..w]);
    let bottom = field.clone().slice([0..b, 0..c, 0..(h - 1), 0..w]);
    let grad_h = top - bottom;

    // Pad to maintain original size
    let zeros_h = Tensor::zeros([b, c, 1, w], &device);
    let grad_h = Tensor::cat(vec![grad_h, zeros_h], 2);

    // Gradient in W direction (horizontal)
    let left = field.clone().slice([0..b, 0..c, 0..h, 1..w]);
    let right = field.slice([0..b, 0..c, 0..h, 0..(w - 1)]);
    let grad_w = left - right;

    // Pad to maintain original size
    let zeros_w = Tensor::zeros([b, c, h, 1], &device);
    let grad_w = Tensor::cat(vec![grad_w, zeros_w], 3);

    (grad_h, grad_w)
}

/// Spatial gradients via forward finite differences on a 3-D field `[B, C, D, H, W]`.
fn spatial_gradient_volumetric<B: Backend>(
    field: Tensor<B, 5>,
) -> (Tensor<B, 5>, Tensor<B, 5>, Tensor<B, 5>) {
    let [b, c, d, h, w] = field.dims();
    let device = field.device();

    // Gradient in D direction (depth)
    let front = field.clone().slice([0..b, 0..c, 1..d, 0..h, 0..w]);
    let back = field.clone().slice([0..b, 0..c, 0..(d - 1), 0..h, 0..w]);
    let grad_d = front - back;
    let zeros_d = Tensor::zeros([b, c, 1, h, w], &device);
    let grad_d = Tensor::cat(vec![grad_d, zeros_d], 2);

    // Gradient in H direction (height)
    let top = field.clone().slice([0..b, 0..c, 0..d, 1..h, 0..w]);
    let bottom = field.clone().slice([0..b, 0..c, 0..d, 0..(h - 1), 0..w]);
    let grad_h = top - bottom;
    let zeros_h = Tensor::zeros([b, c, d, 1, w], &device);
    let grad_h = Tensor::cat(vec![grad_h, zeros_h], 3);

    // Gradient in W direction (width)
    let left = field.clone().slice([0..b, 0..c, 0..d, 0..h, 1..w]);
    let right = field.slice([0..b, 0..c, 0..d, 0..h, 0..(w - 1)]);
    let grad_w = left - right;
    let zeros_w = Tensor::zeros([b, c, d, h, 1], &device);
    let grad_w = Tensor::cat(vec![grad_w, zeros_w], 4);

    (grad_d, grad_h, grad_w)
}

/// Second-order spatial Laplacian via 4-point stencil on a 2-D field `[B, C, H, W]`.
fn spatial_laplacian_planar<B: Backend>(field: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = field.dims();
    let device = field.device();

    // Central region
    let center = field.clone().slice([0..b, 0..c, 1..(h - 1), 1..(w - 1)]);
    let left = field.clone().slice([0..b, 0..c, 1..(h - 1), 0..(w - 2)]);
    let right = field.clone().slice([0..b, 0..c, 1..(h - 1), 2..w]);
    let top = field.clone().slice([0..b, 0..c, 0..(h - 2), 1..(w - 1)]);
    let bottom = field.slice([0..b, 0..c, 2..h, 1..(w - 1)]);

    let laplacian_center = left + right + top + bottom - center.mul_scalar(4.0);

    // Pad back to original size
    let zeros_h_front = Tensor::zeros([b, c, 1, w - 2], &device);
    let zeros_h_back = Tensor::zeros([b, c, 1, w - 2], &device);
    let laplacian = Tensor::cat(vec![zeros_h_front, laplacian_center, zeros_h_back], 2);
    let zeros_w_front = Tensor::zeros([b, c, h, 1], &device);
    let zeros_w_back = Tensor::zeros([b, c, h, 1], &device);

    Tensor::cat(vec![zeros_w_front, laplacian, zeros_w_back], 3)
}

/// Spatial Laplacian via 6-point stencil on a 3-D field `[B, C, D, H, W]`.
fn spatial_laplacian_volumetric<B: Backend>(field: Tensor<B, 5>) -> Tensor<B, 5> {
    let [b, c, d, h, w] = field.dims();
    let device = field.device();

    // Central region (exclude boundaries)
    let center = field
        .clone()
        .slice([0..b, 0..c, 1..(d - 1), 1..(h - 1), 1..(w - 1)]);

    // Neighbors in each dimension
    let front = field
        .clone()
        .slice([0..b, 0..c, 0..(d - 2), 1..(h - 1), 1..(w - 1)]);
    let back = field
        .clone()
        .slice([0..b, 0..c, 2..d, 1..(h - 1), 1..(w - 1)]);
    let top = field
        .clone()
        .slice([0..b, 0..c, 1..(d - 1), 0..(h - 2), 1..(w - 1)]);
    let bottom = field
        .clone()
        .slice([0..b, 0..c, 1..(d - 1), 2..h, 1..(w - 1)]);
    let left = field
        .clone()
        .slice([0..b, 0..c, 1..(d - 1), 1..(h - 1), 0..(w - 2)]);
    let right = field.slice([0..b, 0..c, 1..(d - 1), 1..(h - 1), 2..w]);

    // 3D Laplacian: sum of second derivatives
    let laplacian_center = front + back + top + bottom + left + right - center.mul_scalar(6.0);

    // Pad back to original size
    let zeros_d_front = Tensor::zeros([b, c, 1, h - 2, w - 2], &device);
    let zeros_d_back = Tensor::zeros([b, c, 1, h - 2, w - 2], &device);
    let laplacian = Tensor::cat(vec![zeros_d_front, laplacian_center, zeros_d_back], 2);
    let zeros_h_front = Tensor::zeros([b, c, d, 1, w - 2], &device);
    let zeros_h_back = Tensor::zeros([b, c, d, 1, w - 2], &device);
    let laplacian = Tensor::cat(vec![zeros_h_front, laplacian, zeros_h_back], 3);
    let zeros_w_front = Tensor::zeros([b, c, d, h, 1], &device);
    let zeros_w_back = Tensor::zeros([b, c, d, h, 1], &device);

    Tensor::cat(vec![zeros_w_front, laplacian, zeros_w_back], 4)
}

#[cfg(test)]
#[path = "tests_dispatch.rs"]
mod tests;
