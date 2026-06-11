//! Spatial gradient and Laplacian operators for regularization.
//!
//! Used exclusively by `regularization/dispatch.rs`. All functions operate
//! on 4-D (2-D field, shape `[B, C, H, W]`) or 5-D (3-D field, shape
//! `[B, C, D, H, W]`) displacement tensors.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Compute spatial gradients using forward finite differences on a 2-D field.
///
/// # Arguments
/// * `field` - Input tensor with shape `[B, C, H, W]`
///
/// # Returns
/// `(grad_h, grad_w)` — gradient in H and W directions, both shape `[B, C, H, W]`.
pub(crate) fn spatial_gradient_2d<B: Backend>(field: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
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

/// Compute spatial gradients using forward finite differences on a 3-D field.
///
/// # Arguments
/// * `field` - Input tensor with shape `[B, C, D, H, W]`
///
/// # Returns
/// `(grad_d, grad_h, grad_w)` — gradients in D, H, and W directions.
pub(crate) fn spatial_gradient_3d<B: Backend>(
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

/// Compute second-order spatial Laplacian on a 2-D field.
///
/// Uses a 4-point stencil (5-point total with center) with zero-padding at boundaries.
///
/// # Arguments
/// * `field` - Input tensor with shape `[B, C, H, W]`
pub(crate) fn spatial_laplacian_2d<B: Backend>(field: Tensor<B, 4>) -> Tensor<B, 4> {
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

/// Compute spatial Laplacian using a 6-point stencil on a 3-D field.
///
/// # Arguments
/// * `field` - Input tensor with shape `[B, C, D, H, W]`
pub(crate) fn spatial_laplacian_3d<B: Backend>(field: Tensor<B, 5>) -> Tensor<B, 5> {
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
