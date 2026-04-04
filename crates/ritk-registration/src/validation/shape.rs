//! Explicit discrete shape geometry constraints avoiding dimensional divergence.

use crate::error::{RegistrationError, Result};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;

/// Validate that two images strictly mirror shape geometry sizes precisely avoiding structural scaling shifts internally.
pub fn validate_image_shapes<B: Backend, const D: usize>(
    fixed: &Image<B, D>,
    moving: &Image<B, D>,
) -> Result<()> {
    let fixed_shape = fixed.shape();
    let moving_shape = moving.shape();

    if fixed_shape != moving_shape {
        return Err(RegistrationError::ShapeMismatch {
            expected: fixed_shape.to_vec(),
            actual: moving_shape.to_vec(),
        });
    }

    Ok(())
}
