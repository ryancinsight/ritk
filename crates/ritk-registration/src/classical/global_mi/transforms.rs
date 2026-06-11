//! Transform-to-matrix conversion helpers and intensity range estimation.

use crate::types::AffineTransform;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::image::Image;
use ritk_transform::{
    AffineTransform as CoreAffineTransform, RigidTransform, TranslationTransform,
};

// ─── Intensity Range Estimation ───────────────────────────────────────────────

/// Estimate the intensity range [min, max] of an image for MI binning.
///
/// Adds a 1% margin to each side to avoid boundary artifacts in Parzen
/// window estimation.
pub(crate) fn estimate_intensity_range<B: Backend, const D: usize>(
    image: &Image<B, D>,
) -> (f32, f32) {
    let slice = image.data_slice();
    let min_val = slice.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    let margin = range * 0.01;
    (min_val - margin, max_val + margin)
}

// ─── Matrix Extraction ────────────────────────────────────────────────────────

/// Extract a 4×4 homogeneous matrix from a rigid transform.
pub(crate) fn rigid_matrix_to_homogeneous<B: Backend>(
    transform: &RigidTransform<B, 3>,
) -> AffineTransform {
    let matrix_3x4 = transform.matrix();
    let data = matrix_3x4.to_data();
    let slice = data
        .as_slice::<f32>()
        .expect("RigidTransform matrix data must be f32");
    let mut result = [0.0f64; 16];
    for (i, &v) in slice.iter().enumerate() {
        result[i] = v as f64;
    }
    AffineTransform(result)
}

/// Extract a 4×4 homogeneous matrix from an affine transform.
pub(crate) fn affine_matrix_to_homogeneous<B: Backend>(
    transform: &CoreAffineTransform<B, 3>,
) -> AffineTransform {
    let mat = transform.matrix();
    let t = transform.translation();
    let mat_data = mat.to_data();
    let t_data = t.to_data();
    let mat_slice = mat_data
        .as_slice::<f32>()
        .expect("rotation matrix tensor data must be contiguous");
    let t_slice = t_data
        .as_slice::<f32>()
        .expect("translation vector tensor data must be contiguous");
    let mut result = [0.0f64; 16];
    for r in 0..3 {
        for c in 0..3 {
            result[r * 4 + c] = mat_slice[r * 3 + c] as f64;
        }
        result[r * 4 + 3] = t_slice[r] as f64;
    }
    result[15] = 1.0;
    AffineTransform(result)
}

/// Extract a 4×4 homogeneous matrix from a translation transform.
pub(crate) fn translation_matrix_to_homogeneous<B: Backend, const D: usize>(
    transform: &TranslationTransform<B, D>,
) -> AffineTransform {
    let t = transform.translation();
    let t_data = t.to_data();
    let t_slice = t_data
        .as_slice::<f32>()
        .expect("translation vector tensor data must be contiguous");
    let mut result = [0.0f64; 16];
    result[0] = 1.0;
    result[5] = 1.0;
    result[10] = 1.0;
    result[15] = 1.0;
    for i in 0..D.min(3) {
        result[i * 4 + 3] = t_slice[i] as f64;
    }
    AffineTransform(result)
}

// ─── Center Computation ───────────────────────────────────────────────────────

/// Compute the physical center of an image from its shape and metadata.
pub(crate) fn compute_image_center<B: Backend, const D: usize>(image: &Image<B, D>) -> [f64; 3] {
    let shape = image.shape();
    let center_indices: Vec<f32> = (0..3)
        .map(|d| shape.get(d).copied().unwrap_or(1) as f32 / 2.0)
        .collect();
    let device = image.data().device();
    let center_tensor = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::from(center_indices.as_slice()),
        &device,
    );
    let physical = image.index_to_world_tensor(center_tensor.unsqueeze_dim(0));
    let physical_flat: Tensor<B, 1> = physical.squeeze();
    let data = physical_flat.into_data();
    let slice = data
        .as_slice::<f32>()
        .expect("image center coordinates tensor data must be contiguous");
    [slice[0] as f64, slice[1] as f64, slice[2] as f64]
}
