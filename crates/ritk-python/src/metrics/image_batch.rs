use anyhow::{bail, Result};
use pyo3::prelude::PyRef;

use crate::image::{image_to_vec, PyImage};

pub(super) fn collect_image_vectors(
    images: &[PyRef<'_, PyImage>],
) -> Result<(Vec<Vec<f32>>, [usize; 3])> {
    if images.is_empty() {
        bail!("images list must not be empty");
    }

    let mut vectors = Vec::with_capacity(images.len());
    let mut reference_shape = [0usize; 3];

    for (index, image) in images.iter().enumerate() {
        let (values, shape) = image_to_vec(&image.inner);
        if index == 0 {
            reference_shape = shape;
        } else if shape != reference_shape {
            bail!(
                "shape mismatch: images[0] {:?} != images[{}] {:?}",
                reference_shape,
                index,
                shape
            );
        }
        vectors.push(values);
    }

    Ok((vectors, reference_shape))
}
