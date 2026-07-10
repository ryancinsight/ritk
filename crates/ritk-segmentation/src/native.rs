//! Coeus-native segmentation boundaries over the canonical flat-buffer cores.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;

use crate::labeling::{connected_components_values, Connectivity, LabelStatistics};
use crate::threshold::apply_binary_threshold_to_slice;

/// Apply an inclusive binary threshold to a Coeus-backed image.
pub fn binary_threshold<B, const D: usize>(
    image: &Image<f32, B, D>,
    lower: f32,
    upper: f32,
    inside_value: f32,
    outside_value: f32,
    backend: &B,
) -> Result<Image<f32, B, D>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    assert!(
        lower <= upper,
        "lower bound {lower} must be <= upper bound {upper}"
    );
    assert!(
        inside_value.is_finite(),
        "inside_value must be finite, got {inside_value}"
    );
    assert!(
        outside_value.is_finite(),
        "outside_value must be finite, got {outside_value}"
    );

    let output = apply_binary_threshold_to_slice(
        image.data_slice()?,
        lower,
        upper,
        inside_value,
        outside_value,
    );
    Image::from_flat_on(
        output,
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        backend,
    )
}

/// Label connected foreground components in a Coeus-backed 3-D image.
pub fn connected_components<B>(
    mask: &Image<f32, B, 3>,
    connectivity: Connectivity,
    background_value: f32,
    backend: &B,
) -> Result<(Image<f32, B, 3>, Vec<LabelStatistics>)>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let dims = mask.shape();
    let (labels, statistics) =
        connected_components_values(mask.data_slice()?, dims, connectivity, background_value);
    let image = Image::from_flat_on(
        labels,
        dims,
        *mask.origin(),
        *mask.spacing(),
        *mask.direction(),
        backend,
    )?;
    Ok((image, statistics))
}

#[cfg(test)]
mod tests {
    use coeus_core::SequentialBackend;
    use ritk_core::{Direction, Point, Spacing};

    use super::*;

    fn image(values: Vec<f32>, dims: [usize; 3]) -> Image<f32, SequentialBackend, 3> {
        Image::from_flat_on(
            values,
            dims,
            Point::new([2.0, 3.0, 4.0]),
            Spacing::new([0.5, 1.0, 2.0]),
            Direction::identity(),
            &SequentialBackend,
        )
        .unwrap()
    }

    #[test]
    fn threshold_preserves_values_shape_and_metadata() {
        let source = image(vec![-1.0, 0.0, 50.0, 100.0, 101.0], [1, 1, 5]);
        let result = binary_threshold(&source, 0.0, 100.0, 1.0, 0.0, &SequentialBackend).unwrap();

        assert_eq!(result.data_slice().unwrap(), &[0.0, 1.0, 1.0, 1.0, 0.0]);
        assert_eq!(result.shape(), source.shape());
        assert_eq!(result.origin(), source.origin());
        assert_eq!(result.spacing(), source.spacing());
        assert_eq!(result.direction(), source.direction());
    }

    #[test]
    fn connected_components_reports_exact_labels_and_counts() {
        let source = image(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 2, 2]);
        let (labels, stats) =
            connected_components(&source, Connectivity::Six, 0.0, &SequentialBackend).unwrap();

        assert_eq!(
            labels.data_slice().unwrap(),
            &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]
        );
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].label, 1);
        assert_eq!(stats[0].voxel_count, 2);
        assert_eq!(stats[1].label, 2);
        assert_eq!(stats[1].voxel_count, 1);
    }
}
