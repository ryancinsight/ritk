//! Coeus-native segmentation boundaries over the canonical flat-buffer cores.

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image;

use crate::labeling::{
    connected_components_values, relabel::relabel_values, Connectivity, LabelStatistics,
    RelabelStatistics,
};
use crate::region_growing::connected_threshold::flood_fill;
use crate::threshold::apply_binary_threshold_to_slice;
use crate::threshold::multi_otsu::apply_multi_otsu_to_slice;
use ritk_core::spatial::VoxelIndex;

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

/// Re-label connected components by decreasing voxel count on a Coeus-backed image.
pub fn relabel_components<B>(
    labels: &Image<f32, B, 3>,
    minimum_object_size: usize,
    backend: &B,
) -> Result<(Image<f32, B, 3>, Vec<RelabelStatistics>)>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (output, statistics) = relabel_values(labels.data_slice()?, minimum_object_size);
    let image = Image::from_flat_on(
        output,
        labels.shape(),
        *labels.origin(),
        *labels.spacing(),
        *labels.direction(),
        backend,
    )?;
    Ok((image, statistics))
}

/// Segment a Coeus-backed image with Multi-Otsu class labels.
pub fn multi_otsu<B, const D: usize>(
    image: &Image<f32, B, D>,
    num_classes: usize,
    num_bins: usize,
    backend: &B,
) -> Result<Image<f32, B, D>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let output = apply_multi_otsu_to_slice(image.data_slice()?, num_classes, num_bins);
    Image::from_flat_on(
        output,
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        backend,
    )
}

/// Grow a six-connected threshold region on a Coeus-backed image.
pub fn connected_threshold<B>(
    image: &Image<f32, B, 3>,
    seed: VoxelIndex,
    lower: f32,
    upper: f32,
    backend: &B,
) -> Result<Image<f32, B, 3>>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    assert!(
        lower <= upper,
        "lower bound {lower} must be <= upper bound {upper}"
    );
    let shape = image.shape();
    assert!(
        seed[0] < shape[0] && seed[1] < shape[1] && seed[2] < shape[2],
        "seed {:?} is out of bounds for image shape {:?}",
        seed.as_array(),
        shape
    );
    Image::from_flat_on(
        flood_fill(image.data_slice()?, shape, seed, lower, upper),
        shape,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        backend,
    )
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

    #[test]
    fn relabel_components_orders_sizes_and_preserves_metadata() {
        let source = image(vec![1.0, 1.0, 2.0, 2.0, 2.0, 0.0], [1, 1, 6]);
        let (output, statistics) = relabel_components(&source, 0, &SequentialBackend).unwrap();

        assert_eq!(
            output.data_slice().unwrap(),
            &[2.0, 2.0, 1.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(statistics.len(), 2);
        assert_eq!(statistics[0].original_label, 2);
        assert_eq!(statistics[0].new_label, 1);
        assert_eq!(statistics[0].voxel_count, 3);
        assert_eq!(statistics[1].original_label, 1);
        assert_eq!(statistics[1].new_label, 2);
        assert_eq!(statistics[1].voxel_count, 2);
        assert_eq!(output.shape(), source.shape());
        assert_eq!(output.origin(), source.origin());
        assert_eq!(output.spacing(), source.spacing());
        assert_eq!(output.direction(), source.direction());
    }

    #[test]
    fn multi_otsu_assigns_ordered_classes_and_preserves_metadata() {
        let source = image(vec![0.0, 0.0, 10.0, 10.0, 100.0, 100.0], [1, 1, 6]);
        let output = multi_otsu(&source, 3, 256, &SequentialBackend).unwrap();

        assert_eq!(
            output.data_slice().unwrap(),
            &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        );
        assert_eq!(output.shape(), source.shape());
        assert_eq!(output.origin(), source.origin());
        assert_eq!(output.spacing(), source.spacing());
        assert_eq!(output.direction(), source.direction());
    }
}
