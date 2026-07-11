//! Rank promotion for the legacy application filter graph.

use ritk_image::Image;

/// Promote a planar filter result to one depth slice for the active filter graph.
pub(crate) fn elevate_to_volume<B: ritk_image::tensor::Backend>(
    image_2d: Image<B, 2>,
) -> anyhow::Result<Image<B, 3>> {
    let (tensor_2d, origin_2d, spacing_2d, _direction_2d) = image_2d.into_parts();
    let [rows, columns] = tensor_2d
        .shape()
        .dims
        .try_into()
        .expect("invariant: planar filter result has rank two");
    let device = tensor_2d.device();
    let values = tensor_2d
        .into_data()
        .into_vec::<f32>()
        .expect("invariant: Snap filter graph operates on f32 images");
    let tensor = ritk_image::tensor::Tensor::<B, 3>::from_data(
        ritk_image::tensor::TensorData::new(
            values,
            ritk_image::tensor::Shape::new([1, rows, columns]),
        ),
        &device,
    );
    Ok(Image::new(
        tensor,
        ritk_spatial::Point::new([0.0, origin_2d[0], origin_2d[1]]),
        ritk_spatial::Spacing::new([1.0, spacing_2d[0], spacing_2d[1]]),
        ritk_spatial::Direction::identity(),
    ))
}
