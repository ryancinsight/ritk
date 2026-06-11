use ritk_image::Image;

/// Promote a 2-D `Image<B, 2>` to 3-D `Image<B, 3>` by prepending a Z axis
/// with extent 1, spacing 1.0, and origin 0.0.
///
/// The 2-D spatial axes [Y, X] map to the 3-D axes [Y, X] (Z=0).
/// This enables filters that produce 2-D output (e.g. CPR) to be stored in
/// the `Study<B, 3>` type used by `ViewerCore`.
pub(crate) fn promote_2d_to_3d<B: burn::tensor::backend::Backend>(
    image_2d: Image<B, 2>,
) -> anyhow::Result<Image<B, 3>> {
    let (tensor_2d, origin_2d, spacing_2d, _dir_2d) = image_2d.into_parts();
    let [nr, nc] = tensor_2d
        .shape()
        .dims
        .try_into()
        .expect("2-D output must have rank 2");
    let device = tensor_2d.device();
    let vals = tensor_2d
        .into_data()
        .into_vec::<f32>()
        .expect("promote_2d_to_3d requires f32 backend");
    let td_3d = burn::tensor::TensorData::new(vals, burn::tensor::Shape::new([1, nr, nc]));
    let tensor_3d = burn::tensor::Tensor::<B, 3>::from_data(td_3d, &device);
    let origin_3d = ritk_spatial::Point::new([0.0, origin_2d[0], origin_2d[1]]);
    let spacing_3d = ritk_spatial::Spacing::new([1.0, spacing_2d[0], spacing_2d[1]]);
    let dir_3d = ritk_spatial::Direction::identity();
    Ok(Image::new(tensor_3d, origin_3d, spacing_3d, dir_3d))
}
