use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

fn make_mask_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    let device = Default::default();
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn make_mask_2d(data: Vec<f32>, dims: [usize; 2]) -> Image<TestBackend, 2> {
    let device = Default::default();
    let tensor =
        Tensor::<TestBackend, 2>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0]),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    )
}

fn make_mask_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
    let n = data.len();
    let device = Default::default();
    let tensor =
        Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
    Image::new(
        tensor,
        Point::new([0.0]),
        Spacing::new([1.0]),
        Direction::identity(),
    )
}

mod overlap;
mod quality;
mod surface;
