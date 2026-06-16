use burn_ndarray::NdArray;
use ritk_image::test_support;
use ritk_image::Image;

type TestBackend = NdArray<f32>;

pub(super) fn make_image<const D: usize>(data: Vec<f32>, dims: [usize; D]) -> Image<TestBackend, D> {
    test_support::make_image(data, dims)
}

fn make_mask_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
    let n = data.len();
    test_support::make_image(data, [n])
}

fn make_mask_2d(data: Vec<f32>, dims: [usize; 2]) -> Image<TestBackend, 2> {
    test_support::make_image(data, dims)
}

fn make_mask_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    test_support::make_image(data, dims)
}

pub(super) const F32_TOL: f32 = 1e-5;

mod overlap;
mod quality;
mod surface;
