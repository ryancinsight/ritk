use burn_ndarray::NdArray;
use ritk_image::test_support;
use ritk_image::Image;

type TestBackend = NdArray<f32>;

pub(super) fn make_image<const D: usize>(data: Vec<f32>, dims: [usize; D]) -> Image<TestBackend, D> {
    test_support::make_image(data, dims)
}

pub(super) const F32_TOL: f32 = 1e-5;

mod overlap;
mod quality;
mod surface;
