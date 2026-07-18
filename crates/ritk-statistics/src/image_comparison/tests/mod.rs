use coeus_core::SequentialBackend;
use ritk_image::test_support;
use ritk_image::Image;

type TestBackend = SequentialBackend;

pub(super) fn make_image<const D: usize>(
    data: Vec<f32>,
    dims: [usize; D],
) -> Image<f32, TestBackend, D> {
    test_support::make_image(data, dims)
}

pub(super) const F32_TOL: f32 = 1e-5;

mod native;
mod overlap;
mod quality;
mod surface;
