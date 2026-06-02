use crate::filter::ops::{extract_vec_infallible as extract_vec, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Intensity inversion filter.
///
/// # Mathematical Specification
///
/// `out(x) = maximum - in(x)`
///
/// where `maximum` is either user-specified or derived from the image:
///   `maximum = max_{x} in(x)` (ITK default).
///
/// The mapping is an affine reflection of the intensity range about its midpoint.
/// The input's maximum voxel maps to `0.0`; the input's minimum voxel maps to
/// `maximum - min(in)`.
///
/// # Properties
/// - `InvertIntensity(InvertIntensity(I, M), M) = I` (involution when M is fixed).
/// - Constant image with value `c` maps to all-zero output (using auto maximum).
///
/// # References
/// - ITK `itk::InvertIntensityImageFilter<TImage>`.
/// - `SimpleITK::InvertIntensity(image, maximum)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct InvertIntensityFilter {
    /// Fixed inversion maximum.  When `None`, the image maximum is used.
    maximum: Option<f32>,
}

impl InvertIntensityFilter {
    /// Construct with automatic maximum (derived from the input image).
    pub fn new() -> Self {
        Self { maximum: None }
    }

    /// Construct with a fixed maximum value.
    ///
    /// `maximum` must be finite; the result is undefined if `maximum` is NaN or
    /// infinite (no error is raised; values saturate silently).
    pub fn with_maximum(maximum: f32) -> Self {
        Self {
            maximum: Some(maximum),
        }
    }

    /// Apply intensity inversion to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let max_val = self
            .maximum
            .unwrap_or_else(|| vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        let out: Vec<f32> = vals.into_iter().map(|v| max_val - v).collect();
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let t = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data_vec()
    }

    /// Auto maximum: [1,2,3] → max=3, out=[2,1,0].
    #[test]
    fn invert_auto_max() {
        let img = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let out = InvertIntensityFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(
            v,
            vec![2.0_f32, 1.0, 0.0],
            "[1,2,3] with auto max=3 must invert to [2,1,0]"
        );
    }

    /// Fixed maximum: [1,4,7] with max=10 → [9,6,3].
    #[test]
    fn invert_fixed_max() {
        let img = make_image(vec![1.0, 4.0, 7.0], [1, 1, 3]);
        let out = InvertIntensityFilter::with_maximum(10.0).apply(&img);
        let v = vals(&out);
        assert_eq!(
            v,
            vec![9.0_f32, 6.0, 3.0],
            "[1,4,7] inverted with max=10 must yield [9,6,3]"
        );
    }

    /// Minimum maps to (max - min), maximum maps to 0.
    #[test]
    fn invert_max_maps_to_zero_min_maps_to_range() {
        let img = make_image(vec![2.0, 5.0], [1, 1, 2]);
        let out = InvertIntensityFilter::new().apply(&img);
        let v = vals(&out);
        // auto max = 5.0; 5 - 5 = 0, 5 - 2 = 3
        assert_eq!(v[0], 3.0_f32, "minimum voxel 2 with max=5 → 5-2=3");
        assert_eq!(v[1], 0.0_f32, "maximum voxel 5 with max=5 → 5-5=0");
    }

    /// Constant image with auto max → all zero.
    #[test]
    fn invert_constant_auto_max_all_zero() {
        let img = make_image(vec![4.0, 4.0, 4.0], [1, 1, 3]);
        let out = InvertIntensityFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "constant image with auto max → 0 everywhere");
        }
    }

    /// Spatial metadata is preserved.
    #[test]
    fn invert_preserves_metadata() {
        let sp = Spacing::new([1.5, 2.5, 3.5]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![1.0_f32, 3.0], Shape::new([1usize, 1, 2]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(t, Point::new([0.0, 0.0, 0.0]), sp, Direction::identity());
        let out = InvertIntensityFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }
}
