use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;

/// Downsample filter.
///
/// Reduces the image size by integer factors by keeping every Nth pixel.
/// Updates spacing to reflect the new resolution.
pub struct DownsampleFilter<B: Backend> {
    factors: Vec<usize>,
    _b: std::marker::PhantomData<fn() -> B>,
}

impl<B: Backend> DownsampleFilter<B> {
    /// Create a new downsample filter.
    ///
    /// # Arguments
    /// * `factors` - Downsampling factor for each dimension (must be >= 1).
    pub fn new(factors: Vec<usize>) -> Self {
        Self {
            factors,
            _b: std::marker::PhantomData,
        }
    }

    /// Apply the filter to an image.
    pub fn apply<const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        let (data, origin, mut spacing, direction) = image.clone().into_parts();
        let dims: [usize; D] = data
            .shape()
            .try_into()
            .expect("DownsampleFilter preserves the const-generic image rank");
        let mut output_dims = dims;
        // Origin remains the same if we start sampling at index 0
        // (Physical location of first pixel is unchanged)

        for d in 0..D {
            let factor = if d < self.factors.len() {
                self.factors[d]
            } else {
                self.factors[0]
            };

            if factor <= 1 {
                continue;
            }

            output_dims[d] = dims[d].div_ceil(factor);

            // Update spacing
            spacing[d] *= factor as f64;
        }

        let source = data.to_vec();
        let mut source_strides = [1usize; D];
        let mut output_strides = [1usize; D];
        for axis in (0..D.saturating_sub(1)).rev() {
            source_strides[axis] = source_strides[axis + 1] * dims[axis + 1];
            output_strides[axis] = output_strides[axis + 1] * output_dims[axis + 1];
        }
        let factors: [usize; D] = std::array::from_fn(|axis| {
            self.factors
                .get(axis)
                .copied()
                .unwrap_or(self.factors[0])
                .max(1)
        });
        let output = (0..output_dims.iter().product())
            .map(|linear| {
                let source_index = (0..D)
                    .map(|axis| {
                        let coordinate = (linear / output_strides[axis]) % output_dims[axis];
                        coordinate * factors[axis] * source_strides[axis]
                    })
                    .sum::<usize>();
                source[source_index]
            })
            .collect::<Vec<_>>();

        Image::new(
            Tensor::<f32, B>::from_slice(output_dims, &output),
            origin,
            spacing,
            direction,
        )
    }
}

#[cfg(test)]
#[path = "tests_downsample.rs"]
mod tests;
