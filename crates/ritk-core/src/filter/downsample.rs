use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use crate::image::Image;

/// Downsample filter.
///
/// Reduces the image size by integer factors by keeping every Nth pixel.
/// Updates spacing to reflect the new resolution.
pub struct DownsampleFilter<B: Backend> {
    factors: Vec<usize>,
    _b: std::marker::PhantomData<B>,
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
    pub fn apply<const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let mut data = image.data().clone();
        let device = data.device();
        let dims: [usize; D] = data.shape().dims();
        
        let mut new_spacing = *image.spacing();
        // Origin remains the same if we start sampling at index 0
        // (Physical location of first pixel is unchanged)
        
        for d in 0..D {
            let factor = if d < self.factors.len() { self.factors[d] } else { self.factors[0] };
            
            if factor <= 1 {
                continue;
            }
            
            let size = dims[d];
            let _new_size = (size + factor - 1) / factor; // ceil division? or just floor? 
            // Standard downsample usually floors: 0, factor, 2*factor...
            // If size is 10, factor 2: 0, 2, 4, 6, 8. Count = 5.
            
            let indices_vec: Vec<i32> = (0..size).step_by(factor).map(|x| x as i32).collect();
            let indices = Tensor::<B, 1, burn::tensor::Int>::from_ints(indices_vec.as_slice(), &device);
            
            data = data.select(d, indices);
            
            // Update spacing
            new_spacing[d] *= factor as f64;
        }

        Image::new(
            data,
            image.origin().clone(),
            new_spacing,
            image.direction().clone(),
        )
    }
}
