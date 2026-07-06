//! White and black top-hat filters for 3-D grayscale images.
//!
//! WTH_B(f)(x) = f(x) - opening_B(f)(x) = f(x) - D_B(E_B(f))(x)
//! BTH_B(f)(x) = closing_B(f)(x) - f(x) = E_B(D_B(f))(x) - f(x)
//!
//! Properties:
//! - WTH and BTH of constant images are 0.
//! - WTH(f) >= 0 (opening is anti-extensive).
//! - BTH(f) >= 0 (closing is extensive).
//!
//! References:
//! - Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press.
//! - Soille, P. (2003). Morphological Image Analysis, 2nd ed. Springer.

use super::grayscale_dilation::GrayscaleDilation;
use super::grayscale_erosion::GrayscaleErosion;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

/// White top-hat filter: WTH_B(f) = f - opening_B(f).
/// Isolates bright structures smaller than the structuring element.
#[derive(Debug, Clone)]
pub struct WhiteTopHatFilter {
    pub radius: usize,
}
impl WhiteTopHatFilter {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let eroded = GrayscaleErosion::new(self.radius).apply(image)?;
        let opened = GrayscaleDilation::new(self.radius).apply(&eroded)?;
        sub_clamp(image, &opened)
    }
}

/// Black top-hat filter: BTH_B(f) = closing_B(f) - f.
/// Isolates dark structures smaller than the structuring element.
#[derive(Debug, Clone)]
pub struct BlackTopHatFilter {
    pub radius: usize,
}
impl BlackTopHatFilter {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dilated = GrayscaleDilation::new(self.radius).apply(image)?;
        let closed = GrayscaleErosion::new(self.radius).apply(&dilated)?;
        sub_clamp(&closed, image)
    }
}

fn sub_clamp<B: Backend>(a: &Image<B, 3>, b: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    let (av, dims) = extract_vec(a)?;
    let (bv, _) = extract_vec(b)?;
    let result: Vec<f32> = av
        .iter()
        .zip(bv.iter())
        .map(|(&ai, &bi)| (ai - bi).max(0.0))
        .collect();
    let device = a.data().device();
    let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
    Ok(Image::new(t, *a.origin(), *a.spacing(), *a.direction()))
}

#[cfg(test)]
#[path = "tests_top_hat.rs"]
mod tests_top_hat;
