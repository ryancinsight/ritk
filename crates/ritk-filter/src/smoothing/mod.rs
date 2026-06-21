//! Smoothing filters for 3-D images.
pub mod binomial_blur;
pub mod box_mean;
pub mod box_rank;
pub mod box_sigma;
pub mod convolution;
pub mod local_noise;
pub mod mean;

pub use binomial_blur::BinomialBlurImageFilter;
pub use box_mean::BoxMeanImageFilter;
pub use box_rank::RankImageFilter;
pub use box_sigma::BoxSigmaImageFilter;
pub use convolution::SpatialConvolutionFilter;
pub use local_noise::NoiseImageFilter;
pub use mean::MeanImageFilter;

#[cfg(test)]
mod tests_convolution;
