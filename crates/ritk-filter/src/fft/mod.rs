//! FFT / frequency-domain filter suite (GAP-262-FLT-01).
//!
//! This module provides canonical Fast Fourier Transform filters for medical
//! image processing. All filters operate on 2-D or 3-D images with f32 data.
//!
//! # Components
//!
//! | Filter | Description |
//! |---|---|
//! | [`ForwardFftFilter`] | Transform image to frequency domain (R² → C²) |
//! | [`InverseFftFilter`] | Transform frequency domain back to spatial (C² → R²) |
//! | [`FftShiftFilter`] | Swap quadrants of frequency-domain data for display/analysis |
//! | [`FftConvolutionFilter`] | FFT-based convolution: O(N log N) via frequency multiplication |
//! | [`FftNormalizedCorrelationFilter`] | FFT-based normalized cross-correlation for template matching |
//!
//! # Implementation notes
//!
//! - Uses `rustfft` for pure-Rust FFT (no external C/Fortran dependencies)
//! - Composite (non-power-of-two) dimensions are handled by padding to the next
//!   power-of-two in each axis, computing the transform, then cropping back
//! - In-place transforms are used wherever possible to minimize memory allocations
//! - For 3-D images, a separable 2-D FFT is applied per slice along the depth axis,
//!   then a 1-D FFT is applied along the depth axis for each frequency pair
//!
//! # FFT conventions
//!
//! Forward transform (ITK convention):
//!   F(u) = Σ_{x} f(x) · e^{−2πi ⟨u,x⟩}
//! Inverse transform:
//!   f(x) = (1/N) Σ_{u} F(u) · e^{^{+2πi ⟨u,x⟩}}
//!
//! After forward transform, low-frequency components (DC, near-DC) are at the
//! beginning of the array. [`FftShiftFilter`] moves them to the centre for
//! human-readable display and for filters that need centred kernels.

pub mod convolution;
pub mod forward;
pub mod frequency_filter;
pub mod inverse;
pub mod shift;

pub use convolution::{
    fft_nd, FftConvolution3DFilter, FftConvolutionFilter, FftNormalizedCorrelation3DFilter,
    FftNormalizedCorrelationFilter,
};
pub use forward::{ForwardFftFilter, RealToHalfHermitianForwardFftFilter};
pub use frequency_filter::{
    ButterworthHighPass, ButterworthLowPass, FftFilterKind, FrequencyDomainFilter,
    FrequencyResponse, IdealHighPass, IdealLowPass,
};
pub use inverse::{HalfHermitianToRealInverseFftFilter, InverseFftFilter};
pub use shift::FftShiftFilter;
