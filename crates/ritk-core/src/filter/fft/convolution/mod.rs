//! FFT-based convolution and normalized cross-correlation filters.
//!
//! # Convolution theorem
//!
//! ```text
//! conv(f, g) = IFFT(FFT(f) · FFT(g))
//! ```
//!
//! # Algorithm (2-D "same" convolution)
//!
//! 1. Pad image `[h, w]` and kernel `[kr, kc]` to `[pad_r, pad_c]` where
//!    `pad_r = next_power_of_two(h + kr − 1)` and
//!    `pad_c = next_power_of_two(w + kc − 1)`.
//!    This padding prevents circular aliasing in the linear convolution result.
//! 2. Place both arrays at the **top-left origin** of the padded buffer (no
//!    centring shift), so the kernel phase is zero and no quadrant swap is needed.
//! 3. Apply separable 2-D forward FFT: row-wise, then column-wise.
//! 4. Multiply element-wise in the frequency domain.
//! 5. Apply separable 2-D inverse FFT; normalize by `1 / (pad_r · pad_c)`.
//! 6. Extract the "same" output: a `[h, w]` window starting at
//!    `(⌊kr/2⌋, ⌊kc/2⌋)`.
//!
//! # Proof of "same" crop offset
//!
//! With kernel placed at origin, the circular convolution at position `(r, c)` is
//!
//! ```text
//! C[r, c] = Σ_{r'=r−(kr−1)}^{r} Σ_{c'=c−(kc−1)}^{c} f[r', c'] · g[r−r', c−c']
//! ```
//!
//! The full linear convolution occupies `[0, h+kr−2] × [0, w+kc−2]`. The
//! "same" window of size `[h, w]` that centres output pixel `(r, c)` over input
//! pixel `(r, c)` starts at `(⌊kr/2⌋, ⌊kc/2⌋)`. For the Dirac delta
//! `δ[⌊kr/2⌋, ⌊kc/2⌋] = 1`, the crop recovers `f[r, c]` exactly.

mod conv2d;
mod conv3d;
mod helpers;
mod ncc2d;
mod ncc3d;

pub use conv2d::FftConvolutionFilter;
pub use conv3d::FftConvolution3DFilter;
pub use helpers::{fft2d, fft3d, FftDir};
pub use ncc2d::FftNormalizedCorrelationFilter;
pub use ncc3d::FftNormalizedCorrelation3DFilter;

#[cfg(test)]
mod tests_convolution;
