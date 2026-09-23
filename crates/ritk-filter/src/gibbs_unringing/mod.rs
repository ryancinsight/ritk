//! Gibbs-ringing removal by local subvoxel shifts — the `mrdegibbs` role.
//!
//! # Mathematical specification
//!
//! Kellner, Dhital, Kiselev & Reisert (2016), "Gibbs-ringing artifact removal
//! based on local subvoxel-shifts", *Magnetic Resonance in Medicine* 76,
//! 1574–1581 (preprint arXiv:1501.07758v1, whose section and equation numbers
//! are cited below). The reference implementation is MRtrix3 `mrdegibbs`
//! (`cmd/mrdegibbs.cpp`, derived from the authors' `unring` code).
//!
//! Truncating k-space convolves the image with a sinc; sampling that sinc at
//! its extrema produces the ringing, sampling it at its zero crossings removes
//! it (Introduction, Fig. 1).
//!
//! **One-dimensional case** (Methods). From a line `I₀` with Fourier
//! coefficients `c₀(k)`, the `2M + 1` shifted lines
//!
//! ```text
//! I_s(x) = (1/N) Σ_k c₀(k) · e^{i2πk(x + s/(2M))/N},   s = −M, …, M      (Eq. 2)
//! ```
//!
//! are formed by phase ramps (for an even `N` the unpaired Nyquist bin is
//! dropped from every shifted line, as the reference implementation does). The
//! oscillation on each side of voxel `x` is the total variation over the
//! window `K = [k₁, k₂]` ([`TvWindow`]):
//!
//! ```text
//! TV⁺_s(x) = Σ_{t=k₁}^{k₂} |I_s(x + t) − I_s(x + t + 1)|
//! TV⁻_s(x) = Σ_{t=k₁}^{k₂} |I_s(x − t) − I_s(x − t − 1)|
//! ```
//!
//! This is the reference implementation's indexing, which realises the
//! paper's stated intent that the central voxel take no part in its own
//! measure when `k₁ ≥ 1`. The shift `r(x)` minimises `min(TV⁺, TV⁻)` over all
//! candidates (Eq. 3–4), and the line is interpolated back to the grid
//! linearly, `I_unring(x) = I_r(x − r/(2M))`, from the neighbour on the side
//! the shift came from. Lines wrap periodically, as the discrete Fourier
//! series does. The defaults are `M = 20` (the reference implementation's
//! `nshifts`) and `K = [1, 3]` (Results, "Numerical Phantoms").
//!
//! **Two-dimensional case** (Methods, Eq. 5–6). With `c_a = 1 + cos k_a` per
//! in-plane frequency, the slice spectrum splits into the part
//! `G_x = c_y / (c_x + c_y)` corrected along `x` and `G_y = c_x / (c_x + c_y)`
//! corrected along `y`, and the two corrected parts are summed. Eq. 5 writes
//! the weighting after the two one-dimensional corrections; this module
//! follows the reference implementation, which splits the spectrum first so
//! that each one-dimensional pass sees the part whose ringing runs along its
//! own axis. Because `G_x + G_y = 1` (kept at the `0/0` joint Nyquist bin by an
//! even split), an artifact-free slice is returned unchanged under either
//! order, up to the linear back-interpolation error.
//!
//! Slices are the planes spanned by the two axes other than the
//! [`SliceAxis`]; every slice of every volume is corrected independently, in
//! parallel on the stack's parallel provider (`moirai`) with one reused
//! workspace per worker, so the output is bitwise independent of the worker
//! count.
//!
//! # Accuracy on smooth input
//!
//! On a band-limited line `A·cos(ωx + φ)` the selected shift may be nonzero,
//! and linear interpolation over one voxel then errs by at most
//! `max_s s(1 − s)/2 · Aω² = Aω²/8` (the Taylor remainder of linear
//! interpolation with unit spacing). The Fourier transforms add rounding of
//! order `ε·log₂N` relative to the amplitude.
//!
//! # Precision
//!
//! Every transform, phase ramp, total variation, and interpolation runs in the
//! series' scalar type `T` through the first-party FFT provider
//! (`apollo_fft`).

mod error;
mod line;
mod params;
mod slice;
mod unring;

pub use error::GibbsError;
pub use params::{SliceAxis, TvWindow};
pub use unring::GibbsUnringer;

#[cfg(test)]
mod tests;
