//! Marchenko-Pastur principal component analysis (MP-PCA) denoising of a
//! volume series — the `dwidenoise` role.
//!
//! # Mathematical specification
//!
//! Veraart, Novikov, Christiaens, Ades-aron, Sijbers & Fieremans (2016),
//! "Denoising of diffusion MRI using random matrix theory", *NeuroImage* 142,
//! 394–406, §2.1–§2.2.
//!
//! A window of `V` voxels across `D` volumes forms the Casorati matrix
//! `Y ∈ ℝ^{V×D}` (§2.2). With `m = min(V, D)` and `n = max(V, D)`, the
//! eigenvalues `λ₁ ≥ … ≥ λ_m` of the `m × m` Gram matrix divided by `n`
//! (Eq. 2) are, for i.i.d. noise of variance `σ²`, distributed per the
//! Marchenko-Pastur law (Eq. 3) with ratio `γ = m/n` and support
//! `λ± = σ²(1 ± √γ)²` (Eq. 4). The support width is `λ₊ − λ₋ = 4σ²√γ`.
//!
//! The number of signal components `P̂` is the smallest `p` for which the
//! trailing `m − p` eigenvalues are consistent with that law (Eq. 10–11):
//!
//! ```text
//! Σ_{i=p+1}^{m} λᵢ ≥ (m − p) · σ̂²(p),   σ̂²(p) = (λ_{p+1} − λ_m) / (4 √γ_p),   γ_p = (m − p)/n
//! ```
//!
//! and the noise variance is the mean of those trailing eigenvalues (Eq. 12),
//! `σ̂² = Σ_{i>P̂} λᵢ / (m − P̂)`. The window is reconstructed from its top
//! `P̂` principal components — every eigenvalue inside the noise bulk is
//! nullified — and each voxel's output averages the reconstructions of all
//! windows that contain it, the overlapping-window choice §2.2 adopts. No
//! constant in the threshold is empirical: the only inputs are `m`, `n`, and
//! the spectrum.
//!
//! [`MpEstimator`] selects `γ_p`: Veraart's `(m − p)/n`, or the default
//! Cordero-Grande et al. (2019) refinement `(m − p)/(n − p)`, which accounts
//! for the degrees of freedom the `p` removed components take from the larger
//! dimension. Both keep Eq. 12's mean, so `σ̂²` carries the `(n − P̂)/n`
//! downward bias of averaging a rank-reduced residual over `n`.
//!
//! Windows are the configured extent placed around each voxel and shifted
//! inward at the borders, so every window holds exactly `V` voxels. The noise
//! map and component map report each voxel's own window.
//!
//! # Parallelism
//!
//! Windows are reconstructed in parallel on the stack's parallel provider
//! (`moirai`), each worker reusing one window workspace, and their
//! reconstructions are added to the overlap sums in window-centre order, so
//! every voxel averages its windows in the order a sequential sweep would:
//! the output is bitwise independent of the worker count.
//!
//! # Precision
//!
//! Every Gram accumulation, eigendecomposition, and reconstruction runs in the
//! series' scalar type `T`; a caller wanting double-precision work on
//! single-precision data instantiates at `f64`. The eigendecomposition is
//! Householder tridiagonalization plus implicit-shift QL
//! ([`leto_ops::SymmetricEigenWorkspace`], reused per worker), whose computed
//! spectrum is exact for a Gram perturbation `‖E‖₂ ≤ p(m)·ε·‖G‖₂`: each
//! eigenvalue moves by at most `‖E‖₂` (Weyl) and the signal subspace by at
//! most `‖E‖₂` over the gap `λ_P̂ − λ_{P̂+1}` (Davis–Kahan).

mod denoise;
mod error;
mod threshold;
mod window;

pub use denoise::{MpPcaDenoiser, MpPcaOutput};
pub use error::MpPcaError;
pub use threshold::MpEstimator;
pub use window::PatchExtent;

#[cfg(test)]
mod tests;
