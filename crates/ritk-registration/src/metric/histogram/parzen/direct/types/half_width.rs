//! Half-width computation and constants (SSOT for the ±3σ rule).

/// Maximum Parzen bins a single sample can touch on one axis.
///
/// ±3σ with σ≈5.2 bins → half_width=15 → range≤31. Rounded to 32 for
/// 128-byte SIMD alignment (32×f32 = four AVX2 `__m256` registers).
#[cfg(test)]
pub(crate) const MAX_PARZEN_BINS: usize = 31;

/// Minimum support half-width. Ensures ≥3 bins per side even for σ<1 bin
/// (B-spline-like continuity).
pub(crate) const MIN_HALF_WIDTH: usize = 3;

// ── Half-width computation (SSOT) ─────────────────────────────────────────

/// Support half-width from sigma² via the ±3σ rule.
///
/// Returns `ceil(3*sqrt(sigma_sq)).max(MIN_HALF_WIDTH)` — captures >99.7%
/// of Gaussian mass. SSOT: `sparse.rs` duplicate is `#[cfg(test)]`-only
/// and delegates here.
#[inline]
pub fn compute_half_width(sigma_sq: f32) -> usize {
    let sigma = sigma_sq.sqrt();
    let computed = (3.0 * sigma).ceil() as usize;
    computed.max(MIN_HALF_WIDTH)
}
