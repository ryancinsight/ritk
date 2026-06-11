//! Validated Gaussian sigma parameter.

/// Validated Gaussian sigma (standard deviation) parameter.
///
/// # Invariant
/// `sigma > 0.0`. Construction via [`GaussianSigma::new`] enforces this.
/// The inner value is accessible via [`GaussianSigma::get`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct GaussianSigma(f64);

impl GaussianSigma {
    /// Create a `GaussianSigma`, returning `None` if `sigma <= 0.0`.
    pub fn new(sigma: f64) -> Option<Self> {
        if sigma > 0.0 {
            Some(Self(sigma))
        } else {
            None
        }
    }

    /// Create a `GaussianSigma` without validation.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `sigma <= 0.0`.
    pub fn new_unchecked(sigma: f64) -> Self {
        debug_assert!(sigma > 0.0, "GaussianSigma requires sigma > 0, got {sigma}");
        Self(sigma)
    }

    /// Return the inner sigma value.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Default for GaussianSigma {
    /// Default sigma: 1.0 (unit standard deviation).
    fn default() -> Self {
        Self(1.0)
    }
}
