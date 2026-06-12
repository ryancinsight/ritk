//! Validated intensity range `[min, max]`.

/// A validated intensity range `[min, max]` where `min ≤ max`.
///
/// # Invariant
/// `self.min() <= self.max()`, enforced at construction via [`IntensityRange::new`].
/// [`IntensityRange::new_unchecked`] bypasses validation for const contexts;
/// callers must uphold the invariant.
///
/// # Example
/// ```rust
/// use ritk_core::statistics::IntensityRange;
///
/// let r = IntensityRange::new(0.0_f32, 1.0).unwrap();
/// assert_eq!(r.min(), 0.0);
/// assert_eq!(r.max(), 1.0);
/// assert!((r.span() - 1.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntensityRange<T: PartialOrd + Copy> {
    min: T,
    max: T,
}

impl<T: PartialOrd + Copy> IntensityRange<T> {
    /// Construct a validated range.
    ///
    /// Returns `None` if `min > max`.
    pub fn new(min: T, max: T) -> Option<Self> {
        if min <= max {
            Some(Self { min, max })
        } else {
            None
        }
    }

    /// Construct a range without validation.
    ///
    /// # Safety invariant
    /// Caller must ensure `min <= max`.
    pub fn new_unchecked(min: T, max: T) -> Self {
        Self { min, max }
    }

    /// Return the lower bound.
    #[inline]
    pub fn min(&self) -> T {
        self.min
    }

    /// Return the upper bound.
    #[inline]
    pub fn max(&self) -> T {
        self.max
    }

    /// Return `max − min`.
    #[inline]
    pub fn span(&self) -> T
    where
        T: std::ops::Sub<Output = T>,
    {
        self.max - self.min
    }
}

impl<T: PartialOrd + Copy + std::fmt::Display> std::fmt::Display for IntensityRange<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}]", self.min, self.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid_range_returns_some() {
        let r = IntensityRange::new(0.0_f32, 1.0);
        assert!(r.is_some(), "valid range must return Some");
        let r = r.unwrap();
        assert_eq!(r.min(), 0.0);
        assert_eq!(r.max(), 1.0);
    }

    #[test]
    fn test_new_equal_bounds_returns_some() {
        // min == max is a degenerate but valid range.
        let r = IntensityRange::new(5.0_f32, 5.0);
        assert!(r.is_some(), "equal bounds must be accepted (min <= max)");
    }

    #[test]
    fn test_new_inverted_bounds_returns_none() {
        let r = IntensityRange::new(1.0_f32, 0.0);
        assert!(r.is_none(), "inverted range must return None");
    }

    #[test]
    fn test_span_equals_max_minus_min() {
        let r = IntensityRange::new_unchecked(-1.0_f32, 1.0);
        assert!((r.span() - 2.0).abs() < 1e-9, "span must equal max - min");
    }

    #[test]
    fn test_display_format() {
        let r = IntensityRange::new_unchecked(0.0_f32, 255.0);
        assert_eq!(format!("{r}"), "[0, 255]");
    }
}
