//! Validation and convergence-window helpers — single source of truth (SSOT)
//! for the per-engine image-dimension and CC-variance convergence boilerplate
//! previously duplicated across every registration engine.
//!
//! Centralising these helpers ensures that:
//! - the [`RegistrationError::DimensionMismatch`](crate::error::RegistrationError::DimensionMismatch)
//!   error wording stays in lockstep across all engines,
//! - the rolling CC-variance convergence test produces identical numerics in
//!   greedy SyN and multi-resolution SyN (the two engines that use it), and
//! - new engines pick up the canonical pre-condition check and convergence
//!   test without re-implementing either.

use std::collections::VecDeque;

use crate::error::RegistrationError;

/// Validate that two image buffers have length `nz * ny * nx`.
///
/// This is the canonical pre-condition check for every deformable-registration
/// engine that takes a `(fixed, moving, dims)` argument triple.  All engines
/// in [`crate::demons`], [`crate::diffeomorphic`], [`crate::lddmm`], and
/// [`crate::atlas`] (via a per-subject loop) call this helper as the first
/// line of their `register*` implementation.
///
/// # Errors
/// - [`RegistrationError::DimensionMismatch`] when `fixed.len() != dims product`
/// - [`RegistrationError::DimensionMismatch`] when `moving.len() != dims product`
pub(crate) fn validate_image_pair(
    fixed: &[f32],
    moving: &[f32],
    dims: [usize; 3],
) -> Result<(), RegistrationError> {
    let n = dims[0] * dims[1] * dims[2];
    if fixed.len() != n {
        return Err(RegistrationError::DimensionMismatch(format!(
            "fixed length {} != dims product {}",
            fixed.len(),
            n
        )));
    }
    if moving.len() != n {
        return Err(RegistrationError::DimensionMismatch(format!(
            "moving length {} != dims product {}",
            moving.len(),
            n
        )));
    }
    Ok(())
}

/// Push `current` into a rolling CC-value window and report whether the
/// variance of the last `window` values is below `threshold`.
///
/// # Returns
/// - `false` until the window is full (so the caller keeps iterating)
/// - `true` exactly when the variance of the last `window` values falls
///   below `threshold` (the caller's `break` condition)
///
/// # Used by
/// - [`crate::diffeomorphic::syn_core`] (greedy SyN)
/// - [`crate::diffeomorphic::multires_syn`] (multi-resolution SyN)
///
/// Both engines previously open-coded this same rolling-window mean+variance
/// check; consolidating it here ensures the two paths produce identical
/// convergence behaviour.
pub(crate) fn cc_converged(
    history: &mut VecDeque<f64>,
    current: f64,
    window: usize,
    threshold: f64,
) -> bool {
    history.push_back(current);
    if history.len() > window {
        history.pop_front();
    }
    if history.len() < window {
        return false;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    let var = history
        .iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f64>()
        / history.len() as f64;
    var < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate_image_pair ──────────────────────────────────────────────

    #[test]
    fn validate_pair_matching_lengths_ok() {
        let dims = [4usize, 4, 4];
        let n = 64;
        let a = vec![0.0_f32; n];
        let b = vec![0.0_f32; n];
        assert!(validate_image_pair(&a, &b, dims).is_ok());
    }

    #[test]
    fn validate_pair_fixed_length_mismatch() {
        let dims = [4usize, 4, 4];
        let a = vec![0.0_f32; 64];
        let b = vec![0.0_f32; 63];
        let err = validate_image_pair(&a, &b, dims).unwrap_err();
        assert!(
            matches!(err, RegistrationError::DimensionMismatch(_)),
            "expected DimensionMismatch, got {err:?}"
        );
    }

    #[test]
    fn validate_pair_moving_length_mismatch() {
        let dims = [4usize, 4, 4];
        let a = vec![0.0_f32; 63];
        let b = vec![0.0_f32; 64];
        let err = validate_image_pair(&a, &b, dims).unwrap_err();
        assert!(matches!(err, RegistrationError::DimensionMismatch(_)));
    }

    #[test]
    fn validate_pair_zero_dim_product_is_handled() {
        // dims product = 0; the helper still reports DimensionMismatch rather
        // than dividing by zero or accepting empty buffers as valid.
        let dims = [0usize, 4, 4];
        let empty: Vec<f32> = vec![];
        let err = validate_image_pair(&empty, &empty, dims).unwrap_err();
        assert!(matches!(err, RegistrationError::DimensionMismatch(_)));
    }

    // ── cc_converged ─────────────────────────────────────────────────────

    #[test]
    fn cc_converged_window_not_full_returns_false() {
        let mut hist = VecDeque::new();
        // window = 3, only 1 value pushed → not full → false
        assert!(!cc_converged(&mut hist, 0.5, 3, 1e-6));
        assert_eq!(hist.len(), 1);
    }

    #[test]
    fn cc_converged_full_window_high_variance_returns_false() {
        let mut hist = VecDeque::new();
        // 4 wildly different values → variance >> 1e-6 → false
        for &v in &[0.1, 0.9, 0.1, 0.9] {
            assert!(!cc_converged(&mut hist, v, 4, 1e-6));
        }
        assert_eq!(hist.len(), 4);
    }

    #[test]
    fn cc_converged_full_window_zero_variance_returns_true() {
        let mut hist = VecDeque::new();
        // 4 identical values → variance = 0 < 1e-6 → true on the 4th push
        for _ in 0..3 {
            assert!(!cc_converged(&mut hist, 0.5, 4, 1e-6));
        }
        assert!(cc_converged(&mut hist, 0.5, 4, 1e-6));
    }

    #[test]
    fn cc_converged_rolling_window_evicts_oldest() {
        let mut hist = VecDeque::new();
        // Push 6 values into a window of size 4 — the oldest 2 should be evicted.
        for &v in &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] {
            let _ = cc_converged(&mut hist, v, 4, 1e-6);
        }
        assert_eq!(hist.len(), 4, "history must be capped at window size");
        // The retained entries are the last 4: 0.3, 0.4, 0.5, 0.6.
        let expected = [0.3, 0.4, 0.5, 0.6];
        for (a, b) in hist.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "expected {b}, got {a}");
        }
    }
}
