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

/// Validate that a single image buffer has length `nz * ny * nx`.
///
/// The single-image validator.  Used by [`crate::atlas`] to validate each
/// subject in a per-element loop (the atlas has N subject images rather
/// than a fixed/moving pair), and as the inner check for
/// [`validate_image_pair`].
pub(crate) fn validate_image(
    image: &[f32],
    dims: [usize; 3],
) -> Result<(), RegistrationError> {
    let n = dims[0] * dims[1] * dims[2];
    if image.len() != n {
        return Err(RegistrationError::DimensionMismatch(format!(
            "image length {} != dims product {}",
            image.len(),
            n
        )));
    }
    Ok(())
}

/// Validate that two image buffers have length `nz * ny * nx`.
///
/// This is the canonical pre-condition check for every deformable-registration
/// engine that takes a `(fixed, moving, dims)` argument triple.  All engines
/// in [`crate::demons`], [`crate::diffeomorphic`], and [`crate::lddmm`] call
/// this helper as the first line of their `register*` implementation.
///
/// # Errors
/// - [`RegistrationError::DimensionMismatch`] when `fixed.len() != dims product`
/// - [`RegistrationError::DimensionMismatch`] when `moving.len() != dims product`
pub(crate) fn validate_image_pair(
    fixed: &[f32],
    moving: &[f32],
    dims: [usize; 3],
) -> Result<(), RegistrationError> {
    validate_image(fixed, dims)?;
    validate_image(moving, dims)?;
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
    // `window == 0` is a degenerate case: `len > 0` (the eviction guard) is
    // always false, so the history would grow without bound; and `len < 0`
    // (the window-full check) is also always false (usize underflow), so
    // the function would never report convergence.  Treat as "not converged"
    // so the caller keeps iterating and the misconfiguration shows up as a
    // `max_iterations` cap rather than a hang or memory blowup.
    if window == 0 {
        return false;
    }
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
    fn validate_pair_zero_dim_product_with_nonzero_buffer_fails() {
        // dims product = 0 but the buffers are non-empty — the helper must
        // catch the mismatch (don't divide by zero, don't accept silently).
        let dims = [0usize, 4, 4];
        let a = vec![0.0_f32; 1];
        let b = vec![0.0_f32; 1];
        let err = validate_image_pair(&a, &b, dims).unwrap_err();
        assert!(matches!(err, RegistrationError::DimensionMismatch(_)));
    }

    #[test]
    fn validate_pair_zero_dim_product_with_empty_buffers_is_ok() {
        // Degenerate but consistent: dims product = 0 and both buffers empty.
        // The helper accepts this as a valid (empty) volume.
        let dims = [0usize, 4, 4];
        let empty: Vec<f32> = vec![];
        assert!(validate_image_pair(&empty, &empty, dims).is_ok());
    }

    // ── validate_image (single-image variant used by the atlas) ──────────

    #[test]
    fn validate_image_matching_length_ok() {
        let dims = [4usize, 4, 4];
        let a = vec![0.0_f32; 64];
        assert!(validate_image(&a, dims).is_ok());
    }

    #[test]
    fn validate_image_length_mismatch_err() {
        let dims = [4usize, 4, 4];
        let a = vec![0.0_f32; 63];
        let err = validate_image(&a, dims).unwrap_err();
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
    fn cc_converged_variance_at_threshold_strictly_does_not_converge() {
        // Pins the strict `<` semantics: a window whose variance equals the
        // threshold exactly is NOT yet converged (the engines want the
        // "variance strictly below" check, not `<=`).
        let mut hist = VecDeque::new();
        // Two values with variance exactly 1.0: (0.0, 2.0) → mean = 1.0,
        // variance = 0.5 * ((0-1)² + (2-1)²) = 1.0.
        for _ in 0..3 {
            assert!(!cc_converged(&mut hist, 0.0, 2, 1.0));
        }
        // 4th push brings variance to 1.0 exactly; threshold = 1.0 → not converged.
        assert!(!cc_converged(&mut hist, 2.0, 2, 1.0));
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
